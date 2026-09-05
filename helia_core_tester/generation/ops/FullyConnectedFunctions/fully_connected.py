"""
FullyConnected operation implementation with dtype-aware quantization.
"""

from typing import Dict, Any, Optional
import numpy as np
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.bias_init import SignedMagnitudeUniform
from helia_core_tester.generation.kernel_dispatch import resolve_fully_connected_kernel
from helia_core_tester.core.cpu_targets import get_cpu_profile
import keras
from pathlib import Path


class OpFullyConnected(OperationBase):
    """
    FullyConnected operation.
    """

    def needs_keras_model(self) -> bool:
        return str(self.desc.get("weight_dtype", "S8")).upper() != "S4"
    
    def build_keras_model(self):
        """Build Keras model for FullyConnected operation."""
        input_shape = self.desc['input_shape']
        filter_shape = self.desc['filter_shape']
        
        # Handle both 2D [batch, features] and 4D [batch, h, w, c] input shapes
        if len(input_shape) == 2:
            # 2D input: [batch, features]
            input_features = input_shape[1]
            batch_size = input_shape[0]
            needs_flatten = False
        else:
            # 4D input: [batch, h, w, c]
            # Calculate total features: h * w * c
            input_features = input_shape[1] * input_shape[2] * input_shape[3]
            batch_size = input_shape[0]
            needs_flatten = True
        
        # Extract output units from filter_shape
        if len(filter_shape) == 2:
            output_units = filter_shape[0]
        else:
            output_units = filter_shape[0]
        
        # Get activation and use_bias from descriptor
        activation_str = self.desc.get('activation', 'NONE')
        use_bias = self.desc.get('use_bias', True)
        
        # Build model with input layer matching descriptor shape
        if needs_flatten:
            # 4D input: [batch, h, w, c]
            inputs = keras.layers.Input(shape=input_shape[1:], batch_size=batch_size, name='input')
            # Flatten to [batch, h*w*c]
            x = keras.layers.Flatten()(inputs)
        else:
            # 2D input: [batch, features]
            inputs = keras.layers.Input(shape=(input_features,), batch_size=batch_size, name='input')
            x = inputs
        
        # A zero bias_initializer (Dense's default) produces an all-zero bias
        # tensor, which the TFLite converter's constant-folding optimizer
        # strips from the graph entirely -- the generated CMSIS-NN test then
        # calls the kernel with a NULL bias pointer, leaving the bias-add
        # path completely untested. Use a nonzero uniform bias, deterministic
        # from the case seed, so real bias data flows through the golden and
        # the kernel call.
        #
        # The quantized magnitude has to clear one output quantization step,
        # or a dropped bias-add still reproduces the golden bit for bit. FC
        # sums far fewer terms than conv does, so its calibrated output range
        # is much narrower and the floor scales down with it: every
        # bias-carrying channel of an int FC case clears at least one output
        # step with margin, while the bias costs only a small fraction of the
        # calibrated dynamic range. The float range is unchanged.
        _case_is_float = str(self.tensor_dtype("input", default="S8")).upper() in {"FP32", "FP16"}
        if not use_bias:
            bias_initializer = 'zeros'
        elif _case_is_float:
            bias_initializer = keras.initializers.RandomUniform(minval=-0.25, maxval=0.25, seed=self.seed)
        else:
            bias_initializer = SignedMagnitudeUniform(minval=0.125, maxval=0.25, seed=self.seed)

        # Dense layer without activation (we'll apply activation separately if needed)
        x = keras.layers.Dense(
            output_units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.GlorotUniform(seed=1234),
            bias_initializer=bias_initializer,
            name='dense'
        )(x)
        
        # Apply activation if specified
        if activation_str == 'RELU':
            x = keras.layers.ReLU()(x)
        elif activation_str == 'RELU6':
            x = keras.layers.ReLU(max_value=6)(x)
        elif activation_str == 'TANH':
            x = keras.layers.Activation('tanh')(x)
        elif activation_str == 'SIGMOID':
            x = keras.layers.Activation('sigmoid')(x)
        elif activation_str != 'NONE':
            raise ValueError(f"Unsupported activation: {activation_str}")
        
        model = keras.models.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        weight_dtype = str(self.desc.get("weight_dtype", "S8")).upper()
        if weight_dtype == "S4":
            from helia_core_tester.generation.utils.litert_builder import build_fully_connected_s4_op

            extras = self.desc.get("hint", {}).get("extras", {})
            input_scale = float(extras.get("input_scale", 4.0))
            input_zp = int(extras.get("input_zero_point", 3))
            weight_scale = float(extras.get("weight_scale", 1.0))
            output_scale = float(extras.get("output_scale", 4.0))
            output_zp = int(extras.get("output_zero_point", 0))

            input_shape = tuple(self.desc["input_shape"])
            fs = tuple(self.desc["filter_shape"])  # O, I
            filter_shape = fs
            out_ch = filter_shape[0]

            weights_int4 = self.rng.integers(-8, 8, size=filter_shape).astype(np.int8)
            biases = None
            if self.desc.get("use_bias", True):
                biases = self.rng.integers(-128, 128, size=(out_ch,), dtype=np.int32)

            # Ensure output_scale is compatible with input_scale * weight_scale to satisfy TFLite checks.
            effective_scale = input_scale * weight_scale
            if effective_scale > 0:
                output_scale = float(effective_scale)

            tflite_model = build_fully_connected_s4_op(
                input_shape=input_shape,
                filter_shape=filter_shape,
                use_bias=self.desc.get("use_bias", True),
                input_quant=([input_scale], [input_zp]),
                weight_quant=([weight_scale], [0]),
                output_quant=([output_scale], [output_zp]),
                weights_int4=weights_int4,
                biases=biases,
            )
            with open(out_path, "wb") as f:
                f.write(tflite_model)
            return

        import tensorflow as tf
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization based on activation_dtype
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        elif activation_dtype == 'FP16':
            converter.optimizations = []
            converter.target_spec.supported_types = [tf.float16]
        elif activation_dtype == 'FP32':
            converter.optimizations = []
        
        # Force per-tensor quantization when requested
        force_per_tensor = bool(self.desc.get("hint", {}).get("force_per_tensor", False))
        if force_per_tensor and hasattr(converter, "_experimental_disable_per_channel"):
            converter._experimental_disable_per_channel = True

        # Generate representative dataset
        def representative_data_gen():
            for _ in range(100):
                if 'input_shape' in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc['input_shape']).astype(np.float32)
                    yield [inputs]
                elif 'input_1_shape' in self.desc and 'input_2_shape' in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_1_shape']).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc['input_2_shape']).astype(np.float32)
                    yield [inputs1, inputs2]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_fc_kernel(self) -> Dict[str, str]:
        return resolve_fully_connected_kernel(
            activation_dtype=self.desc.get('activation_dtype', 'S8'),
            weight_dtype=self.desc.get('weight_dtype', 'S8'),
            cpu=self.target_cpu,
        )

    def _find_fully_connected_op_index(self, model: Any, subgraph: Any) -> int:
        """Find the FULLY_CONNECTED operator index in the subgraph (fallback to 0)."""
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")

        try:
            from ai_edge_litert import schema_py_generated as litert

            for op_idx, op in enumerate(subgraph.operators):
                opcode = model.operatorCodes[op.opcodeIndex]
                if opcode.builtinCode == litert.BuiltinOperator.FULLY_CONNECTED:
                    return op_idx
        except Exception:
            pass

        return 0
    
    def _get_zero_point(self, quant_dict: Dict[str, Any]) -> int:
        """Get zero point from quantization dictionary."""
        if quant_dict is None:
            return 0
        zp = quant_dict.get('zero_point', 0)
        if isinstance(zp, (list, np.ndarray)):
            return int(zp[0]) if len(zp) > 0 else 0
        return int(zp)
    
    def _compute_activation_range(self, output_quant: Dict[str, Any], output_dtype: np.dtype) -> tuple[int, int]:
        """Compute activation min/max based on output quantization and dtype."""
        activation_str = self.desc.get('activation', 'NONE')
        output_zp = self._get_zero_point(output_quant)
        output_scale = output_quant.get('scale', 1.0)
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        
        if output_dtype == np.int16:
            default_min, default_max = -32768, 32767
        else:  # int8
            default_min, default_max = -128, 127
        
        if activation_str == 'RELU':
            activation_min = max(0, default_min)
            activation_max = default_max
        elif activation_str == 'RELU6':
            # RELU6: clamp to [0, 6] in float, then quantize
            relu6_max_float = 6.0
            relu6_max_quantized = int(np.round(relu6_max_float / output_scale + output_zp))
            activation_min = max(0, default_min)
            activation_max = min(relu6_max_quantized, default_max)
        else:  # NONE, TANH, SIGMOID, etc.
            activation_min = default_min
            activation_max = default_max
        
        # Override with descriptor values if present
        if 'activation_min' in self.desc:
            activation_min = int(self.desc['activation_min'])
        if 'activation_max' in self.desc:
            activation_max = int(self.desc['activation_max'])
        
        return activation_min, activation_max
    
    def _compute_weight_sum_size(self, weights: Optional[np.ndarray], output_dtype: np.dtype) -> int:
        """Compute the size of the weight sum tensor."""
        if weights is None:
            return 0
        if output_dtype == np.int8 and self._supports_weight_sum():
            # Weight sum size = output_units (number of rows in weight matrix)
            return weights.shape[0] if len(weights.shape) == 2 else 0
        return 0
    
    def _supports_weight_sum(self) -> bool:
        """Check if platform supports weight sum optimization.

        arm_nn_vec_mat_mult_t_s8 reads the precomputed kernel sum only under
        ARM_MATH_MVEI; every other build reads the bias pointer instead. The
        generator folds the bias into the kernel sum and then passes a NULL
        bias, so claiming support on a non-MVE target drops the bias-add.
        """
        return get_cpu_profile(self.target_cpu).has_mve
    
    def _should_precompute_weight_sum(self, weights: Optional[np.ndarray], output_dtype: np.dtype) -> bool:
        """Determine if weight sum should be precomputed."""
        return (
            output_dtype == np.int8
            and self._supports_weight_sum()
            and weights is not None
            and weights.size > 0
        )
    
    def _compute_fixed_point_multipliers(
        self,
        input_scale: float,
        weight_scales: np.ndarray,
        output_scale: float
    ) -> list[Dict[str, Any]]:
        """
        Compute fixed-point multipliers and shifts for each output channel.
        
        Returns:
            List of dictionaries with 'multiplier' and 'shift' keys
        """
        from helia_core_tester.generation.utils.tflite_utils import calculate_per_channel_multiplier_shift
        
        # Compute effective scales: (input_scale * weight_scale) / output_scale
        if isinstance(weight_scales, np.ndarray):
            effective_scales = (input_scale * weight_scales) / output_scale
        else:
            effective_scales = np.array([(input_scale * weight_scales) / output_scale])
        
        multipliers, shifts = calculate_per_channel_multiplier_shift(
            effective_scales,
            reduce_to_q15=False  # Kernel handles reduction internally
        )

        return [
            {'multiplier': int(m), 'shift': int(s)}
            for m, s in zip(multipliers, shifts)
        ]

    def _compute_fc_reference_output_s8(
        self,
        input_q: np.ndarray,
        weights: np.ndarray,
        biases: Optional[np.ndarray],
        quant_params: Dict[str, Any],
        fc_params: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute an s8 fully-connected reference output using CMSIS-style arithmetic.

        This is used when descriptor overrides (e.g. force_filter_offset) intentionally
        diverge from the TFLite model quantization, so LiteRT inference can no longer
        be used as the expected output source.
        """
        from helia_core_tester.generation.utils.tflite_utils import requantize_np

        if weights is None:
            raise ValueError("Weights are required to compute fully connected reference output")

        # Flatten input to [batch, features] to match FC kernel expectations.
        in_features = int(weights.shape[1])
        input_2d = input_q.reshape(input_q.shape[0], in_features).astype(np.int32)
        weights_2d = weights.astype(np.int32)

        input_offset = int(fc_params["input_offset"])
        filter_offset = int(fc_params["filter_offset"])
        output_offset = int(fc_params["output_offset"])
        activation_min = int(fc_params["activation_min"])
        activation_max = int(fc_params["activation_max"])

        # Accumulate: sum((input + input_offset) * (weight + filter_offset))
        accum = (input_2d + input_offset) @ (weights_2d + filter_offset).T

        if biases is not None and biases.size > 0:
            accum = accum + np.asarray(biases, dtype=np.int32).reshape(1, -1)

        if quant_params.get("per_channel", False):
            multipliers = np.asarray(quant_params["multiplier"], dtype=np.int32)
            shifts = np.asarray(quant_params["shift"], dtype=np.int32)
            requantized = np.empty_like(accum, dtype=np.int32)
            for ch in range(accum.shape[1]):
                requantized[:, ch] = requantize_np(
                    accum[:, ch], int(multipliers[ch]), int(shifts[ch])
                )
        else:
            requantized = requantize_np(
                accum,
                int(quant_params["multiplier"]),
                int(quant_params["shift"]),
            )

        requantized = requantized + output_offset
        requantized = np.clip(requantized, activation_min, activation_max)
        return requantized.astype(np.int8)
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for FullyConnected operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_fc_kernel()
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        fc_op_index = self._find_fully_connected_op_index(model, subgraph)
        op_tensors = get_operator_tensors_from_litert(model, subgraph, fc_op_index)
        
        # Extract shapes from LiteRT
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        # Extract quantization parameters from LiteRT
        input_quant_litert = op_tensors['inputs'][0]['quantization']
        output_quant_litert = op_tensors['outputs'][0]['quantization']
        
        input_quant = {
            'scale': input_quant_litert.get('scale', 1.0),
            'zero_point': input_quant_litert.get('zero_point', 0),
            'per_channel': input_quant_litert.get('per_channel', False)
        }
        output_quant = {
            'scale': output_quant_litert.get('scale', 1.0),
            'zero_point': output_quant_litert.get('zero_point', 0),
            'per_channel': output_quant_litert.get('per_channel', False)
        }
        
        # Extract weights and biases for the actual FULLY_CONNECTED op.
        # For 4D inputs, op[0] may be RESHAPE, so using operator index 0 can be wrong.
        weights = op_tensors.get('weights')
        biases = op_tensors.get('biases')
        biases_for_reference = (
            np.asarray(biases, dtype=np.int32).copy()
            if biases is not None and biases.size > 0
            else None
        )
        
        # Get weight quantization from LiteRT
        from helia_core_tester.generation.utils.litert_utils import (
            get_tensor_data_from_litert, get_tensor_quantization_from_litert,
            get_tensor_shape_from_litert
        )
        
        weight_quant = None
        if weights is not None:
            # Search all tensors to find the one matching our weights
            input_indices = set(subgraph.inputs)
            output_indices = set(subgraph.outputs)
            
            for tensor_idx, tensor in enumerate(subgraph.tensors):
                if tensor_idx in input_indices or tensor_idx in output_indices:
                    continue
                
                tensor_data = get_tensor_data_from_litert(tensor, model)
                tensor_shape = get_tensor_shape_from_litert(tensor)
                
                if (tensor_data is not None and tensor_shape is not None and 
                    len(tensor_shape) > 1 and tensor_data.shape == weights.shape and
                    np.array_equal(tensor_data, weights)):
                    weight_quant = get_tensor_quantization_from_litert(tensor)
                    break
            
            # Fallback: check operator inputs for weight tensor
            if weight_quant is None:
                for input_tensor_info in op_tensors['inputs']:
                    if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                        weight_quant = input_tensor_info.get('quantization')
                        break
        
        # Prepare weight quantization dict
        if weight_quant is None:
            # No weight-tensor quantization could be recovered from the
            # converted model. Silently substituting the output tensor's
            # quantization would produce incorrect per-channel/per-tensor
            # weight scales and a wrong multiplier, potentially masking a real
            # kernel/golden mismatch. Fail loudly instead.
            raise RuntimeError(
                f"FullyConnected descriptor '{name}' weight quantization could "
                "not be recovered from the converted TFLite model; refusing to "
                "substitute unrelated output quantization"
            )
        
        weight_quant_dict = {
            'scale': weight_quant.get('scale', 1.0),
            'zero_point': weight_quant.get('zero_point', 0),
            'per_channel': weight_quant.get('per_channel', False)
        }
        
        weight_dtype = str(self.desc.get("weight_dtype", "S8")).upper()

        # Validate weights shape
        if weight_dtype == "S4":
            fs = tuple(self.desc['filter_shape'])
            if len(fs) != 2:
                raise ValueError(f"Unsupported filter_shape in descriptor: {fs}")
            filter_dims = {
                'n': int(fs[1]),  # input_features
                'h': 1,
                'w': 1,
                'c': int(fs[0])   # output_units
            }
        elif weights is not None:
            filter_shape = tuple(weights.shape)
            if len(filter_shape) == 1:
                # Try to infer 2D shape
                if len(output_shape) == 2:
                    output_units = output_shape[1]
                    input_features = filter_shape[0] // output_units if filter_shape[0] % output_units == 0 else filter_shape[0]
                    if filter_shape[0] == output_units * input_features:
                        weights = weights.reshape(output_units, input_features)
                        filter_shape = tuple(weights.shape)
                    else:
                        raise ValueError(f"Cannot infer 2D shape from 1D weights shape {filter_shape}")
                else:
                    raise ValueError(f"Unsupported filter shape: {filter_shape} (1D)")
            
            if len(filter_shape) != 2:
                raise ValueError(f"Unsupported filter shape: {filter_shape}")
            
            if not float_kernel and weights.dtype != np.int8:
                weights = weights.astype(np.int8)
            
            # TFLite format: [output_units, input_features]
            filter_dims = {
                'n': int(filter_shape[1]),  # input_features (col_dim)
                'h': 1,
                'w': 1,
                'c': int(filter_shape[0])   # output_units (row_dim)
            }
        else:
            # Fallback: descriptor format
            fs = tuple(self.desc['filter_shape'])
            if len(fs) != 2:
                raise ValueError(f"Unsupported filter_shape in descriptor: {fs}")
            filter_dims = {
                'n': int(fs[1]),  # input_features
                'h': 1,
                'w': 1,
                'c': int(fs[0])   # output_units
            }
        
        builder = TemplateContextBuilder()
        
        # Compute input dimensions
        if len(input_shape) == 2:
            input_dims = {
                'n': int(input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1])
            }
        elif len(input_shape) == 4:
            # Flatten: [batch, h, w, c] -> features = h * w * c
            input_dims = {
                'n': int(input_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(input_shape[1] * input_shape[2] * input_shape[3])
            }
        else:
            input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Compute output dimensions - use weights shape to get correct output_units
        if weight_dtype == "S4":
            fs = tuple(self.desc['filter_shape'])
            batch_size = int(output_shape[0]) if len(output_shape) >= 1 else int(input_shape[0])
            output_dims = {
                'n': batch_size,
                'h': 1,
                'w': 1,
                'c': int(fs[0])
            }
        elif weights is not None and len(weights.shape) == 2:
            correct_output_units = int(weights.shape[0])
            batch_size = int(output_shape[0]) if len(output_shape) >= 1 else int(input_shape[0])

            if len(output_shape) == 2 and output_shape[1] != correct_output_units:
                # The converter/LiteRT-reported output shape disagrees with the
                # weight tensor's output-unit dimension. Silently rewriting
                # output_dims from the weight shape would hide a real
                # converter/kernel contract error behind an auto-corrected
                # harness. Fail loudly instead.
                raise RuntimeError(
                    f"FullyConnected descriptor '{name}' has LiteRT "
                    f"output_shape[1] ({output_shape[1]}) that disagrees with "
                    f"weights.shape[0] ({correct_output_units}); refusing to "
                    "silently override output dims"
                )

            output_dims = {
                'n': batch_size,
                'h': 1,
                'w': 1,
                'c': correct_output_units
            }
        elif len(output_shape) == 2:
            output_dims = {
                'n': int(output_shape[0]),
                'h': 1,
                'w': 1,
                'c': int(output_shape[1])
            }
        else:
            output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        float_kernel = kernel_info["input_c_type"] in {"float", "float16_t"}
        if float_kernel:
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            if weights is not None and weights.dtype != float_dtype:
                weights = weights.astype(float_dtype)
            has_biases = biases is not None and biases.size > 0
            if has_biases and biases.dtype != float_dtype:
                biases = biases.astype(float_dtype)

            fc_params = builder.build_fc_params(
                self.desc,
                input_quant,
                weight_quant_dict,
                output_quant,
            )

            input_data = np.asarray(self._sample_uniform(input_shape), dtype=float_dtype)
            from helia_core_tester.generation.utils.litert_utils import run_inference_litert
            interpreter_input_dtype = self.load_litert_interpreter(str(tflite_path)).get_input_details()[0]['dtype']
            output_data = run_inference_litert(
                str(tflite_path),
                input_data.astype(interpreter_input_dtype),
                subgraph_index=0,
            )

            weights_array_str = builder.format_array_as_c_literal(weights) if weights is not None else ""
            biases_array_str = builder.format_array_as_c_literal(biases) if has_biases else ""
            input_data_array_str = builder.format_array_as_c_literal(np.asarray(input_data, dtype=float_dtype).flatten())
            expected_output_array_str = builder.format_array_as_c_literal(np.asarray(output_data, dtype=float_dtype).flatten())
            element_size = np.dtype(float_dtype).itemsize
            buffer_size_max = max(
                1024,
                int(
                    (
                        input_dims['n'] * input_dims['c']
                        + filter_dims['n'] * filter_dims['c']
                        + output_dims['n'] * output_dims['c']
                    ) * element_size
                ),
            )

            context = {
                'name': name,
                'input_dims': input_dims,
                'filter_dims': filter_dims,
                'output_dims': output_dims,
                'fc_params': fc_params,
                'weights_array': weights_array_str,
                'biases_array': biases_array_str,
                'has_biases': has_biases,
                'has_bias_array': has_biases,
                'input_data_array': input_data_array_str,
                'expected_output_array': expected_output_array_str,
                'input_dtype': kernel_info["input_c_type"],
                'output_dtype': kernel_info["output_c_type"],
                'weight_dtype': kernel_info.get("weight_c_type", kernel_info["input_c_type"]),
                'bias_dtype': kernel_info["bias_c_type"],
                'kernel_fn': kernel_info["kernel_fn"],
                'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
                'call_style': kernel_info.get("call_style", "baseline"),
                'buffer_size_max': buffer_size_max,
                'weight_sum_array': "",
                'has_weight_sum': False,
                'float_kernel': True,
                'fc_params_type': kernel_info.get("fc_params_type", 'cmsis_nn_fc_params_f32'),
                'kernel_layout': kernel_info.get("layout", "ARM_NN_LAYOUT_NHWC"),
                'fc_activation_min_literal': builder.format_float_literal(fc_params['activation_min']),
                'fc_activation_max_literal': builder.format_float_literal(fc_params['activation_max']),
                'validation_mode': 'float',
            }

            includes_api_dir = output_dir / "includes"
            includes_api_dir.mkdir(parents=True, exist_ok=True)
            
            h_content = self.render_template("FullyConnectedFunctions/fully_connected/fully_connected.h.j2", context)
            h_path = includes_api_dir / f"{name}_fully_connected.h"
            with open(h_path, 'w') as f:
                f.write(h_content)
            
            c_content = self.render_template("FullyConnectedFunctions/fully_connected/fully_connected.c.j2", context)
            c_path = output_dir / f"{name}_fully_connected.c"
            with open(c_path, 'w') as f:
                f.write(c_content)
            
            cmake_context = {
                'name': name,
                'operator': self.desc.get('operator', 'FullyConnected'),
                'operator_name': 'fully_connected'
            }
            cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
            cmake_path = output_dir / "CMakeLists.txt"
            with open(cmake_path, 'w') as f:
                f.write(cmake_content)
            return
        
        # Get scales as arrays for per-channel computation
        input_scale = input_quant['scale']
        if isinstance(input_scale, (list, np.ndarray)):
            input_scale = float(input_scale[0])
        else:
            input_scale = float(input_scale)
        
        weight_scale = weight_quant_dict['scale']
        output_scale = output_quant['scale']
        force_per_tensor = bool(self.desc.get("hint", {}).get("force_per_tensor", False))
        per_channel = bool(weight_quant_dict.get('per_channel', False)) and not force_per_tensor
        
        # Convert scales to numpy arrays for per-channel computation
        if per_channel and isinstance(weight_scale, (list, np.ndarray)):
            weight_scales = np.array(weight_scale, dtype=np.float64)
        else:
            if isinstance(weight_scale, (list, np.ndarray)):
                weight_scales = np.array([float(weight_scale[0])], dtype=np.float64)
            else:
                weight_scales = np.array([float(weight_scale)], dtype=np.float64)
        
        if isinstance(output_scale, (list, np.ndarray)):
            output_scale = float(output_scale[0])
        else:
            output_scale = float(output_scale)
        
        # Compute fixed-point multipliers and shifts
        outputs_fp = self._compute_fixed_point_multipliers(
            input_scale,
            weight_scales,
            output_scale
        )
        
        # Build quantization parameters dict
        if per_channel and len(outputs_fp) > 1:
            multipliers = [fp['multiplier'] for fp in outputs_fp]
            shifts = [fp['shift'] for fp in outputs_fp]
            quant_params_dict = {
                'multiplier': multipliers,
                'shift': shifts,
                'multiplier_array': ', '.join(map(str, multipliers)),
                'shift_array': ', '.join(map(str, shifts)),
                'per_channel': True
            }
        else:
            # Per-tensor
            if len(outputs_fp) > 0:
                multiplier = outputs_fp[0]['multiplier']
                shift = outputs_fp[0]['shift']
            else:
                # Fallback calculation
                effective_scale = (input_scale * float(weight_scales[0])) / output_scale
                from helia_core_tester.generation.utils.tflite_utils import calculate_per_channel_multiplier_shift
                mults, shfts = calculate_per_channel_multiplier_shift(
                    np.array([effective_scale]),
                    reduce_to_q15=False
                )
                multiplier = int(mults[0])
                shift = int(shfts[0])

            quant_params_dict = {
                'multiplier': multiplier,
                'shift': shift,
                'per_channel': False
            }
        
        # Compute activation range
        output_dtype = np.int16 if kernel_info["output_c_type"] == "int16_t" else np.int8
        activation_min, activation_max = self._compute_activation_range(output_quant, output_dtype)
        
        # Build FC parameters
        input_zp = self._get_zero_point(input_quant)
        weight_zp = self._get_zero_point(weight_quant_dict)
        output_zp = self._get_zero_point(output_quant)
        extras = self.desc.get("hint", {}).get("extras", {})
        filter_offset_override = extras.get("force_filter_offset")
        if filter_offset_override is not None:
            filter_offset_override = int(filter_offset_override)

        activation_dtype = self.desc.get('activation_dtype', 'S8')
        weight_dtype = str(self.desc.get("weight_dtype", "S8")).upper()
        if activation_dtype == 'S16':
            # S16 uses symmetric quantization (zero points are 0)
            fc_params = {
                'input_offset': 0,
                'filter_offset': 0,
                'output_offset': 0,
                'activation_min': activation_min,
                'activation_max': activation_max,
            }
        elif weight_dtype == "S4":
            fc_params = {
                'input_offset': int(-input_zp),
                'filter_offset': 0,
                'output_offset': int(output_zp),
                'activation_min': activation_min,
                'activation_max': activation_max,
            }
        else:
            # S8 uses zero points as offsets
            fc_params = {
                'input_offset': int(-input_zp),
                'filter_offset': int(filter_offset_override) if filter_offset_override is not None else int(-weight_zp),
                'output_offset': int(output_zp),
                'activation_min': activation_min,
                'activation_max': activation_max,
            }
        
        # Compute weight sum if needed
        weight_sum = None
        has_weight_sum = False
        folded_bias = None
        if weight_dtype != "S4" and self._should_precompute_weight_sum(weights, output_dtype):
            from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import vector_sum_s8
            
            vector_rows = weights.shape[0]  # output_units
            vector_cols = weights.shape[1]  # input_features

            lhs_offset = -input_zp
            rhs_offset = int(filter_offset_override) if filter_offset_override is not None else -weight_zp
            
            bias_data = None
            if biases is not None and biases.size > 0:
                if biases.dtype != np.int32:
                    bias_data = biases.astype(np.int32)
                else:
                    bias_data = biases
            
            weight_sum = vector_sum_s8(
                vector_data=weights,
                vector_cols=vector_cols,
                vector_rows=vector_rows,
                lhs_offset=lhs_offset,
                rhs_offset=rhs_offset,
                bias_data=bias_data,
            ).astype(np.int32)
            
            has_weight_sum = True
            
            # If weight_sum is precomputed, biases are consumed into it
            if bias_data is not None:
                folded_bias = bias_data
                biases = None
        
        # Format arrays
        if weight_dtype == "S4":
            from helia_core_tester.generation.utils.litert_utils import get_tensor_data_packed_from_litert
            packed_weights = None
            for input_tensor_info in op_tensors['inputs']:
                if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                    packed_weights = get_tensor_data_packed_from_litert(input_tensor_info['tensor'], model)
                    break
            if packed_weights is None:
                raise ValueError("Packed S4 weights not found in LiteRT model")
            weights = packed_weights.astype(np.int8)
        weights_array_str = builder.format_array_as_c_literal(weights) if weights is not None else ""
        
        has_biases = False
        biases_array_str = ""
        if not has_weight_sum and weight_dtype != "S4":
            has_biases = biases is not None and biases.size > 0
            if has_biases:
                # Convert biases to appropriate type
                if kernel_info["bias_c_type"] == "int64_t":
                    if biases.dtype != np.int64:
                        biases = biases.astype(np.int64)
                else:  # int32_t
                    if biases.dtype != np.int32:
                        biases = biases.astype(np.int32)
                biases_array_str = builder.format_array_as_c_literal(biases)
        elif weight_dtype == "S4":
            has_biases = biases is not None and biases.size > 0
            if has_biases and biases.dtype != np.int32:
                biases = biases.astype(np.int32)
            if has_biases:
                biases_array_str = builder.format_array_as_c_literal(biases)

        # A bias folded into the kernel sum still has to appear as an array in
        # the header. The kernel keeps taking a NULL bias pointer, but the
        # perf-stream bridge reads the bias back out of the header decl, and a
        # NULL decl is indistinguishable there from a zero bias, so the bridge
        # would rebuild the kernel sum without the bias term.
        has_bias_array = has_biases
        if has_weight_sum and folded_bias is not None:
            biases_array_str = builder.format_array_as_c_literal(folded_bias.astype(np.int32))
            has_bias_array = True

        weight_sum_array_str = builder.format_array_as_c_literal(weight_sum) if weight_sum is not None else ""
        
        # Generate input data and run inference
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.generate_input_data()
        self.rng.__setstate__(rng_state)
        
        # Quantize input - keep original shape for inference (model has Flatten layer)
        input_q = np.round(input_data / input_scale + input_zp).astype(np.int32)
        if kernel_info["input_c_type"] == "int8_t":
            input_q = np.clip(input_q, -128, 127).astype(np.int8)
        else:  # int16_t
            input_q = np.clip(input_q, -32768, 32767).astype(np.int16)
        
        # Run inference using LiteRT; fall back to numpy for S4 if LiteRT rejects scales.
        from helia_core_tester.generation.utils.litert_utils import run_inference_litert
        output_data = None
        if weight_dtype == "S4":
            try:
                output_data = run_inference_litert(str(tflite_path), input_q, subgraph_index=0)
            except Exception:
                output_data = None
        else:
            output_data = run_inference_litert(str(tflite_path), input_q, subgraph_index=0)

        # LiteRT golden output does not include descriptor-side filter offset overrides.
        # Recompute expected output with CMSIS arithmetic when forced RHS offset is used.
        if (
            weight_dtype != "S4"
            and kernel_info["output_c_type"] == "int8_t"
            and filter_offset_override is not None
        ):
            output_data = self._compute_fc_reference_output_s8(
                input_q=input_q,
                weights=weights,
                biases=biases_for_reference,
                quant_params=quant_params_dict,
                fc_params=fc_params,
            )

        if output_data is None and weight_dtype == "S4":
            from helia_core_tester.generation.utils.tflite_utils import requantize_np
            # Use unpacked weights from LiteRT extraction
            weights_unpacked = op_tensors['weights']
            if weights_unpacked is None:
                raise ValueError("Unpacked S4 weights not found for fallback inference")
            if weights_unpacked.dtype != np.int8:
                weights_unpacked = weights_unpacked.astype(np.int8)
            if biases is None:
                biases = np.zeros((weights_unpacked.shape[0],), dtype=np.int32)
            if biases.dtype != np.int32:
                biases = biases.astype(np.int32)
            multiplier = int(quant_params_dict['multiplier'])
            shift = int(quant_params_dict['shift'])
            input_offset = int(fc_params['input_offset'])
            output_offset = int(fc_params['output_offset'])
            batch = input_q.shape[0]
            out_ch = weights_unpacked.shape[0]
            in_feat = weights_unpacked.shape[1]
            inp_flat = input_q.reshape(batch, in_feat).astype(np.int32)
            out = np.zeros((batch, out_ch), dtype=np.int32)
            for b in range(batch):
                acc = (inp_flat[b] + input_offset).astype(np.int32)
                # (out_ch, in_feat) dot (in_feat,)
                prod = weights_unpacked.astype(np.int32) @ acc
                prod = prod + biases
                requant = requantize_np(prod, multiplier, shift)
                requant = requant + output_offset
                requant = np.clip(requant, -128, 127).astype(np.int8)
                out[b, :] = requant.astype(np.int32)
            output_data = out.astype(np.int8)
        
        # Format arrays - format_array_as_c_literal automatically flattens
        input_data_array_str = builder.format_array_as_c_literal(input_q.flatten())
        # Ensure output_data is properly shaped and flattened for C array
        if kernel_info["output_c_type"] == "int16_t":
            expected_output_array_str = builder.format_array_as_c_literal(output_data.flatten().astype(np.int16))
        else:
            expected_output_array_str = builder.format_array_as_c_literal(output_data.flatten().astype(np.int8))
        
        # Calculate buffer size max
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        is_s16 = (activation_dtype == 'S16' or kernel_info["input_c_type"] == "int16_t")
        
        if is_s16 and quant_params_dict.get('per_channel', False):
            buffer_size_max = output_dims['c'] * 4  # sizeof(int32_t) = 4
        else:
            if weight_dtype == "S4":
                buffer_size_max = 0
            else:
                buffer_size_max = builder.calculate_fc_buffer_size_max(
                    filter_dims,
                    output_dtype=activation_dtype
                )
        
        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'filter_dims': filter_dims,
            'output_dims': output_dims,
            'fc_params': fc_params,
            'quant_params': quant_params_dict,
            'weights_array': weights_array_str,
            'biases_array': biases_array_str,
            'has_biases': has_biases,
            'has_bias_array': has_bias_array,
            'input_data_array': input_data_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'bias_dtype': kernel_info["bias_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'kernel_get_buffer_size_fn': kernel_info["kernel_get_buffer_size_fn"],
            'call_style': kernel_info.get("call_style", "baseline"),
            'buffer_size_max': buffer_size_max,
            'weight_sum_array': weight_sum_array_str,
            'has_weight_sum': has_weight_sum,
        }
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("FullyConnectedFunctions/fully_connected/fully_connected.h.j2", context)
        h_path = includes_api_dir / f"{name}_fully_connected.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("FullyConnectedFunctions/fully_connected/fully_connected.c.j2", context)
        c_path = output_dir / f"{name}_fully_connected.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'FullyConnected'),
            'operator_name': 'fully_connected'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
