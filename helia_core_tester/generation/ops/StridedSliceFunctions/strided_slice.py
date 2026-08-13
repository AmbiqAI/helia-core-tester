"""
StridedSlice operation implementation.
"""

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpStridedSlice(OperationBase):
    """
    StridedSlice operation.
    """

    def needs_keras_model(self) -> bool:
        # Float dtypes use a LiteRT-only single-op model (Keras/TFLiteConverter
        # has no reliable float activation path for single-op slices);
        # quantized dtypes keep the existing Keras-based pipeline.
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        return activation_dtype not in ('FP16', 'FP32')

    def build_keras_model(self) -> tf.keras.Model:
        """Build Keras model for StridedSlice operation."""
        input_shape = self.desc['input_shape']
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        input_dtype = tf.int32 if activation_dtype == 'S32' else tf.float32
        inputs = tf.keras.Input(
            shape=input_shape[1:],
            batch_size=input_shape[0],
            dtype=input_dtype,
            name='input',
        )
        
        # Get begin, end, and strides from descriptor
        begin = self.desc.get('begin', [0, 0, 0, 0])
        end = self.desc.get('end', None)
        strides = self.desc.get('strides', [1, 1, 1, 1])
        shrink_axis_mask = self.desc.get('shrink_axis_mask', 0)
        
        # If end is not provided, use default (all dimensions)
        if end is None:
            end = [-1] * len(begin)
        
        # StridedSlice operation
        def strided_slice_op(x):
            return tf.strided_slice(
                x,
                begin=begin,
                end=end,
                strides=strides,
                shrink_axis_mask=shrink_axis_mask
            )
        
        x = tf.keras.layers.Lambda(strided_slice_op, name='strided_slice')(inputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """Convert Keras model to TFLite with quantization."""
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()

        if activation_dtype in ('FP16', 'FP32'):
            from helia_core_tester.generation.utils.litert_builder import build_strided_slice_op

            input_shape = tuple(self.desc['input_shape'])
            begin = self.desc.get('begin', [0, 0, 0, 0])
            end = self.desc.get('end', None)
            strides = self.desc.get('strides', [1, 1, 1, 1])
            shrink_axis_mask = int(self.desc.get('shrink_axis_mask', 0))
            if end is None:
                end = list(input_shape)

            model_bytes = build_strided_slice_op(
                input_shape=input_shape,
                begin=begin,
                end=end,
                strides=strides,
                shrink_axis_mask=shrink_axis_mask,
                dtype="float16" if activation_dtype == 'FP16' else "float32",
            )
            with open(out_path, "wb") as f:
                f.write(model_bytes)
            return

        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply quantization based on activation_dtype
        if activation_dtype == 'S8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            # Ensure all operations use int8 (including StridedSlice)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        elif activation_dtype == 'S16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        elif activation_dtype == 'S32':
            converter.optimizations = []
        else:
            raise NotImplementedError(f"Unsupported StridedSlice dtype: {activation_dtype}")
        
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
        
        if activation_dtype in {'S8', 'S16'}:
            converter.representative_dataset = representative_data_gen
        
        # Convert and save
        tflite_model = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    
    def _select_cmsis_strided_slice_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for StridedSlice operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = str(self.desc.get('activation_dtype', 'S8')).upper()
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_strided_slice_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t'
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_strided_slice_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t'
            }
        elif activation_dtype == 'S32':
            return {
                'kernel_fn': 'arm_strided_slice_s32',
                'input_c_type': 'int32_t',
                'output_c_type': 'int32_t'
            }
        elif activation_dtype == 'FP32':
            return {
                'kernel_fn': 'arm_strided_slice_f32',
                'input_c_type': 'float',
                'output_c_type': 'float'
            }
        elif activation_dtype == 'FP16':
            return {
                'kernel_fn': 'arm_strided_slice_f16',
                'input_c_type': 'float16_t',
                'output_c_type': 'float16_t'
            }
        else:
            raise NotImplementedError(f"Unsupported StridedSlice dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for StridedSlice operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_strided_slice_kernel()
        
        # Load LiteRT model for shape extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT
        input_shape = op_tensors['inputs'][0]['shape']
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input_shape is not None:
            input_shape = tuple(input_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Handle shrink_axis_mask: when dimensions are shrunk, TFLite reduces the rank
        # but CMSIS-NN still expects 4D, so we need to reconstruct the 4D shape
        shrink_axis_mask = self.desc.get('shrink_axis_mask', 0)
        if shrink_axis_mask != 0:
            # Reconstruct 4D output shape from reduced-rank TFLite output
            # by inserting size-1 dimensions where axes were shrunk
            output_shape_4d = list(input_shape)  # Start with input shape
            output_rank = len(output_shape)
            input_rank = len(input_shape)
            
            # Determine which dimensions were shrunk
            shrunk_dims = []
            for i in range(min(4, input_rank)):
                if shrink_axis_mask & (1 << i):
                    shrunk_dims.append(i)
                    output_shape_4d[i] = 1  # Shrunk dimension becomes size 1
            
            # Now map the TFLite output shape back to 4D
            # TFLite output has reduced rank, we need to map it correctly
            output_idx = 0
            for i in range(4):
                if i not in shrunk_dims:
                    if output_idx < output_rank:
                        output_shape_4d[i] = output_shape[output_idx]
                        output_idx += 1
                # else: already set to 1 above
            
            output_dims = builder.nhwc_to_cmsis_dims(tuple(output_shape_4d))
        else:
            output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        # Extract begin, end, and strides from descriptor
        begin = self.desc.get('begin', [0, 0, 0, 0])
        end = self.desc.get('end', None)
        strides = self.desc.get('strides', [1, 1, 1, 1])
        
        # Ensure begin and strides are 4D (NHWC)
        if len(begin) < 4:
            begin = list(begin) + [0] * (4 - len(begin))
        if len(strides) < 4:
            strides = list(strides) + [1] * (4 - len(strides))
        
        # Normalize negative begin indices (e.g., -1 means last element)
        begin_normalized = []
        for i, b in enumerate(begin[:len(input_shape)]):
            if b < 0:
                # Negative index: count from the end
                begin_normalized.append(input_shape[i] + b)
            else:
                begin_normalized.append(b)
        # Pad to 4D if needed
        while len(begin_normalized) < 4:
            begin_normalized.append(0)
        
        # Convert to CMSIS dims format (NHWC -> N, H, W, C)
        begin_dims = builder.nhwc_to_cmsis_dims(begin_normalized[:4])
        stride_dims = builder.nhwc_to_cmsis_dims(strides[:4])
        
        # Generate input data and quantize when required.
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] in ("float16_t", "float"):
            # StridedSlice is pure data movement, so the golden output can be
            # computed directly via numpy slicing without invoking a TFLite
            # interpreter (which has no reliable float activation path).
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            input_q = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(float_dtype)
        elif kernel_info["input_c_type"] == "int32_t":
            input_q = self.rng.integers(-1000, 1001, size=input_shape, dtype=np.int32)
        else:
            input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)

            # Extract quantization from LiteRT
            input_quant = op_tensors['inputs'][0]['quantization']
            input_scale = input_quant.get('scale', 1.0)
            input_zp = input_quant.get('zero_point', 0)

            # Handle per-channel quantization (convert to scalar)
            if isinstance(input_scale, (list, np.ndarray)):
                input_scale = float(input_scale[0]) if len(input_scale) > 0 else 1.0
            if isinstance(input_zp, (list, np.ndarray)):
                input_zp = int(input_zp[0]) if len(input_zp) > 0 else 0

            input_scale = float(input_scale)
            input_zp = int(input_zp)

            # Quantize inputs
            if kernel_info["input_c_type"] == "int8_t":
                np_in_dtype = np.int8
                qmin, qmax = -128, 127
            elif kernel_info["input_c_type"] == "int16_t":
                np_in_dtype = np.int16
                qmin, qmax = -32768, 32767
            else:
                raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

            input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)

        self.rng.__setstate__(rng_state)

        if kernel_info["input_c_type"] in ("float16_t", "float"):
            end_resolved = list(end) if end is not None else list(input_shape)
            while len(end_resolved) < len(input_shape):
                end_resolved.append(input_shape[len(end_resolved)])
            slices = tuple(
                slice(begin_normalized[i], end_resolved[i], strides[i])
                for i in range(len(input_shape))
            )
            output_data = input_q[slices]
            if shrink_axis_mask:
                squeeze_axes = tuple(
                    i for i in range(len(input_shape)) if shrink_axis_mask & (1 << i)
                )
                output_data = np.squeeze(output_data, axis=squeeze_axes)
            output_data = np.array(output_data).astype(
                np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            )
        else:
            # Run inference using LiteRT interpreter
            interpreter = self.load_litert_interpreter(str(tflite_path))
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], input_q)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.array(output_data)

            # Convert output_data to the expected dtype before formatting.
            if kernel_info["output_c_type"] == "int32_t":
                output_data = output_data.astype(np.int32)
            elif kernel_info["output_c_type"] == "int16_t":
                output_data = output_data.astype(np.int16)
            else:  # int8_t
                output_data = output_data.astype(np.int8)
        
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'prefix': name,
            'input_dims': input_dims,
            'output_dims': output_dims,
            'begin_dims': begin_dims,
            'stride_dims': stride_dims,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        if kernel_info["input_c_type"] in ("float16_t", "float"):
            context["validation_mode"] = "float"
        
        # Render templates
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        
        h_content = self.render_template("StridedSliceFunctions/strided_slice/strided_slice.h.j2", context)
        h_path = includes_api_dir / f"{name}_strided_slice.h"
        with open(h_path, 'w') as f:
            f.write(h_content)
        
        c_content = self.render_template("StridedSliceFunctions/strided_slice/strided_slice.c.j2", context)
        c_path = output_dir / f"{name}_strided_slice.c"
        with open(c_path, 'w') as f:
            f.write(c_content)
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'StridedSlice'),
            'operator_name': 'strided_slice'
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated C/H files and CMakeLists.txt for {name}")
