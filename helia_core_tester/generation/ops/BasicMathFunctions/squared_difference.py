"""SquaredDifference operation implementation."""

from pathlib import Path
from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase
from helia_core_tester.generation.utils.litert_builder import build_binary_broadcast_op
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Storage dtype -> kernel suffix of the flat float entry point.
FLOAT_KERNEL_SUFFIX = {"FP16": "f16", "FP32": "f32"}

# The s8 inputs carry a moderate asymmetric zero point rather than the -128 the
# output legitimately uses (squared difference being non-negative). -128 would
# pin every input lane to a non-negative post-offset value and hide the
# sign-dependent kernel paths, while 0 would leave the input offset term dead in
# every s8 case; -40 does neither (hct#81).
SQUARED_DIFFERENCE_QUANT_PRESETS = {
    "int8": {
        "input_1_quant": ([1.0 / 128.0], [-40]),
        "input_2_quant": ([1.0 / 256.0], [-40]),
        "output_quant": ([1.0 / 64.0], [-128]),
    },
    "int16": {
        "input_1_quant": ([1.0 / 32768.0], [0]),
        "input_2_quant": ([1.0 / 65536.0], [0]),
        "output_quant": ([1.0 / 32768.0], [0]),
    },
}


class QuantizedSquaredDifference(layers.Layer):
    """SquaredDifference followed by int16 fake-quant simulation."""

    def __init__(self, min_val: float = -32768.0, max_val: float = 32767.0, **kwargs):
        super().__init__(**kwargs)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def call(self, inputs):
        tensor_a, tensor_b = inputs
        sq_diff = tf.math.squared_difference(tensor_a, tensor_b)
        return tf.quantization.fake_quant_with_min_max_vars(
            sq_diff,
            min=self.min_val,
            max=self.max_val,
            num_bits=16,
            narrow_range=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"min_val": self.min_val, "max_val": self.max_val})
        return config


# A relu-shaped input range: the zero point sits at the bottom of the domain, so
# every post-offset lane is non-negative. Two s8 cases keep it deliberately, to
# hold the regime the upstream reference vectors for this kernel were captured
# in; they declare operand_sign_span_exempt for it.
SQUARED_DIFFERENCE_QUANT_PRESET_VARIANTS = {
    "relu_range": {
        "int8": {
            "input_1_quant": ([1.0 / 128.0], [-128]),
            "input_2_quant": ([1.0 / 256.0], [-128]),
            "output_quant": ([1.0 / 64.0], [-128]),
        },
    },
}


def squared_difference_quant_preset(dtype: str, quant_preset: str = "default") -> dict:
    """Quantization preset for a SquaredDifference case, by dtype and variant."""
    if quant_preset == "default":
        return SQUARED_DIFFERENCE_QUANT_PRESETS[dtype]
    try:
        return SQUARED_DIFFERENCE_QUANT_PRESET_VARIANTS[quant_preset][dtype]
    except KeyError:
        raise ValueError(
            f"SquaredDifference has no '{quant_preset}' quantization preset for {dtype}; "
            f"variants: {sorted(SQUARED_DIFFERENCE_QUANT_PRESET_VARIANTS)}"
        ) from None


def build_squared_difference_op(
    *, input_1_shape, input_2_shape, dtype: str = "int8", quant_preset: str = "default"
) -> bytes:
    quant = squared_difference_quant_preset(dtype, quant_preset)
    return build_binary_broadcast_op(
        op_name="SQUARED_DIFFERENCE",
        input_1_shape=input_1_shape,
        input_2_shape=input_2_shape,
        dtype=dtype,
        input_1_quant=quant["input_1_quant"],
        input_2_quant=quant["input_2_quant"],
        output_quant=quant["output_quant"],
    )


class OpSquaredDifference(BinaryBasicMathBase):
    """SquaredDifference operation."""

    SIGN_SPAN_OPERANDS = ("input_1", "input_2")
    # Argument faults the flat float kernel diagnoses with ARM_CMSIS_NN_ARG_ERROR
    # (ns-cmsis-nn#490): each NULL pointer on its own, so every operand of the
    # guard's short-circuit chain is the one that trips it, plus the two sides of
    # `block_size < 1`.
    FAULT_KINDS = ("null_input_1", "null_input_2", "null_output", "zero_block", "negative_block")

    def needs_keras_model(self) -> bool:
        return self._use_s16_fake_quant_keras_path()

    def build_keras_model(self):
        if not self._use_s16_fake_quant_keras_path():
            raise NotImplementedError("SquaredDifference uses LiteRT-only model generation.")

        input_1_shape = tuple(self.desc["input_1_shape"])
        input_2_shape = tuple(self.desc["input_2_shape"])
        min_val, max_val = self._s16_fake_quant_range()

        input_a = tf.keras.Input(shape=input_1_shape[1:], dtype=tf.float32, name="input1")
        input_b = tf.keras.Input(shape=input_2_shape[1:], dtype=tf.float32, name="input2")

        output = QuantizedSquaredDifference(min_val=min_val, max_val=max_val, name="squared_difference")(
            [input_a, input_b]
        )
        return tf.keras.Model(inputs=[input_a, input_b], outputs=output, name="SquaredDifferenceS16FakeQuant")

    def _use_s16_fake_quant_keras_path(self) -> bool:
        if self.desc.get("activation_dtype", "S8") != "S16":
            return False

        hint = self.desc.get("hint", {}) or {}
        mode = str(hint.get("s16_builder", hint.get("generation_mode", ""))).strip().lower()
        if mode in {"keras_fake_quant", "fake_quant", "keras"}:
            return True

        return bool(self.desc.get("s16_use_fake_quant", False))

    def _s16_fake_quant_range(self) -> tuple[float, float]:
        hint = self.desc.get("hint", {}) or {}
        min_val = hint.get("s16_fake_quant_min", self.desc.get("s16_fake_quant_min", -32768.0))
        max_val = hint.get("s16_fake_quant_max", self.desc.get("s16_fake_quant_max", 32767.0))
        return float(min_val), float(max_val)

    def _convert_with_litert_builder(self, out_path: str) -> None:
        activation_dtype = self.tensor_dtype("input", default=str(self.desc.get("activation_dtype", "S8")))
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        elif activation_dtype in FLOAT_KERNEL_SUFFIX:
            # Float tensors carry no quantization; the builder's default quant is
            # None for float tensor types.
            model_bytes = build_binary_broadcast_op(
                op_name="SQUARED_DIFFERENCE",
                input_1_shape=tuple(self.desc["input_1_shape"]),
                input_2_shape=tuple(self.desc["input_2_shape"]),
                dtype="float16" if activation_dtype == "FP16" else "float32",
            )
            self._write_tflite_bytes(out_path, model_bytes)
            return
        else:
            raise NotImplementedError(f"Unsupported SquaredDifference dtype: {activation_dtype}")

        model_bytes = build_squared_difference_op(
            input_1_shape=tuple(self.desc["input_1_shape"]),
            input_2_shape=tuple(self.desc["input_2_shape"]),
            dtype=dtype,
            quant_preset=str(self.desc.get("quant_preset", "default")),
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_cmsis_squared_difference_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for SquaredDifference operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.tensor_dtype("input", default=str(self.desc.get('activation_dtype', 'S8')))
        
        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_squared_difference_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'float_kernel': False,
            }
        elif activation_dtype == 'S16':
            call_style = self.desc.get("hint", {}).get("call_style", "")
            if str(call_style).lower() == "elementwise":
                return {
                    'kernel_fn': 'arm_elementwise_squared_difference_s16',
                    'input_c_type': 'int16_t',
                    'output_c_type': 'int16_t',
                    'float_kernel': False,
                }
            return {
                'kernel_fn': 'arm_squared_difference_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'float_kernel': False,
            }
        elif activation_dtype in FLOAT_KERNEL_SUFFIX:
            # The float entry point is flat (no dims, no broadcast, no clamp):
            # arm_elementwise_squared_difference_f16 (ns-cmsis-nn#490).
            suffix = FLOAT_KERNEL_SUFFIX[activation_dtype]
            c_type = 'float16_t' if activation_dtype == 'FP16' else 'float'
            return {
                'kernel_fn': f"arm_elementwise_squared_difference_{suffix}",
                'input_c_type': c_type,
                'output_c_type': c_type,
                'float_kernel': True,
            }
        else:
            raise NotImplementedError(f"Unsupported SquaredDifference dtype: {activation_dtype}")

    def _check_fault_reachable(self, kind: str, kernel_info: Dict[str, Any]) -> None:
        """Reject fault kinds the selected kernel does not diagnose.

        The fault template drives the flat float entry point, whose guard is the
        one `if` in the kernel: any NULL pointer or a block_size below 1 returns
        ARM_CMSIS_NN_ARG_ERROR. The int dims-taking kernels have their own guard
        shape (dims pointers and broadcast validity) and no block_size, and the
        int elementwise kernels have no guard at all, so neither is wired here.
        """
        if not kernel_info["float_kernel"]:
            raise self.fault_unreachable(
                kind, f"{kernel_info['kernel_fn']} is not covered by the float fault template"
            )

    @staticmethod
    def _float_reference(float_dtype) -> Any:
        """Return the IEEE-754 model of the flat float kernel for `float_dtype`.

        Both kernel legs compute `(a - b)` and then `d * d` in the storage
        format with one rounding per operation (MVE vsubq/vmulq on halves; the
        scalar leg on `_Float16`). For binary16 the exact difference of two
        halves fits in 40 bits and the exact square of a half in 22, so float64
        intermediates with one narrowing per operation reproduce that bit for
        bit. NumPy's own half arithmetic widens to float32 per operation and
        would double-round. binary32 is computed in float32 directly, which is
        already one rounding per operation.
        """
        if float_dtype == np.float16:

            def reference(operands: Sequence[np.ndarray]) -> np.ndarray:
                a = np.asarray(operands[0], dtype=np.float64)
                b = np.asarray(operands[1], dtype=np.float64)
                # Overflow to Inf on narrowing is the IEEE result being modelled.
                with np.errstate(over="ignore"):
                    diff = (a - b).astype(np.float16).astype(np.float64)
                    return (diff * diff).astype(np.float16)

            return reference

        def reference32(operands: Sequence[np.ndarray]) -> np.ndarray:
            a = np.asarray(operands[0], dtype=np.float32)
            b = np.asarray(operands[1], dtype=np.float32)
            with np.errstate(over="ignore"):
                diff = (a - b).astype(np.float32)
                return (diff * diff).astype(np.float32)

        return reference32

    def _float_operands(
        self, input1_shape: Tuple[int, ...], input2_shape: Tuple[int, ...], float_dtype
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw the two float operands, or take them verbatim from the descriptor.

        `hint.extras.input_1_values` / `input_2_values` pin an operand element for
        element (flat, NHWC order). A pinned operand is the case -- overflow to
        +Inf, the largest finite square, subnormal squares, identical operands --
        so it is emitted exactly as written and never steered or swept. Either
        operand may be pinned on its own; the other is drawn as usual.
        """
        extras = (self.desc.get("hint", {}) or {}).get("extras", {}) or {}
        drawn_1, drawn_2 = self._sample_dual_uniform_inputs(input1_shape, input2_shape)
        pinned = {"input_1_values": input1_shape, "input_2_values": input2_shape}
        if self.input_mode() == "nonfinite_sweep" and "input_1_values" in extras:
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} pins input_1_values and requests "
                "input_mode 'nonfinite_sweep'; the sweep overwrites the left operand, so "
                "pin the tokens in the values instead of combining the two"
            )
        operands = [drawn_1, drawn_2]
        for index, (key, shape) in enumerate(pinned.items()):
            if key not in extras:
                continue
            values = np.asarray(extras[key], dtype=np.float64).flatten()
            expected = int(np.prod(shape))
            if values.size != expected:
                raise ValueError(
                    f"Descriptor {self.desc.get('name')!r}: {key} has {values.size} entries, "
                    f"expected {expected} to match shape {list(shape)}"
                )
            with np.errstate(over="ignore"):
                pinned_values = values.astype(float_dtype)
            widened = pinned_values.astype(np.float64)
            finite = np.isfinite(values)
            if not np.array_equal(widened[finite], values[finite]):
                raise ValueError(
                    f"Descriptor {self.desc.get('name')!r}: {key} holds values that are not "
                    f"exactly representable in {np.dtype(float_dtype).name}; write the "
                    "rounded value so the emitted operand is the one the golden was computed from"
                )
            operands[index] = pinned_values.reshape(shape)
        return operands[0].astype(float_dtype), operands[1].astype(float_dtype)

    def _generate_float_c_files(
        self,
        output_dir: Path,
        *,
        name: str,
        kernel_info: Dict[str, Any],
        input1_shape: Tuple[int, ...],
        input2_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        if input1_shape != input2_shape:
            raise NotImplementedError(
                f"Descriptor {self.desc.get('name')!r}: {kernel_info['kernel_fn']} is a flat "
                f"kernel with no broadcast entry point; input_1_shape {list(input1_shape)} and "
                f"input_2_shape {list(input2_shape)} must match"
            )
        builder = TemplateContextBuilder()
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32

        input1_q, input2_q = self._float_operands(input1_shape, input2_shape, float_dtype)
        reference = self._float_reference(float_dtype)
        output_data = reference([input1_q, input2_q])
        output_data, nonfinite_context = self.apply_nonfinite_policy(
            output_data, reference=reference, inputs=[input1_q, input2_q]
        )

        context: Dict[str, Any] = {
            'name': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            'block_size': int(np.prod(output_shape)),
            'call_style': str(self.desc.get("hint", {}).get("call_style", "")),
            'input1_data_array': builder.format_array_as_c_literal(input1_q),
            'input2_data_array': builder.format_array_as_c_literal(input2_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': True,
            'validation_mode': 'float',
        }
        context.update(nonfinite_context)

        c_template = "BasicMathFunctions/squared_difference/squared_difference.c.j2"
        fault = self.fault_kind()
        if fault:
            self._check_fault_reachable(fault, kernel_info)
            context.update(self.fault_context())
            c_template = "BasicMathFunctions/squared_difference/squared_difference_fault.c.j2"

        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'SquaredDifference'),
            'operator_name': 'squared_difference'
        }
        self._write_op_outputs(
            output_dir,
            "squared_difference",
            "BasicMathFunctions/squared_difference/squared_difference.h.j2",
            c_template,
            context,
            cmake_context,
        )

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        if self._use_s16_fake_quant_keras_path():
            if model is None:
                raise ValueError("Expected Keras model for S16 FakeQuant SquaredDifference path.")

            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
                ]
                converter.inference_input_type = tf.int16
                converter.inference_output_type = tf.int16

                rng = np.random.default_rng(rep_seed)
                in1_shape = tuple(self.desc["input_1_shape"])
                in2_shape = tuple(self.desc["input_2_shape"])

                def representative_data_gen():
                    for _ in range(100):
                        x1 = rng.uniform(-1.0, 1.0, size=in1_shape).astype(np.float32)
                        x2 = rng.uniform(-1.0, 1.0, size=in2_shape).astype(np.float32)
                        yield [x1, x2]

                converter.representative_dataset = representative_data_gen
                tflite_model = converter.convert()
                self._write_tflite_bytes(out_path, tflite_model)
                return
            except Exception:
                hint = self.desc.get("hint", {}) or {}
                if bool(hint.get("s16_builder_strict", self.desc.get("s16_builder_strict", False))):
                    raise
                # Converter support for 16x8 FakeQuant graphs can be incomplete.
                # Fall back to explicit LiteRT construction to keep generation robust.
                self._convert_with_litert_builder(out_path)
                return

        self._convert_with_litert_builder(out_path)

    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for SquaredDifference operation.
        """
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import (
            scalar_scale_zp,
            activation_bounds,
            elementwise_squared_difference_quant_params,
        )
        
        name = self.desc['name']
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
        
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_squared_difference_kernel()
        
        # Load LiteRT model for shape and quantization extraction
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Extract shapes from LiteRT (multi-input operator)
        input1_shape = op_tensors['inputs'][0]['shape']
        input2_shape = op_tensors['inputs'][1]['shape'] if len(op_tensors['inputs']) > 1 else input1_shape
        output_shape = op_tensors['outputs'][0]['shape']
        
        # Ensure shapes are tuples
        if input1_shape is not None:
            input1_shape = tuple(input1_shape)
        if input2_shape is not None:
            input2_shape = tuple(input2_shape)
        if output_shape is not None:
            output_shape = tuple(output_shape)

        if kernel_info["float_kernel"]:
            self._generate_float_c_files(
                output_dir,
                name=name,
                kernel_info=kernel_info,
                input1_shape=input1_shape,
                input2_shape=input2_shape,
                output_shape=output_shape,
            )
            return
        if self.fault_kind():
            self._check_fault_reachable(self.fault_kind(), kernel_info)
        
        # Extract quantization from LiteRT
        input1_quant = op_tensors['inputs'][0]['quantization']
        input2_quant = op_tensors['inputs'][1]['quantization'] if len(op_tensors['inputs']) > 1 else input1_quant
        output_quant = op_tensors['outputs'][0]['quantization']
        
        input1_scale, input1_zp = scalar_scale_zp(input1_quant)
        input2_scale, input2_zp = scalar_scale_zp(input2_quant)
        output_scale, output_zp = scalar_scale_zp(output_quant)
        
        builder = TemplateContextBuilder()
        
        # Convert shapes to CMSIS dims
        input1_dims = builder.nhwc_to_cmsis_dims(input1_shape)
        input2_dims = builder.nhwc_to_cmsis_dims(input2_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        activation_min, activation_max = activation_bounds(activation_dtype)
        sqdiff_qparams = elementwise_squared_difference_quant_params(
            input1_scale=float(input1_scale),
            input2_scale=float(input2_scale),
            output_scale=float(output_scale),
            activation_dtype=activation_dtype,
        )
        mult1 = sqdiff_qparams["input1_mult"]
        shift1 = sqdiff_qparams["input1_shift"]
        mult2 = sqdiff_qparams["input2_mult"]
        shift2 = sqdiff_qparams["input2_shift"]
        output_mult = sqdiff_qparams["out_mult"]
        output_shift = sqdiff_qparams["out_shift"]
        left_shift = sqdiff_qparams["left_shift"]
        
        # Generate input data and quantize
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        
        input1_data = self.rng.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32)
        input2_data = self.rng.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32)
        
        self.rng.__setstate__(rng_state)
        
        # Quantize inputs
        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        
        input1_q = np.round(input1_data / float(input1_scale) + float(input1_zp)).astype(np.int32)
        input1_q = np.clip(input1_q, qmin, qmax).astype(np_in_dtype)
        
        input2_q = np.round(input2_data / float(input2_scale) + float(input2_zp)).astype(np.int32)
        input2_q = np.clip(input2_q, qmin, qmax).astype(np_in_dtype)
        input1_q, input2_q = self._enforce_int_operand_sign_span(
            (("input_1", input1_q, input1_zp), ("input_2", input2_q, input2_zp)),
            steerable=("input_1", "input_2"),
        )
        
        # Run inference using LiteRT interpreter when shapes match for int8.
        # LiteRT does not currently invoke INT16 SQUARED_DIFFERENCE reliably,
        # and broadcasting can abort in some runtimes, so use the local
        # quantized simulation for those cases.
        if input1_shape == input2_shape and activation_dtype != "S16":
            interpreter = self.load_litert_interpreter(str(tflite_path))
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], input1_q)
            interpreter.set_tensor(input_details[1]['index'], input2_q)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.array(output_data)
        else:
            output_data = self._simulate_squared_difference_quantized(
                input1_q,
                input2_q,
                input1_offset=-int(input1_zp),
                input2_offset=-int(input2_zp),
                input1_mult=int(mult1),
                input1_shift=int(shift1),
                input2_mult=int(mult2),
                input2_shift=int(shift2),
                left_shift=int(left_shift),
                out_offset=int(output_zp),
                out_mult=int(output_mult),
                out_shift=int(output_shift),
                out_activation_min=int(activation_min),
                out_activation_max=int(activation_max),
                out_dtype=np_in_dtype,
            )
        
        # Format arrays
        input1_array_str = builder.format_array_as_c_literal(input1_q)
        input2_array_str = builder.format_array_as_c_literal(input2_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)
        
        # Build template context
        context = {
            'name': name,
            'input1_dims': input1_dims,
            'input2_dims': input2_dims,
            'output_dims': output_dims,
            'input1_offset': -int(input1_zp),
            'input1_mult': int(mult1),
            'input1_shift': int(shift1),
            'input2_offset': -int(input2_zp),
            'input2_mult': int(mult2),
            'input2_shift': int(shift2),
            'left_shift': int(left_shift),
            'out_offset': int(output_zp),
            'out_mult': int(output_mult),
            'out_shift': int(output_shift),
            'out_activation_min': int(activation_min),
            'out_activation_max': int(activation_max),
            'block_size': int(np.prod(output_shape)),
            'call_style': str(self.desc.get("hint", {}).get("call_style", "")),
            'input1_data_array': input1_array_str,
            'input2_data_array': input2_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': False,
        }
        
        cmake_context = {
            'name': name,
            'operator': self.desc.get('operator', 'SquaredDifference'),
            'operator_name': 'squared_difference'
        }
        self._write_op_outputs(output_dir, "squared_difference", "BasicMathFunctions/squared_difference/squared_difference.h.j2", "BasicMathFunctions/squared_difference/squared_difference.c.j2", context, cmake_context)

    @classmethod
    def _simulate_squared_difference_quantized(
        cls,
        input1_q: np.ndarray,
        input2_q: np.ndarray,
        *,
        input1_offset: int,
        input2_offset: int,
        input1_mult: int,
        input1_shift: int,
        input2_mult: int,
        input2_shift: int,
        left_shift: int,
        out_offset: int,
        out_mult: int,
        out_shift: int,
        out_activation_min: int,
        out_activation_max: int,
        out_dtype: np.dtype,
    ) -> np.ndarray:
        a = (input1_q.astype(np.int32) + int(input1_offset)) << int(left_shift)
        b = (input2_q.astype(np.int32) + int(input2_offset)) << int(left_shift)
        a = cls._requantize_np(a, int(input1_mult), int(input1_shift))
        b = cls._requantize_np(b, int(input2_mult), int(input2_shift))
        diff = (a - b) ** 2
        diff = cls._requantize_np(diff, int(out_mult), int(out_shift))
        diff = diff + int(out_offset)
        diff = np.clip(diff, int(out_activation_min), int(out_activation_max))
        return diff.astype(out_dtype)
