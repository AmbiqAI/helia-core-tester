"""
Abs operation implementation.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


def abs_golden(input_q: np.ndarray, input_zp: int, output_zp: int, mult: int, shift: int, rescale: bool) -> np.ndarray:
    """TFLite's integer ABS: |x - zp_in|, requantized when the scales differ, plus zp_out, saturated."""
    from helia_core_tester.generation.utils.tflite_utils import requantize_np

    info = np.iinfo(input_q.dtype)
    value = np.abs(input_q.astype(np.int64) - int(input_zp))
    if rescale:
        value = requantize_np(value, int(mult), int(shift)).astype(np.int64)
    return np.clip(value + int(output_zp), info.min, info.max).astype(input_q.dtype)


class OpAbs(OperationBase):
    """
    Abs operation.
    """

    SIGN_SPAN_OPERANDS = ("input",)

    # The (scale, zero point) the one-op LiteRT builder gave input and output; kept so
    # the goldens do not move.
    FIXED_QUANT = {"S8": (0.125, 0), "S16": (1.0 / 32768.0, 0)}

    def needs_tflite(self) -> bool:
        return False

    def _select_cmsis_abs_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            return {
                "kernel_fn": "arm_abs_s8",
                "input_c_type": "int8_t",
                "output_c_type": "int8_t",
                "float_kernel": False,
            }
        if activation_dtype == "S16":
            return {
                "kernel_fn": "arm_abs_s16",
                "input_c_type": "int16_t",
                "output_c_type": "int16_t",
                "float_kernel": False,
            }
        if activation_dtype == "FP32":
            return {
                "kernel_fn": "arm_nn_abs_f32",
                "input_c_type": "float",
                "output_c_type": "float",
                "float_kernel": True,
            }
        if activation_dtype == "FP16":
            return {
                "kernel_fn": "arm_nn_abs_f16",
                "input_c_type": "float16_t",
                "output_c_type": "float16_t",
                "float_kernel": True,
            }
        raise NotImplementedError(f"Unsupported Abs dtype: {activation_dtype}")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import (
            calculate_multiplier_shift,
            activation_bounds,
        )

        name = self.desc["name"]
        kernel_info = self._select_cmsis_abs_kernel()
        input_shape = output_shape = tuple(int(d) for d in self.desc["input_shape"])

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        activation_dtype = self.desc.get("activation_dtype", "S8")

        if kernel_info["float_kernel"]:
            # arm_nn_abs_f32/f16 are pure (input, output, block_size) kernels
            # with no quantization or activation parameters.
            float_dtype = np.float16 if kernel_info["input_c_type"] == "float16_t" else np.float32
            input_q = self._sample_uniform(input_shape, dtype=float_dtype)

            def float_reference(operands, _dtype=float_dtype):
                return np.abs(operands[0]).astype(_dtype)

            output_data = float_reference([input_q])
            input_zp = output_zp = output_mult = output_shift = needs_rescale = 0
            activation_min = activation_max = 0
        else:
            input_scale, input_zp = output_scale, output_zp = self.FIXED_QUANT[activation_dtype]

            activation_min, activation_max = activation_bounds(activation_dtype)

            effective_scale = float(input_scale) / float(output_scale)
            output_mult, output_shift = calculate_multiplier_shift(effective_scale)
            needs_rescale = 0 if abs(effective_scale - 1.0) < 1e-6 else 1
            if bool(self.desc.get("hint", {}).get("force_rescale", False)):
                needs_rescale = 1

            rng_state = self.rng.__getstate__()
            self.rng = np.random.default_rng(self.seed)
            input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
            self.rng.__setstate__(rng_state)

            qmin, qmax = activation_bounds(activation_dtype)
            np_in_dtype = np.int16 if activation_dtype == "S16" else np.int8
            input_q = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
            input_q = np.clip(input_q, qmin, qmax).astype(np_in_dtype)
            # abs is the sign-defining int kernel: its two branches are chosen
            # by the sign of value - zero_point, so an operand that lands on
            # one side exercises exactly one of them.
            (input_q,) = self._enforce_int_operand_sign_span(
                (("input", input_q, input_zp),),
                steerable=("input",),
            )

            output_data = abs_golden(input_q, input_zp, output_zp, output_mult, output_shift, bool(needs_rescale))

        if kernel_info["float_kernel"]:
            output_data, nonfinite_context = self.apply_nonfinite_policy(
                output_data, reference=float_reference, inputs=[input_q]
            )
        else:
            nonfinite_context = {}
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)

        block_size = int(np.prod(output_shape))

        context = {
            "name": name,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "input_offset": -int(input_zp),
            "output_offset": int(output_zp),
            "out_mult": int(output_mult),
            "out_shift": int(output_shift),
            "needs_rescale": int(needs_rescale),
            "out_activation_min": int(activation_min),
            "out_activation_max": int(activation_max),
            "block_size": int(block_size),
            "input_data_array": input_array_str,
            "expected_output_array": expected_output_array_str,
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "kernel_fn": kernel_info["kernel_fn"],
            "float_kernel": kernel_info["float_kernel"],
        }
        context.update(nonfinite_context)
        if kernel_info["float_kernel"]:
            context["validation_mode"] = "float"

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "Abs"),
            "operator_name": "abs",
        }
        self._write_op_outputs(output_dir, "abs", "BasicMathFunctions/abs/abs.h.j2", "BasicMathFunctions/abs/abs.c.j2", context, cmake_context)


from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool  # noqa: E402


@harness_pool("BasicMathFunctions/abs/abs.c.j2", label="Abs")
def abs_argument_pool(context):
    values = {"block_size": context["block_size"]}
    if not context.get("float_kernel"):
        values.update(input_offset=context["input_offset"], out_offset=context["output_offset"],
                      out_mult=context["out_mult"], out_shift=context["out_shift"],
                      needs_rescale=context["needs_rescale"], out_activation_min=context["out_activation_min"],
                      out_activation_max=context["out_activation_max"])
    return tensor_case_pool(context, values, output_count=dims_count(context["output_dims"]))
