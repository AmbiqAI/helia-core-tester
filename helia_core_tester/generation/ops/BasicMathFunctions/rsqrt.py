"""
Rsqrt operation implementation.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool

RSQRT_CALL_STYLES = ("per_op", "universal")


def rsqrt_argument_pool(context: Dict[str, Any]) -> ArgumentPool:
    """The universal kernel takes the requantisation and a 32-bit LUT; per-op takes a 16-bit LUT."""
    n = context["name"]
    style = str(context.get("call_style", "per_op"))
    if style not in RSQRT_CALL_STYLES:
        raise ValueError(f"{n}: Rsqrt call_style {style!r} is not one of {RSQRT_CALL_STYLES}")
    values = {"input_offset": context["input_offset"], "out_offset": context["output_offset"],
              "out_activation_min": context["out_activation_min"], "out_activation_max": context["out_activation_max"],
              "block_size": context["block_size"], "lut": f"{n}_rsqrt_lut"}
    if style == "universal":
        values.update(out_mult=context["out_mult"], out_shift=context["out_shift"],
                      needs_rescale=context["needs_rescale"])
    lut = Declaration(f"{n}_rsqrt_lut", context["lut_dtype"], ArrayLiteral(context["rsqrt_lut_array"]), array=True,
                      comment="Reciprocal square-root lookup table")
    return tensor_case_pool(context, values, extra_header=(lut,), output_count=dims_count(context["output_dims"]))


RSQRT_CANONICAL_OUTPUT_SCALE = 1.0 / 32768.0
# Wider inputs keep rsqrt mostly unsaturated.
RSQRT_INPUT_SCALE = 1.0 / 512.0
# Default input draw, in quantized units.
RSQRT_INPUT_Q_RANGE = (4096, 32767)
RSQRT_LUT_SIZE = 513
RSQRT_SLOT_SHIFT = 7
RSQRT_BASE_STEP_SHIFT = 6


def _quant_param_to_scalar(value, name: str, cast):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Rsqrt expects scalar quantization for {name}, got shape {arr.shape}")
    return cast(arr.reshape(-1)[0])


def _requantize_like_cmsis(value: int, multiplier: int, shift: int) -> int:
    if multiplier == 0 or value == 0:
        return 0
    result = int(round((value * multiplier) / float(1 << (31 - shift))))
    return result


def make_rsqrt_per_op_lut(input_scale, output_scale, output_zp) -> np.ndarray:
    input_scale = _quant_param_to_scalar(input_scale, "input_scale", float)
    output_scale = _quant_param_to_scalar(output_scale, "output_scale", float)
    output_zp = _quant_param_to_scalar(output_zp, "output_zero_point", int)

    lut = np.zeros(RSQRT_LUT_SIZE, dtype=np.int16)
    for index in range(RSQRT_LUT_SIZE):
        q_value = -32768 + (index << RSQRT_SLOT_SHIFT)
        if q_value <= 0:
            lut[index] = np.int16(32767)
            continue
        real_value = input_scale * float(q_value)
        real_rsqrt = 1.0 / sqrt(real_value)
        quantized = int(round(real_rsqrt / output_scale)) + output_zp
        lut[index] = np.int16(np.clip(quantized, -32768, 32767))
    return lut


def make_rsqrt_universal_lut(input_scale) -> np.ndarray:
    input_scale = _quant_param_to_scalar(input_scale, "input_scale", float)

    lut = np.zeros(RSQRT_LUT_SIZE, dtype=np.int32)
    for index in range(RSQRT_LUT_SIZE):
        # Kernel reads entry ceil(q / 64).
        q_value = index << RSQRT_BASE_STEP_SHIFT
        if q_value <= 0:
            lut[index] = 32767
            continue
        real_value = input_scale * float(q_value)
        real_rsqrt = 1.0 / sqrt(real_value)
        quantized = int(round(real_rsqrt / RSQRT_CANONICAL_OUTPUT_SCALE))
        lut[index] = int(np.clip(quantized, -32768, 32767))
    return lut


def derive_rsqrt_universal_quant_params(output_scale: float) -> Dict[str, int]:
    from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift

    output_scale = float(output_scale)
    effective_scale = RSQRT_CANONICAL_OUTPUT_SCALE / output_scale
    out_mult, out_shift = calculate_multiplier_shift(effective_scale)
    needs_rescale = 0 if abs(effective_scale - 1.0) < 1e-9 else 1
    return {
        "out_mult": int(out_mult),
        "out_shift": int(out_shift),
        "needs_rescale": int(needs_rescale),
    }


class OpRsqrt(OperationBase):
    """Rsqrt operation: goldens from the C reference (TFLite's int16 LUT Rsqrt; float in binary64)."""

    def _expect_arg_error(self) -> bool:
        return bool(self.desc.get("hint", {}).get("force_negative_input_case", False))

    def uses_reference(self) -> bool:
        return not self.status_only()

    def status_only(self) -> bool:
        # A negative-domain case checks the kernel's ARG_ERROR, not an output.
        return self.tensor_dtype("input") not in ("FP16", "FP32") and self._expect_arg_error()

    def _variant(self) -> str:
        call_style = str(self.desc.get("hint", {}).get("call_style", "per_op"))
        if call_style not in {"per_op", "universal"}:
            raise ValueError(f"Unsupported Rsqrt call_style: {call_style}")
        return call_style

    def _select_cmsis_rsqrt_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S16")
        if activation_dtype != "S16":
            raise NotImplementedError(f"Unsupported Rsqrt dtype: {activation_dtype}")

        call_style = self._variant()
        if call_style == "universal":
            return {
                "kernel_fn": "arm_rsqrt_s16_universal",
                "input_c_type": "int16_t",
                "output_c_type": "int16_t",
                "lut_dtype": "int32_t",
            }
        return {
            "kernel_fn": "arm_rsqrt_s16_per_op",
            "input_c_type": "int16_t",
            "output_c_type": "int16_t",
            "lut_dtype": "int16_t",
        }

    def _generate_positive_float_input(self, shape: Tuple[int, ...], input_scale: float) -> np.ndarray:
        # LUT error grows below q = 4096.
        low, high = self.desc.get("hint", {}).get("input_q_range", RSQRT_INPUT_Q_RANGE)
        scale = float(input_scale)
        return self._sample_uniform(shape, low=low * scale, high=high * scale, dtype=np.float32)

    def _generate_negative_domain_input(self, shape: Tuple[int, ...], input_zp: int) -> np.ndarray:
        fill = np.int32(input_zp) - 1
        clipped = np.clip(fill, -32768, 32767)
        return np.full(shape, clipped, dtype=np.int16)

    def _ensure_positive_domain_input(self, input_q: np.ndarray, input_zp: int) -> None:
        if np.any(input_q.astype(np.int32) - int(input_zp) < 0):
            raise ValueError("Rsqrt test inputs must stay in the non-negative post-offset domain")

    def generate_c_files(self, output_dir: Path) -> None:
        if self.tensor_dtype("input") in ("FP16", "FP32"):
            from helia_core_tester.generation.ops._shared.sqrt_float import generate_sqrt_float

            generate_sqrt_float(self, output_dir, reciprocal=True)
            return

        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import activation_bounds

        name = self.desc["name"]
        kernel_info = self._select_cmsis_rsqrt_kernel()
        input_shape = output_shape = tuple(int(d) for d in self.desc["input_shape"])
        quant = self.desc.get("quantization") or {}
        in_quant = policy.descriptor_quant(quant.get("input"), "s16") or policy.TensorQuant(RSQRT_INPUT_SCALE, 0, "s16")
        out_quant = (policy.descriptor_quant(quant.get("output"), "s16")
                     or policy.TensorQuant(RSQRT_CANONICAL_OUTPUT_SCALE, 0, "s16"))
        input_scale, input_zp = in_quant.scale, in_quant.zero_point
        output_scale, output_zp = out_quant.scale, out_quant.zero_point

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        out_activation_min, out_activation_max = activation_bounds("S16")

        if self.status_only():
            input_q = self._generate_negative_domain_input(input_shape, input_zp)
            expected_status = "ARM_CMSIS_NN_ARG_ERROR"
            output_data = np.zeros(output_shape, dtype=np.int16)
        else:
            input_data = self._generate_positive_float_input(input_shape, input_scale)
            input_q = policy.quantize(input_data, in_quant)
            self._ensure_positive_domain_input(input_q, input_zp)
            output_data = self.reference_golden(ReferenceCall(
                "rsqrt_s16",
                {"input_scale": input_scale, "input_zero_point": input_zp,
                 "output_scale": output_scale, "output_zero_point": output_zp},
                {"input": np.ascontiguousarray(input_q)}, {"output": output_shape},
                quant={"input": in_quant.to_json(), "output": out_quant.to_json()},
            ))
            expected_status = "ARM_CMSIS_NN_SUCCESS"

        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)

        call_style = self._variant()
        if call_style == "universal":
            rsqrt_lut = make_rsqrt_universal_lut(input_scale)
            quant_params = derive_rsqrt_universal_quant_params(output_scale)
        else:
            rsqrt_lut = make_rsqrt_per_op_lut(input_scale, output_scale, output_zp)
            quant_params = {"out_mult": 0, "out_shift": 0, "needs_rescale": 0}

        context = {
            "name": name,
            "call_style": call_style,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "input_offset": int(input_zp),
            "output_offset": int(output_zp),
            "out_mult": int(quant_params["out_mult"]),
            "out_shift": int(quant_params["out_shift"]),
            "needs_rescale": int(quant_params["needs_rescale"]),
            "out_activation_min": int(out_activation_min),
            "out_activation_max": int(out_activation_max),
            "block_size": int(np.prod(output_shape)),
            "input_data_array": input_array_str,
            "expected_output_array": expected_output_array_str,
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "kernel_fn": kernel_info["kernel_fn"],
            "lut_dtype": kernel_info["lut_dtype"],
            "rsqrt_lut_array": builder.format_array_as_c_literal(rsqrt_lut),
            "expected_status": expected_status,
        }

        self.render_harness_case(
            output_dir, stem="rsqrt", context=context, pool=rsqrt_argument_pool(context),
            validation_key="BasicMathFunctions/rsqrt/rsqrt.c.j2", label="Rsqrt", operator="Rsqrt", sidecar=True,
        )
