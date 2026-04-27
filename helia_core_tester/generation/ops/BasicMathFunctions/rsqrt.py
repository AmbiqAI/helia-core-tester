"""
Rsqrt operation implementation.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_unary_same_shape_op


RSQRT_CANONICAL_OUTPUT_SCALE = 1.0 / 32768.0
RSQRT_LUT_SIZE = 513
RSQRT_SLOT_SHIFT = 7


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
        q_value = -32768 + (index << RSQRT_SLOT_SHIFT)
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


def build_rsqrt_op(
    *,
    input_shape,
    dtype: str = "int16",
) -> bytes:
    return build_unary_same_shape_op(
        op_name="RSQRT",
        input_shape=input_shape,
        dtype=dtype,
    )


class OpRsqrt(OperationBase):
    """Rsqrt operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Rsqrt uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        activation_dtype = self.desc.get("activation_dtype", "S16")
        if activation_dtype != "S16":
            raise NotImplementedError(f"Unsupported Rsqrt dtype: {activation_dtype}")

        input_shape = tuple(self.desc["input_shape"])
        model_bytes = build_rsqrt_op(input_shape=input_shape, dtype="int16")
        self._write_tflite_bytes(out_path, model_bytes)

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
        low = max(float(input_scale), 1.0e-3)
        return self._sample_uniform(shape, low=low, high=1.0, dtype=np.float32)

    def _generate_negative_domain_input(self, shape: Tuple[int, ...], input_zp: int) -> np.ndarray:
        fill = np.int32(input_zp) - 1
        clipped = np.clip(fill, -32768, 32767)
        return np.full(shape, clipped, dtype=np.int16)

    def _quantize_input(self, input_data: np.ndarray, input_scale: float, input_zp: int) -> np.ndarray:
        quantized = np.round(input_data / float(input_scale) + float(input_zp)).astype(np.int32)
        quantized = np.clip(quantized, -32768, 32767)
        return quantized.astype(np.int16)

    def _ensure_positive_domain_input(self, input_q: np.ndarray, input_zp: int) -> None:
        if np.any(input_q.astype(np.int32) - int(input_zp) < 0):
            raise ValueError("Rsqrt test inputs must stay in the non-negative post-offset domain")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.tflite_utils import activation_bounds, scalar_scale_zp

        name = self.desc["name"]
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        kernel_info = self._select_cmsis_rsqrt_kernel()

        model, subgraph = self.load_litert_model(str(tflite_path))
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)

        input_shape = self._ensure_shape_tuple(op_tensors["inputs"][0]["shape"])
        output_shape = self._ensure_shape_tuple(op_tensors["outputs"][0]["shape"])
        input_quant = op_tensors["inputs"][0]["quantization"]
        output_quant = op_tensors["outputs"][0]["quantization"]
        input_scale, input_zp = scalar_scale_zp(input_quant)
        output_scale, output_zp = scalar_scale_zp(output_quant)

        builder = TemplateContextBuilder()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)
        out_activation_min, out_activation_max = activation_bounds("S16")

        expect_arg_error = bool(self.desc.get("hint", {}).get("force_negative_input_case", False))
        if expect_arg_error:
            input_q = self._generate_negative_domain_input(input_shape, input_zp)
            expected_status = "ARM_CMSIS_NN_ARG_ERROR"
            output_data = np.zeros(output_shape, dtype=np.int16)
        else:
            input_data = self._generate_positive_float_input(input_shape, input_scale)
            input_q = self._quantize_input(input_data, input_scale, input_zp)
            self._ensure_positive_domain_input(input_q, input_zp)

            interpreter = self.load_litert_interpreter(str(tflite_path))
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], input_q)
            interpreter.invoke()
            output_data = np.array(interpreter.get_tensor(output_details[0]["index"]))
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
            "prefix": name,
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

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "Rsqrt"),
            "operator_name": "rsqrt",
        }
        self._write_op_outputs(
            output_dir,
            "rsqrt",
            "BasicMathFunctions/rsqrt/rsqrt.h.j2",
            "BasicMathFunctions/rsqrt/rsqrt.c.j2",
            context,
            cmake_context,
        )
