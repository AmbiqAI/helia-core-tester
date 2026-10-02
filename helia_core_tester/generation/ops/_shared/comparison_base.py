"""Shared implementation for CMSIS-NN comparison generation."""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase


_OP_MAP = {
    "equal": ("EQUAL", "ARM_COMPARE_EQUAL", "arm_equal"),
    "not_equal": ("NOT_EQUAL", "ARM_COMPARE_NOT_EQUAL", "arm_not_equal"),
    "greater": ("GREATER", "ARM_COMPARE_GREATER", "arm_greater"),
    "greater_equal": ("GREATER_EQUAL", "ARM_COMPARE_GREATER_EQUAL", "arm_greater_equal"),
    "less": ("LESS", "ARM_COMPARE_LESS", "arm_less"),
    "less_equal": ("LESS_EQUAL", "ARM_COMPARE_LESS_EQUAL", "arm_less_equal"),
}

_DTYPE_INFO = {
    "S8": ("int8", "int8_t", np.int8, -128, 127, "s8"),
    "S16": ("int16", "int16_t", np.int16, -32768, 32767, "s16"),
}


class ComparisonFamilyBase(OperationBase):
    """Shared implementation for the generic CMSIS-NN comparison API."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Comparison uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_comparison_op

        activation_dtype = self.desc.get("activation_dtype", "S8")
        dtype_info = _DTYPE_INFO.get(activation_dtype)
        if dtype_info is None:
            raise NotImplementedError(f"Unsupported Comparison dtype: {activation_dtype}")
        dtype, _, _, _, _, _ = dtype_info

        op = str(self.desc.get("operation", "equal")).lower()
        if op not in _OP_MAP:
            raise ValueError(f"Unsupported comparison operation: {op}")
        litert_op, _, _ = _OP_MAP[op]

        input_1_shape = tuple(self.desc["input_1_shape"])
        input_2_shape = tuple(self.desc["input_2_shape"])

        model_bytes = build_comparison_op(
            input_1_shape=input_1_shape,
            input_2_shape=input_2_shape,
            op_name=litert_op,
            dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    @staticmethod
    def _requantize_np(values: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
        left_shift = shift if shift > 0 else 0
        right_shift = -shift if shift < 0 else 0
        prod = values.astype(np.int64) * (1 << left_shift)
        mult = (1 << 30) + (prod * int(multiplier))
        res = (mult >> 31).astype(np.int64)
        if right_shift == 0:
            return res.astype(np.int32)
        remainder_mask = (1 << right_shift) - 1
        remainder = res & remainder_mask
        result = res >> right_shift
        threshold = remainder_mask >> 1
        threshold = threshold + (result < 0)
        result = result + (remainder > threshold)
        return result.astype(np.int32)

    def _simulate_compare(self, input1_q: np.ndarray, input2_q: np.ndarray, operation: str, params: Dict[str, int]) -> np.ndarray:
        left_shift = params["left_shift"]
        a = (input1_q.astype(np.int32) + params["input_1_offset"]) << left_shift
        b = (input2_q.astype(np.int32) + params["input_2_offset"]) << left_shift
        a = self._requantize_np(a, params["input_1_mult"], params["input_1_shift"])
        b = self._requantize_np(b, params["input_2_mult"], params["input_2_shift"])

        if operation == "ARM_COMPARE_EQUAL":
            out = a == b
        elif operation == "ARM_COMPARE_NOT_EQUAL":
            out = a != b
        elif operation == "ARM_COMPARE_GREATER":
            out = a > b
        elif operation == "ARM_COMPARE_GREATER_EQUAL":
            out = a >= b
        elif operation == "ARM_COMPARE_LESS":
            out = a < b
        elif operation == "ARM_COMPARE_LESS_EQUAL":
            out = a <= b
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        return out.astype(np.uint8)

    def _quant_params(self, tflite_path: Path) -> Dict[str, int]:
        """Derive compare params like the runtime."""
        from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift

        inputs = self.load_primary_operator_tensors(str(tflite_path))["inputs"]
        params = {"left_shift": 8}
        for index, tensor in enumerate(inputs[:2], start=1):
            quant = tensor["quantization"]
            mult, shift = calculate_multiplier_shift(float(self._quant_param_scalar(quant, "scale", 1.0)))
            params[f"input_{index}_offset"] = -int(self._quant_param_scalar(quant, "zero_point", 0))
            params[f"input_{index}_mult"] = int(mult)
            params[f"input_{index}_shift"] = int(shift)
        return params

    def _sample_operands(self, shape_1, shape_2, output_shape, qmin, qmax, np_dtype):
        """Draw spread operands with forced ties."""
        rng = self._seeded_rng()
        # Few levels so equal pairs occur.
        levels = np.round(np.linspace(qmin, qmax, 7)).astype(np.int32)
        input_1 = rng.choice(levels, size=shape_1)
        input_2 = rng.choice(levels, size=shape_2)
        # Tie every third output element.
        if tuple(shape_2) == tuple(output_shape):
            tied = np.broadcast_to(input_1, output_shape).reshape(-1)
            input_2.reshape(-1)[::3] = tied[::3]
        elif tuple(shape_1) == tuple(output_shape):
            tied = np.broadcast_to(input_2, output_shape).reshape(-1)
            input_1.reshape(-1)[::3] = tied[::3]
        return input_1.astype(np_dtype), input_2.astype(np_dtype)

    def generate_c_files(self, output_dir) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.litert_builder import _broadcast_shape

        name = self.desc["name"]
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        activation_dtype = self.desc.get("activation_dtype", "S8")
        dtype_info = _DTYPE_INFO.get(activation_dtype)
        if dtype_info is None:
            raise NotImplementedError(f"Unsupported Comparison dtype: {activation_dtype}")
        _, c_type, np_in_dtype, qmin, qmax, kernel_suffix = dtype_info

        op = str(self.desc.get("operation", "equal")).lower()
        if op not in _OP_MAP:
            raise ValueError(f"Unsupported comparison operation: {op}")
        _, op_enum, wrapper_prefix = _OP_MAP[op]
        kernel_fn = f"{wrapper_prefix}_{kernel_suffix}"

        input_shape_1 = tuple(self.desc["input_1_shape"])
        input_shape_2 = tuple(self.desc["input_2_shape"])
        output_shape = _broadcast_shape(input_shape_1, input_shape_2)

        builder = TemplateContextBuilder()
        input_1_dims = builder.nhwc_to_cmsis_dims(input_shape_1)
        input_2_dims = builder.nhwc_to_cmsis_dims(input_shape_2)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        params = self._quant_params(tflite_path)
        input_1_q, input_2_q = self._sample_operands(input_shape_1, input_shape_2, output_shape, qmin, qmax, np_in_dtype)
        expected = self._simulate_compare(input_1_q, input_2_q, op_enum, params)

        context = {
            "name": name,
            "input_1_dims": input_1_dims,
            "input_2_dims": input_2_dims,
            "output_dims": output_dims,
            "input_1_data_array": builder.format_array_as_c_literal(input_1_q),
            "input_2_data_array": builder.format_array_as_c_literal(input_2_q),
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "input_dtype": c_type,
            "kernel_fn": kernel_fn,
            "output_size": int(np.prod(output_shape)),
            **params,
        }

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "Comparison"),
            "operator_name": "comparison",
        }
        self._write_op_outputs(
            Path(output_dir),
            "comparison",
            "ComparisonFunctions/comparison/comparison.h.j2",
            "ComparisonFunctions/comparison/comparison.c.j2",
            context,
            cmake_context,
        )
