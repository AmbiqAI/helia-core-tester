"""
Comparison operation implementation for direct arm_comparison_* coverage.
"""

from typing import Dict, Any
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


_OP_MAP = {
    "equal": ("EQUAL", "ARM_COMPARE_EQUAL"),
    "not_equal": ("NOT_EQUAL", "ARM_COMPARE_NOT_EQUAL"),
    "greater": ("GREATER", "ARM_COMPARE_GREATER"),
    "greater_equal": ("GREATER_EQUAL", "ARM_COMPARE_GREATER_EQUAL"),
    "less": ("LESS", "ARM_COMPARE_LESS"),
    "less_equal": ("LESS_EQUAL", "ARM_COMPARE_LESS_EQUAL"),
}


class OpComparison(OperationBase):
    """
    Comparison operation (direct arm_comparison_* coverage).
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Comparison uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_comparison_op

        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        else:
            raise NotImplementedError(f"Unsupported Comparison dtype: {activation_dtype}")

        op = str(self.desc.get("operation", "equal")).lower()
        if op not in _OP_MAP:
            raise ValueError(f"Unsupported comparison operation: {op}")
        litert_op, _ = _OP_MAP[op]

        input_1_shape = tuple(self.desc["input_1_shape"])
        input_2_shape = tuple(self.desc["input_2_shape"])

        model_bytes = build_comparison_op(
            input_1_shape=input_1_shape,
            input_2_shape=input_2_shape,
            op_name=litert_op,
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)

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

    def _simulate_compare(self, input1_q: np.ndarray, input2_q: np.ndarray, operation: str) -> np.ndarray:
        input_1_offset = int(self.desc.get("input_1_offset", 0))
        input_2_offset = int(self.desc.get("input_2_offset", 0))
        input_1_mult = int(self.desc.get("input_1_mult", 1))
        input_2_mult = int(self.desc.get("input_2_mult", 1))
        input_1_shift = int(self.desc.get("input_1_shift", 0))
        input_2_shift = int(self.desc.get("input_2_shift", 0))
        left_shift = int(self.desc.get("left_shift", 0))

        a = (input1_q.astype(np.int32) + input_1_offset) << left_shift
        b = (input2_q.astype(np.int32) + input_2_offset) << left_shift
        a = self._requantize_np(a, input_1_mult, input_1_shift)
        b = self._requantize_np(b, input_2_mult, input_2_shift)

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

    def generate_c_files(self, output_dir) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        from helia_core_tester.generation.utils.litert_builder import _broadcast_shape

        name = self.desc["name"]
        tflite_path = Path(output_dir) / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            kernel_fn = "arm_comparison_s16"
            c_type = "int16_t"
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            kernel_fn = "arm_comparison_s8"
            c_type = "int8_t"
            np_in_dtype = np.int8
            qmin, qmax = -128, 127

        op = str(self.desc.get("operation", "equal")).lower()
        if op not in _OP_MAP:
            raise ValueError(f"Unsupported comparison operation: {op}")
        _, op_enum = _OP_MAP[op]

        input_shape_1 = tuple(self.desc["input_1_shape"])
        input_shape_2 = tuple(self.desc["input_2_shape"])
        output_shape = _broadcast_shape(input_shape_1, input_shape_2)

        builder = TemplateContextBuilder()
        input_1_dims = builder.nhwc_to_cmsis_dims(input_shape_1)
        input_2_dims = builder.nhwc_to_cmsis_dims(input_shape_2)
        output_dims = builder.nhwc_to_cmsis_dims(output_shape)

        # Default quantization: LiteRT builder uses scale 0.125 (s8) or 1/32768 (s16), zero point 0.
        input_zp_1 = 0
        input_zp_2 = 0

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_1_f = self.rng.uniform(-1.0, 1.0, size=input_shape_1).astype(np.float32)
        input_2_f = self.rng.uniform(-1.0, 1.0, size=input_shape_2).astype(np.float32)
        self.rng.__setstate__(rng_state)

        input_1_q = np.round(input_1_f).astype(np.int32)
        input_1_q = np.clip(input_1_q, qmin, qmax).astype(np_in_dtype)
        input_2_q = np.round(input_2_f).astype(np.int32)
        input_2_q = np.clip(input_2_q, qmin, qmax).astype(np_in_dtype)

        expected = self._simulate_compare(input_1_q, input_2_q, op_enum)

        context = {
            "name": name,
            "prefix": name,
            "input_1_dims": input_1_dims,
            "input_2_dims": input_2_dims,
            "output_dims": output_dims,
            "input_1_data_array": builder.format_array_as_c_literal(input_1_q),
            "input_2_data_array": builder.format_array_as_c_literal(input_2_q),
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "input_dtype": c_type,
            "kernel_fn": kernel_fn,
            "output_size": int(np.prod(output_shape)),
            "input_1_offset": int(-input_zp_1),
            "input_1_mult": 1,
            "input_1_shift": 0,
            "input_2_offset": int(-input_zp_2),
            "input_2_mult": 1,
            "input_2_shift": 0,
            "left_shift": 0,
            "operation": op_enum,
        }

        includes_api_dir = Path(output_dir) / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        h_content = self.render_template("comparison/comparison.h.j2", context)
        (includes_api_dir / f"{name}_comparison.h").write_text(h_content)
        c_content = self.render_template("comparison/comparison.c.j2", context)
        (Path(output_dir) / f"{name}_comparison.c").write_text(c_content)

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "Comparison"),
            "operator_name": "comparison",
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (Path(output_dir) / "CMakeLists.txt").write_text(cmake_content)
