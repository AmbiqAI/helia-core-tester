"""Shared implementation for CMSIS-NN comparison generation."""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.tflite_utils import simulate_compare


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


def _near_step(params: Dict[str, int]) -> int:
    """Smallest input gap the rescale keeps."""
    gains = [
        params[f"input_{i}_mult"] * 2.0 ** (params[f"input_{i}_shift"] + params["left_shift"] - 31)
        for i in (1, 2)
    ]
    return max(1, int(np.ceil(1.0 / min(gains))))


def _pin_near(rng, source, target, output_shape, qmin, qmax, gap) -> None:
    """Tie or nudge target to source."""
    src_ids = np.broadcast_to(np.arange(source.size).reshape(source.shape), output_shape).reshape(-1)
    dst_ids = np.broadcast_to(np.arange(target.size).reshape(target.shape), output_shape).reshape(-1)
    # One output per target element.
    order = np.argsort(dst_ids, kind="stable")
    counts = np.bincount(dst_ids, minlength=target.size)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    picks = order[starts + np.arange(target.size) % counts]
    src = source.reshape(-1)[src_ids[picks]]
    flat = target.reshape(-1)
    flat[::3] = src[::3]
    step = gap * rng.choice([-1, 1], size=flat[1::3].size)
    near = src[1::3] + step
    # Step inward at the dtype bounds.
    flat[1::3] = np.where((near < qmin) | (near > qmax), src[1::3] - step, near)


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

    def _quant_params(self, tflite_path: Path) -> Dict[str, int]:
        """Read operand scales from the model."""
        from helia_core_tester.generation.utils.tflite_utils import comparison_quant_params, scalar_scale_zp

        inputs = self.load_primary_operator_tensors(str(tflite_path))["inputs"]
        (scale_1, zp_1), (scale_2, zp_2) = (scalar_scale_zp(t["quantization"]) for t in inputs[:2])
        return comparison_quant_params(scale_1, zp_1, scale_2, zp_2)

    def _sample_operands(self, rng, shape_1, shape_2, output_shape, qmin, qmax, np_dtype, gap):
        """Draw spread operands with forced ties."""
        input_1 = rng.integers(qmin, qmax + 1, size=shape_1, dtype=np.int32)
        input_2 = rng.integers(qmin, qmax + 1, size=shape_2, dtype=np.int32)
        # A scalar at the median splits outputs.
        if input_1.size == 1:
            input_1[...] = np.median(input_2)
        elif input_2.size == 1:
            input_2[...] = np.median(input_1)
        # Per three outputs: tie, near, free.
        if input_2.size >= input_1.size:
            _pin_near(rng, input_1, input_2, output_shape, qmin, qmax, gap)
        else:
            _pin_near(rng, input_2, input_1, output_shape, qmin, qmax, gap)
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
        rng = self._seeded_rng()
        # Redraw until the output mixes.
        for _ in range(16):
            input_1_q, input_2_q = self._sample_operands(
                rng, input_shape_1, input_shape_2, output_shape, qmin, qmax, np_in_dtype, _near_step(params)
            )
            expected = simulate_compare(input_1_q, input_2_q, operation=op_enum, **params)
            if np.unique(expected).size > 1:
                break

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
