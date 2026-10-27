"""Float DIV tests against the C reference, retaining compact broadcast operands."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.binary_basic_math_base import BinaryBasicMathBase
# Div renders through Mul's binary harness pool, which that module registers.
from helia_core_tester.generation.ops.BasicMathFunctions import mul as _mul  # noqa: F401
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


class OpDiv(BinaryBasicMathBase):
    """Generate flat and NHWC broadcast division for FP32 and FP16 storage."""

    def validate_descriptor(self) -> None:
        self._dtype()
        shapes = [tuple(self.desc[key]) for key in ("input_1_shape", "input_2_shape")]
        if any(not 1 <= len(shape) <= 4 or any(dim <= 0 for dim in shape) for shape in shapes):
            raise ValueError("Div requires positive shapes of rank 1 through 4")
        np.broadcast_shapes(*shapes)
        dtype = self._dtype()
        lo = dtype.type(self.desc.get("act_min", -1000))
        hi = dtype.type(self.desc.get("act_max", 1000))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError("Div requires finite ordered activation bounds")

    def uses_reference(self) -> bool:
        return True

    def _dtype(self):
        dtype = self.tensor_dtype("input")
        if dtype not in ("FP16", "FP32") or self.tensor_dtype("output") != dtype:
            raise ValueError("Div requires matching FP16 or FP32 input/output storage")
        return np.dtype("float16" if dtype == "FP16" else "float32")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference.call import ReferenceCall

        self.validate_descriptor()
        dtype = self._dtype()
        suffix = "f16" if dtype == np.dtype("float16") else "f32"
        c_type = "float16_t" if suffix == "f16" else "float"
        name = self.desc["name"]
        shape_a = tuple(self.desc["input_1_shape"])
        shape_b = tuple(self.desc["input_2_shape"])
        shape_out = np.broadcast_shapes(shape_a, shape_b)
        broadcast = self._float_broadcast_call(auto_on_shape_mismatch=True)
        rng = self._seeded_rng()
        a = rng.uniform(-8, 8, shape_a).astype(dtype)
        b = rng.uniform(0.25, 4, shape_b).astype(dtype)
        b.flat[::2] *= -1
        lo = dtype.type(self.desc.get("act_min", -1000))
        hi = dtype.type(self.desc.get("act_max", 1000))
        # The binary64 quotient of the stored operands, rounded once, then clamped.
        expected = self.reference_golden(ReferenceCall(
            f"div_{suffix}", {"activation_min": float(lo), "activation_max": float(hi)},
            {"input1": np.ascontiguousarray(a), "input2": np.ascontiguousarray(b)}, {"output": tuple(shape_out)}))
        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "input1_dims": builder.nhwc_to_cmsis_dims((1,) * (4 - len(shape_a)) + shape_a),
            "input2_dims": builder.nhwc_to_cmsis_dims((1,) * (4 - len(shape_b)) + shape_b),
            "output_dims": builder.nhwc_to_cmsis_dims((1,) * (4 - len(shape_out)) + shape_out),
            "input1_data_array": builder.format_array_as_c_literal(a),
            "input2_data_array": builder.format_array_as_c_literal(b),
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "input_dtype": c_type,
            "output_dtype": c_type,
            "block_size": int(np.prod(shape_out)),
            "kernel_fn": f"arm_elementwise_div_{'broadcast_' if broadcast else ''}{suffix}",
            "float_kernel": True,
            "float_broadcast": broadcast,
            "out_activation_min_literal": builder.format_float_literal(float(lo)),
            "out_activation_max_literal": builder.format_float_literal(float(hi)),
            "validation_label": "Div",
            "validation_mode": "float",
            "validation_atol": 0.0,
            "validation_rtol": 0.0,
            "expected_status": self.expected_status(),
        }
        # Mul's existing template accepts a binary float kernel and its call
        # style; reusing it keeps the guarded invocation and assertion in one place.
        self._write_op_outputs(
            output_dir, "mul", "BasicMathFunctions/mul/mul.h.j2",
            "BasicMathFunctions/mul/mul.c.j2", context,
            {"name": name, "operator": "Div", "operator_name": "mul"},
        )
