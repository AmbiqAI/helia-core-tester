"""Fill operation implementation (TFLite FILL: one value splatted over a shape).

Formerly a tester-only extension with no generated templates; ns-cmsis-nn#475 ships
arm_nn_fill_f32/arm_nn_fill_f16 under Source/BasicMathFunctions, so Fill is now a CMSIS
parity operator. The kernel is a bit copy of the value (sign and NaN payload included);
the golden comes from numpy.
"""

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_fill_op


_FLOAT_KERNELS: Dict[str, Dict[str, object]] = {
    "FP32": {"kernel_fn": "arm_nn_fill_f32", "c_type": "float", "np_dtype": np.float32, "litert": "float32"},
    "FP16": {"kernel_fn": "arm_nn_fill_f16", "c_type": "float16_t", "np_dtype": np.float16, "litert": "float16"},
}

# Descriptor spellings of the non-finite and signed-zero fill values. YAML's own
# `.nan`/`.inf`/`-.inf` also parse, but the strings keep the descriptor greppable.
_VALUE_TOKENS = {
    "nan": math.nan,
    "-nan": -math.nan,
    "inf": math.inf,
    "+inf": math.inf,
    "-inf": -math.inf,
    "-0": -0.0,
    "-0.0": -0.0,
}


def parse_fill_value(raw: Any) -> float:
    """Descriptor `value` to a Python float; strings name the non-finite tokens."""
    if isinstance(raw, bool):
        raise ValueError(f"Fill value must be a number, got {raw!r}")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in _VALUE_TOKENS:
            return _VALUE_TOKENS[token]
        try:
            return float(token)
        except ValueError as exc:
            raise ValueError(f"Unsupported Fill value {raw!r}") from exc
    raise ValueError(f"Unsupported Fill value {raw!r}")


class OpFill(OperationBase):
    """Fill operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Fill uses LiteRT-only model generation.")

    def _kernel(self) -> Dict[str, object]:
        dtype = self.tensor_dtype("output")
        try:
            return _FLOAT_KERNELS[dtype]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported Fill dtype: {dtype}") from exc

    def _output_shape(self):
        shape = tuple(int(dim) for dim in self.desc["output_shape"])
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Fill output_shape must be non-negative, got {shape}")
        return shape

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        model_bytes = build_fill_op(
            output_shape=self._output_shape(),
            dtype=str(self._kernel()["litert"]),
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        kernel = self._kernel()
        np_dtype = kernel["np_dtype"]
        output_shape = self._output_shape()
        value = np.array(parse_fill_value(self.desc["value"]), dtype=np_dtype)
        builder = TemplateContextBuilder()

        expected = np.full(output_shape, value, dtype=np_dtype)
        block_size = int(expected.size)
        # A block_size of 0 is a documented no-op; keep one element of storage so the
        # arrays stay non-empty, and validate over 0 elements.
        golden = expected if block_size > 0 else np.zeros(1, dtype=np_dtype)

        context = {
            "name": name,
            "kernel_fn": kernel["kernel_fn"],
            "output_dtype": kernel["c_type"],
            "block_size": block_size,
            # The value goes through a static initializer rather than a literal in the
            # call: that is the shape in which NAN/INFINITY are observed to keep their
            # bits at -Ofast (README, "Non-finite inputs").
            "fill_value_array": builder.format_array_as_c_literal(value.reshape(1)),
            "fill_value_repr": repr(float(value)),
            "expected_output_array": builder.format_array_as_c_literal(golden),
            "validation_mode": "float",
        }
        cmake_context = {"name": name, "operator": self.desc.get("operator", "Fill"), "operator_name": "fill"}
        self._write_op_outputs(
            output_dir,
            "fill",
            "BasicMathFunctions/fill/fill.h.j2",
            "BasicMathFunctions/fill/fill.c.j2",
            context,
            cmake_context,
        )
