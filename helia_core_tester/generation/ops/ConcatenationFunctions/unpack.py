"""Unpack operation implementation (TFLite UNPACK: unstack a tensor along one axis)."""

from pathlib import Path
from typing import Dict

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_unpack_op


_FLOAT_KERNELS: Dict[str, Dict[str, object]] = {
    # arm_unpack_f32/f16 (ns-cmsis-nn#475): bit copy, any rank >= 1, any axis.
    "FP32": {"kernel_fn": "arm_unpack_f32", "c_type": "float", "np_dtype": np.float32, "litert": "float32"},
    "FP16": {"kernel_fn": "arm_unpack_f16", "c_type": "float16_t", "np_dtype": np.float16, "litert": "float16"},
}


class OpUnpack(OperationBase):
    """Unpack operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Unpack uses LiteRT-only model generation.")

    def _kernel(self) -> Dict[str, object]:
        dtype = self.tensor_dtype("input")
        try:
            return _FLOAT_KERNELS[dtype]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported Unpack dtype: {dtype}") from exc

    def _geometry(self):
        input_shape = tuple(int(dim) for dim in self.desc["input_shape"])
        if not input_shape:
            raise ValueError("Unpack requires a rank >= 1 input_shape")
        axis = int(self.desc["axis"])
        if axis < 0:
            axis += len(input_shape)
        if not 0 <= axis < len(input_shape):
            raise ValueError(f"Unpack axis {self.desc['axis']} out of range for rank {len(input_shape)}")
        if input_shape[axis] < 1:
            raise ValueError("Unpack axis must have at least one slice")
        return input_shape, axis

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        input_shape, axis = self._geometry()
        model_bytes = build_unpack_op(
            input_shape=input_shape,
            axis=axis,
            dtype=str(self._kernel()["litert"]),
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        kernel = self._kernel()
        np_dtype = kernel["np_dtype"]
        input_shape, axis = self._geometry()
        builder = TemplateContextBuilder()

        input_data = self._sample_uniform(input_shape, dtype=np_dtype)
        slices = [
            np.ascontiguousarray(np.take(input_data, index, axis=axis)).astype(np_dtype)
            for index in range(input_shape[axis])
        ]

        outputs = []
        for index, slab in enumerate(slices):
            size = int(slab.size)
            # A zero-element slice (some other axis has extent 0) keeps one element of
            # storage and a non-empty golden; the validation runs over size 0.
            golden = slab if size > 0 else np.zeros(1, dtype=np_dtype)
            outputs.append({
                "name": f"{name}_out_{index}",
                "expected_output_array": builder.format_array_as_c_literal(golden),
                "size": size,
            })

        context = {
            "name": name,
            "kernel_fn": kernel["kernel_fn"],
            "input_dtype": kernel["c_type"],
            "output_dtype": kernel["c_type"],
            "input_dims_count": len(input_shape),
            "axis": axis,
            "num_outputs": len(outputs),
            "input_shape_array": builder.format_array_as_c_literal(np.array(input_shape, dtype=np.int32)),
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "outputs": outputs,
            "validation_mode": "float",
        }
        cmake_context = {"name": name, "operator": self.desc.get("operator", "Unpack"), "operator_name": "unpack"}
        self._write_op_outputs(
            output_dir,
            "unpack",
            "ConcatenationFunctions/unpack/unpack.h.j2",
            "ConcatenationFunctions/unpack/unpack.c.j2",
            context,
            cmake_context,
        )
