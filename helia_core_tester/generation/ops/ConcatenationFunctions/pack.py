"""Pack operation implementation (TFLite PACK: stack equal-shape tensors along a new axis)."""

from pathlib import Path
from typing import Dict

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_pack_op


_FLOAT_KERNELS: Dict[str, Dict[str, object]] = {
    # arm_pack_f32/f16 (ns-cmsis-nn#475): bit copy, any rank (0 included), any axis.
    "FP32": {"kernel_fn": "arm_pack_f32", "c_type": "float", "np_dtype": np.float32, "litert": "float32"},
    "FP16": {"kernel_fn": "arm_pack_f16", "c_type": "float16_t", "np_dtype": np.float16, "litert": "float16"},
}


class OpPack(OperationBase):
    """Pack operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Pack uses LiteRT-only model generation.")

    def _kernel(self) -> Dict[str, object]:
        dtype = self.tensor_dtype("input")
        try:
            return _FLOAT_KERNELS[dtype]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported Pack dtype: {dtype}") from exc

    def _geometry(self):
        input_shape = tuple(int(dim) for dim in self.desc["input_shape"])
        num_inputs = int(self.desc["num_inputs"])
        axis = int(self.desc["axis"])
        if axis < 0:
            axis += len(input_shape) + 1
        if not 0 <= axis <= len(input_shape):
            raise ValueError(f"Pack axis {self.desc['axis']} out of range for rank {len(input_shape)} inputs")
        if num_inputs < 1:
            raise ValueError("Pack requires num_inputs >= 1")
        return input_shape, num_inputs, axis

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        input_shape, num_inputs, axis = self._geometry()
        model_bytes = build_pack_op(
            input_shape=input_shape,
            num_inputs=num_inputs,
            axis=axis,
            dtype=str(self._kernel()["litert"]),
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        kernel = self._kernel()
        np_dtype = kernel["np_dtype"]
        input_shape, num_inputs, axis = self._geometry()
        builder = TemplateContextBuilder()

        # One seeded draw covers every operand so the inputs differ from each other;
        # only the first carries a non-finite sweep, the rest stay finite neighbours
        # that must arrive untouched (the concatenation descriptors do the same).
        rng = self._seeded_rng()
        inputs = []
        for index in range(num_inputs):
            operand = rng.uniform(-1.0, 1.0, size=input_shape).astype(np_dtype)
            if index == 0:
                operand = self._maybe_apply_input_mode(operand)
            inputs.append(operand)

        expected = np.stack(inputs, axis=axis).astype(np_dtype)
        output_shape = tuple(int(dim) for dim in expected.shape)

        context = {
            "name": name,
            "kernel_fn": kernel["kernel_fn"],
            "input_dtype": kernel["c_type"],
            "output_dtype": kernel["c_type"],
            "num_inputs": num_inputs,
            "input_dims_count": len(input_shape),
            "axis": axis,
            "input_shape_array": builder.format_array_as_c_literal(np.array(input_shape, dtype=np.int32)),
            "output_shape_array": builder.format_array_as_c_literal(np.array(output_shape, dtype=np.int32)),
            "output_size": int(expected.size),
            "input_data_arrays": [builder.format_array_as_c_literal(operand) for operand in inputs],
            "expected_output_array": builder.format_array_as_c_literal(expected),
            "validation_mode": "float",
        }
        cmake_context = {"name": name, "operator": self.desc.get("operator", "Pack"), "operator_name": "pack"}
        self._write_op_outputs(
            output_dir,
            "pack",
            "ConcatenationFunctions/pack/pack.h.j2",
            "ConcatenationFunctions/pack/pack.c.j2",
            context,
            cmake_context,
        )
