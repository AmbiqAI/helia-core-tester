"""Pack operation implementation (TFLite PACK: stack equal-shape tensors along a new axis)."""

from pathlib import Path
from typing import Dict

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase


_FLOAT_KERNELS: Dict[str, Dict[str, object]] = {
    # arm_pack_f32/f16 (ns-cmsis-nn#475): bit copy, any rank (0 included), any axis.
    "FP32": {"kernel_fn": "arm_pack_f32", "c_type": "float", "np_dtype": np.float32},
    "FP16": {"kernel_fn": "arm_pack_f16", "c_type": "float16_t", "np_dtype": np.float16},
}


class OpPack(OperationBase):
    """Pack operation."""

    def needs_tflite(self) -> bool:
        # The golden is computed in numpy; nothing reads a .tflite.
        return False

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
            "",  # the registered pool renders the header; no header template exists
            "ConcatenationFunctions/pack/pack.c.j2",
            context,
            cmake_context,
        )


from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, HarnessInput  # noqa: E402
from helia_core_tester.generation.harness.registry import harness_pool  # noqa: E402


@harness_pool("ConcatenationFunctions/pack/pack.c.j2", label="Pack")
def pack_argument_pool(context):
    """Pack takes an array of input pointers; the case lists them at file scope."""
    n, dtype, count = context["name"], context["input_dtype"], int(context["num_inputs"])
    header = []
    if int(context["input_dims_count"]) > 0:
        header.append(Declaration(f"{n}_input_shape", "int32_t", ArrayLiteral(context["input_shape_array"]), array=True))
    header.append(Declaration(f"{n}_output_shape", "int32_t", ArrayLiteral(context["output_shape_array"]), array=True))
    header += [Declaration(f"{n}_input{i + 1}", dtype, ArrayLiteral(context["input_data_arrays"][i]), array=True)
               for i in range(count)]
    header.append(Declaration(f"{n}_expected_output", context["output_dtype"],
                              ArrayLiteral(context["expected_output_array"]), array=True))
    pointers = Declaration(f"{n}_input_ptrs", f"{dtype}*", ArrayLiteral("\n".join(f"    {n}_input{i + 1}," for i in range(count))),
                           array=True, comment="Array of input pointers")
    values = {"num_inputs": str(count), "input_dims": str(context["input_dims_count"]),
              "input_shape": f"{n}_input_shape" if int(context["input_dims_count"]) > 0 else "NULL", "axis": str(context["axis"])}
    return ArgumentPool(
        name=n, values=values, header=header, source=(pointers,),
        inputs=(HarnessInput("input_data", "input_ptrs", f"{n}_input_ptrs", f"{dtype}* const"),),
        output_count=f"({context['output_size']})", benchmark=False, scratch_buffer=False,
    )
