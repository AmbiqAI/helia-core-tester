"""
Gather operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.io.dtypes import (
    descriptor_dtype_to_c_type,
    get_resolved_tensor_dtype,
)
from helia_core_tester.generation.ops._shared.base import OperationBase


# The kernel entry point per resolved element dtype. Gather moves values and
# never computes one, so the float entry points differ from the integer ones
# only in the element type they copy.
_GATHER_KERNEL_BY_DTYPE = {
    "S8": "arm_gather_s8",
    "S16": "arm_gather_s16",
    "FP32": "arm_gather_f32",
    "FP16": "arm_gather_f16",
}


class OpGather(OperationBase):
    """
    Gather operation - gathers slices along an axis.
    """

    def needs_tflite(self) -> bool:
        # The golden is computed in numpy; nothing reads a .tflite.
        return False

    def _element_dtype(self) -> str:
        """The resolved element dtype, which for a copy operator is input and output alike.

        Input and output must agree. Gather moves values without converting them, so a
        descriptor asking for different element types is expressing something the kernel
        cannot do; accepting it would silently gather at the input type and compare against
        a golden built at the same type, proving nothing about the mismatch it asked for.
        """
        # No default: resolve_tensor_dtypes already raises on a descriptor with no input
        # dtype, so a fallback here would be unreachable code that reads like a safety net.
        dtype = get_resolved_tensor_dtype(self.desc, "input")
        output_dtype = get_resolved_tensor_dtype(self.desc, "output", dtype)
        if output_dtype != dtype:
            raise ValueError(
                f"Gather copies elements and cannot convert them: descriptor "
                f"{self.desc.get('name')!r} asks for input {dtype} with output {output_dtype}."
            )
        if dtype not in _GATHER_KERNEL_BY_DTYPE:
            raise NotImplementedError(f"Unsupported Gather dtype: {dtype}")
        return dtype

    def _select_cmsis_gather_kernel(self) -> Dict[str, str]:
        dtype = self._element_dtype()
        c_type = descriptor_dtype_to_c_type(dtype)
        return {
            "kernel_fn": _GATHER_KERNEL_BY_DTYPE[dtype],
            "input_c_type": c_type,
            "output_c_type": c_type,
        }

    def _draw_indices(self, indices_shape: tuple[int, ...], axis_size: int) -> np.ndarray:
        """Draw the index array, distinct wherever the shape allows it.

        Distinctness is the property that lets an index case fail. Drawing uniformly left
        it to the seed, and one seed produced {0, 0} for the outer-axis case -- the only
        case covering that copy shape -- so a kernel that ignored the index array and
        always read slice zero passed it.

        When there are more indices than the axis has slices, repeats are forced by
        pigeonhole and are the point rather than a weakness: that is how the repeated-index
        case gets its repeats, and it is the case that catches a kernel consuming a source
        slice or advancing the source pointer once per output slice.
        """
        index_count = int(np.prod(indices_shape))
        if index_count <= axis_size:
            flat_indices = self.rng.choice(axis_size, size=index_count, replace=False)
        else:
            flat_indices = self.rng.integers(0, axis_size, size=index_count)
        return flat_indices.astype(np.int32).reshape(indices_shape)

    @staticmethod
    def _shape_to_dims(shape: tuple[int, ...]) -> Dict[str, int]:
        if len(shape) == 1:
            return {"n": int(shape[0]), "h": 1, "w": 1, "c": 1}
        if len(shape) == 2:
            return {"n": int(shape[0]), "h": int(shape[1]), "w": 1, "c": 1}
        if len(shape) == 3:
            return {"n": int(shape[0]), "h": int(shape[1]), "w": int(shape[2]), "c": 1}
        if len(shape) == 4:
            return {"n": int(shape[0]), "h": int(shape[1]), "w": int(shape[2]), "c": int(shape[3])}
        raise ValueError(f"Unsupported shape length: {len(shape)}")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]

        kernel_info = self._select_cmsis_gather_kernel()
        builder = TemplateContextBuilder()

        input_shape = tuple(self.desc["input_shape"])
        indices_shape = tuple(self.desc["indices_shape"])
        axis = int(self.desc.get("axis", 0))
        batch_dims = int(self.desc.get("batch_dims", 0))

        input_rank = len(input_shape)
        coords_rank = len(indices_shape)

        if axis < 0:
            axis += input_rank

        if batch_dims != 0:
            raise ValueError("Only batch_dims=0 is supported in the current Gather generator.")

        output_shape = input_shape[:axis] + indices_shape[batch_dims:] + input_shape[axis + 1 :]

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        element_dtype = self._element_dtype()
        if element_dtype == "S8":
            np_in_dtype = np.int8
            input_q = self.rng.integers(-128, 128, size=input_shape, dtype=np_in_dtype)
        elif element_dtype == "S16":
            np_in_dtype = np.int16
            input_q = self.rng.integers(-32768, 32768, size=input_shape, dtype=np_in_dtype)
        else:
            # Whole numbers in [-2000, 2000] are exactly representable in both
            # float32 and float16, so the golden array is the same value the
            # kernel copies and zero tolerance is meaningful. The wide range
            # also keeps elements distinct, which is what makes a misplaced
            # index visible: a narrow range would let a wrong element compare
            # equal by coincidence.
            np_in_dtype = np.float32 if element_dtype == "FP32" else np.float16
            input_q = self.rng.integers(-2000, 2001, size=input_shape).astype(np_in_dtype)

        axis_size = int(input_shape[axis])
        indices_q = self._draw_indices(indices_shape, axis_size)

        self.rng.__setstate__(rng_state)

        output_q = np.take(input_q, indices_q, axis=axis)

        input_dims = self._shape_to_dims(input_shape)
        indices_dims = self._shape_to_dims(indices_shape)
        output_dims = self._shape_to_dims(output_shape)

        context = {
            "name": name,
            "kernel_fn": kernel_info["kernel_fn"],
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "input_dims": input_dims,
            "indices_dims": indices_dims,
            "output_dims": output_dims,
            "input_rank": input_rank,
            "coords_rank": coords_rank,
            "axis": axis,
            "batch_dims": batch_dims,
            "input_shape_array": builder.format_array_as_c_literal(np.array(input_shape, dtype=np.int32)),
            "indices_shape_array": builder.format_array_as_c_literal(np.array(indices_shape, dtype=np.int32)),
            "output_shape_array": builder.format_array_as_c_literal(np.array(output_shape, dtype=np.int32)),
            "input_data_array": builder.format_array_as_c_literal(input_q),
            "indices_data_array": builder.format_array_as_c_literal(indices_q),
            "expected_output_array": builder.format_array_as_c_literal(output_q),
            "output_size": int(np.prod(output_shape)),
        }

        self.render_harness_case(
            Path(output_dir), stem="gather", context=context, pool=gather_argument_pool(context),
            validation_key="GatherFunctions/gather/gather.c.j2", label="Gather", operator="Gather",
        )



from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, HarnessInput  # noqa: E402
from helia_core_tester.generation.harness.simple import dims_declaration  # noqa: E402


def gather_argument_pool(context):
    n = context["name"]
    dims = ("input_dims", "indices_dims", "output_dims")
    header = [dims_declaration(f"{n}_{d}", context[d]) for d in dims]
    header += [Declaration(f"{n}_{k}_shape", "int32_t", ArrayLiteral(context[f"{k}_shape_array"]), array=True)
               for k in ("input", "indices", "output")]
    header += [
        Declaration(f"{n}_input", context["input_dtype"], ArrayLiteral(context["input_data_array"]), array=True),
        Declaration(f"{n}_indices", "int32_t", ArrayLiteral(context["indices_data_array"]), array=True),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]), array=True),
    ]
    params = Declaration(f"{n}_params", "cmsis_nn_gather_params",
                         {k: str(context[k]) for k in ("axis", "batch_dims", "input_rank", "coords_rank")}, storage="static")
    return ArgumentPool(
        name=n, values={**{d: f"&{n}_{d}" for d in dims}, "params": f"&{n}_params"}, header=header, source=(params,),
        inputs=(HarnessInput("input_data", "input", f"{n}_input"),
                HarnessInput("indices_data", "indices", f"{n}_indices", "int32_t")),
        output_count=f"({context['output_size']})", benchmark=False, scratch_buffer=False,
    )
