"""ScatterNd operation implementation."""
from pathlib import Path

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpScatterNd(OperationBase):
    """ScatterNd operation."""

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_scatter_nd_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_scatter_nd_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        output_shape = list(self.desc["input_shape"])
        indices = np.array(self.desc["indices"], dtype=np.int32)
        updates_raw = self.desc["updates"]

        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        updates = np.array(updates_raw, dtype=np_dtype)

        num_updates = indices.shape[0]
        index_depth = indices.shape[1] if indices.ndim > 1 else 1
        slice_size = int(np.prod(updates.shape[1:])) if updates.ndim > 1 else 1
        output_size = int(np.prod(output_shape))

        # Compute output strides for each indexed dimension.
        # strides[d] = product of output_shape[d+1:]
        output_strides = []
        for d in range(index_depth):
            output_strides.append(int(np.prod(output_shape[d + 1:])))

        output_data = scatter_nd(indices, updates, output_shape)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "rank": len(output_shape),
            "output_shape": output_shape,
            "num_updates": num_updates,
            "index_depth": index_depth,
            "slice_size": slice_size,
            "output_size": output_size,
            "output_strides": output_strides,
            "indices_array": builder.format_array_as_c_literal(indices.flatten()),
            "updates_array": builder.format_array_as_c_literal(updates.flatten()),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "c_type": ki["c_type"],
            "kernel_fn": ki["kernel_fn"],
        }

        self.render_harness_case(
            Path(output_dir), stem="scatter_nd", context=context, pool=scatter_nd_argument_pool(context),
            validation_key="ScatterFunctions/scatter_nd/scatter_nd.c.j2", label="ScatterNd", operator="ScatterNd",
        )


from helia_core_tester.generation.harness.simple import shaped_case_pool  # noqa: E402


def scatter_nd(indices: np.ndarray, updates: np.ndarray, shape) -> np.ndarray:
    """TFLite SCATTER_ND: zero output, updates added at each index (duplicates accumulate,
    wrapping in the tensor type)."""
    idx = indices.reshape(indices.shape[0], -1)
    depth = idx.shape[1]
    if depth > len(shape) or updates.shape[0] != idx.shape[0]:
        raise ValueError(f"indices {indices.shape} / updates {updates.shape} do not fit output {list(shape)}")
    if np.any(idx < 0) or np.any(idx >= np.asarray(shape[:depth])):
        raise ValueError(f"scatter index outside output {list(shape)}")
    out = np.zeros(shape, dtype=np.int64)
    for row, update in zip(idx, updates.astype(np.int64)):
        out[tuple(row)] += update
    return out.astype(updates.dtype)


def scatter_nd_argument_pool(context):
    """ScatterND accumulates into its output, so the case zero-seeds it before the call."""
    n = context["name"]
    return shaped_case_pool(
        context, shapes=(("output_strides", "output_strides"),), params_type="cmsis_nn_scatter_nd_params",
        params={"num_updates": context["num_updates"], "index_depth": context["index_depth"],
                "slice_size": context["slice_size"], "output_size": context["output_size"],
                "output_strides": f"{n}_output_strides"},
        inputs=(("indices", "indices", "indices_array"), ("updates", "updates", "updates_array")),
        indices_ctype="int32_t", output_count=str(context["output_size"]), includes=("<string.h>",),
        test_prologue=f"    memset({n}_output, 0, sizeof({n}_output));")
