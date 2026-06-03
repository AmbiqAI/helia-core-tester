"""ScatterNd operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpScatterNd(OperationBase):
    """ScatterNd operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("ScatterNd uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        # ScatterNd does not have a standard TFLite builtin that maps cleanly.
        # We generate test data directly in generate_c_files using numpy reference.
        # Write a minimal placeholder tflite.
        from helia_core_tester.generation.utils.litert_builder import (
            build_shape_transform_op, TensorSpec,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get("activation_dtype", "S8")
        dtype = "int16" if activation_dtype == "S16" else "int8"
        input_shape = tuple(self.desc["input_shape"])

        # Build a trivial identity-like model as placeholder
        model_bytes = build_shape_transform_op(
            op_name="RESHAPE",
            input_shape=input_shape,
            output_shape=input_shape,
            dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_scatter_nd_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_scatter_nd_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        output_shape = list(self.desc["input_shape"])  # scatter_nd output shape = input_shape in descriptor
        indices = np.array(self.desc["indices"], dtype=np.int32)
        updates_raw = self.desc["updates"]

        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        updates = np.array(updates_raw, dtype=np_dtype)

        num_updates = indices.shape[0]
        index_depth = indices.shape[1] if indices.ndim > 1 else 1
        slice_size = int(np.prod(updates.shape[1:])) if updates.ndim > 1 else 1
        output_size = int(np.prod(output_shape))

        # Compute output strides
        output_strides = []
        stride = 1
        for d in reversed(output_shape[:index_depth]):
            output_strides.insert(0, stride)
            stride *= d

        # Reference: scatter into zeros
        output_data = np.zeros(output_shape, dtype=np_dtype)
        for i in range(num_updates):
            idx = tuple(indices[i]) if indices.ndim > 1 else (int(indices[i]),)
            if updates.ndim > 1:
                output_data[idx] = updates[i]
            else:
                output_data[idx] = updates[i]

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
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

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("ScatterFunctions/scatter_nd/scatter_nd.h.j2", context)
        (includes_dir / f"{name}_scatter_nd.h").write_text(h_content)

        c_content = self.render_template("ScatterFunctions/scatter_nd/scatter_nd.c.j2", context)
        (output_dir / f"{name}_scatter_nd.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            "name": name, "operator": "ScatterNd", "operator_name": "scatter_nd"
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
