"""SelectV2 operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpSelectV2(OperationBase):
    """SelectV2 operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("SelectV2 uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import (
            build_shape_transform_op,
        )
        activation_dtype = self.desc.get("activation_dtype", "S8")
        dtype = "int16" if activation_dtype == "S16" else "int8"
        output_shape = tuple(self.desc["input_shape"])
        model_bytes = build_shape_transform_op(
            op_name="RESHAPE",
            input_shape=output_shape,
            output_shape=output_shape,
            dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_select_v2_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_select_v2_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def _compute_broadcast_strides(self, tensor_shape, output_shape):
        """Compute broadcast strides for a tensor relative to output shape."""
        rank = len(output_shape)
        # Pad tensor_shape to match rank
        padded = [1] * (rank - len(tensor_shape)) + list(tensor_shape)
        strides = [0] * rank
        stride = 1
        for i in range(rank - 1, -1, -1):
            if padded[i] == output_shape[i]:
                strides[i] = stride
            else:
                strides[i] = 0  # broadcast dimension
            stride *= padded[i]
        return strides

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        output_shape = list(self.desc["input_shape"])
        condition_shape = list(self.desc.get("condition_shape", output_shape))
        x_shape = list(self.desc.get("x_shape", output_shape))
        y_shape = list(self.desc.get("y_shape", output_shape))
        rank = len(output_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8

        condition = rng.integers(0, 2, size=condition_shape, dtype=np.int8)
        x_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=x_shape, dtype=np_dtype)
        y_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=y_shape, dtype=np_dtype)

        # Broadcast and compute reference
        cond_bc = np.broadcast_to(condition, output_shape)
        x_bc = np.broadcast_to(x_data, output_shape)
        y_bc = np.broadcast_to(y_data, output_shape)
        output_data = np.where(cond_bc != 0, x_bc, y_bc).astype(np_dtype)

        cond_strides = self._compute_broadcast_strides(condition_shape, output_shape)
        x_strides = self._compute_broadcast_strides(x_shape, output_shape)
        y_strides = self._compute_broadcast_strides(y_shape, output_shape)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
            "rank": rank,
            "output_shape": output_shape,
            "condition_shape": condition_shape,
            "x_shape": x_shape,
            "y_shape": y_shape,
            "cond_strides": cond_strides,
            "x_strides": x_strides,
            "y_strides": y_strides,
            "condition_size": int(np.prod(condition_shape)),
            "x_size": int(np.prod(x_shape)),
            "y_size": int(np.prod(y_shape)),
            "output_size": int(np.prod(output_shape)),
            "condition_array": builder.format_array_as_c_literal(condition),
            "x_data_array": builder.format_array_as_c_literal(x_data),
            "y_data_array": builder.format_array_as_c_literal(y_data),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "c_type": ki["c_type"],
            "kernel_fn": ki["kernel_fn"],
        }

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("SelectFunctions/select_v2/select_v2.h.j2", context)
        (includes_dir / f"{name}_select_v2.h").write_text(h_content)

        c_content = self.render_template("SelectFunctions/select_v2/select_v2.c.j2", context)
        (output_dir / f"{name}_select_v2.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            "name": name, "operator": "SelectV2", "operator_name": "select_v2"
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
