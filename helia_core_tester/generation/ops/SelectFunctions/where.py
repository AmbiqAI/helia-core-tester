"""Where operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpWhere(OperationBase):
    """Where operation - returns coordinates of non-zero elements."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Where uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_shape_transform_op
        activation_dtype = self.desc.get("activation_dtype", "S8")
        dtype = "int16" if activation_dtype == "S16" else "int8"
        input_shape = tuple(self.desc["input_shape"])
        model_bytes = build_shape_transform_op(
            op_name="RESHAPE", input_shape=input_shape, output_shape=input_shape, dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_where_s16", "c_type": "int16_t", "cond_c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_where_s8", "c_type": "int8_t", "cond_c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        input_shape = list(self.desc["input_shape"])
        rank = len(input_shape)
        total_elements = int(np.prod(input_shape))

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        # Generate condition with ~50% non-zero
        condition = rng.integers(-5, 6, size=input_shape, dtype=np_dtype)

        # Reference: find non-zero coordinates
        nonzero_coords = np.argwhere(condition != 0)
        num_true = len(nonzero_coords)
        # Output is int32 array of shape [num_true, rank]
        output_data = nonzero_coords.astype(np.int32)
        max_output_size = total_elements * rank  # worst case all true

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
            "rank": rank,
            "input_shape": input_shape,
            "total_elements": total_elements,
            "num_true": num_true,
            "max_output_size": max_output_size,
            "condition_array": builder.format_array_as_c_literal(condition),
            "expected_output_array": builder.format_array_as_c_literal(output_data.flatten()),
            "cond_c_type": ki["cond_c_type"],
            "kernel_fn": ki["kernel_fn"],
        }

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("SelectFunctions/where/where.h.j2", context)
        (includes_dir / f"{name}_where.h").write_text(h_content)

        c_content = self.render_template("SelectFunctions/where/where.c.j2", context)
        (output_dir / f"{name}_where.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            "name": name, "operator": "Where", "operator_name": "where"
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
