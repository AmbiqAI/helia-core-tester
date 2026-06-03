"""DynamicUpdateSlice operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpDynamicUpdateSlice(OperationBase):
    """DynamicUpdateSlice operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("DynamicUpdateSlice uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_shape_transform_op
        activation_dtype = self.desc.get("activation_dtype", "S8")
        dtype = "int16" if activation_dtype == "S16" else "int8"
        operand_shape = tuple(self.desc["operand_shape"])
        model_bytes = build_shape_transform_op(
            op_name="RESHAPE", input_shape=operand_shape, output_shape=operand_shape, dtype=dtype,
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_dynamic_update_slice_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_dynamic_update_slice_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        operand_shape = list(self.desc["operand_shape"])
        update_shape = list(self.desc["update_shape"])
        start_indices = list(self.desc["start_indices"])
        rank = len(operand_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        operand_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=operand_shape, dtype=np_dtype)
        update_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=update_shape, dtype=np_dtype)

        # Clamp start indices
        clamped_starts = [
            min(max(0, start_indices[i]), operand_shape[i] - update_shape[i])
            for i in range(rank)
        ]

        # Reference: copy operand, then overwrite slice
        output_data = operand_data.copy()
        slices = tuple(slice(s, s + u) for s, u in zip(clamped_starts, update_shape))
        output_data[slices] = update_data

        # Compute operand strides
        operand_strides = []
        stride = 1
        for d in reversed(operand_shape):
            operand_strides.insert(0, stride)
            stride *= d

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
            "rank": rank,
            "operand_shape": operand_shape,
            "update_shape": update_shape,
            "start_indices": start_indices,
            "operand_size": int(np.prod(operand_shape)),
            "update_size": int(np.prod(update_shape)),
            "operand_strides": operand_strides,
            "operand_data_array": builder.format_array_as_c_literal(operand_data),
            "update_data_array": builder.format_array_as_c_literal(update_data),
            "start_indices_array": builder.format_array_as_c_literal(np.array(start_indices, dtype=np.int32)),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "c_type": ki["c_type"],
            "kernel_fn": ki["kernel_fn"],
        }

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("DynamicUpdateSliceFunctions/dynamic_update_slice/dynamic_update_slice.h.j2", context)
        (includes_dir / f"{name}_dynamic_update_slice.h").write_text(h_content)

        c_content = self.render_template("DynamicUpdateSliceFunctions/dynamic_update_slice/dynamic_update_slice.c.j2", context)
        (output_dir / f"{name}_dynamic_update_slice.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            "name": name, "operator": "DynamicUpdateSlice", "operator_name": "dynamic_update_slice"
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
