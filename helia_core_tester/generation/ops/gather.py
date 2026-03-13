"""
Gather operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpGather(OperationBase):
    """
    Gather operation - gathers slices along an axis.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Gather uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_gather_op

        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        else:
            raise NotImplementedError(f"Unsupported Gather dtype: {activation_dtype}")

        input_shape = tuple(self.desc["input_shape"])
        indices_shape = tuple(self.desc["indices_shape"])
        axis = int(self.desc.get("axis", 0))
        batch_dims = int(self.desc.get("batch_dims", 0))

        model_bytes = build_gather_op(
            input_shape=input_shape,
            indices_shape=indices_shape,
            axis=axis,
            batch_dims=batch_dims,
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)

    def _select_cmsis_gather_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            return {
                "kernel_fn": "arm_gather_s8",
                "input_c_type": "int8_t",
                "output_c_type": "int8_t",
            }
        if activation_dtype == "S16":
            return {
                "kernel_fn": "arm_gather_s16",
                "input_c_type": "int16_t",
                "output_c_type": "int16_t",
            }
        raise NotImplementedError(f"Unsupported Gather dtype: {activation_dtype}")

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
        tflite_path = output_dir / f"{name}.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

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

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            input_q = self.rng.integers(-128, 128, size=input_shape, dtype=np_in_dtype)
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            input_q = self.rng.integers(-32768, 32768, size=input_shape, dtype=np_in_dtype)
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        axis_size = int(input_shape[axis])
        indices_q = self.rng.integers(0, axis_size, size=indices_shape, dtype=np.int32)

        self.rng.__setstate__(rng_state)

        output_q = np.take(input_q, indices_q, axis=axis)

        input_dims = self._shape_to_dims(input_shape)
        indices_dims = self._shape_to_dims(indices_shape)
        output_dims = self._shape_to_dims(output_shape)

        context = {
            "name": name,
            "prefix": name,
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

        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("gather/gather.h.j2", context)
        h_path = includes_api_dir / f"{name}_gather.h"
        with open(h_path, "w") as f:
            f.write(h_content)

        c_content = self.render_template("gather/gather.c.j2", context)
        c_path = output_dir / f"{name}_gather.c"
        with open(c_path, "w") as f:
            f.write(c_content)

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "Gather"),
            "operator_name": "gather",
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, "w") as f:
            f.write(cmake_content)

