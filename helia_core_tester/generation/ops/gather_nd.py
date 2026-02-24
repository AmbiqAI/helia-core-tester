"""
GatherND operation implementation.
"""

from typing import Dict
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops.base import OperationBase


class OpGatherND(OperationBase):
    """
    GatherND operation - gathers slices from params using indices.
    """

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("GatherND uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import build_gather_nd_op

        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            dtype = "int8"
        elif activation_dtype == "S16":
            dtype = "int16"
        else:
            raise NotImplementedError(f"Unsupported GatherND dtype: {activation_dtype}")

        params_shape = tuple(self.desc["input_shape"])
        indices_shape = tuple(self.desc["indices_shape"])

        model_bytes = build_gather_nd_op(
            params_shape=params_shape,
            indices_shape=indices_shape,
            dtype=dtype,
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)

    def _select_cmsis_gather_nd_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S8":
            return {
                "kernel_fn": "arm_gather_nd_s8",
                "input_c_type": "int8_t",
                "output_c_type": "int8_t",
            }
        if activation_dtype == "S16":
            return {
                "kernel_fn": "arm_gather_nd_s16",
                "input_c_type": "int16_t",
                "output_c_type": "int16_t",
            }
        raise NotImplementedError(f"Unsupported GatherND dtype: {activation_dtype}")

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

        kernel_info = self._select_cmsis_gather_nd_kernel()
        builder = TemplateContextBuilder()

        params_shape = tuple(self.desc["input_shape"])
        indices_shape = tuple(self.desc["indices_shape"])
        batch_dims = int(self.desc.get("batch_dims", 0))

        params_rank = len(params_shape)
        indices_rank = len(indices_shape)
        indices_nd = int(indices_shape[-1])

        if batch_dims != 0:
            raise ValueError("Only batch_dims=0 is supported in the current GatherND generator.")

        output_shape = indices_shape[:-1] + params_shape[indices_nd:]

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            params_q = self.rng.integers(-128, 128, size=params_shape, dtype=np_in_dtype)
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            params_q = self.rng.integers(-32768, 32768, size=params_shape, dtype=np_in_dtype)
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")

        # Build indices within valid range for each dimension
        indices_q = np.zeros(indices_shape, dtype=np.int32)
        for i in range(indices_nd):
            dim_size = int(params_shape[i])
            rand_vals = self.rng.integers(0, dim_size, size=indices_shape[:-1], dtype=np.int32)
            indices_q[..., i] = rand_vals

        self.rng.__setstate__(rng_state)

        # Compute expected output
        output_q = np.zeros(output_shape, dtype=np_in_dtype)
        flat_indices = indices_q.reshape(-1, indices_nd)
        flat_output = output_q.reshape(-1, *params_shape[indices_nd:])

        for out_idx, idx_tuple in enumerate(flat_indices):
            src = params_q[tuple(idx_tuple)]
            flat_output[out_idx] = src

        output_q = flat_output.reshape(output_shape)

        params_dims = self._shape_to_dims(params_shape)
        indices_dims = self._shape_to_dims(indices_shape)
        output_dims = self._shape_to_dims(output_shape)

        context = {
            "name": name,
            "prefix": name,
            "kernel_fn": kernel_info["kernel_fn"],
            "input_dtype": kernel_info["input_c_type"],
            "output_dtype": kernel_info["output_c_type"],
            "params_dims": params_dims,
            "indices_dims": indices_dims,
            "output_dims": output_dims,
            "params_rank": params_rank,
            "indices_rank": indices_rank,
            "batch_dims": batch_dims,
            "params_shape_array": builder.format_array_as_c_literal(np.array(params_shape, dtype=np.int32)),
            "indices_shape_array": builder.format_array_as_c_literal(np.array(indices_shape, dtype=np.int32)),
            "output_shape_array": builder.format_array_as_c_literal(np.array(output_shape, dtype=np.int32)),
            "params_data_array": builder.format_array_as_c_literal(params_q),
            "indices_data_array": builder.format_array_as_c_literal(indices_q),
            "expected_output_array": builder.format_array_as_c_literal(output_q),
            "output_size": int(np.prod(output_shape)),
        }

        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("gather_nd/gather_nd.h.j2", context)
        h_path = includes_api_dir / f"{name}_gather_nd.h"
        with open(h_path, "w") as f:
            f.write(h_content)

        c_content = self.render_template("gather_nd/gather_nd.c.j2", context)
        c_path = output_dir / f"{name}_gather_nd.c"
        with open(c_path, "w") as f:
            f.write(c_content)

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "GatherND"),
            "operator_name": "gather_nd",
        }
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        cmake_path = output_dir / "CMakeLists.txt"
        with open(cmake_path, "w") as f:
            f.write(cmake_content)

        print(f"Generated C/H files and CMakeLists.txt for {name}")
