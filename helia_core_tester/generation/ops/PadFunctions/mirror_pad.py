"""MirrorPad operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpMirrorPad(OperationBase):
    """MirrorPad operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("MirrorPad uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import (
            build_shape_transform_op, TensorSpec,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get("activation_dtype", "S8")
        dtype = "int16" if activation_dtype == "S16" else "int8"
        input_shape = tuple(self.desc["input_shape"])
        paddings = self.desc["paddings"]
        output_shape = tuple(
            input_shape[i] + paddings[i][0] + paddings[i][1]
            for i in range(len(input_shape))
        )

        paddings_flat = []
        for p in paddings:
            paddings_flat.extend(p)
        paddings_tensor = TensorSpec(
            name="paddings",
            shape=(len(paddings), 2),
            tensor_type=litert.TensorType.INT32,
            is_input=False,
            data=np.array(paddings_flat, dtype=np.int32),
        )

        model_bytes = build_shape_transform_op(
            op_name="MIRROR_PAD",
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=dtype,
            extra_input_tensors=[paddings_tensor],
        )
        self._write_tflite_bytes(out_path, model_bytes)

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_mirror_pad_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_mirror_pad_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        input_shape = list(self.desc["input_shape"])
        paddings = self.desc["paddings"]
        mode_str = self.desc.get("mode", "reflect")
        mode_int = 0 if mode_str == "reflect" else 1

        pad_before = [p[0] for p in paddings]
        output_shape = [input_shape[i] + paddings[i][0] + paddings[i][1] for i in range(len(input_shape))]
        rank = len(input_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        input_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=input_shape, dtype=np_dtype)

        np_mode = "reflect" if mode_str == "reflect" else "symmetric"
        pad_width = [(p[0], p[1]) for p in paddings]
        output_data = np.pad(input_data, pad_width, mode=np_mode)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
            "rank": rank,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "pad_before": pad_before,
            "mode": mode_int,
            "input_size": int(np.prod(input_shape)),
            "output_size": int(np.prod(output_shape)),
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "c_type": ki["c_type"],
            "kernel_fn": ki["kernel_fn"],
        }

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("PadFunctions/mirror_pad/mirror_pad.h.j2", context)
        (includes_dir / f"{name}_mirror_pad.h").write_text(h_content)

        c_content = self.render_template("PadFunctions/mirror_pad/mirror_pad.c.j2", context)
        (output_dir / f"{name}_mirror_pad.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            "name": name, "operator": "MirrorPad", "operator_name": "mirror_pad"
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
