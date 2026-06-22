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
            LiteRtSingleOpBuilder, TensorSpec, _default_quant,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get("activation_dtype", "S8")
        tensor_type = litert.TensorType.INT16 if activation_dtype == "S16" else litert.TensorType.INT8

        output_shape = tuple(self.desc["input_shape"])
        condition_shape = tuple(self.desc.get("condition_shape", output_shape))
        x_shape = tuple(self.desc.get("x_shape", output_shape))
        y_shape = tuple(self.desc.get("y_shape", output_shape))

        builder = LiteRtSingleOpBuilder(op_name="SELECT_V2")
        cond_idx = builder.add_tensor(TensorSpec(
            name="condition", shape=condition_shape, tensor_type=litert.TensorType.BOOL, is_input=True,
        ))
        x_idx = builder.add_tensor(TensorSpec(
            name="x", shape=x_shape, tensor_type=tensor_type, is_input=True,
            quantization=_default_quant(tensor_type),
        ))
        y_idx = builder.add_tensor(TensorSpec(
            name="y", shape=y_shape, tensor_type=tensor_type, is_input=True,
            quantization=_default_quant(tensor_type),
        ))
        output_idx = builder.add_tensor(TensorSpec(
            name="output", shape=output_shape, tensor_type=tensor_type, is_output=True,
            quantization=_default_quant(tensor_type),
        ))
        builder.add_operator("SELECT_V2", inputs=[cond_idx, x_idx, y_idx],
            outputs=[output_idx], options=None, options_type=litert.BuiltinOptions.SelectV2Options)
        self._write_tflite_bytes(out_path, builder.build())

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
            elif padded[i] == 1:
                strides[i] = 0  # broadcast dimension
            else:
                raise ValueError(f"Shapes not broadcastable: {tensor_shape} vs {output_shape}")
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

        condition = rng.integers(0, 2, size=condition_shape, dtype=np.bool_)
        x_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=x_shape, dtype=np_dtype)
        y_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=y_shape, dtype=np_dtype)

        # Use TFLite interpreter for reference output (fallback to numpy if unsupported)
        tflite_path = str(output_dir / f"{name}.tflite")
        try:
            interpreter = self.load_litert_interpreter(tflite_path)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], condition)
            interpreter.set_tensor(input_details[1]["index"], x_data)
            interpreter.set_tensor(input_details[2]["index"], y_data)
            interpreter.invoke()
            output_data = np.array(interpreter.get_tensor(output_details[0]["index"]))
        except (ValueError, RuntimeError):
            output_data = np.where(condition, x_data, y_data)

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
            "condition_c_type": "bool",
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
