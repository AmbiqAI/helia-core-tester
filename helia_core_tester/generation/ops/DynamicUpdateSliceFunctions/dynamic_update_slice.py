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
        from helia_core_tester.generation.utils.litert_builder import (
            LiteRtSingleOpBuilder, TensorSpec, _default_quant,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get("activation_dtype", "S8")
        tensor_type = litert.TensorType.INT16 if activation_dtype == "S16" else litert.TensorType.INT8
        operand_shape = tuple(self.desc["operand_shape"])
        update_shape = tuple(self.desc["update_shape"])
        rank = len(operand_shape)

        builder = LiteRtSingleOpBuilder(op_name="DYNAMIC_UPDATE_SLICE")
        operand_idx = builder.add_tensor(TensorSpec(
            name="operand", shape=operand_shape, tensor_type=tensor_type, is_input=True,
            quantization=_default_quant(tensor_type),
        ))
        update_idx = builder.add_tensor(TensorSpec(
            name="update", shape=update_shape, tensor_type=tensor_type, is_input=True,
            quantization=_default_quant(tensor_type),
        ))
        start_idx = builder.add_tensor(TensorSpec(
            name="start_indices", shape=(rank,), tensor_type=litert.TensorType.INT32, is_input=True,
        ))
        output_idx = builder.add_tensor(TensorSpec(
            name="output", shape=operand_shape, tensor_type=tensor_type, is_output=True,
            quantization=_default_quant(tensor_type),
        ))

        opts = litert.DynamicUpdateSliceOptionsT()
        builder.add_operator("DYNAMIC_UPDATE_SLICE", inputs=[operand_idx, update_idx, start_idx],
            outputs=[output_idx], options=opts, options_type=litert.BuiltinOptions.DynamicUpdateSliceOptions)
        self._write_tflite_bytes(out_path, builder.build())

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

        # Use TFLite interpreter for reference output (fallback to numpy if unsupported)
        tflite_path = str(output_dir / f"{name}.tflite")
        try:
            interpreter = self.load_litert_interpreter(tflite_path)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], operand_data)
            interpreter.set_tensor(input_details[1]["index"], update_data)
            interpreter.set_tensor(input_details[2]["index"], np.array(start_indices, dtype=np.int32))
            interpreter.invoke()
            output_data = np.array(interpreter.get_tensor(output_details[0]["index"]))
        except (ValueError, RuntimeError):
            clamped_starts = [
                min(max(0, start_indices[i]), operand_shape[i] - update_shape[i])
                for i in range(rank)
            ]
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
