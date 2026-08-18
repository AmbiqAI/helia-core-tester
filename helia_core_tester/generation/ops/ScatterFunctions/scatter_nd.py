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
        from helia_core_tester.generation.utils.litert_builder import (
            LiteRtSingleOpBuilder, TensorSpec, _default_quant,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get("activation_dtype", "S8")
        tensor_type = litert.TensorType.INT16 if activation_dtype == "S16" else litert.TensorType.INT8

        output_shape = tuple(self.desc["input_shape"])
        indices = np.array(self.desc["indices"], dtype=np.int32)
        updates_raw = self.desc["updates"]
        np_dtype = np.int16 if activation_dtype == "S16" else np.int8
        updates = np.array(updates_raw, dtype=np_dtype)

        num_updates = indices.shape[0]
        index_depth = indices.shape[1] if indices.ndim > 1 else 1
        indices_shape = (num_updates, index_depth) if indices.ndim > 1 else (num_updates,)
        updates_shape = tuple(updates.shape)

        builder = LiteRtSingleOpBuilder(op_name="SCATTER_ND")
        indices_idx = builder.add_tensor(TensorSpec(
            name="indices", shape=indices_shape, tensor_type=litert.TensorType.INT32, is_input=True,
        ))
        updates_idx = builder.add_tensor(TensorSpec(
            name="updates", shape=updates_shape, tensor_type=tensor_type, is_input=True,
            quantization=_default_quant(tensor_type),
        ))
        shape_idx = builder.add_tensor(TensorSpec(
            name="shape", shape=(len(output_shape),), tensor_type=litert.TensorType.INT32,
            is_input=False, data=np.array(output_shape, dtype=np.int32),
        ))
        output_idx = builder.add_tensor(TensorSpec(
            name="output", shape=output_shape, tensor_type=tensor_type, is_output=True,
            quantization=_default_quant(tensor_type),
        ))
        builder.add_operator("SCATTER_ND", inputs=[indices_idx, updates_idx, shape_idx],
            outputs=[output_idx], options=None, options_type=litert.BuiltinOptions.NONE)
        self._write_tflite_bytes(out_path, builder.build())

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

        # Use TFLite interpreter for reference output; fall back to INT32 model if type unsupported
        tflite_path = str(output_dir / f"{name}.tflite")
        try:
            interpreter = self.load_litert_interpreter(tflite_path)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], indices)
            interpreter.set_tensor(input_details[1]["index"], updates)
            interpreter.invoke()
            output_data = np.array(interpreter.get_tensor(output_details[0]["index"]))
        except (ValueError, RuntimeError):
            # Rebuild with INT32 updates (SCATTER_ND doesn't support INT16)
            from ai_edge_litert.interpreter import Interpreter
            from helia_core_tester.generation.utils.litert_builder import LiteRtSingleOpBuilder, TensorSpec
            import ai_edge_litert.schema_py_generated as litert
            b = LiteRtSingleOpBuilder(op_name="SCATTER_ND")
            i_idx = b.add_tensor(TensorSpec(name="indices", shape=indices.shape, tensor_type=litert.TensorType.INT32, is_input=True))
            u_idx = b.add_tensor(TensorSpec(name="updates", shape=updates.shape, tensor_type=litert.TensorType.INT32, is_input=True))
            s_idx = b.add_tensor(TensorSpec(name="shape", shape=(len(output_shape),), tensor_type=litert.TensorType.INT32,
                is_input=False, data=np.array(output_shape, dtype=np.int32)))
            o_idx = b.add_tensor(TensorSpec(name="output", shape=tuple(output_shape), tensor_type=litert.TensorType.INT32, is_output=True))
            b.add_operator("SCATTER_ND", inputs=[i_idx, u_idx, s_idx], outputs=[o_idx], options=None, options_type=litert.BuiltinOptions.NONE)
            interp = Interpreter(model_content=bytes(b.build()))
            interp.allocate_tensors()
            inp_d = interp.get_input_details()
            out_d = interp.get_output_details()
            interp.set_tensor(inp_d[0]["index"], indices)
            interp.set_tensor(inp_d[1]["index"], updates.astype(np.int32))
            interp.invoke()
            output_data = interp.get_tensor(out_d[0]["index"]).astype(np_dtype)

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
