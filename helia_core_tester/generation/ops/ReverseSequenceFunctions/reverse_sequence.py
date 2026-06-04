"""ReverseSequence operation implementation."""

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpReverseSequence(OperationBase):
    """ReverseSequence operation."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("ReverseSequence uses LiteRT-only model generation.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        from helia_core_tester.generation.utils.litert_builder import (
            LiteRtSingleOpBuilder, TensorSpec, _default_quant,
        )
        import ai_edge_litert.schema_py_generated as litert

        activation_dtype = self.desc.get("activation_dtype", "S8")
        tensor_type = litert.TensorType.INT16 if activation_dtype == "S16" else litert.TensorType.INT8
        input_shape = tuple(self.desc["input_shape"])
        seq_lengths = self.desc["seq_lengths"]
        seq_dim = int(self.desc["seq_dim"])
        batch_dim = int(self.desc["batch_dim"])
        batch_size = input_shape[batch_dim]

        builder = LiteRtSingleOpBuilder(op_name="REVERSE_SEQUENCE")
        input_idx = builder.add_tensor(TensorSpec(
            name="input", shape=input_shape, tensor_type=tensor_type, is_input=True,
            quantization=_default_quant(tensor_type),
        ))
        seq_len_idx = builder.add_tensor(TensorSpec(
            name="seq_lengths", shape=(batch_size,), tensor_type=litert.TensorType.INT32, is_input=True,
        ))
        output_idx = builder.add_tensor(TensorSpec(
            name="output", shape=input_shape, tensor_type=tensor_type, is_output=True,
            quantization=_default_quant(tensor_type),
        ))

        opts = litert.ReverseSequenceOptionsT()
        opts.seqDim = seq_dim
        opts.batchDim = batch_dim
        builder.add_operator("REVERSE_SEQUENCE", inputs=[input_idx, seq_len_idx],
            outputs=[output_idx], options=opts, options_type=litert.BuiltinOptions.ReverseSequenceOptions)
        self._write_tflite_bytes(out_path, builder.build())

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_reverse_sequence_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_reverse_sequence_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        ki = self._select_kernel()
        input_shape = list(self.desc["input_shape"])
        seq_lengths = list(self.desc["seq_lengths"])
        seq_dim = int(self.desc["seq_dim"])
        batch_dim = int(self.desc["batch_dim"])
        rank = len(input_shape)

        rng = self._seeded_rng()
        np_dtype = np.int16 if ki["np_dtype"] == "int16" else np.int8
        input_data = rng.integers(ki["qmin"], ki["qmax"] + 1, size=input_shape, dtype=np_dtype)

        # Use TFLite interpreter with INT32 model (runtime doesn't support INT8/INT16 for this op)
        from ai_edge_litert.interpreter import Interpreter
        from helia_core_tester.generation.utils.litert_builder import LiteRtSingleOpBuilder, TensorSpec
        import ai_edge_litert.schema_py_generated as litert

        ref_builder = LiteRtSingleOpBuilder(op_name="REVERSE_SEQUENCE")
        inp_idx = ref_builder.add_tensor(TensorSpec(
            name="input", shape=tuple(input_shape), tensor_type=litert.TensorType.INT32, is_input=True,
        ))
        seq_idx = ref_builder.add_tensor(TensorSpec(
            name="seq_lengths", shape=(input_shape[batch_dim],), tensor_type=litert.TensorType.INT32, is_input=True,
        ))
        out_idx = ref_builder.add_tensor(TensorSpec(
            name="output", shape=tuple(input_shape), tensor_type=litert.TensorType.INT32, is_output=True,
        ))
        opts = litert.ReverseSequenceOptionsT()
        opts.seqDim = seq_dim
        opts.batchDim = batch_dim
        ref_builder.add_operator("REVERSE_SEQUENCE", inputs=[inp_idx, seq_idx], outputs=[out_idx],
            options=opts, options_type=litert.BuiltinOptions.ReverseSequenceOptions)
        interp = Interpreter(model_content=bytes(ref_builder.build()))
        interp.allocate_tensors()
        inp_details = interp.get_input_details()
        out_details = interp.get_output_details()
        interp.set_tensor(inp_details[0]["index"], input_data.astype(np.int32))
        interp.set_tensor(inp_details[1]["index"], np.array(seq_lengths, dtype=np.int32))
        interp.invoke()
        output_data = interp.get_tensor(out_details[0]["index"]).astype(np_dtype)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
            "prefix": name,
            "rank": rank,
            "input_shape": input_shape,
            "seq_lengths": seq_lengths,
            "seq_dim": seq_dim,
            "batch_dim": batch_dim,
            "input_size": int(np.prod(input_shape)),
            "output_size": int(np.prod(input_shape)),
            "num_batches": input_shape[batch_dim],
            "input_data_array": builder.format_array_as_c_literal(input_data),
            "seq_lengths_array": builder.format_array_as_c_literal(np.array(seq_lengths, dtype=np.int32)),
            "expected_output_array": builder.format_array_as_c_literal(output_data),
            "c_type": ki["c_type"],
            "kernel_fn": ki["kernel_fn"],
        }

        includes_dir = output_dir / "includes"
        includes_dir.mkdir(parents=True, exist_ok=True)

        h_content = self.render_template("ReverseSequenceFunctions/reverse_sequence/reverse_sequence.h.j2", context)
        (includes_dir / f"{name}_reverse_sequence.h").write_text(h_content)

        c_content = self.render_template("ReverseSequenceFunctions/reverse_sequence/reverse_sequence.c.j2", context)
        (output_dir / f"{name}_reverse_sequence.c").write_text(c_content)

        cmake_content = self.render_template("common/CMakeLists.txt.j2", {
            "name": name, "operator": "ReverseSequence", "operator_name": "reverse_sequence"
        })
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
