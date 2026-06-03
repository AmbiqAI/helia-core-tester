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

        # Reference implementation
        output_data = input_data.copy()
        batch_size = input_shape[batch_dim]
        for b in range(batch_size):
            seq_len = seq_lengths[b]
            # Build index slices for this batch
            idx = [slice(None)] * rank
            idx[batch_dim] = b
            batch_slice = input_data[tuple(idx)]
            # Reverse along seq_dim (relative to batch_slice which has one fewer dim)
            actual_seq_dim = seq_dim if seq_dim < batch_dim else seq_dim - 1
            rev_idx = [slice(None)] * (rank - 1)
            rev_idx[actual_seq_dim] = slice(None, seq_len)
            result = batch_slice.copy()
            chunk = result[tuple(rev_idx)]
            result[tuple(rev_idx)] = np.flip(chunk, axis=actual_seq_dim)
            output_data[tuple(idx)] = result

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
            "num_batches": batch_size,
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
