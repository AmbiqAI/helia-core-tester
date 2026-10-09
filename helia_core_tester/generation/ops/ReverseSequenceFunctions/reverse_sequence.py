"""ReverseSequence operation implementation."""
from pathlib import Path

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpReverseSequence(OperationBase):
    """ReverseSequence operation."""

    def needs_tflite(self) -> bool:
        # The golden is computed in numpy; nothing reads a .tflite.
        return False

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

        output_data = reverse_sequence(input_data, seq_lengths, seq_dim=seq_dim, batch_dim=batch_dim)

        builder = TemplateContextBuilder()
        context = {
            "name": name,
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

        self.render_harness_case(
            Path(output_dir), stem="reverse_sequence", context=context, pool=reverse_sequence_argument_pool(context),
            validation_key="ReverseSequenceFunctions/reverse_sequence/reverse_sequence.c.j2", label="ReverseSequence", operator="ReverseSequence",
        )


from helia_core_tester.generation.harness.simple import shaped_case_pool  # noqa: E402


def reverse_sequence(data: np.ndarray, seq_lengths, *, seq_dim: int, batch_dim: int) -> np.ndarray:
    """TFLite REVERSE_SEQUENCE: for each index b along batch_dim, reverse the first
    seq_lengths[b] elements along seq_dim and leave the rest in place."""
    rank = data.ndim
    if not (0 <= seq_dim < rank and 0 <= batch_dim < rank) or seq_dim == batch_dim:
        raise ValueError(f"invalid seq_dim {seq_dim} / batch_dim {batch_dim} for rank {rank}")
    if len(seq_lengths) != data.shape[batch_dim]:
        raise ValueError(f"{len(seq_lengths)} seq_lengths for batch extent {data.shape[batch_dim]}")
    out = data.copy()
    axis = seq_dim - (1 if seq_dim > batch_dim else 0)
    for b, length in enumerate(int(v) for v in seq_lengths):
        if not 0 <= length <= data.shape[seq_dim]:
            raise ValueError(f"seq_lengths[{b}] = {length} outside [0, {data.shape[seq_dim]}]")
        index = [slice(None)] * rank
        index[batch_dim] = b
        index[seq_dim] = slice(0, length)
        out[tuple(index)] = np.flip(data[tuple(index)], axis=axis)
    return out


def reverse_sequence_argument_pool(context):
    n = context["name"]
    return shaped_case_pool(
        context, shapes=(("shape", "input_shape"),), params_type="cmsis_nn_reverse_sequence_params",
        params={"rank": context["rank"], "shape": f"{n}_shape", "seq_dim": context["seq_dim"],
                "batch_dim": context["batch_dim"]},
        inputs=(("input", "input", "input_data_array"), ("seq_lengths", "seq_lengths", "seq_lengths_array")),
        seq_lengths_ctype="int32_t", output_count=str(context["output_size"]))
