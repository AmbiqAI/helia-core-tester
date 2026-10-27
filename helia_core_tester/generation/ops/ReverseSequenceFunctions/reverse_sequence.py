"""ReverseSequence operation implementation."""
from pathlib import Path

from typing import Dict
import numpy as np
from pathlib import Path as _Path
from helia_core_tester.generation.ops._shared.base import OperationBase


class OpReverseSequence(OperationBase):
    """ReverseSequence operation."""

    def uses_reference(self) -> bool:
        return True

    def _select_kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get("activation_dtype", "S8")
        if activation_dtype == "S16":
            return {"kernel_fn": "arm_reverse_sequence_s16", "c_type": "int16_t", "np_dtype": "int16", "qmin": -32768, "qmax": 32767}
        return {"kernel_fn": "arm_reverse_sequence_s8", "c_type": "int8_t", "np_dtype": "int8", "qmin": -128, "qmax": 127}

    def generate_c_files(self, output_dir: _Path) -> None:
        from helia_core_tester.generation.reference.call import ReferenceCall
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

        output_data = self.reference_golden(ReferenceCall(
            "reverse_sequence", {"seq_dim": seq_dim, "batch_dim": batch_dim},
            {"input": input_data, "seq_lengths": np.array(seq_lengths, dtype=np.int32)},
            {"output": tuple(input_shape)}))

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


def reverse_sequence_argument_pool(context):
    n = context["name"]
    return shaped_case_pool(
        context, shapes=(("shape", "input_shape"),), params_type="cmsis_nn_reverse_sequence_params",
        params={"rank": context["rank"], "shape": f"{n}_shape", "seq_dim": context["seq_dim"],
                "batch_dim": context["batch_dim"]},
        inputs=(("input", "input", "input_data_array"), ("seq_lengths", "seq_lengths", "seq_lengths_array")),
        seq_lengths_ctype="int32_t", output_count=str(context["output_size"]))
