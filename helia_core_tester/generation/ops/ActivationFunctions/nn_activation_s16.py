"""
NN activation S16 operation implementation (arm_nn_activation_s16).
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from helia_core_tester.generation.harness.simple import tensor_case_pool
from helia_core_tester.generation.ops._shared.base import OperationBase


def nn_activation_reference_params(left_shift: int) -> Dict[str, int]:
    """arm_nn_activation_s16's left_shift as the tanh/logistic reference's (multiplier, shift):
    the kernel scales by 3 << left_shift, or by 3 with a rounding right shift of -left_shift."""
    left_shift = int(left_shift)
    if left_shift >= 0:
        return {"input_multiplier": 3 << left_shift, "input_left_shift": 0}
    return {"input_multiplier": 3, "input_left_shift": -left_shift}


class OpNNActivationS16(OperationBase):
    """
    NN activation (sigmoid/tanh) for int16, on the C reference's TFLite tanh/logistic.
    """

    def uses_reference(self) -> bool:
        return True

    def _activation(self) -> Tuple[str, str]:
        act_type = str(self.desc.get('activation_type', 'TANH')).upper()
        if act_type not in ("SIGMOID", "TANH"):
            raise ValueError(f"Unsupported activation_type: {act_type}")
        return act_type, "logistic_s16" if act_type == "SIGMOID" else "tanh_s16"

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        left_shift = int(self.desc.get('left_shift', 0))
        act_type, entry = self._activation()

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_q = self.rng.integers(-32768, 32768, size=input_shape, dtype=np.int16)
        self.rng.__setstate__(rng_state)

        expected_output = self.reference_golden(ReferenceCall(
            entry, nn_activation_reference_params(left_shift), {"input": np.ascontiguousarray(input_q)},
            {"output": input_shape}, quant={"left_shift": left_shift},
        ))

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'input_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'output_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'left_shift': left_shift,
            'activation_type': f"ARM_{act_type}",
            'output_size': int(np.prod(input_shape)),
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(expected_output),
            'input_dtype': 'int16_t',
            'output_dtype': 'int16_t',
            'kernel_fn': 'arm_nn_activation_s16',
        }

        self.render_harness_case(
            output_dir, stem="nn_activation", context=context, pool=tensor_case_pool(context, {"size": context["output_size"], "left_shift": context["left_shift"], "type": context["activation_type"]}),
            validation_key="ActivationFunctions/nn_activation/nn_activation.c.j2", label="NN activation", operator="NNActivationS16",
        )
