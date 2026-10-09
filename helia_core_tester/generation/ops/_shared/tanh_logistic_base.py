"""Shared golden path of int16 Tanh and Logistic: TFLite's reference_integer_ops on the C reference."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool
from helia_core_tester.generation.ops._shared.base import OperationBase


class TanhLogisticBase(OperationBase):
    """arm_tanh_s16 (OPERATOR "Tanh") and arm_logistic_s16 (OPERATOR "Logistic"); CMSIS-NN has no s8 form."""

    OPERATOR = "Tanh"
    ENTRY = "tanh"

    def uses_reference(self) -> bool:
        return True

    def kernel_fn(self) -> str:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype != 'S16':
            raise NotImplementedError(f"Unsupported {self.OPERATOR} dtype: {activation_dtype} (only S16 supported)")
        return f'arm_{self.ENTRY}_s16'

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.bindings import get_bindings
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        kernel_fn = self.kernel_fn()
        name = self.desc['name']
        shape = tuple(int(d) for d in self.desc['input_shape'])
        if not shape or any(d < 1 for d in shape):
            raise ValueError(f"{name}: invalid input_shape {shape}")

        # The table covers |x| < 10.7 (tanh) / 21.3 (logistic); [-1, 1] is the range the
        # converter used to calibrate over. The output scale is fixed at 2^-15 by the op.
        in_quant = self.activation_quant("input", (-1.0, 1.0), "s16")
        out_quant = (policy.descriptor_quant((self.desc.get("quantization") or {}).get("output"), "s16")
                     or policy.TensorQuant(2.0 ** -15, 0, "s16"))
        input_q = policy.quantize(self._sample_uniform(shape, low=-1.0, high=1.0), in_quant)
        params = get_bindings().prepare(f"{self.ENTRY}_prepare", {
            "input_scale": in_quant.scale, "input_zero_point": in_quant.zero_point,
            "output_scale": out_quant.scale, "output_zero_point": out_quant.zero_point})
        output = self.reference_golden(ReferenceCall(
            f"{self.ENTRY}_s16", params, {"input": np.ascontiguousarray(input_q)}, {"output": shape},
            quant={"input": in_quant.to_json(), "output": out_quant.to_json()},
        ))

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'input_dims': builder.nhwc_to_cmsis_dims(shape),
            'output_dims': builder.nhwc_to_cmsis_dims(shape),
            'output_size': int(np.prod(shape)),
            'input_multiplier': params["input_multiplier"],
            'input_left_shift': params["input_left_shift"],
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output),
            'input_dtype': 'int16_t',
            'output_dtype': 'int16_t',
            'kernel_fn': kernel_fn,
        }
        self.render_harness_case(
            output_dir, stem=self.ENTRY, context=context,
            pool=tensor_case_pool(context, {"input_size": context["output_size"],
                                            "input_multiplier": context["input_multiplier"],
                                            "input_left_shift": context["input_left_shift"]},
                                  output_count=dims_count(context["output_dims"])),
            validation_key=f"ActivationFunctions/{self.ENTRY}/{self.ENTRY}.c.j2", label=self.OPERATOR,
            operator=self.OPERATOR,
        )
