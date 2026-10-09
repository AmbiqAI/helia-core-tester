"""Shared implementation of the int16 LUT activations (Tanh, Logistic)."""

from pathlib import Path
from typing import Dict

import numpy as np

from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool
from helia_core_tester.generation.ops._shared.base import OperationBase

# TFLite fixes the output of both at zero point 0, scale 2^-15.
OUTPUT_SCALE = 1.0 / 32768.0


class LutActivationS16Base(OperationBase):
    """Tanh / Logistic on int16: the TFLM prepare gives the kernel parameters and the
    TFLM reference kernel the golden. CMSIS-NN has no int8 variant of either."""

    OPERATOR_NAME = "Tanh"
    STEM = "tanh"
    LOGISTIC = False

    def uses_reference(self) -> bool:
        self._kernel()
        return True

    def _kernel(self) -> Dict[str, str]:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype != 'S16':
            raise NotImplementedError(
                f"Unsupported {self.OPERATOR_NAME} dtype: {activation_dtype} (only S16 supported)"
            )
        return {'kernel_fn': f'arm_{self.STEM}_s16', 'input_c_type': 'int16_t', 'output_c_type': 'int16_t'}

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference import bindings as b
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.case import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._kernel()
        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        if not input_shape or any(d < 1 for d in input_shape):
            raise ValueError(f"{name}: invalid input_shape {input_shape}")

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        input_data = self.rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        self.rng.__setstate__(rng_state)

        quant = self.activation_quant("input", input_data, "s16")
        if quant.zero_point != 0:
            raise ValueError(f"{name}: int16 {self.OPERATOR_NAME} needs zero point 0, got {quant.zero_point}")
        input_q = policy.quantize(input_data, quant)
        params = b.get_bindings().tanh_logistic_s16_prepare(self.LOGISTIC, quant.scale, OUTPUT_SCALE)
        prepared = b.struct_to_dict(params)
        call = ReferenceCall(
            f"{self.STEM}_s16", prepared, {"input": input_q}, input_shape, "int16",
            quant={"input": quant.to_json(), "output": {"scale": OUTPUT_SCALE, "zero_point": 0, "dtype": "s16"}},
        )
        output_data = self.reference_golden(call)

        builder = TemplateContextBuilder()
        dims = builder.nhwc_to_cmsis_dims(input_shape)
        context = {
            'name': name,
            'input_dims': dims,
            'output_dims': dims,
            'output_size': int(np.prod(input_shape)),
            'input_multiplier': prepared["input_multiplier"],
            'input_left_shift': prepared["input_left_shift"],
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        values = {"input_size": context["output_size"], "input_multiplier": context["input_multiplier"],
                  "input_left_shift": context["input_left_shift"]}
        self.render_harness_case(
            output_dir, stem=self.STEM, context=context,
            pool=tensor_case_pool(context, values, output_count=dims_count(context["output_dims"])),
            validation_key=f"ActivationFunctions/{self.STEM}/{self.STEM}.c.j2", label=self.OPERATOR_NAME,
            operator=self.OPERATOR_NAME,
        )
