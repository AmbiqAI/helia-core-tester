"""Shared golden path of the hard-swish operators, on the C reference.

Compat s8 is TFLite's HardSwish<int8_t>; precise s8/s16 is CMSIS-NN's int32 variant, a named
reference entry since TFLite has no counterpart; f32/f16 are the exact result rounded once.
"""

from typing import Dict, Tuple
import numpy as np
from pathlib import Path
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness.simple import tensor_case_pool


def hard_swish_values(context: Dict, variant: str) -> Dict:
    """The kernel's scalars by parameter name: the compat kernel takes fixed-point/exponent pairs,
    the precise and float kernels a requantisation plus the two ReLU thresholds and a prescale."""
    if variant == "compat":
        return {"input_offset": context["input_offset"], "output_offset": context["output_offset"],
                "output_multiplier_fp": context["output_multiplier_fp"],
                "output_multiplier_exp": context["output_multiplier_exp"],
                "relu_multiplier_fp": context["relu_multiplier_fp"], "relu_multiplier_exp": context["relu_multiplier_exp"],
                "output_size": context["output_size"]}
    return {"input_offset": context["input_offset"], "output_offset": context["output_offset"],
            "output_multiplier": context["output_mult"], "output_shift": context["output_shift"],
            "relu_q3": context["relu_q3"], "relu_q6": context["relu_q6"], "prescale": context["prescale"],
            "output_size": context["output_size"]}


class HardSwishFamilyBase(OperationBase):
    """Shared implementation for precise and compat hard-swish generation."""

    VARIANT = "precise"
    OPERATOR_NAME = "HardSwishPrecise"
    # The range the converter used to calibrate over; hard_swish maps it onto [-3/8, 8].
    INPUT_RANGE = (-8.0, 8.0)
    OUTPUT_RANGE = (-0.375, 8.0)

    def uses_reference(self) -> bool:
        return True

    def variant_name(self) -> str:
        return self.VARIANT

    def _select_cmsis_hard_swish_kernel(self) -> Dict[str, str]:
        activation_dtype = self.tensor_dtype("input", default="S8")
        variant = self.variant_name()
        if variant == "compat" and activation_dtype != "S8":
            raise NotImplementedError("HardSwishCompat is only supported for S8.")
        if activation_dtype == 'S8':
            return {'kernel_fn': f'arm_hard_swish_{variant}_s8', 'input_c_type': 'int8_t', 'output_c_type': 'int8_t'}
        if activation_dtype == 'S16':
            return {'kernel_fn': 'arm_hard_swish_precise_s16', 'input_c_type': 'int16_t', 'output_c_type': 'int16_t'}
        if activation_dtype == 'FP32':
            return {'kernel_fn': 'arm_hard_swish_f32', 'input_c_type': 'float', 'output_c_type': 'float',
                    'float_kernel': True}
        if activation_dtype == 'FP16':
            return {'kernel_fn': 'arm_hard_swish_f16', 'input_c_type': 'float16_t', 'output_c_type': 'float16_t',
                    'float_kernel': True}
        raise NotImplementedError(f"Unsupported HardSwish dtype: {activation_dtype}")

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference import quant as ref_quant
        from helia_core_tester.generation.reference.bindings import get_bindings
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        kernel_info = self._select_cmsis_hard_swish_kernel()
        if kernel_info.get('float_kernel'):
            self._generate_float_c_files(output_dir, kernel_info)
            return

        name = self.desc['name']
        variant = self.variant_name()
        kind = "s16" if kernel_info["input_c_type"] == "int16_t" else "s8"
        input_shape = tuple(int(d) for d in self.desc['input_shape'])
        if not input_shape or any(d < 1 for d in input_shape):
            raise ValueError(f"{name}: invalid input_shape {input_shape}")

        in_quant = self.activation_quant("input", self.INPUT_RANGE, kind)
        out_quant = self.activation_quant("output", self.OUTPUT_RANGE, kind)
        qmin, qmax = policy.dtype_range(kind)
        # Stay inside the representable input range.
        bound = min(self.INPUT_RANGE[1], min(qmax - in_quant.zero_point, in_quant.zero_point - qmin) * in_quant.scale)
        input_q = policy.quantize(self._sample_uniform(input_shape, low=-bound, high=bound), in_quant)

        entry = f"hard_swish_{kind}" if variant == "compat" else f"hard_swish_precise_{kind}"
        params = get_bindings().prepare(
            "hard_swish_prepare" if variant == "compat" else "hard_swish_precise_prepare",
            {"dtype": ref_quant.hct_dtype(kind.upper()), "input_scale": in_quant.scale,
             "input_zero_point": in_quant.zero_point, "output_scale": out_quant.scale,
             "output_zero_point": out_quant.zero_point})
        output_data = self.reference_golden(ReferenceCall(
            entry, params, {"input": np.ascontiguousarray(input_q)}, {"output": input_shape},
            quant={"input": in_quant.to_json(), "output": out_quant.to_json()},
        ))

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'input_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'output_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'output_size': int(np.prod(input_shape)),
            'input_data_array': builder.format_array_as_c_literal(input_q),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
        }
        if variant == "compat":
            context.update({
                'input_offset': params["input_zero_point"],
                'output_offset': params["output_zero_point"],
                'output_multiplier_fp': params["output_multiplier_fixedpoint_int16"],
                'output_multiplier_exp': params["output_multiplier_exponent"],
                'relu_multiplier_fp': params["reluish_multiplier_fixedpoint_int16"],
                'relu_multiplier_exp': params["reluish_multiplier_exponent"],
            })
            validation_key = "ActivationFunctions/hard_swish/hard_swish_compat.c.j2"
        else:
            context.update({
                'input_offset': params["input_offset"],
                'output_offset': params["output_offset"],
                'output_mult': params["output_multiplier"],
                'output_shift': params["output_shift"],
                'relu_q3': params["relu_q3"],
                'relu_q6': params["relu_q6"],
                'prescale': params["prescale"],
            })
            validation_key = "ActivationFunctions/hard_swish/hard_swish.c.j2"

        self.render_harness_case(
            output_dir, stem="hard_swish", context=context, pool=tensor_case_pool(context, hard_swish_values(context, variant)),
            validation_key=validation_key, label="HardSwish", operator=self.OPERATOR_NAME, sidecar=True,
        )

    def _generate_float_c_files(self, output_dir: Path, kernel_info: Dict[str, str]) -> None:
        """arm_hard_swish_f32/f16 (ns-cmsis-nn #413) against the exact result rounded once.

        The kernels compute x * clamp(fma(x, 1/6, 0.5), 0, 1): x >= 3 returns x and x <= -3
        exact zero, matching the reference bit-exactly; the curved region sits well inside
        the float tolerances (#413 measured 1.24e-7 f32 / 3.4e-4 f16)."""
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        is_f16 = kernel_info["input_c_type"] == "float16_t"
        float_dtype = np.float16 if is_f16 else np.float32
        input_shape = tuple(int(d) for d in self.desc['input_shape'])

        # [-8, 8] reaches the zero region, the curved region and the identity region.
        input_data = self._sample_uniform(
            input_shape,
            low=float(self.desc.get("input_min", -8.0)),
            high=float(self.desc.get("input_max", 8.0)),
            dtype=float_dtype,
        )
        output_data = self.reference_golden(ReferenceCall(
            "hard_swish_f16" if is_f16 else "hard_swish_f32", {"unused": 0},
            {"input": np.ascontiguousarray(input_data)}, {"output": input_shape},
        ))

        builder = TemplateContextBuilder()
        context = {
            'name': name,
            'input_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'output_dims': builder.nhwc_to_cmsis_dims(input_shape),
            'output_size': int(np.prod(input_shape)),
            'input_data_array': builder.format_array_as_c_literal(input_data),
            'expected_output_array': builder.format_array_as_c_literal(output_data),
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'float_kernel': True,
            'validation_mode': 'float',
        }
        self.render_harness_case(
            output_dir, stem="hard_swish", context=context,
            pool=tensor_case_pool(context, {"size": context["output_size"]}),
            validation_key="ActivationFunctions/hard_swish/hard_swish_float.c.j2", label="HardSwish",
            operator=self.OPERATOR_NAME, sidecar=True,
        )
