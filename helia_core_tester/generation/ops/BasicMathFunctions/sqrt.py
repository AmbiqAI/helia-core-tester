"""Sqrt operation implementation."""

from math import sqrt
from pathlib import Path
from typing import Any, Dict

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.harness import ArrayLiteral, Declaration
from helia_core_tester.generation.harness.simple import tensor_case_pool


def sqrt_argument_pool(context):
    """Sqrt reads a lookup table the case carries as the file-scope `sqrt_lut`."""
    lut = context["sqrt_lut"]
    rows = "\n".join(", ".join(str(v) for v in lut[i:i + 8]) + "," for i in range(0, len(lut), 8))
    table = Declaration("sqrt_lut", context["lut_c_type"], ArrayLiteral(rows), storage="static", array=True,
                        extent=str(context["lut_size"]))
    return tensor_case_pool(context, {"sqrt_lut": "sqrt_lut"}, dims=("input_dims",), extra_header=(table,))


SQRT_S16_LUT_SIZE = 513
SQRT_S16_SLOT_SHIFT = 7
SQRT_S16_SLOT_HALF_STEP = 1 << (SQRT_S16_SLOT_SHIFT - 1)
SQRT_S16_ACCURATE_FROM = 2048


def clamp_f32(x, min_val, max_val):
    '''Clamp a float32 value to the specified range.'''
    return max(min(x, max_val), min_val)

def _quant_param_to_scalar(value, name: str, cast):
    """Normalize a quantization value to a scalar."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Sqrt expects scalar quantization for {name}, got shape {arr.shape}")
    return cast(arr.reshape(-1)[0])


def _sqrt_quantized_real(real_value: float) -> float:
    """Return sqrt(real_value) while clamping the invalid negative domain to zero."""
    if real_value <= 0.0:
        return 0.0
    return float(sqrt(real_value))


def _quantize_s16_sqrt_output(real_value: float, output_scale: float, output_zp: int) -> int:
    """Truncate on requantizing, as TFLite's int16 sqrt does."""
    quantized_output = int(np.trunc(np.float32(real_value / np.float32(output_scale)))) + int(output_zp)
    return int(np.clip(quantized_output, -32768, 32767))

def make_sqrt_lut_s8(input_scale, input_zp, output_scale, output_zp) -> np.ndarray:
    """Generate the uint8-addressed LUT required by arm_sqrt_s8."""
    input_scale = _quant_param_to_scalar(input_scale, "input_scale", float)
    input_zp = _quant_param_to_scalar(input_zp, "input_zero_point", int)
    output_scale = _quant_param_to_scalar(output_scale, "output_scale", float)
    output_zp = _quant_param_to_scalar(output_zp, "output_zero_point", int)

    lut = np.zeros(256, dtype=np.int8)
    for i in range(-128, 128):
        final_val = output_zp
        x = np.float32(input_scale) * np.float32(i - input_zp)

        if x > np.float32(0.0):
            res = np.float32(sqrt(float(x)))
            quantized_output = int(np.trunc(np.float32(res / np.float32(output_scale)))) + int(
                output_zp
            )
            final_val = min(max(quantized_output, -128), 127)

        # Mimic C's (uint8_t)i indexing with two's complement wrap.
        lut[i & 0xFF] = np.int8(final_val)

    return lut


def make_sqrt_lut_s16(input_scale, input_zp, output_scale, output_zp) -> np.ndarray:
    """Generate the 513-entry interpolated LUT required by arm_sqrt_s16."""
    input_scale = _quant_param_to_scalar(input_scale, "input_scale", float)
    input_zp = _quant_param_to_scalar(input_zp, "input_zero_point", int)
    output_scale = _quant_param_to_scalar(output_scale, "output_scale", float)
    output_zp = _quant_param_to_scalar(output_zp, "output_zero_point", int)

    lut = np.zeros(SQRT_S16_LUT_SIZE, dtype=np.int16)
    for index in range(SQRT_S16_LUT_SIZE):
        q_value = -32768 + (index << SQRT_S16_SLOT_SHIFT)
        real_value = float(np.float32(input_scale) * np.float32(q_value - input_zp))
        compensated_sqrt = _sqrt_quantized_real(real_value)

        # arm_sqrt_s16 linearly interpolates between neighboring LUT anchors.
        # Bias interior positive anchors upward by the local midpoint sag of sqrt(x)
        # so the piecewise-linear approximation tracks the concave curve better.
        if 0 < index < (SQRT_S16_LUT_SIZE - 1) and real_value > 0.0:
            next_q_value = q_value + (1 << SQRT_S16_SLOT_SHIFT)
            midpoint_q_value = q_value + SQRT_S16_SLOT_HALF_STEP
            next_real_value = float(np.float32(input_scale) * np.float32(next_q_value - input_zp))
            midpoint_real_value = float(np.float32(input_scale) * np.float32(midpoint_q_value - input_zp))
            next_sqrt = _sqrt_quantized_real(next_real_value)
            midpoint_sqrt = _sqrt_quantized_real(midpoint_real_value)
            compensated_sqrt += midpoint_sqrt - 0.5 * (compensated_sqrt + next_sqrt)

        lut[index] = np.int16(
            _quantize_s16_sqrt_output(compensated_sqrt, output_scale, output_zp)
        )

    return lut


def make_sqrt_lut(input_scale, input_zp, output_scale, output_zp, activation_dtype: str) -> np.ndarray:
    """Generate a dtype-specific lookup table for the Sqrt operation."""
    if activation_dtype == "S8":
        return make_sqrt_lut_s8(input_scale, input_zp, output_scale, output_zp)
    if activation_dtype == "S16":
        return make_sqrt_lut_s16(input_scale, input_zp, output_scale, output_zp)
    raise NotImplementedError(f"Unsupported Sqrt dtype: {activation_dtype}")


class OpSqrt(OperationBase):
    """
    Sqrt operation: goldens from the C reference (TFLite's SqrtEvalQuantized; float in binary64).
    """

    def uses_reference(self) -> bool:
        return True

    def _select_cmsis_sqrt_kernel(self) -> Dict[str, str]:
        """
        Select appropriate CMSIS-NN kernel function for Sqrt operation.
        
        Returns:
            Dictionary with kernel_fn, input_c_type, output_c_type
        """
        activation_dtype = self.desc.get('activation_dtype', 'S8')

        if activation_dtype == 'S8':
            return {
                'kernel_fn': 'arm_sqrt_s8',
                'input_c_type': 'int8_t',
                'output_c_type': 'int8_t',
                'lut_c_type': 'int8_t',
                'lut_size': 256,
            }
        elif activation_dtype == 'S16':
            return {
                'kernel_fn': 'arm_sqrt_s16',
                'input_c_type': 'int16_t',
                'output_c_type': 'int16_t',
                'lut_c_type': 'int16_t',
                'lut_size': SQRT_S16_LUT_SIZE,
            }
        else:
            raise NotImplementedError(f"Unsupported Sqrt dtype: {activation_dtype}")
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates for Sqrt operation.
        """
        if self.tensor_dtype("input") in ("FP16", "FP32"):
            from helia_core_tester.generation.ops._shared.sqrt_float import generate_sqrt_float

            generate_sqrt_float(self, output_dir, reciprocal=False)
            return

        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference import quant as ref_quant
        from helia_core_tester.generation.reference.call import ReferenceCall
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc['name']
        kernel_info = self._select_cmsis_sqrt_kernel()
        activation_dtype = self.desc.get("activation_dtype", "S8")
        kind = ref_quant.kind(activation_dtype)
        input_shape = tuple(int(d) for d in self.desc["input_shape"])

        builder = TemplateContextBuilder()
        comparison = self.comparison_config()
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)

        # The non-negative codes: TFLite refuses a negative dequantized input. arm_sqrt_s16
        # interpolates a 513-entry table, which is within one code of sqrt only from code
        # SQRT_S16_ACCURATE_FROM up (up to 7 codes off from 512, hundreds near 0); the default
        # draw stays there, and hint.input_q_range covers the steep low end with its own tolerance.
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        np_in_dtype = np.int8 if kind == "s8" else np.int16
        qmax = int(np.iinfo(np_in_dtype).max)
        low, high = self.desc.get("hint", {}).get("input_q_range", (SQRT_S16_ACCURATE_FROM if kind == "s16" else 0, qmax))
        if not 0 <= int(low) <= int(high) <= qmax:
            raise ValueError(f"{name}: input_q_range must lie in [0, {qmax}], got [{low}, {high}]")
        input_q = self.rng.integers(int(low), int(high) + 1, size=input_shape).astype(np_in_dtype)
        self.rng.__setstate__(rng_state)

        preset = policy.TensorQuant(*ref_quant.preset_quant(activation_dtype), kind)
        quant = self.desc.get("quantization") or {}
        in_quant = policy.descriptor_quant(quant.get("input"), kind) or preset
        out_quant = policy.descriptor_quant(quant.get("output"), kind) or preset
        input_scale, input_zp = in_quant.scale, in_quant.zero_point
        output_scale, output_zp = out_quant.scale, out_quant.zero_point
        output_data = self.reference_golden(ReferenceCall(
            f"sqrt_{kind}",
            {"input_scale": input_scale, "input_zero_point": input_zp,
             "output_scale": output_scale, "output_zero_point": output_zp},
            {"input": np.ascontiguousarray(input_q)}, {"output": input_shape},
            quant={"input": in_quant.to_json(), "output": out_quant.to_json()},
        ))
        output_shape = input_shape
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)

        sqrt_lut = make_sqrt_lut(
            input_scale=input_scale,
            input_zp=input_zp,
            output_scale=output_scale,
            output_zp=output_zp,
            activation_dtype=activation_dtype,
        )

        # Build template context
        context = {
            'name': name,
            'input_dims': input_dims,
            'input_data_array': input_array_str,
            'expected_output_array': expected_output_array_str,
            'input_dtype': kernel_info["input_c_type"],
            'output_dtype': kernel_info["output_c_type"],
            'kernel_fn': kernel_info["kernel_fn"],
            'output_size': int(np.prod(output_shape)),
            'lut_c_type': kernel_info["lut_c_type"],
            'lut_size': kernel_info["lut_size"],
            'sqrt_lut': sqrt_lut,
        }
        if comparison.get("mode") == "tolerant_int":
            context["validation_mode"] = "tolerant_int"
            context["comparison_tolerance"] = int(comparison.get("tolerance", 1))
        
        self.render_harness_case(
            output_dir, stem="sqrt", context=context, pool=sqrt_argument_pool(context),
            validation_key="BasicMathFunctions/sqrt/sqrt.c.j2", label="Sqrt", operator="Sqrt",
        )
        
