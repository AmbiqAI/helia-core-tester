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
# Below eight LUT slots the chord of sqrt sags too far for a 2 LSB bound (503 LSB at q = 29).
SQRT_S16_SANITY_FLOOR = 8 << SQRT_S16_SLOT_SHIFT
SQRT_S16_SANITY_LSB = 2
# The (scale, zero point) the one-op LiteRT builder gave input and output.
SQRT_FIXED_QUANT = {"S8": (0.125, 0), "S16": (1.0 / 32768.0, 0)}


def clamp_f32(x, min_val, max_val):
    '''Clamp a float32 value to the specified range.'''
    return max(min(x, max_val), min_val)

def _quant_param_to_scalar(value, name: str, cast):
    """Normalize LiteRT quantization values to a scalar."""
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
    """Match LiteRT's int16 sqrt output more closely with truncation-based requantization."""
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


def sqrt_s8_golden(input_q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """arm_sqrt_s8 / TFLite's int8 SQRT: a 256-entry table indexed by the uint8 bit pattern."""
    if lut.shape != (256,) or lut.dtype != np.int8:
        raise ValueError(f"sqrt s8 LUT must be 256 int8 entries, got {lut.shape} {lut.dtype}")
    return lut[input_q.astype(np.uint8)]


def sqrt_s16_golden(input_q: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """arm_sqrt_s16's interpolation, exactly: TFLite has no int16 SQRT and TFLM no integer
    one, so the kernel's own arithmetic is the oracle (bounded by check_sqrt_s16_golden)."""
    if lut.shape != (SQRT_S16_LUT_SIZE,):
        raise ValueError(f"sqrt s16 LUT must have {SQRT_S16_LUT_SIZE} entries, got {lut.shape}")
    value = input_q.astype(np.int32)
    index = 256 + (value >> SQRT_S16_SLOT_SHIFT)
    offset = value & 0x7F
    base = lut.astype(np.int32)[index]
    slope = lut.astype(np.int32)[index + 1] - base
    return (base + ((slope * offset + 64) >> 7)).astype(np.int16)


def check_sqrt_s16_golden(input_q: np.ndarray, golden: np.ndarray, input_scale: float, output_scale: float) -> None:
    """Fail when the interpolated golden strays more than SQRT_S16_SANITY_LSB from float sqrt
    anywhere at or above SQRT_S16_SANITY_FLOOR."""
    q = input_q.astype(np.int64).ravel()
    keep = q >= SQRT_S16_SANITY_FLOOR
    real = np.sqrt(q[keep] * float(input_scale)) / float(output_scale)
    expected = np.clip(np.floor(real + 0.5), -32768, 32767)
    worst = np.abs(golden.ravel()[keep].astype(np.int64) - expected).max(initial=0)
    if worst > SQRT_S16_SANITY_LSB:
        raise ValueError(f"sqrt s16 golden is {int(worst)} LSB from float sqrt (bound {SQRT_S16_SANITY_LSB})")


class OpSqrt(OperationBase):
    """
    Sqrt operation.
    """

    def needs_tflite(self) -> bool:
        return False

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

        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
        
        name = self.desc['name']
        # Select CMSIS kernel + types
        kernel_info = self._select_cmsis_sqrt_kernel()
        
        input_shape = tuple(self.desc["input_shape"])
        
        builder = TemplateContextBuilder()
        comparison = self.comparison_config()
        
        # Convert shapes to CMSIS dims
        input_dims = builder.nhwc_to_cmsis_dims(input_shape)
        
        # Generate deterministic integer input data
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)

        if kernel_info["input_c_type"] == "int8_t":
            np_in_dtype = np.int8
            qmin, qmax = -128, 127
        elif kernel_info["input_c_type"] == "int16_t":
            np_in_dtype = np.int16
            qmin, qmax = -32768, 32767
        else:
            raise ValueError(f"Unsupported input_c_type: {kernel_info['input_c_type']}")
        input_q = self.rng.integers(0, qmax + 1, size=input_shape, dtype=np_in_dtype)
        self.rng.__setstate__(rng_state)

        activation_dtype = self.desc.get("activation_dtype", "S8")
        input_scale, input_zp = output_scale, output_zp = SQRT_FIXED_QUANT[activation_dtype]
        sqrt_lut = make_sqrt_lut(input_scale, input_zp, output_scale, output_zp, activation_dtype)
        if activation_dtype == "S8":
            output_data = sqrt_s8_golden(input_q, sqrt_lut)
        else:
            output_data = sqrt_s16_golden(input_q, sqrt_lut)
            check_sqrt_s16_golden(input_q, output_data, input_scale, output_scale)
        output_shape = tuple(output_data.shape)
        # Format arrays
        input_array_str = builder.format_array_as_c_literal(input_q)
        expected_output_array_str = builder.format_array_as_c_literal(output_data)


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
        
