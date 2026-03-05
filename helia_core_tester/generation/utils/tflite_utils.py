"""
Quantization utilities for CMSIS-NN.
These functions convert quantization scales to multiplier/shift format.
"""

import numpy as np
from typing import Dict, Tuple, Any


def scalar_scale_zp(quant_dict: Dict[str, Any]) -> Tuple[float, int]:
    """
    Extract scalar scale and zero_point from a quantization dict (per-tensor or per-channel).
    For per-channel, uses the first element.
    """
    scale = quant_dict.get("scale", 1.0)
    zp = quant_dict.get("zero_point", 0)
    if isinstance(scale, (list, np.ndarray)):
        scale = float(scale[0])
    if isinstance(zp, (list, np.ndarray)):
        zp = int(zp[0])
    return float(scale), int(zp)


def qp_scalar(quant_dict: Dict[str, Any], key: str, default: Any) -> Any:
    value = quant_dict.get(key, default)
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return default
        return value[0]
    return value


def activation_bounds(activation_dtype: str) -> Tuple[int, int]:
    """Return (min, max) int bounds for activation_dtype (e.g. S8, S16)."""
    activation_dtype = str(activation_dtype).upper()
    if activation_dtype == "S8":
        return -128, 127
    if activation_dtype == "S16":
        return -32768, 32767
    return -128, 127


def elementwise_addsub_quant_params(
    input1_scale: float,
    input2_scale: float,
    output_scale: float,
    activation_dtype: str,
) -> Dict[str, int]:
    """
    Calculate CMSIS/TFL-style quant params for elementwise Add/Sub.

    This matches the reference approach used by CMSIS unit-test generation:
      - left_shift = 20 for S8, 15 for S16
      - input multipliers based on input_scale / (2 * max_input_scale)
      - output multiplier based on:
        (2 * max_input_scale) / ((1 << left_shift) * output_scale)
    """
    activation_dtype = str(activation_dtype).upper()
    left_shift = 15 if activation_dtype == "S16" else 20

    twice_max_input_scale = 2.0 * max(float(input1_scale), float(input2_scale))
    input1_mult, input1_shift = calculate_multiplier_shift(float(input1_scale) / twice_max_input_scale)
    input2_mult, input2_shift = calculate_multiplier_shift(float(input2_scale) / twice_max_input_scale)
    output_mult, output_shift = calculate_multiplier_shift(
        twice_max_input_scale / ((1 << left_shift) * float(output_scale))
    )

    return {
        "left_shift": int(left_shift),
        "input1_mult": int(input1_mult),
        "input1_shift": int(input1_shift),
        "input2_mult": int(input2_mult),
        "input2_shift": int(input2_shift),
        "out_mult": int(output_mult),
        "out_shift": int(output_shift),
    }


def calculate_multiplier_shift(scale: float) -> Tuple[int, int]:
    """
    Calculate multiplier and shift from quantization scale.
    
    This converts a floating-point scale to integer multiplier and shift
    for CMSIS-NN quantization format.
    
    - Uses math.frexp() to decompose scale
    - Uses round-half-up rounding (math.floor(fraction * (1 << 31) + 0.5))
    - Formula: scale ≈ multiplier / (2^(31 - shift))
    
    Args:
        scale: Quantization scale (float)
        
    Returns:
        Tuple of (multiplier, shift) where:
        - multiplier is in Q31 format (0 to 2^31-1)
        - shift is the exponent from frexp
        - scale ≈ multiplier / (2^(31 - shift))
    """
    import math
    
    if scale == 0.0:
        return 0, 0
    
    # Decompose scale into fraction and exponent such that:
    #   scale = fraction * 2^exponent,  with fraction in [0.5, 1)
    fraction, exponent = math.frexp(scale)
    
    # Convert fraction to a Q31 fixed-point value.
    # Use round-half-up instead of banker's rounding:
    quantized_multiplier = int(math.floor(fraction * (1 << 31) + 0.5))
    
    # Handle the corner-case where rounding might push the value to 1<<31.
    if quantized_multiplier == (1 << 31):
        quantized_multiplier //= 2
        exponent += 1
    
    shift = exponent
    
    return quantized_multiplier, shift


def reduce_multiplier_q31_to_q15(multiplier: int) -> int:
    """Reduce a multiplier value for int16 quantization.

    Certain CMSIS functions for int16 quantized models require the
    multiplier to be reduced from Q31 format to Q15 format. This function
    performs that reduction.
    
    Args:
        multiplier (int): The original multiplier value in Q31 format.

    Returns:
        int: The reduced multiplier value in Q15 format.
    """
    if multiplier < 0x7FFF0000:
        return (multiplier + (1 << 15)) >> 16
    return 0x7FFF


def calculate_per_channel_multiplier_shift(scales: np.ndarray, reduce_to_q15: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate multiplier and shift arrays for per-channel quantization.
    
    Args:
        scales: Array of quantization scales (one per channel)
        reduce_to_q15: If True, reduce multipliers from Q31 to Q15 (for S16)
        
    Returns:
        Tuple of (multiplier_array, shift_array)
    """
    multipliers = []
    shifts = []
    
    for scale in scales:
        mult, shift = calculate_multiplier_shift(float(scale))
        if reduce_to_q15:
            mult = reduce_multiplier_q31_to_q15(mult)
        multipliers.append(mult)
        shifts.append(shift)
    
    return np.array(multipliers, dtype=np.int32), np.array(shifts, dtype=np.int32)


def default_input_scale(activation_dtype: str) -> float:
    activation_dtype = str(activation_dtype).upper()
    if activation_dtype == "S16":
        return 1.0 / 32768.0
    return 0.125


def comparison_quant_params(
    lhs_scale: float,
    rhs_scale: float,
    activation_dtype: str,
) -> Tuple[int, int, int, int, int]:
    activation_dtype = str(activation_dtype).upper()
    left_shift = 15 if activation_dtype == "S16" else 20
    twice_max_input_scale = 2.0 * max(lhs_scale, rhs_scale)
    lhs_multiplier = lhs_scale / twice_max_input_scale
    rhs_multiplier = rhs_scale / twice_max_input_scale
    lhs_mult, lhs_shift = calculate_multiplier_shift(lhs_multiplier)
    rhs_mult, rhs_shift = calculate_multiplier_shift(rhs_multiplier)
    return left_shift, lhs_mult, lhs_shift, rhs_mult, rhs_shift


def requantize_np(values: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    prod = values.astype(np.int64) * (1 << left_shift)
    mult = (1 << 30) + (prod * int(multiplier))
    res = (mult >> 31).astype(np.int64)
    if right_shift == 0:
        return res.astype(np.int32)
    remainder_mask = (1 << right_shift) - 1
    remainder = res & remainder_mask
    result = res >> right_shift
    threshold = remainder_mask >> 1
    threshold = threshold + (result < 0)
    result = result + (remainder > threshold)
    return result.astype(np.int32)


def simulate_compare(
    input1_q: np.ndarray,
    input2_q: np.ndarray,
    *,
    operation: str,
    input_1_offset: int,
    input_1_mult: int,
    input_1_shift: int,
    input_2_offset: int,
    input_2_mult: int,
    input_2_shift: int,
    left_shift: int,
) -> np.ndarray:
    a = (input1_q.astype(np.int32) + int(input_1_offset)) << int(left_shift)
    b = (input2_q.astype(np.int32) + int(input_2_offset)) << int(left_shift)
    a = requantize_np(a, input_1_mult, input_1_shift)
    b = requantize_np(b, input_2_mult, input_2_shift)

    if operation == "ARM_COMPARE_EQUAL":
        out = a == b
    elif operation == "ARM_COMPARE_NOT_EQUAL":
        out = a != b
    elif operation == "ARM_COMPARE_GREATER":
        out = a > b
    elif operation == "ARM_COMPARE_GREATER_EQUAL":
        out = a >= b
    elif operation == "ARM_COMPARE_LESS":
        out = a < b
    elif operation == "ARM_COMPARE_LESS_EQUAL":
        out = a <= b
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    return out.astype(np.uint8)
