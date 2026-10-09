"""Kernel parameters derived exactly as TFLite/TFLM prepare them.

Every fixed-point conversion goes through the shim's own TFLM code
(QuantizeMultiplier, PreprocessSoftmaxScaling, ...), so Python never
re-implements that arithmetic. What stays here is the glue the TFLM prepare
functions (micro/kernels/*_common.cc, kernels/padding.h) perform around it, in
the same precision: scales are float32 tensor parameters, products and
quotients of them are formed in double.

Conventions follow cmsis_nn_*_params: input_offset = -zp, output_offset = +zp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from helia_core_tester.generation.reference import bindings as b

DTYPE_RANGES: Dict[str, Tuple[int, int]] = {
    "s4": (-8, 7),
    "s8": (-128, 127),
    "s16": (-32768, 32767),
}

ACTIVATIONS = {
    "NONE": b.ACT_NONE,
    "RELU": b.ACT_RELU,
    "RELU_N1_TO_1": b.ACT_RELU_N1_TO_1,
    "RELU6": b.ACT_RELU6,
}


def _f32(value: float) -> float:
    """A tensor scale as TFLite stores it (float32), widened back to Python float."""
    scale = float(np.float32(value))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {value!r}")
    return scale


def dtype_range(dtype: str) -> Tuple[int, int]:
    try:
        return DTYPE_RANGES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"no integer range for dtype {dtype!r}") from exc


def quantize_multiplier(real_multiplier: float) -> Tuple[int, int]:
    return b.get_bindings().quantize_multiplier(real_multiplier)


def quantize_multiplier_smaller_than_one_exp(real_multiplier: float) -> Tuple[int, int]:
    return b.get_bindings().quantize_multiplier_smaller_than_one_exp(real_multiplier)


def per_channel(input_scale: float, weight_scales: Sequence[float], output_scale: float) -> b.PerChannel:
    """Per-channel requantization as PopulateConvolutionQuantizationParams forms it.

    effective_c = double(input_scale) * double(weight_scale_c) / double(output_scale)
    """
    in_s, out_s = _f32(input_scale), _f32(output_scale)
    scales = np.asarray(weight_scales, dtype=np.float32)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("weight_scales must be a non-empty 1-D sequence")
    lib = b.get_bindings()
    multipliers = np.empty(scales.size, dtype=np.int32)
    shifts = np.empty(scales.size, dtype=np.int32)
    for c, w_s in enumerate(scales):
        effective = in_s * _f32(float(w_s)) / out_s
        multipliers[c], shifts[c] = lib.quantize_multiplier(effective)
    return b.PerChannel(multipliers, shifts)


def per_tensor(input_scale: float, weight_scale: float, output_scale: float) -> b.PerChannel:
    """Single-entry requantization (the per-tensor FC kernel)."""
    return per_channel(input_scale, [weight_scale], output_scale)


def activation_range(activation: str, scale: float, zero_point: int, dtype: str) -> Tuple[int, int]:
    """Fused-activation clamp as CalculateActivationRangeQuantized computes it."""
    try:
        code = ACTIVATIONS[str(activation).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported fused activation {activation!r}") from exc
    qmin, qmax = dtype_range(dtype)
    return b.get_bindings().activation_range_quantized(code, _f32(scale), int(zero_point), qmin, qmax)


def float_activation_range(activation: str) -> Tuple[float, float]:
    """CalculateActivationRange for float kernels."""
    bounds = {
        "NONE": (-np.inf, np.inf),
        "RELU": (0.0, np.inf),
        "RELU_N1_TO_1": (-1.0, 1.0),
        "RELU6": (0.0, 6.0),
    }
    try:
        return bounds[str(activation).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported fused activation {activation!r}") from exc


# ---- padding (tensorflow/lite/kernels/padding.h) ----


@dataclass(frozen=True)
class Padding:
    pad: int
    offset: int


def out_size(padding: str, in_size: int, filter_size: int, stride: int, dilation: int = 1) -> int:
    """ComputeOutSize."""
    if stride < 1 or dilation < 1 or filter_size < 1 or in_size < 1:
        raise ValueError("sizes, stride and dilation must be positive")
    effective = (filter_size - 1) * dilation + 1
    mode = padding.upper()
    if mode == "SAME":
        return (in_size + stride - 1) // stride
    if mode == "VALID":
        size = (in_size + stride - effective) // stride
        if size < 1:
            raise ValueError(f"VALID window {effective} exceeds input {in_size}")
        return size
    raise ValueError(f"unknown padding {padding!r}")


def padding_with_offset(stride: int, dilation: int, in_size: int, filter_size: int, out: int) -> Padding:
    """ComputePaddingWithOffset."""
    effective = (filter_size - 1) * dilation + 1
    total = max((out - 1) * stride + effective - in_size, 0)
    return Padding(total // 2, total % 2)


def conv_geometry(
    padding: str,
    in_hw: Tuple[int, int],
    filter_hw: Tuple[int, int],
    stride_hw: Tuple[int, int],
    dilation_hw: Tuple[int, int] = (1, 1),
) -> Tuple[Tuple[int, int], Padding, Padding]:
    """Output (h, w) and the height/width paddings TFLite derives for a conv/pool."""
    out_h = out_size(padding, in_hw[0], filter_hw[0], stride_hw[0], dilation_hw[0])
    out_w = out_size(padding, in_hw[1], filter_hw[1], stride_hw[1], dilation_hw[1])
    pad_h = padding_with_offset(stride_hw[0], dilation_hw[0], in_hw[0], filter_hw[0], out_h)
    pad_w = padding_with_offset(stride_hw[1], dilation_hw[1], in_hw[1], filter_hw[1], out_w)
    return (out_h, out_w), pad_h, pad_w


def tconv_padding(
    padding: str, out_hw: Tuple[int, int], filter_hw: Tuple[int, int], stride_hw: Tuple[int, int]
) -> Tuple[Padding, Padding]:
    """TransposeConv padding: ComputePaddingHeightWidth with input and output swapped."""
    pads = []
    for axis in range(2):
        in_size = out_hw[axis]
        unused_out = out_size(padding, in_size, filter_hw[axis], stride_hw[axis])
        pads.append(padding_with_offset(stride_hw[axis], 1, in_size, filter_hw[axis], unused_out))
    return pads[0], pads[1]


# ---- elementwise / softmax / reduce prepare (micro/kernels/*_common.cc) ----


@dataclass(frozen=True)
class AddSubParams:
    left_shift: int
    input1_offset: int
    input1_multiplier: int
    input1_shift: int
    input2_offset: int
    input2_multiplier: int
    input2_shift: int
    output_offset: int
    output_multiplier: int
    output_shift: int


def addsub_params(
    dtype: str,
    input1_scale: float,
    input1_zp: int,
    input2_scale: float,
    input2_zp: int,
    output_scale: float,
    output_zp: int,
) -> AddSubParams:
    """CalculateOpDataAdd (Sub is identical): the max of the two float32 scales,
    doubled in double precision."""
    left_shift = {"s8": 20, "s16": 15}.get(dtype.lower())
    if left_shift is None:
        raise ValueError(f"add/sub params are defined for s8/s16, got {dtype!r}")
    s1, s2, so = _f32(input1_scale), _f32(input2_scale), _f32(output_scale)
    twice_max = 2.0 * max(s1, s2)
    m1, sh1 = quantize_multiplier_smaller_than_one_exp(s1 / twice_max)
    m2, sh2 = quantize_multiplier_smaller_than_one_exp(s2 / twice_max)
    mo, sho = quantize_multiplier_smaller_than_one_exp(twice_max / ((1 << left_shift) * so))
    return AddSubParams(left_shift, -int(input1_zp), m1, sh1, -int(input2_zp), m2, sh2, int(output_zp), mo, sho)


def mul_params(input1_scale: float, input2_scale: float, output_scale: float) -> Tuple[int, int]:
    """CalculateOpDataMul: QuantizeMultiplier(s1 * s2 / so)."""
    return quantize_multiplier(_f32(input1_scale) * _f32(input2_scale) / _f32(output_scale))


@dataclass(frozen=True)
class SoftmaxParams:
    input_multiplier: int
    input_left_shift: int
    diff_min: Optional[int]


_SOFTMAX_SCALED_DIFF_INTEGER_BITS = 5


def softmax_params_s8(input_scale: float, beta: float = 1.0) -> SoftmaxParams:
    """CalculateSoftmaxParams, int8 input."""
    lib = b.get_bindings()
    multiplier, left_shift = lib.preprocess_softmax_scaling(
        float(np.float32(beta)), _f32(input_scale), _SOFTMAX_SCALED_DIFF_INTEGER_BITS
    )
    radius = lib.calculate_input_radius(_SOFTMAX_SCALED_DIFF_INTEGER_BITS, left_shift, 31)
    return SoftmaxParams(multiplier, left_shift, -radius)


def softmax_params_s16(input_scale: float, beta: float = 1.0) -> SoftmaxParams:
    """CalculateSoftmaxParams, int16 input: [-65535, 0] maps onto [-10, 0]."""
    rescale = _f32(input_scale) * float(np.float32(beta)) / (10.0 / 65535.0)
    multiplier, left_shift = quantize_multiplier(rescale)
    return SoftmaxParams(multiplier, left_shift, None)


def mean_params(input_scale: float, output_scale: float) -> Tuple[int, int]:
    """PrepareMeanOrSumHelper: QuantizeMultiplier(si / so); the 1/N fold happens
    inside the reference kernel, not here."""
    return quantize_multiplier(_f32(input_scale) / _f32(output_scale))
