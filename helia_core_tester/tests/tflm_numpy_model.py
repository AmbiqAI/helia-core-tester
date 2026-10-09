"""Independent numpy model of the TFLM reference arithmetic, for cross-checking
the hct_ref shim. Written from the TFLM sources' semantics, not from the shim:
a shim that passes its arguments in the wrong order or the wrong convention
disagrees with it."""

from __future__ import annotations

import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _trunc_div(a: np.ndarray, b: int) -> np.ndarray:
    """C integer division (truncation toward zero) on int64 arrays."""
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def srdhm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """gemmlowp SaturatingRoundingDoublingHighMul for int32."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    overflow = (a == b) & (a == INT32_MIN)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    high = _trunc_div(ab + nudge, 1 << 31)
    return np.where(overflow, INT32_MAX, high)


def rounding_divide_by_pot(x: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    exponent = np.asarray(exponent, dtype=np.int64)
    mask = (np.int64(1) << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> exponent) + (remainder > threshold).astype(np.int64)


def mbqm32(x: np.ndarray, multiplier: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Double-rounding MultiplyByQuantizedMultiplier(int32 x)."""
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    scaled = np.asarray(x, dtype=np.int64) * (np.int64(1) << left)
    return rounding_divide_by_pot(srdhm(scaled, multiplier), right)


def mbqm64(x: np.ndarray, multiplier: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """MultiplyByQuantizedMultiplier(int64 x): the multiplier reduced to 16 bits."""
    multiplier = np.asarray(multiplier, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    reduced = np.where(multiplier < 0x7FFF0000, (multiplier + (1 << 15)) >> 16, 0x7FFF)
    total = 15 - shift
    acc = np.asarray(x, dtype=np.int64) * reduced + (np.int64(1) << (total - 1))
    return acc >> total


def conv_nhwc(
    x: np.ndarray,
    w: np.ndarray,
    stride=(1, 1),
    dilation=(1, 1),
    pad=(0, 0),
    out_hw=None,
    input_offset: int = 0,
) -> np.ndarray:
    """Integer (or float) NHWC x OHWI convolution accumulators, groups supported.

    Returns [N, OH, OW, O] int64 (or float64) before bias and requantization.
    """
    n, h, wd, c = x.shape
    o, kh, kw, ci = w.shape
    groups = c // ci
    oh, ow = out_hw
    floating = np.issubdtype(x.dtype, np.floating)
    acc_t = np.float64 if floating else np.int64
    xs = x.astype(acc_t) + (0 if floating else input_offset)
    ws = w.astype(acc_t)
    out = np.zeros((n, oh, ow, o), dtype=acc_t)
    per_group = o // groups
    for oy in range(oh):
        for ox in range(ow):
            for fy in range(kh):
                iy = oy * stride[0] - pad[0] + fy * dilation[0]
                if not 0 <= iy < h:
                    continue
                for fx in range(kw):
                    ix = ox * stride[1] - pad[1] + fx * dilation[1]
                    if not 0 <= ix < wd:
                        continue
                    for g in range(groups):
                        patch = xs[:, iy, ix, g * ci:(g + 1) * ci]  # [N, ci]
                        kernel = ws[g * per_group:(g + 1) * per_group, fy, fx, :]  # [per_group, ci]
                        out[:, oy, ox, g * per_group:(g + 1) * per_group] += patch @ kernel.T
    return out


def dwconv_nhwc(x, w, depth_multiplier, stride=(1, 1), dilation=(1, 1), pad=(0, 0), out_hw=None, input_offset=0):
    """Depthwise accumulators; filter [1, KH, KW, C*M], output channel = c*M + m."""
    n, h, wd, c = x.shape
    _, kh, kw, oc = w.shape
    oh, ow = out_hw
    floating = np.issubdtype(x.dtype, np.floating)
    acc_t = np.float64 if floating else np.int64
    xs = x.astype(acc_t) + (0 if floating else input_offset)
    ws = w.astype(acc_t)[0]
    out = np.zeros((n, oh, ow, oc), dtype=acc_t)
    rep = np.repeat(np.arange(c), depth_multiplier)
    for oy in range(oh):
        for ox in range(ow):
            for fy in range(kh):
                iy = oy * stride[0] - pad[0] + fy * dilation[0]
                if not 0 <= iy < h:
                    continue
                for fx in range(kw):
                    ix = ox * stride[1] - pad[1] + fx * dilation[1]
                    if not 0 <= ix < wd:
                        continue
                    out[:, oy, ox, :] += xs[:, iy, ix, rep] * ws[fy, fx, :]
    return out


def requantize(acc, bias, multiplier, shift, output_offset, act_min, act_max, wide: bool = False):
    """bias add + MultiplyByQuantizedMultiplier per output channel (last axis) + clamp."""
    acc = np.asarray(acc, dtype=np.int64)
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=np.int64)
    fn = mbqm64 if wide else mbqm32
    scaled = fn(acc, np.asarray(multiplier), np.asarray(shift)) + output_offset
    return np.clip(scaled, act_min, act_max)


def pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack int4 values (int8 storage, -8..7) two per byte, low nibble first."""
    flat = np.asarray(values, dtype=np.int8).ravel()
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    lo = flat[0::2].astype(np.uint8) & 0x0F
    hi = (flat[1::2].astype(np.uint8) & 0x0F) << 4
    return (lo | hi).astype(np.int8)
