"""Independent Python models of TFLite's integer arithmetic, for testing the C reference.

Written on unbounded Python ints with floor division, deliberately unlike the C
(int64 products, truncating division, nudges), so agreement is evidence.
"""

from __future__ import annotations

import numpy as np


def srdhm(a: int, b: int) -> int:
    """gemmlowp SaturatingRoundingDoublingHighMul: floor(ab / 2^31 + 1/2), saturating MIN*MIN."""
    if a == b == -(2**31):
        return 2**31 - 1
    return (a * b + 2**30) >> 31


def rdbpot(x: int, e: int) -> int:
    """RoundingDivideByPOT: x / 2^e, halves away from zero."""
    if e == 0:
        return x
    q, r = divmod(x, 1 << e)
    half = 1 << (e - 1)
    return q + (1 if (r > half or (r == half and x >= 0)) else 0)


def wrap32(x: int) -> int:
    return (x + 2**31) % 2**32 - 2**31


def mbqm(x: int, m: int, shift: int) -> int:
    """MultiplyByQuantizedMultiplier, double rounding; the left shift wraps as int32 math does."""
    left, right = max(shift, 0), max(-shift, 0)
    return rdbpot(srdhm(wrap32(x << left), m), right)


def broadcast_pairs(a: np.ndarray, b: np.ndarray):
    """Flattened operand pairs in the numpy broadcast order, and the output shape."""
    shape = np.broadcast_shapes(a.shape, b.shape)
    return np.broadcast_to(a, shape).ravel().tolist(), np.broadcast_to(b, shape).ravel().tolist(), shape


def sqrt_float_reference(bits: np.ndarray, reciprocal: bool) -> np.ndarray:
    """ns-cmsis-nn#295 float sqrt/rsqrt on bit patterns: finite positives in float64 rounded once,
    with the public special-value contract encoded bitwise."""
    half = bits.dtype == np.uint16
    dtype = np.float16 if half else np.float32
    sign, inf, quiet = (
        (0x8000, 0x7C00, 0x0200) if half else (0x80000000, 0x7F800000, 0x00400000)
    )
    magnitude = bits & (sign - 1)
    with np.errstate(all="ignore"):
        values = bits.view(dtype).astype(np.float64)
        result = np.sqrt(values)
        if reciprocal:
            result = 1.0 / result
        output = result.astype(dtype).view(bits.dtype).copy()
    output[(bits & sign != 0) & (magnitude != 0)] = inf | quiet
    nan = magnitude > inf
    output[nan] = bits[nan] | quiet
    zero = magnitude == 0
    output[zero] = bits[zero] | (inf if reciprocal else 0)
    output[bits == inf] = 0 if reciprocal else inf
    return output


def lut_populate_int16(in_scale, in_zp, out_scale, out_zp, fn):
    """TFLM detail::LUTPopulateInt16<float>, step for step in float32."""
    f = np.float32

    def rnd(x):
        return f(np.sign(x) * np.floor(np.abs(x) + f(0.5)))

    in_scale, out_scale = f(in_scale), f(out_scale)
    imin, imax = in_scale * f(-32768 - in_zp), in_scale * f(32767 - in_zp)
    omin, omax = out_scale * f(-32768 - out_zp), out_scale * f(32767 - out_zp)
    step = (imax - imin) / f(512)
    half, inv = step / f(2), f(65536) / (omax - omin)
    lut = []
    for i in range(512):
        val, mid, nxt = fn(imin + f(i) * step), fn(imin + f(i) * step + half), fn(imin + f(i + 1) * step)
        sample = rnd(val * inv)
        bias = rnd((rnd((nxt * inv + rnd(val * inv)) / f(2)) - rnd(mid * inv)) / f(2))
        lut.append(int(min(max(sample - bias, -32768), 32767)))
    lut.append(int(min(max(rnd(fn(imax) * inv), -32768), 32767)))
    return lut


def lut_lookup_int16(v: int, lut) -> int:
    """LUTLookup(int16_t): linear interpolation between 513 anchors."""
    i, off = 256 + (v >> 7), v & 0x7F
    return lut[i] + (((lut[i + 1] - lut[i]) * off + 64) >> 7)


# ---- arm_nn_activation_f32/f16 (numpy; float16 tanh as CMSIS-NN tabulates it) ----


def _fp16(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float16)


# float16 tanh LUT sampled over x in [0, 4] with 256 intervals, matching
# arm_nn_tanh_lut_f16 in Source/NNSupportFunctions/arm_nntables_flt.c:
#   arm_nn_tanh_lut_f16[i] = float16(tanh(4 * i / 256))
TANH_LUT256_F16 = np.tanh(4.0 * np.arange(257, dtype=np.float64) / 256.0).astype(np.float16)


def tanh_reference_f16(input_data: np.ndarray) -> np.ndarray:
    """Scalar LUT reference with separately rounded half-precision operations.

    Models round-to-nearest without flushing subnormals. Optimized scalar code
    may contract interpolation, so off-grid bitwise parity is not promised.
    The table is independently sampled from tanh, not read from the kernel.
    """
    x = _fp16(input_data)
    is_nan = (x.view(np.uint16) & 0x7fff) > 0x7c00
    # Classify before indexing: NaNs must never undergo an integer conversion.
    ax = _fp16(np.abs(np.where(is_nan, _fp16(0.0), x)))
    saturate = ax > _fp16(4.0)
    t = _fp16(np.minimum(ax, _fp16(4.0)) * _fp16(64.0))
    idx = np.minimum(t.astype(np.uint16), np.uint16(255))
    frac = _fp16(t - idx.astype(np.float16))
    y0 = TANH_LUT256_F16[idx]
    diff = _fp16(TANH_LUT256_F16[idx + 1] - y0)
    interp = _fp16(y0 + _fp16(diff * frac))
    magnitude = np.where(saturate, _fp16(1.0), interp)
    return _fp16(np.where(is_nan, x, np.copysign(magnitude, x)))


def tanh_reference_f16_mve(input_data: np.ndarray) -> np.ndarray:
    """Mirror the float16 MVE tanh path (arm_nn_vtanh_lut_direct_mve_f16).

    Reproduces the Helium kernel's LUT + linear interpolation and float16 rounding
    so generated expectations match Cortex-M55 MVE output bit-for-bit.
    """
    x = _fp16(input_data)
    ax = _fp16(np.abs(x))
    saturate = ax > _fp16(4.0)
    ax = _fp16(np.minimum(ax, _fp16(4.0)))
    # t = ax * (256 / 4); idx = floor(t) clamped to [0, 255]; frac = t - idx
    t = _fp16(ax * _fp16(64.0))
    idx = np.minimum(t.astype(np.uint16), np.uint16(255))
    frac = _fp16(t - idx.astype(np.float16))
    y0 = TANH_LUT256_F16[idx]
    y1 = TANH_LUT256_F16[idx + 1]
    diff = _fp16(y1 - y0)
    # vfmaq performs a single-rounded multiply-add; evaluate exactly then round once.
    interp = _fp16(y0.astype(np.float64) + diff.astype(np.float64) * frac.astype(np.float64))
    magnitude = np.where(saturate, _fp16(1.0), interp)
    result = np.where(x < _fp16(0.0), _fp16(-magnitude), magnitude)
    return result.astype(np.float16)


def activation_reference(
    input_data: np.ndarray,
    activation_type: str,
    act_param: float,
    activation_dtype: str = "FP32",
    *,
    use_mve_tanh: bool = False,
) -> np.ndarray:
    if activation_type == "ARM_NN_FLT_ACT_TANH" and str(activation_dtype).upper() == "FP16":
        if use_mve_tanh:
            return tanh_reference_f16_mve(input_data)
        return tanh_reference_f16(input_data)

    data = input_data.astype(np.float32)
    if activation_type == "ARM_NN_FLT_ACT_SIGMOID":
        return 1.0 / (1.0 + np.exp(-data))
    if activation_type == "ARM_NN_FLT_ACT_TANH":
        return np.tanh(data)
    if activation_type == "ARM_NN_FLT_ACT_HARDSWISH":
        return data * np.clip(data + 3.0, 0.0, 6.0) / 6.0
    if activation_type == "ARM_NN_FLT_ACT_LEAKY_RELU":
        return np.where(data >= 0.0, data, data * float(act_param))
    if activation_type == "ARM_NN_FLT_ACT_RELU":
        return np.maximum(data, 0.0)
    if activation_type == "ARM_NN_FLT_ACT_RELU6":
        return np.clip(data, 0.0, 6.0)
    if activation_type == "ARM_NN_FLT_ACT_NONE":
        return data
    raise ValueError(f"Unsupported float activation type: {activation_type}")
