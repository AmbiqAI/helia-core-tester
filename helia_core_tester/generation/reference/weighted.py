"""Weights, biases and their quantization for the weighted operators (Conv, DepthwiseConv,
TransposeConv, FullyConnected, BatchMatMul), replacing Keras initializers and the converter.

Float weights are drawn Glorot-uniform from the case seed; integer filters are quantized the
way TFLite's converter does it: symmetric, per output channel, scale = max|w_c| / qmax with
qmax 127 (int8) or 7 (int4), zero point 0; biases at scale input_scale * filter_scale[c],
rounded half away from zero, in int32 (int8 activations) or int64 (int16 activations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from helia_core_tester.generation.reference.policy import round_half_away

_FILTER_QMAX = {"S8": 127, "S4": 7}


def glorot_uniform(rng: np.random.Generator, shape: Sequence[int], fan_in: int, fan_out: int,
                   gain: Optional[float] = None) -> np.ndarray:
    """VarianceScaling(gain, fan_avg, uniform); gain None is Glorot (1.0)."""
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(f"fan_in/fan_out must be positive, got {fan_in}/{fan_out}")
    scale = 1.0 if gain is None else float(gain)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"weight_gain must be positive and finite, got {gain!r}")
    limit = float(np.sqrt(3.0 * scale / ((fan_in + fan_out) / 2.0)))
    return rng.uniform(-limit, limit, size=tuple(int(d) for d in shape)).astype(np.float32)


def signed_magnitude(rng: np.random.Generator, n: int, lo: float = 2.0, hi: float = 4.0) -> np.ndarray:
    """Magnitudes in [lo, hi) with a random sign: a quantized bias that clears an output step."""
    if not 0 <= lo < hi:
        raise ValueError(f"invalid magnitude range [{lo}, {hi})")
    mags = rng.uniform(lo, hi, size=int(n))
    signs = np.where(rng.integers(0, 2, size=int(n)) == 0, -1.0, 1.0)
    return (mags * signs).astype(np.float32)


@dataclass(frozen=True)
class QuantizedFilter:
    values: np.ndarray  # int8 storage, same shape as the float filter
    scales: np.ndarray  # float32, one per output channel (axis 0)
    dtype: str  # "S8" or "S4"

    def dequantized(self) -> np.ndarray:
        shape = (-1,) + (1,) * (self.values.ndim - 1)
        return self.values.astype(np.float64) * self.scales.astype(np.float64).reshape(shape)


def quantize_filter(weights: np.ndarray, dtype: str = "S8", channel_axis: int = 0,
                    scales: Optional[Sequence[float]] = None) -> QuantizedFilter:
    """Symmetric per-channel quantization along `channel_axis` (moved to axis 0 in the result's
    scale order, values keep their layout). Explicit `scales` override the max|w| rule."""
    key = str(dtype).upper()
    if key not in _FILTER_QMAX:
        raise ValueError(f"no filter quantization for dtype {dtype!r}")
    qmax = _FILTER_QMAX[key]
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim < 1 or w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("filter must be a non-empty finite array")
    moved = np.moveaxis(w, channel_axis, 0)
    channels = moved.shape[0]
    if scales is None:
        absmax = np.abs(moved.reshape(channels, -1)).max(axis=1)
        s = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    else:
        s = np.asarray(scales, dtype=np.float32).reshape(-1)
        if s.size == 1:
            s = np.full(channels, s[0], dtype=np.float32)
        if s.size != channels or not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError(f"filter scales must be {channels} positive finite values")
    bshape = (-1,) + (1,) * (moved.ndim - 1)
    q = np.clip(round_half_away(moved / s.astype(np.float64).reshape(bshape)), -qmax, qmax)
    values = np.moveaxis(q, 0, channel_axis).astype(np.int8)
    return QuantizedFilter(values, s, key)


def quantize_bias(bias: np.ndarray, input_scale: float, filter_scales: np.ndarray, wide: bool) -> np.ndarray:
    """round(b / (input_scale * filter_scale[c])) in int32, or int64 when `wide` (int16 activations)."""
    b = np.asarray(bias, dtype=np.float64).reshape(-1)
    s = np.float64(input_scale) * np.asarray(filter_scales, dtype=np.float64).reshape(-1)
    if s.size == 1:
        s = np.full(b.size, s[0])
    if s.size != b.size or np.any(s <= 0) or not np.all(np.isfinite(b)):
        raise ValueError("bias needs one positive scale per element and finite values")
    dtype = np.int64 if wide else np.int32
    info = np.iinfo(dtype)
    q = round_half_away(b / s)
    if np.any(q < info.min) or np.any(q > info.max):
        raise ValueError(f"bias does not fit {np.dtype(dtype).name} at these scales")
    return q.astype(dtype)


def pack_int4(values: np.ndarray) -> np.ndarray:
    """TFLite's int4 packing: consecutive elements in pairs, low nibble first; odd count pads 0."""
    v = np.asarray(values).astype(np.int16).reshape(-1)
    if np.any(v < -8) or np.any(v > 7):
        raise ValueError("int4 values must be in [-8, 7]")
    if v.size % 2:
        v = np.concatenate([v, np.zeros(1, np.int16)])
    lo, hi = v[0::2] & 0x0F, v[1::2] & 0x0F
    return ((hi << 4) | lo).astype(np.uint8).view(np.int8)


def same_or_valid(padding: str, in_size: int, kernel: int, stride: int, dilation: int) -> Tuple[int, int]:
    """TFLite's ComputePaddingHeightWidth for one axis: (output size, leading pad)."""
    if stride < 1 or dilation < 1 or kernel < 1 or in_size < 1:
        raise ValueError("stride, dilation, kernel and input size must be positive")
    eff = (kernel - 1) * dilation + 1
    kind = str(padding or "valid").lower()
    if kind == "same":
        out = (in_size + stride - 1) // stride
    elif kind == "valid":
        out = (in_size - eff + stride) // stride
    else:
        raise ValueError(f"unknown padding {padding!r}")
    if out < 1:
        raise ValueError(f"window larger than the input: {in_size} with an effective kernel {eff}")
    return out, max(0, (out - 1) * stride + eff - in_size) // 2


def pair(desc: Mapping, key: str, default: int = 1) -> Tuple[int, int]:
    v = desc.get(key, [default, default])
    if isinstance(v, (int, float)):
        return int(v), int(v)
    if len(v) != 2:
        raise ValueError(f"{key} must be one or two integers, got {v!r}")
    return int(v[0]), int(v[1])


FLOAT_ACTIVATION = {"NONE": (float("-inf"), float("inf")), "RELU": (0.0, float("inf")), "RELU6": (0.0, 6.0),
                    "RELU_N1_TO_1": (-1.0, 1.0)}


def float_activation(name: Optional[str]) -> Tuple[float, float]:
    key = str(name or "NONE").upper()
    if key not in FLOAT_ACTIVATION:
        raise ValueError(f"unsupported fused activation {name!r}")
    return FLOAT_ACTIVATION[key]
