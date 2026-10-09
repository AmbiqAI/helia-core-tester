"""Deterministic quantization policy for reference-golden cases.

Replaces the TFLiteConverter's calibration: every scale and zero point is a
pure function of the drawn float data (or of an explicit descriptor value), so
a case reproduces from its seed alone. Scales are stored as float32, as a
TFLite tensor stores them, and every multiplier is derived from those float32
values.

Rules:
  - s8/s4 activations: asymmetric over a range that is first widened to include
    0, scale = (hi - lo) / (qmax - qmin), zero point nudged into range
    (TFLite's ChooseQuantizationParams).
  - s16 activations: symmetric, zero point 0, scale = max|x| / 32767.
  - weights: symmetric, zero point 0, per output channel by default
    (max|w_c| / 127, or / 7 for s4), per tensor on request.
  - bias: int32 (int64 for s16 activations) at scale input_scale * weight_scale_c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from helia_core_tester.generation.reference.params import dtype_range

# Floor for a range or channel max: an all-zero tensor still needs a positive
# scale, and TFLite substitutes a tiny one the same way.
_MIN_RANGE = 1e-6


@dataclass(frozen=True)
class TensorQuant:
    """Per-tensor quantization of one activation tensor."""

    scale: float
    zero_point: int
    dtype: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale must be positive and finite, got {self.scale}")
        qmin, qmax = dtype_range(self.dtype)
        if not qmin <= self.zero_point <= qmax:
            raise ValueError(f"zero point {self.zero_point} outside {self.dtype} range")

    def to_json(self) -> dict:
        return {"scale": self.scale, "zero_point": self.zero_point, "dtype": self.dtype}


@dataclass(frozen=True)
class WeightQuant:
    """Symmetric weight quantization: one scale per output channel, or one."""

    scales: Tuple[float, ...]
    dtype: str
    axis: int

    @property
    def per_channel(self) -> bool:
        return len(self.scales) > 1

    def to_json(self) -> dict:
        return {"scales": list(self.scales), "zero_point": 0, "dtype": self.dtype, "axis": self.axis}


def _f32(value: float) -> float:
    return float(np.float32(value))


def round_half_away(values) -> np.ndarray:
    """TfLiteRound / std::round: halves round away from zero (np.round is to-even)."""
    v = np.asarray(values, dtype=np.float64)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def asymmetric(lo: float, hi: float, dtype: str = "s8") -> TensorQuant:
    """ChooseQuantizationParams for an asymmetric integer type."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ValueError(f"invalid range [{lo}, {hi}]")
    qmin, qmax = dtype_range(dtype)
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if hi - lo < _MIN_RANGE:
        hi = lo + _MIN_RANGE
    scale = (hi - lo) / (qmax - qmin)
    # Nudge the zero point onto an integer, choosing the endpoint whose
    # error is smaller, exactly as ChooseQuantizationParams does (in double;
    # only the stored scale is float32).
    zp_from_min = qmin - lo / scale
    zp_from_max = qmax - hi / scale
    err_min = abs(qmin) + abs(lo / scale)
    err_max = abs(qmax) + abs(hi / scale)
    initial = zp_from_min if err_min < err_max else zp_from_max
    zero_point = int(qmin if initial < qmin else qmax if initial > qmax else round_half_away(initial))
    return TensorQuant(_f32(scale), zero_point, dtype)


def symmetric(absmax: float, dtype: str = "s16") -> TensorQuant:
    if not np.isfinite(absmax) or absmax < 0:
        raise ValueError(f"invalid absmax {absmax}")
    _, qmax = dtype_range(dtype)
    return TensorQuant(_f32(max(float(absmax), _MIN_RANGE) / qmax), 0, dtype)


def activation_quant(data: np.ndarray, dtype: str, value_range: Optional[Tuple[float, float]] = None) -> TensorQuant:
    """Quantization for an activation tensor from its data or an explicit range."""
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("activation data must be non-empty and finite")
        lo, hi = float(arr.min()), float(arr.max())
    if dtype.lower() == "s16":
        return symmetric(max(abs(lo), abs(hi)), "s16")
    return asymmetric(lo, hi, dtype.lower())


def weight_quant(weights: np.ndarray, dtype: str = "s8", axis: int = 0, per_channel: bool = True) -> WeightQuant:
    """Symmetric weight scales, per output channel along `axis` or per tensor."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("weights must be non-empty and finite")
    _, qmax = dtype_range(dtype)
    if per_channel:
        axis = axis % w.ndim
        reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
        maxima = np.max(np.abs(w), axis=reduce_axes) if reduce_axes else np.abs(w)
    else:
        maxima = np.array([np.max(np.abs(w))])
    scales = tuple(_f32(max(float(m), _MIN_RANGE) / qmax) for m in np.atleast_1d(maxima))
    return WeightQuant(scales, dtype, axis)


def quantize(data: np.ndarray, quant: TensorQuant) -> np.ndarray:
    """round(x / scale) + zp, saturated, in the storage dtype."""
    qmin, qmax = dtype_range(quant.dtype)
    q = round_half_away(np.asarray(data, dtype=np.float64) / quant.scale) + quant.zero_point
    out_dtype = np.int16 if quant.dtype == "s16" else np.int8
    return np.clip(q, qmin, qmax).astype(out_dtype)


def quantize_weights(weights: np.ndarray, quant: WeightQuant) -> np.ndarray:
    qmin, qmax = dtype_range(quant.dtype)
    w = np.asarray(weights, dtype=np.float64)
    scales = np.asarray(quant.scales, dtype=np.float64)
    if quant.per_channel:
        shape = [1] * w.ndim
        shape[quant.axis] = scales.size
        scales = scales.reshape(shape)
    # Symmetric weights may use the full negative end only for s4/s8 storage of
    # -qmax..qmax; TFLite never emits qmin for symmetric weights.
    return np.clip(round_half_away(w / scales), -qmax, qmax).astype(np.int8)


def bias_quant_scales(input_scale: float, weights: WeightQuant, channels: int) -> np.ndarray:
    scales = np.asarray(weights.scales, dtype=np.float64)
    if scales.size == 1:
        scales = np.repeat(scales, channels)
    if scales.size != channels:
        raise ValueError(f"{scales.size} weight scales for {channels} channels")
    return float(np.float32(input_scale)) * scales


def quantize_bias(bias: np.ndarray, input_scale: float, weights: WeightQuant, dtype: type = np.int32) -> np.ndarray:
    b = np.asarray(bias, dtype=np.float64)
    scales = bias_quant_scales(input_scale, weights, b.size)
    q = round_half_away(b / scales)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize(data: np.ndarray, quant: TensorQuant) -> np.ndarray:
    return ((np.asarray(data, dtype=np.float64) - quant.zero_point) * quant.scale).astype(np.float32)


def output_quant_from_float(reference: np.ndarray, dtype: str, headroom: float = 1.0) -> TensorQuant:
    """Output quantization from a float reference run, widened by `headroom`."""
    if headroom < 1.0:
        raise ValueError(f"headroom must be >= 1, got {headroom}")
    arr = np.asarray(reference, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    return activation_quant(arr, dtype, (lo * headroom, hi * headroom))


def descriptor_quant(entry: Optional[dict], dtype: str) -> Optional[TensorQuant]:
    """An explicit `quantization.<role>` descriptor block, or None.

    Accepts {scale, zero_point} or {range: [lo, hi]}; never both.
    """
    if not entry:
        return None
    has_scale = "scale" in entry
    has_range = "range" in entry
    if has_scale == has_range:
        raise ValueError(f"quantization entry needs exactly one of scale/zero_point or range: {entry!r}")
    if has_scale:
        return TensorQuant(_f32(float(entry["scale"])), int(entry.get("zero_point", 0)), dtype.lower())
    lo, hi = entry["range"]
    return activation_quant(np.zeros(1), dtype, (float(lo), float(hi)))


def as_tuple(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)
