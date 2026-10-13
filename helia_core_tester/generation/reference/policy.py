"""Deterministic quantization policy: replaces the TFLiteConverter's calibration.

Every scale and zero point is a pure function of the drawn float data, an explicit
range, or a descriptor `quantization.<role>` block, so a case reproduces from its
seed alone. Scales are stored as float32, as a TFLite tensor stores them.

  - s8: asymmetric over a range first widened to include 0,
    scale = (hi - lo) / (qmax - qmin), zero point nudged into range as TFLite's
    ChooseQuantizationParams nudges it.
  - s16: symmetric, zero point 0, scale = max|x| / 32767.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

# Floor for a range: an all-zero tensor still needs a positive scale.
_MIN_RANGE = 1e-6

_RANGES = {"s8": (-128, 127), "s16": (-32768, 32767)}


def dtype_range(dtype: str) -> Tuple[int, int]:
    key = str(dtype).lower()
    if key not in _RANGES:
        raise ValueError(f"no integer range for dtype {dtype!r}")
    return _RANGES[key]


def _f32(value: float) -> float:
    return float(np.float32(value))


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
            raise ValueError(f"zero point {self.zero_point} outside the {self.dtype} range")
        if self.dtype == "s16" and self.zero_point != 0:
            raise ValueError("s16 tensors are symmetric (zero point 0)")

    def to_json(self) -> dict:
        return {"scale": self.scale, "zero_point": self.zero_point}


def round_half_away(values) -> np.ndarray:
    """TfLiteRound / std::round: halves round away from zero (np.round rounds to even)."""
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
    # The endpoint whose zero point carries the smaller error wins, as in
    # ChooseQuantizationParams (in double; only the stored scale is float32).
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


def activation_quant(dtype: str, value_range: Tuple[float, float]) -> TensorQuant:
    """Quantization of an activation tensor covering `value_range`."""
    lo, hi = float(value_range[0]), float(value_range[1])
    if str(dtype).lower() == "s16":
        return symmetric(max(abs(lo), abs(hi)), "s16")
    return asymmetric(lo, hi, str(dtype).lower())


def data_range(data: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("activation data must be non-empty and finite")
    return float(arr.min()), float(arr.max())


def descriptor_quant(entry: Optional[Mapping], dtype: str) -> Optional[TensorQuant]:
    """An explicit `quantization.<role>` descriptor block, or None.

    Accepts {scale, zero_point} or {range: [lo, hi]}, never both."""
    if not entry:
        return None
    has_scale, has_range = "scale" in entry, "range" in entry
    if has_scale == has_range:
        raise ValueError(f"quantization entry needs exactly one of scale/zero_point or range: {dict(entry)!r}")
    if has_scale:
        return TensorQuant(_f32(float(entry["scale"])), int(entry.get("zero_point", 0)), str(dtype).lower())
    lo, hi = entry["range"]
    return activation_quant(dtype, (float(lo), float(hi)))


def quantize(data: np.ndarray, quant: TensorQuant) -> np.ndarray:
    """round(x / scale) + zp (halves away from zero), saturated, in the storage dtype."""
    qmin, qmax = dtype_range(quant.dtype)
    q = round_half_away(np.asarray(data, dtype=np.float64) / quant.scale) + quant.zero_point
    return np.clip(q, qmin, qmax).astype(np.int16 if quant.dtype == "s16" else np.int8)


def dequantize(data: np.ndarray, quant: TensorQuant) -> np.ndarray:
    return (np.asarray(data, dtype=np.float64) - quant.zero_point) * quant.scale
