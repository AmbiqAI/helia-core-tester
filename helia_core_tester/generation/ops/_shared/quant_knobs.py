"""Descriptor knobs for conv quant edges."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def value_range(desc: Mapping[str, Any], key: str, default: Sequence[float]) -> tuple[float, float]:
    """A [lo, hi] descriptor range, else default."""
    lo, hi = desc.get(key) or default
    return float(lo), float(hi)


def clamp_golden(desc: Mapping[str, Any], output: np.ndarray) -> np.ndarray:
    """Apply the descriptor's activation clamp."""
    if "activation_min" not in desc and "activation_max" not in desc:
        return output
    info = np.iinfo(output.dtype)
    low, high = int(desc.get("activation_min", info.min)), int(desc.get("activation_max", info.max))
    return np.clip(output, low, high).astype(output.dtype)
