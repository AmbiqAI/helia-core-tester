"""arm_nn_requantize in numpy, for the host-side conv models of the case bundle and the fake target."""

from __future__ import annotations

import numpy as np

# Q31 multiplier and shift of a unit scale (frexp(1.0) = 0.5 * 2^1).
UNIT_MULTIPLIER = 1 << 30
UNIT_SHIFT = 1


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
