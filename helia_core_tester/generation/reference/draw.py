"""Float tensor draws for reference-golden cases.

Every draw takes an explicit numpy Generator: a case's tensors are a pure
function of its seed (the run seed folded with the case name), so a failure
reproduces from the `--seed` it prints.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Output quantization steps one bias element is worth (from the hoisted-bias
# rule the converter path used for dilated convs, now applied to every biased case): the floor
# clears a 1 LSB comparison tolerance with margin so a dropped bias-add cannot
# hide inside it, the ceiling keeps the bias a few percent of the output range.
BIAS_MIN_STEPS = 3.0
BIAS_MAX_STEPS = 8.0


def _shape(shape: Sequence[int]) -> Tuple[int, ...]:
    dims = tuple(int(d) for d in shape)
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"shape must be non-empty with positive dims, got {shape!r}")
    return dims


def uniform(rng: np.random.Generator, shape: Sequence[int], low: float, high: float) -> np.ndarray:
    if not (np.isfinite(low) and np.isfinite(high)) or low > high:
        raise ValueError(f"invalid uniform range [{low}, {high}]")
    return rng.uniform(low, high, size=_shape(shape)).astype(np.float32)


def glorot_uniform(
    rng: np.random.Generator, shape: Sequence[int], fan_in: int, fan_out: int, gain: float = 1.0
) -> np.ndarray:
    """Keras' default kernel initializer, U(-l, l) with l = sqrt(6 * gain / (fan_in + fan_out));
    gain != 1 is VarianceScaling(gain, "fan_avg", "uniform"), the `weight_gain` knob."""
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(f"fans must be positive, got {fan_in}, {fan_out}")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")
    limit = float(np.sqrt(6.0 * gain / (fan_in + fan_out)))
    return uniform(rng, shape, -limit, limit)


def signed_magnitude(rng: np.random.Generator, shape: Sequence[int], minval: float, maxval: float) -> np.ndarray:
    """|value| ~ U[minval, maxval] with an independent random sign.

    A plain symmetric uniform can land arbitrarily close to zero, and a bias
    below one output step is indistinguishable from no bias in the golden.
    """
    if minval <= 0 or maxval < minval:
        raise ValueError(f"need 0 < minval <= maxval, got {minval}, {maxval}")
    dims = _shape(shape)
    magnitude = rng.uniform(minval, maxval, size=dims)
    signs = rng.choice((-1.0, 1.0), size=dims)
    return magnitude * signs


def bias_in_output_steps(
    rng: np.random.Generator,
    channels: int,
    output_scale: float,
    accumulator_scales: np.ndarray,
    dtype: type = np.int32,
) -> np.ndarray:
    """Quantized bias worth BIAS_MIN_STEPS..BIAS_MAX_STEPS output steps per channel.

    accumulator_scales is input_scale * weight_scale_c, per channel (or one value).
    """
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    acc = np.asarray(accumulator_scales, dtype=np.float64)
    if acc.size == 1:
        acc = np.repeat(acc, channels)
    if acc.size != channels or not np.all(acc > 0):
        raise ValueError("accumulator scales must be positive, one per channel")
    steps = signed_magnitude(rng, (channels,), BIAS_MIN_STEPS, BIAS_MAX_STEPS)
    info = np.iinfo(dtype)
    bias = np.round(steps * float(output_scale) / acc)  # never a tie: steps is continuous
    if np.any(bias < info.min) or np.any(bias > info.max):
        raise OverflowError(f"bias does not fit {np.dtype(dtype).name}")
    return bias.astype(dtype)
