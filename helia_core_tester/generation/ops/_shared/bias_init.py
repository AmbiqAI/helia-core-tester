"""Bias initializers for operators whose Keras model feeds the quantized pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import keras
import numpy as np


class SignedMagnitudeUniform(keras.initializers.Initializer):
    """Seed-derived values with ``|value|`` uniform in ``[minval, maxval]``.

    A plain uniform over ``[-limit, limit]`` can draw arbitrarily close to
    zero, and a bias element below one output quantization step is
    indistinguishable from no bias at all in the golden. Sampling the
    magnitude and the sign separately keeps every channel above the
    detection floor, which matters most for single-output-channel cases
    where there is no other channel to carry the signal.

    Args:
        minval: Smallest absolute value produced. Must be > 0.
        maxval: Largest absolute value produced. Must be >= ``minval``.
        seed: Seed for the generator; identical seeds give identical tensors.
    """

    def __init__(self, minval: float, maxval: float, seed: Optional[int] = None):
        if minval <= 0:
            raise ValueError(f"minval must be positive, got {minval}")
        if maxval < minval:
            raise ValueError(f"maxval ({maxval}) must be >= minval ({minval})")
        self.minval = float(minval)
        self.maxval = float(maxval)
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rng = np.random.default_rng(self.seed)
        shape = tuple(int(dim) for dim in shape)
        magnitude = rng.uniform(self.minval, self.maxval, size=shape)
        signs = rng.choice((-1.0, 1.0), size=shape)
        return keras.ops.convert_to_tensor(
            (magnitude * signs).astype(np.float32), dtype=dtype or "float32"
        )

    def get_config(self) -> Dict[str, Any]:
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}
