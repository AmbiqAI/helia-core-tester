"""Full-width signed inputs with exact integer boundary values."""

import numpy as np


def sample_int32(rng, shape, *, phase=0):
    values = rng.integers(-(2**31), 2**31, size=shape, dtype=np.int32)
    edges = np.array([-(2**31), 2**31 - 1, 0, -1, 1, 2**24 + 1, 2**24], dtype=np.int32)
    edges = np.roll(edges, phase)
    count = min(values.size, edges.size)
    values.flat[:count] = edges[:count]
    return values
