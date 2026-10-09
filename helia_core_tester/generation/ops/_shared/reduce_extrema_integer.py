"""Boundary-sensitive integer inputs and the selection golden for integer reduce max/min."""

import numpy as np


def boundary_inputs(shape, axes, dtype, kind):
    """Place a unique first/last winner in each NHWC reduction domain."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("int8"), np.dtype("int16")):
        raise ValueError("Integer reduce extrema requires int8 or int16")
    if kind not in ("min", "max"):
        raise ValueError("Expected min or max")
    axes = {axis % len(shape) for axis in axes}
    output_shape = tuple(1 if i in axes else n for i, n in enumerate(shape))
    unit = 1 if dtype == np.dtype("int8") else 256
    sign = 1 if kind == "max" else -1
    data = np.full(shape, -sign * 16 * unit, dtype=dtype)
    for flat, output_index in enumerate(np.ndindex(output_shape)):
        index = tuple(
            (0 if flat % 2 == 0 else shape[i] - 1) if i in axes else coordinate
            for i, coordinate in enumerate(output_index)
        )
        data[index] = sign * (32 + 4 * (flat % 24)) * unit
    return data


def reduce_extrema_golden(data, axes, kind, *, keepdims):
    """Max/min over `axes` on the raw codes: exact whenever input and output share
    quantization, which TFLite requires of REDUCE_MAX/MIN on integer tensors."""
    if kind not in ("min", "max"):
        raise ValueError("Expected min or max")
    if not axes:
        raise ValueError("Reduce extrema needs at least one axis")
    rank = data.ndim
    if any(not -rank <= int(a) < rank for a in axes):
        raise ValueError(f"axes {list(axes)} out of range for rank {rank}")
    resolved = tuple(sorted({int(a) % rank for a in axes}))
    reduce = np.max if kind == "max" else np.min
    return reduce(data, axis=resolved, keepdims=keepdims).astype(data.dtype)
