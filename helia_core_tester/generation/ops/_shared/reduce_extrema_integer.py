"""Boundary-sensitive integer inputs; goldens still come from LiteRT."""

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


def resize_integer_interpreter(interpreter, shape):
    """Honor descriptor batch size and require selection without requantization."""
    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], shape, strict=True
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if tuple(input_details[0]["shape"]) != tuple(shape):
        raise ValueError("LiteRT input shape differs from descriptor")
    if input_details[0]["dtype"] != output_details[0]["dtype"]:
        raise ValueError("Reduce extrema requires matching tensor dtypes")
    for key in ("scales", "zero_points"):
        # Raw-code selection supports per-tensor quantization only. Comparing
        # per-axis arrays alone would ignore which dimensions they describe.
        if any(
            np.asarray(details[0]["quantization_parameters"][key]).size != 1
            for details in (input_details, output_details)
        ):
            raise ValueError("Reduce extrema requires per-tensor quantization")
        if not np.array_equal(
            input_details[0]["quantization_parameters"][key],
            output_details[0]["quantization_parameters"][key],
        ):
            raise ValueError("Reduce extrema requires matching quantization")
    return input_details, output_details
