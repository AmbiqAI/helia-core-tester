"""Independent float sqrt/rsqrt goldens and bit-level contract checks (ns-cmsis-nn#295)."""

from pathlib import Path

import numpy as np


def sqrt_float_reference(bits: np.ndarray, reciprocal: bool) -> np.ndarray:
    """Evaluate finite positives in float64; encode the public special-value contract."""
    half = bits.dtype == np.uint16
    dtype = np.float16 if half else np.float32
    sign, inf, quiet = (
        (0x8000, 0x7C00, 0x0200) if half else (0x80000000, 0x7F800000, 0x00400000)
    )
    magnitude = bits & (sign - 1)
    with np.errstate(all="ignore"):
        values = bits.view(dtype).astype(np.float64)
        result = np.sqrt(values)
        if reciprocal:
            result = 1.0 / result
        output = result.astype(dtype).view(bits.dtype).copy()
    output[(bits & sign != 0) & (magnitude != 0)] = inf | quiet
    nan = magnitude > inf
    output[nan] = bits[nan] | quiet
    zero = magnitude == 0
    output[zero] = bits[zero] | (inf if reciprocal else 0)
    output[bits == inf] = 0 if reciprocal else inf
    return output


def sqrt_float_inputs(dtype: str, count: int, pattern: str, seed: int) -> np.ndarray:
    half = dtype == "FP16"
    word = np.uint16 if half else np.uint32
    inf, one, fraction, sign = (
        (0x7C00, 0x3C00, 10, 0x8000)
        if half
        else (0x7F800000, 0x3F800000, 23, 0x80000000)
    )
    if pattern == "special":
        base = [
            0,
            sign,
            inf,
            sign | inf,
            one | sign,
            one,
            1,
            (1 << fraction) - 1,
            1 << fraction,
            inf - 1,
            inf | 1,
            inf | 3 | sign,
            inf | (1 << (fraction - 1)) | 7,
            inf | (1 << (fraction - 1)) | 11 | sign,
            sign | 1,
            one - 1,
            one + 1,
        ]
        return np.resize(np.array(base, dtype=word), count)
    if pattern == "powers_of_four":
        bias = 15 if half else 127
        exponents = range(-14 if half else -126, 16 if half else 128, 2)
        return np.resize(
            np.array([(e + bias) << fraction for e in exponents], dtype=word), count
        )
    if pattern != "positive":
        raise ValueError(f"Unknown float sqrt input pattern: {pattern}")
    return np.random.default_rng(seed).integers(
        1 << fraction, inf, size=count, dtype=word
    )


def generate_sqrt_float(op, output_dir: Path, reciprocal: bool) -> None:
    dtype = op.tensor_dtype("input")
    if dtype not in ("FP16", "FP32") or op.tensor_dtype("output") != dtype:
        raise ValueError("Float sqrt/rsqrt requires matching FP16 or FP32 tensors")
    hint = op.desc.get("hint", {})
    count = int(np.prod(op.desc["input_shape"]))
    if count < 1:
        raise ValueError("Float sqrt descriptors need positive storage dimensions")
    pattern = hint.get("float_pattern", "positive")
    bits = sqrt_float_inputs(dtype, count, pattern, op.seed)
    expected = sqrt_float_reference(bits, reciprocal)
    error = hint.get("api_error", "")
    if error not in ("", "null_input", "null_output", "zero_block", "negative_block"):
        raise ValueError(f"Unknown float sqrt api_error: {error}")
    half = dtype == "FP16"
    suffix = "rsqrt" if reciprocal else "sqrt"
    kernel = ("arm_rsqrt_" if reciprocal else "arm_nn_sqrt_") + (
        "f16" if half else "f32"
    )
    context = {
        "name": op.desc["name"],
        "kernel_fn": kernel,
        "op_suffix": suffix,
        "input_dtype": "float16_t" if half else "float",
        "output_dtype": "float16_t" if half else "float",
        "word_type": "uint16_t" if half else "uint32_t",
        "half": half,
        "block_size": count,
        "input_bits": [hex(int(x)) for x in bits],
        "expected_bits": [hex(int(x)) for x in expected],
        "api_error": error,
        "in_place": bool(hint.get("in_place", False)),
        "call_size": (
            0 if error == "zero_block" else -1 if error == "negative_block" else count
        ),
        "expected_status": (
            "ARM_CMSIS_NN_ARG_ERROR" if error else "ARM_CMSIS_NN_SUCCESS"
        ),
        "max_ulp": int(reciprocal and not half and pattern != "powers_of_four"),
        "flushed_bits": "0x7f800000" if reciprocal else "0",
        "validation_mode": "float",
    }
    op._write_op_outputs(
        output_dir,
        suffix,
        "BasicMathFunctions/sqrt_float/sqrt_float.h.j2",
        "BasicMathFunctions/sqrt_float/sqrt_float.c.j2",
        context,
        {
            "name": op.desc["name"],
            "operator": op.desc["operator"],
            "operator_name": suffix,
        },
    )
