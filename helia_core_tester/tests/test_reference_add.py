"""Add on the C reference, against independent models.

int8/int16: a Python big-int model of TFLite's quantized add (exact), and the
float64 sum of the dequantized operands (within one output step). float32: numpy.
float16: numpy's binary32 sum rounded once to binary16.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference.abi import activation_code, dtype_code
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings, output_shape_for


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _srdhm(a: int, b: int) -> int:
    if a == b == -(2**31):
        return 2**31 - 1
    return (a * b + 2**30) >> 31  # floor(ab / 2^31 + 1/2): gemmlowp's rounding


def _rdbpot(x: int, e: int) -> int:
    if e == 0:
        return x
    q, r = divmod(x, 1 << e)  # floor division
    half = 1 << (e - 1)
    return q + (1 if (r > half or (r == half and x >= 0)) else 0)


def _mbqm(x: int, m: int, shift: int) -> int:
    left, right = max(shift, 0), max(-shift, 0)
    return _rdbpot(_srdhm(x << left, m), right)


def _model_add(a: np.ndarray, b: np.ndarray, p: dict, dtype) -> np.ndarray:
    out_shape = np.broadcast_shapes(a.shape, b.shape)
    a, b = np.broadcast_to(a, out_shape).ravel(), np.broadcast_to(b, out_shape).ravel()
    res = []
    for x1, x2 in zip(a.tolist(), b.tolist()):
        s1 = _mbqm((x1 + p["input1_offset"]) << p["left_shift"], p["input1_multiplier"], p["input1_shift"])
        s2 = _mbqm((x2 + p["input2_offset"]) << p["left_shift"], p["input2_multiplier"], p["input2_shift"])
        o = _mbqm(s1 + s2, p["output_multiplier"], p["output_shift"]) + p["output_offset"]
        res.append(min(max(o, p["activation_min"]), p["activation_max"]))
    return np.array(res, dtype=dtype).reshape(out_shape)


def _prepare(lib, kind, s1, z1, s2, z2, so, zo, activation="NONE"):
    return lib.prepare("add_prepare", {
        "dtype": dtype_code("int8" if kind == "s8" else "int16"), "activation": activation_code(activation),
        "input1_scale": s1, "input1_zero_point": z1, "input2_scale": s2, "input2_zero_point": z2,
        "output_scale": so, "output_zero_point": zo,
    })


def _run(lib, kind, p, a, b):
    out_shape = output_shape_for("add", a.shape, b.shape)
    return lib.run(f"add_{kind}", p, {"input1": a, "input2": b}, {"output": out_shape})["output"]


SHAPES = [((2, 3, 4, 5), (2, 3, 4, 5)), ((2, 3, 4, 5), (5,)), ((1, 3, 1, 5), (2, 1, 4, 1)), ((), (7,)),
          ((4, 1), ()), ((1, 1, 1, 1, 9), (3, 1, 2, 1, 1)), ((3,), (3,))]


@pytest.mark.parametrize("kind", ["s8", "s16"])
@pytest.mark.parametrize("seed", range(4))
def test_quantized_add_matches_the_tflite_model_and_float(lib, kind, seed) -> None:
    rng = np.random.default_rng(seed)
    dtype = np.int8 if kind == "s8" else np.int16
    info = np.iinfo(dtype)
    for shape_a, shape_b in SHAPES:
        s1, s2, so = (float(np.float32(x)) for x in rng.uniform(0.01, 0.2, 3) * (1 if kind == "s8" else 1 / 256))
        if kind == "s8":
            z1, z2, zo = (int(z) for z in rng.integers(-20, 21, 3))
        else:
            z1 = z2 = zo = 0
        p = _prepare(lib, kind, s1, z1, s2, z2, so, zo)
        a = rng.integers(info.min, info.max + 1, shape_a).astype(dtype)
        b = rng.integers(info.min, info.max + 1, shape_b).astype(dtype)
        got = _run(lib, kind, p, a, b)
        np.testing.assert_array_equal(got, _model_add(a, b, p, dtype))
        real = (a.astype(np.float64) - z1) * s1 + (b.astype(np.float64) - z2) * s2
        ideal = np.clip(np.round(real / so) + zo, info.min, info.max)
        assert np.abs(got.astype(np.float64) - ideal).max() <= 1


def test_prepare_matches_tflite_add_params(lib) -> None:
    # Scales all 0.125: input multipliers 0.5 = 2^30 * 2^-30 (shift 0), output 0.25 / (2^20 * 0.125).
    p = _prepare(lib, "s8", 0.125, 0, 0.125, 0, 0.125, 0)
    assert p == {"left_shift": 20, "input1_offset": 0, "input1_multiplier": 1 << 30, "input1_shift": 0,
                 "input2_offset": 0, "input2_multiplier": 1 << 30, "input2_shift": 0, "output_offset": 0,
                 "output_multiplier": 1 << 30, "output_shift": -18, "activation_min": -128, "activation_max": 127}
    q = _prepare(lib, "s8", 0.1, -3, 0.2, 5, 0.3, 7, activation="RELU")
    assert (q["input1_offset"], q["input2_offset"], q["output_offset"]) == (3, -5, 7)
    assert q["activation_min"] == 7 and q["activation_max"] == 127
    assert _prepare(lib, "s16", 1 / 32768, 0, 1 / 32768, 0, 1 / 32768, 0)["left_shift"] == 15


def test_prepare_agrees_with_the_python_multiplier_helper(lib) -> None:
    from helia_core_tester.generation.utils.tflite_utils import elementwise_addsub_quant_params

    rng = np.random.default_rng(3)
    for _ in range(200):
        s1, s2, so = (float(np.float32(x)) for x in rng.uniform(1e-3, 1.0, 3))
        for kind, dt in (("s8", "S8"), ("s16", "S16")):
            p = _prepare(lib, kind, s1, 0, s2, 0, so, 0)
            q = elementwise_addsub_quant_params(input1_scale=s1, input2_scale=s2, output_scale=so, activation_dtype=dt)
            assert (p["input1_multiplier"], p["input1_shift"], p["input2_multiplier"], p["input2_shift"],
                    p["output_multiplier"], p["output_shift"], p["left_shift"]) == (
                q["input1_mult"], q["input1_shift"], q["input2_mult"], q["input2_shift"],
                q["out_mult"], q["out_shift"], q["left_shift"])


@pytest.mark.parametrize(
    "kind, overrides",
    [
        ("s8", {"input1_scale": 0.0}),
        ("s8", {"output_scale": -0.1}),
        ("s8", {"input2_scale": math.nan}),
        ("s8", {"input1_zero_point": 128}),
        ("s8", {"output_zero_point": -129}),
        ("s16", {"input1_zero_point": 1}),
        ("s16", {"output_zero_point": -1}),
        ("s8", {"activation": 42}),
    ],
)
def test_prepare_rejects_invalid_quantization(lib, kind, overrides) -> None:
    fields = {"dtype": dtype_code("int8" if kind == "s8" else "int16"), "activation": activation_code("NONE"),
              "input1_scale": 0.1, "input1_zero_point": 0, "input2_scale": 0.1, "input2_zero_point": 0,
              "output_scale": 0.1, "output_zero_point": 0, **overrides}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        lib.prepare("add_prepare", fields)


def test_prepare_rejects_a_float_dtype(lib) -> None:
    with pytest.raises(ReferenceKernelError, match="E_DTYPE"):
        lib.prepare("add_prepare", {"dtype": dtype_code("float32"), "activation": 0, "input1_scale": 0.1,
                                    "input1_zero_point": 0, "input2_scale": 0.1, "input2_zero_point": 0,
                                    "output_scale": 0.1, "output_zero_point": 0})


@pytest.mark.parametrize(
    "field, value",
    [("left_shift", 19), ("input1_multiplier", -1), ("output_shift", 1), ("input2_shift", -32),
     ("input1_offset", 129), ("output_offset", 128), ("activation_min", 10), ("activation_max", 200)],
)
def test_kernel_rejects_invalid_params(lib, field, value) -> None:
    p = {**_prepare(lib, "s8", 0.1, 0, 0.1, 0, 0.1, 0), "activation_max": 5, field: value}
    a = np.zeros(4, np.int8)
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, "s8", p, a, a)


def test_int16_kernel_rejects_offsets(lib) -> None:
    p = {**_prepare(lib, "s16", 1e-4, 0, 1e-4, 0, 1e-4, 0), "input2_offset": 1}
    a = np.zeros(4, np.int16)
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, "s16", p, a, a)


def test_saturation_and_activation_clamp(lib) -> None:
    p = _prepare(lib, "s8", 0.125, 0, 0.125, 0, 0.125, 0)
    a = np.array([127, -128, 100, -100, 0], np.int8)
    np.testing.assert_array_equal(_run(lib, "s8", p, a, a), [127, -128, 127, -128, 0])
    relu = _prepare(lib, "s8", 0.125, 0, 0.125, 0, 0.125, 0, activation="RELU")
    np.testing.assert_array_equal(_run(lib, "s8", relu, a, np.zeros(5, np.int8)), [127, 0, 100, 0, 0])


def test_empty_tensors_are_accepted(lib) -> None:
    p = _prepare(lib, "s8", 0.1, 0, 0.1, 0, 0.1, 0)
    assert _run(lib, "s8", p, np.zeros((3, 0), np.int8), np.zeros((1, 0), np.int8)).shape == (3, 0)


@pytest.mark.parametrize("shapes", SHAPES)
def test_float32_add_is_numpy(lib, shapes) -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(shapes[0]).astype(np.float32)
    b = rng.standard_normal(shapes[1]).astype(np.float32)
    got = _run(lib, "f32", {"activation_min": -0.5, "activation_max": 0.75}, a, b)
    np.testing.assert_array_equal(got, np.clip(a + b, np.float32(-0.5), np.float32(0.75)))


def test_float16_add_rounds_once(lib) -> None:
    rng = np.random.default_rng(2)
    a = (rng.standard_normal(4096) * 300).astype(np.float16)
    b = (rng.standard_normal(4096) * 300).astype(np.float16)
    got = _run(lib, "f16", {"activation_min": -math.inf, "activation_max": math.inf}, a, b)
    np.testing.assert_array_equal(got, (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16))


def test_float_nonfinite_values_propagate_and_clamp(lib) -> None:
    a = np.array([np.nan, np.inf, -np.inf, 65504, 1.0], np.float16)
    b = np.array([1.0, 1.0, 1.0, 32.0, np.nan], np.float16)
    got = _run(lib, "f16", {"activation_min": -math.inf, "activation_max": math.inf}, a, b)
    assert np.isnan(got[0]) and np.isposinf(got[1]) and np.isneginf(got[2]) and np.isposinf(got[3])
    assert np.isnan(got[4])
    clamped = _run(lib, "f16", {"activation_min": -6.0, "activation_max": 6.0}, a, b)
    assert np.isnan(clamped[0]) and clamped[1] == 6 and clamped[2] == -6 and clamped[3] == 6
    f32 = _run(lib, "f32", {"activation_min": -1e30, "activation_max": 1e30},
               np.array([np.inf], np.float32), np.array([0.0], np.float32))
    assert f32[0] == np.float32(1e30)
