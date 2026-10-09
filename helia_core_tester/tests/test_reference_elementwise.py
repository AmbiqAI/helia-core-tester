"""Sub, Mul, Maximum/Minimum, SquaredDifference, Comparison, Abs and Clamp on the C reference,
against independent big-int models (tests/reference_models.py) and numpy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference import quant as ref_quant
from helia_core_tester.generation.reference.abi import comparison_code
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings, output_shape_for
from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
from helia_core_tester.tests.reference_models import broadcast_pairs, mbqm

SHAPES = [((2, 3, 4), (2, 3, 4)), ((2, 3, 4), (4,)), ((1, 3, 1), (2, 1, 5)), ((), (6,)), ((3, 1), ())]
KINDS = {"s8": np.int8, "s16": np.int16}


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _run(lib, entry, params, *inputs):
    names = ("input1", "input2") if len(inputs) == 2 else ("input",)
    shape = output_shape_for(entry, *(x.shape for x in inputs))
    return lib.run(entry, params, dict(zip(names, inputs)), {"output": shape})["output"]


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _ints(rng, dtype, shape):
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max + 1, shape).astype(dtype)


def _scales(rng, kind, n):
    base = 1.0 if kind == "s8" else 1.0 / 256
    return [float(np.float32(x)) for x in rng.uniform(0.01, 0.2, n) * base]


def _zps(rng, kind, n):
    return [int(z) for z in rng.integers(-20, 21, n)] if kind == "s8" else [0] * n


def _quant(kind, scales, zps):
    return {"dtype": ref_quant.hct_dtype(kind.upper()), "activation": 0,
            "input1_scale": scales[0], "input1_zero_point": zps[0],
            "input2_scale": scales[1], "input2_zero_point": zps[1],
            "output_scale": scales[2], "output_zero_point": zps[2]}


# ------------------------------------------------------------------- Sub


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(3))
def test_sub_matches_the_model(lib, kind, seed) -> None:
    rng = np.random.default_rng(seed)
    dtype = KINDS[kind]
    info = np.iinfo(dtype)
    for sa, sb in SHAPES:
        scales, zps = _scales(rng, kind, 3), _zps(rng, kind, 3)
        p = lib.prepare("sub_prepare", _quant(kind, scales, zps))
        a, b = _ints(rng, dtype, sa), _ints(rng, dtype, sb)
        xs, ys, shape = broadcast_pairs(a, b)
        want = [min(max(mbqm(mbqm((x + p["input1_offset"]) << p["left_shift"], p["input1_multiplier"],
                                       p["input1_shift"])
                                  - mbqm((y + p["input2_offset"]) << p["left_shift"], p["input2_multiplier"],
                                         p["input2_shift"]),
                                  p["output_multiplier"], p["output_shift"]) + p["output_offset"], info.min),
                        info.max) for x, y in zip(xs, ys)]
        np.testing.assert_array_equal(_run(lib, f"sub_{kind}", p, a, b), np.array(want, dtype).reshape(shape))


def test_sub_and_add_prepare_agree(lib) -> None:
    q = _quant("s8", [0.1, 0.2, 0.3], [1, -2, 3])
    assert lib.prepare("sub_prepare", q) == lib.prepare("add_prepare", q)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_float_sub_and_mul_are_ieee(lib, dtype) -> None:
    rng = np.random.default_rng(4)
    a = (rng.standard_normal(500) * 100).astype(dtype)
    b = (rng.standard_normal(500) * 100).astype(dtype)
    kind = "f32" if dtype == np.float32 else "f16"
    free = {"activation_min": -math.inf, "activation_max": math.inf}
    np.testing.assert_array_equal(_run(lib, f"sub_{kind}", free, a, b),
                                  (a.astype(np.float32) - b.astype(np.float32)).astype(dtype))
    np.testing.assert_array_equal(_run(lib, f"mul_{kind}", free, a, b),
                                  (a.astype(np.float32) * b.astype(np.float32)).astype(dtype))
    clamp = {"activation_min": -1.0, "activation_max": 2.0}
    np.testing.assert_array_equal(_run(lib, f"mul_{kind}", clamp, a, b),
                                  np.clip(a.astype(np.float32) * b.astype(np.float32), -1, 2).astype(dtype))


# ------------------------------------------------------------------- Mul


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(3))
def test_mul_matches_the_model(lib, kind, seed) -> None:
    rng = np.random.default_rng(10 + seed)
    dtype = KINDS[kind]
    info = np.iinfo(dtype)
    for sa, sb in SHAPES:
        scales, zps = _scales(rng, kind, 3), _zps(rng, kind, 3)
        p = lib.prepare("mul_prepare", _quant(kind, scales, zps))
        m, s = calculate_multiplier_shift(scales[0] * scales[1] / scales[2])
        assert (p["output_multiplier"], p["output_shift"]) == (m, s)
        a, b = _ints(rng, dtype, sa), _ints(rng, dtype, sb)
        xs, ys, shape = broadcast_pairs(a, b)
        want = [min(max(p["output_offset"] + mbqm((x + p["input1_offset"]) * (y + p["input2_offset"]),
                                                  p["output_multiplier"], p["output_shift"]), info.min), info.max)
                for x, y in zip(xs, ys)]
        np.testing.assert_array_equal(_run(lib, f"mul_{kind}", p, a, b), np.array(want, dtype).reshape(shape))


def test_mul_exact_ties_round_away_from_zero(lib) -> None:
    # 0.125 * 0.125 / 0.125 = 2^-3: SRDHM is exact here, so the tie lands in the final
    # rounding divide-by-POT, which rounds halves away from zero (as arm_nn_requantize).
    p = lib.prepare("mul_prepare", _quant("s8", [0.125, 0.125, 0.125], [0, 0, 0]))
    a = np.array([2, -2, 6, -6], np.int8)
    b = np.array([2, 2, 2, 2], np.int8)  # products 4, -4, 12, -12 -> 0.5, -0.5, 1.5, -1.5
    np.testing.assert_array_equal(_run(lib, "mul_s8", p, a, b), [1, -1, 2, -2])


def test_mul_rejects_invalid_params(lib) -> None:
    good = lib.prepare("mul_prepare", _quant("s8", [0.1, 0.1, 0.1], [0, 0, 0]))
    a = np.zeros(3, np.int8)
    for field, value in (("output_shift", 31), ("output_multiplier", -1), ("input1_offset", 200),
                         ("activation_min", 200), ("activation_max", 300)):
        assert _status(lambda: _run(lib, "mul_s8", {**good, field: value}, a, a)) == "E_PARAM"
    s16 = lib.prepare("mul_prepare", _quant("s16", [1e-4, 1e-4, 1e-4], [0, 0, 0]))
    assert _status(lambda: _run(lib, "mul_s16", {**s16, "output_offset": 1}, *[np.zeros(3, np.int16)] * 2)) == "E_PARAM"
    assert _status(lambda: lib.prepare("mul_prepare", _quant("s16", [0.1] * 3, [0, 1, 0]))) == "E_PARAM"


# --------------------------------------------------------- Maximum / Minimum


@pytest.mark.parametrize("op", ["maximum", "minimum"])
@pytest.mark.parametrize("kind", ["s8", "s16", "f32", "f16"])
def test_min_max_are_numpy(lib, op, kind) -> None:
    rng = np.random.default_rng(5)
    for sa, sb in SHAPES:
        if kind in KINDS:
            a, b = _ints(rng, KINDS[kind], sa), _ints(rng, KINDS[kind], sb)
        else:
            dtype = np.float32 if kind == "f32" else np.float16
            a, b = rng.standard_normal(sa).astype(dtype), rng.standard_normal(sb).astype(dtype)
        want = np.maximum(a, b) if op == "maximum" else np.minimum(a, b)
        np.testing.assert_array_equal(_run(lib, f"{op}_{kind}", {"unused": 0}, a, b), want)


@pytest.mark.parametrize("op", ["maximum", "minimum"])
@pytest.mark.parametrize("kind", ["f32", "f16"])
def test_min_max_follow_ieee_maximum_and_minimum(lib, op, kind) -> None:
    dtype = np.float32 if kind == "f32" else np.float16
    a = np.array([np.nan, 1.0, np.nan, 0.0, -0.0, np.inf, -np.inf, -0.0, 0.0], dtype)
    b = np.array([1.0, np.nan, np.nan, -0.0, 0.0, 1.0, 1.0, -0.0, 0.0], dtype)
    got = _run(lib, f"{op}_{kind}", {"unused": 0}, a, b)
    assert np.isnan(got[:3]).all()  # NaN propagates
    zero_sign = got[3:5].view(np.uint32 if kind == "f32" else np.uint16) >> (31 if kind == "f32" else 15)
    # -0 orders below +0, whichever operand carries it.
    assert zero_sign.tolist() == ([0, 0] if op == "maximum" else [1, 1])
    assert got[5] == (np.inf if op == "maximum" else 1.0) and got[6] == (1.0 if op == "maximum" else -np.inf)
    assert np.signbit(got[7]) and not np.signbit(got[8])


# ------------------------------------------------------- SquaredDifference


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(3))
def test_squared_difference_matches_the_model(lib, kind, seed) -> None:
    rng = np.random.default_rng(20 + seed)
    dtype = KINDS[kind]
    info = np.iinfo(dtype)
    for sa, sb in SHAPES:
        scales, zps = _scales(rng, kind, 3), _zps(rng, kind, 3)
        p = lib.prepare("squared_difference_prepare", _quant(kind, scales, zps))
        assert p["left_shift"] == (7 if kind == "s8" else 0)
        a, b = _ints(rng, dtype, sa), _ints(rng, dtype, sb)
        xs, ys, shape = broadcast_pairs(a, b)
        want = []
        for x, y in zip(xs, ys):
            d = (mbqm((x + p["input1_offset"]) << p["left_shift"], p["input1_multiplier"], p["input1_shift"])
                 - mbqm((y + p["input2_offset"]) << p["left_shift"], p["input2_multiplier"], p["input2_shift"]))
            o = mbqm(d * d, p["output_multiplier"], p["output_shift"]) + p["output_offset"]
            want.append(min(max(o, info.min), info.max))
        np.testing.assert_array_equal(_run(lib, f"squared_difference_{kind}", p, a, b),
                                      np.array(want, dtype).reshape(shape))


def test_squared_difference_rejects_a_wrong_left_shift(lib) -> None:
    p = lib.prepare("squared_difference_prepare", _quant("s8", [0.1, 0.1, 0.1], [0, 0, 0]))
    a = np.zeros(2, np.int8)
    assert _status(lambda: _run(lib, "squared_difference_s8", {**p, "left_shift": 0}, a, a)) == "E_PARAM"
    assert _status(lambda: _run(lib, "squared_difference_s8", {**p, "input1_shift": 1}, a, a)) == "E_PARAM"


# --------------------------------------------------------------- Comparison

OPS = {"EQUAL": np.equal, "NOT_EQUAL": np.not_equal, "GREATER": np.greater, "GREATER_EQUAL": np.greater_equal,
       "LESS": np.less, "LESS_EQUAL": np.less_equal}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("op", OPS)
def test_comparison_matches_the_model(lib, kind, op) -> None:
    rng = np.random.default_rng(30)
    dtype = KINDS[kind]
    for sa, sb in SHAPES:
        s1, s2 = _scales(rng, kind, 2)
        z1, z2 = _zps(rng, kind, 2)
        p = lib.prepare("comparison_prepare", {"dtype": ref_quant.hct_dtype(kind.upper()),
                                               "operation": comparison_code(op), "input1_scale": s1,
                                               "input1_zero_point": z1, "input2_scale": s2, "input2_zero_point": z2})
        a, b = _ints(rng, dtype, sa), _ints(rng, dtype, sb)
        xs, ys, shape = broadcast_pairs(a, b)
        sx = np.array([mbqm((x + p["input1_offset"]) << 8, p["input1_multiplier"], p["input1_shift"]) for x in xs])
        sy = np.array([mbqm((y + p["input2_offset"]) << 8, p["input2_multiplier"], p["input2_shift"]) for y in ys])
        got = _run(lib, f"comparison_{kind}", p, a, b)
        assert got.dtype == np.uint8 and set(np.unique(got)) <= {0, 1}
        np.testing.assert_array_equal(got.astype(bool), OPS[op](sx, sy).reshape(shape))


def test_comparison_rejects_unknown_operations_and_large_scales(lib) -> None:
    base = {"dtype": ref_quant.hct_dtype("S8"), "operation": 0, "input1_scale": 0.1, "input1_zero_point": 0,
            "input2_scale": 0.1, "input2_zero_point": 0}
    assert _status(lambda: lib.prepare("comparison_prepare", {**base, "operation": 6})) == "E_PARAM"
    assert _status(lambda: lib.prepare("comparison_prepare", {**base, "input1_scale": 1.5})) == "E_PARAM"
    p = lib.prepare("comparison_prepare", base)
    a = np.zeros(2, np.int8)
    assert _status(lambda: _run(lib, "comparison_s8", {**p, "operation": -1}, a, a)) == "E_PARAM"
    assert _status(lambda: _run(lib, "comparison_s8", {**p, "left_shift": 7}, a, a)) == "E_PARAM"


# --------------------------------------------------------------------- Abs


@pytest.mark.parametrize("kind", KINDS)
def test_abs_matches_tflite(lib, kind) -> None:
    rng = np.random.default_rng(40)
    dtype = KINDS[kind]
    info = np.iinfo(dtype)
    for same_scale in (True, False):
        si = _scales(rng, kind, 1)[0]
        so = si if same_scale else _scales(rng, kind, 1)[0]
        zi, zo = _zps(rng, kind, 2)
        p = lib.prepare("abs_prepare", {"dtype": ref_quant.hct_dtype(kind.upper()), "input_scale": si,
                                        "input_zero_point": zi, "output_scale": so, "output_zero_point": zo})
        assert p["needs_rescale"] == (0 if same_scale else 1)
        m, s = calculate_multiplier_shift(float(np.float32(si) / np.float32(so)))
        assert (p["multiplier"], p["shift"]) == (m, s)
        x = _ints(rng, dtype, (3, 7))
        values = [abs(v - zi) for v in x.ravel().tolist()]
        want = [min(max((mbqm(v, m, s) if p["needs_rescale"] else v) + zo, info.min), info.max) for v in values]
        np.testing.assert_array_equal(_run(lib, f"abs_{kind}", p, x), np.array(want, dtype).reshape(x.shape))


def test_abs_rescale_at_equal_scales_is_the_identity(lib) -> None:
    p = lib.prepare("abs_prepare", {"dtype": ref_quant.hct_dtype("S8"), "input_scale": 0.125, "input_zero_point": 3,
                                    "output_scale": 0.125, "output_zero_point": -2})
    x = np.arange(-128, 128, dtype=np.int8)
    np.testing.assert_array_equal(_run(lib, "abs_s8", p, x), _run(lib, "abs_s8", {**p, "needs_rescale": 1}, x))


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_float_abs_clears_the_sign(lib, dtype) -> None:
    x = np.array([-1.5, 0.0, -0.0, np.inf, -np.inf, np.nan, -2.0], dtype)
    got = _run(lib, "abs_f32" if dtype == np.float32 else "abs_f16", {"unused": 0}, x)
    assert not np.signbit(got).any()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(x))
    np.testing.assert_array_equal(got[~np.isnan(x)], np.abs(x)[~np.isnan(x)])


def test_abs_rejects_invalid_params(lib) -> None:
    p = lib.prepare("abs_prepare", {"dtype": ref_quant.hct_dtype("S16"), "input_scale": 1e-4, "input_zero_point": 0,
                                    "output_scale": 2e-4, "output_zero_point": 0})
    x = np.zeros(3, np.int16)
    for field, value in (("needs_rescale", 2), ("input_zero_point", 1), ("shift", 31), ("multiplier", -1)):
        assert _status(lambda: _run(lib, "abs_s16", {**p, field: value}, x)) == "E_PARAM"
    assert _status(lambda: lib.prepare("abs_prepare", {"dtype": ref_quant.hct_dtype("S8"), "input_scale": 0.0,
                                                       "input_zero_point": 0, "output_scale": 0.1,
                                                       "output_zero_point": 0})) == "E_PARAM"


# ------------------------------------------------------------------- Clamp


@pytest.mark.parametrize("kind", KINDS)
def test_clamp_is_clip(lib, kind) -> None:
    dtype = KINDS[kind]
    x = _ints(np.random.default_rng(50), dtype, (4, 9))
    np.testing.assert_array_equal(_run(lib, f"clamp_{kind}", {"activation_min": -10, "activation_max": 20}, x),
                                  np.clip(x, -10, 20))
    assert _status(lambda: _run(lib, f"clamp_{kind}", {"activation_min": 5, "activation_max": 4}, x)) == "E_PARAM"
    over = int(np.iinfo(dtype).max) + 1
    assert _status(lambda: _run(lib, f"clamp_{kind}", {"activation_min": 0, "activation_max": over}, x)) == "E_PARAM"


def test_unary_entries_check_the_output_shape(lib) -> None:
    x = np.zeros((2, 3), np.int8)
    assert _status(lambda: lib.run("clamp_s8", {"activation_min": 0, "activation_max": 1}, {"input": x},
                                   {"output": (3, 2)})) == "E_SHAPE"
    assert _status(lambda: lib.run("abs_f32", {"unused": 0}, {"input": np.zeros(4, np.float32)},
                                   {"output": (1, 4)})) == "E_SHAPE"
