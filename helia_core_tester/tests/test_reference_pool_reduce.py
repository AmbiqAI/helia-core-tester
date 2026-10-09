"""Pooling, Mean, ReduceSum and BatchNorm on the C reference, against independent models."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift
from helia_core_tester.tests.reference_models import mbqm

KINDS = {"s8": np.int8, "s16": np.int16}


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _pool_params(stride, filt, pad, act):
    return {"stride_h": stride[0], "stride_w": stride[1], "filter_h": filt[0], "filter_w": filt[1],
            "pad_h": pad[0], "pad_w": pad[1], "activation_min": act[0], "activation_max": act[1]}


def _windows(x, p, out_hw):
    n, h, w, c = x.shape
    for b in range(n):
        for oy in range(out_hw[0]):
            for ox in range(out_hw[1]):
                y0, x0 = oy * p["stride_h"] - p["pad_h"], ox * p["stride_w"] - p["pad_w"]
                ys = range(max(y0, 0), min(y0 + p["filter_h"], h))
                xs = range(max(x0, 0), min(x0 + p["filter_w"], w))
                yield (b, oy, ox), x[b][np.ix_(list(ys), list(xs))].reshape(-1, c)


def _round_div_away(total: int, count: int) -> int:
    q = (abs(total) + count // 2) // count
    return q if total >= 0 else -q


# ---------------------------------------------------------------- pooling


@pytest.mark.parametrize("kind", ["s8", "s16"])
@pytest.mark.parametrize("stride,filt,pad,out_hw", [((1, 1), (2, 2), (0, 0), (4, 5)), ((2, 2), (3, 3), (1, 1), (3, 3)),
                                                    ((2, 1), (3, 2), (1, 0), (3, 4))])
def test_int_pooling_matches_the_model(lib, kind, stride, filt, pad, out_hw) -> None:
    rng = np.random.default_rng([stride[0], filt[1], len(kind)])
    info = np.iinfo(KINDS[kind])
    x = rng.integers(info.min, info.max + 1, (2, 5, 6, 3)).astype(KINDS[kind])
    p = _pool_params(stride, filt, pad, (int(info.min) + 10, int(info.max) - 10))
    out = (2, *out_hw, 3)
    avg = lib.run(f"avg_pool_{kind}", p, {"input": x}, {"output": out})["output"]
    mx = lib.run(f"max_pool_{kind}", p, {"input": x}, {"output": out})["output"]
    lo, hi = p["activation_min"], p["activation_max"]
    for (b, oy, ox), win in _windows(x, p, out_hw):
        for c in range(3):
            taps = [int(v) for v in win[:, c]]
            assert avg[b, oy, ox, c] == min(max(_round_div_away(sum(taps), len(taps)), lo), hi)
            assert mx[b, oy, ox, c] == min(max(max(taps), lo), hi)


def test_int_average_ties_round_away_from_zero(lib) -> None:
    x = np.array([1, 2, -1, -2], np.int8).reshape(1, 2, 2, 1)
    p = _pool_params((1, 1), (1, 2), (0, 0), (-128, 127))
    got = lib.run("avg_pool_s8", p, {"input": x}, {"output": (1, 2, 1, 1)})["output"]
    assert got.ravel().tolist() == [2, -2]


def test_float_pooling_rounds_the_exact_mean_once_and_max_skips_nan(lib) -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, (1, 4, 4, 2)).astype(np.float32)
    x[0, 0, 1, 0] = np.nan
    p = _pool_params((2, 2), (2, 2), (0, 0), (-2.0, 2.0))
    avg = lib.run("avg_pool_f32", p, {"input": x}, {"output": (1, 2, 2, 2)})["output"]
    mx = lib.run("max_pool_f32", p, {"input": x}, {"output": (1, 2, 2, 2)})["output"]
    for (b, oy, ox), win in _windows(x, p, (2, 2)):
        for c in range(2):
            taps = win[:, c].astype(np.float64)
            mean = np.float32(math.fsum(taps.tolist()) / len(taps)) if not np.isnan(taps).any() else np.nan
            if np.isnan(mean):
                assert np.isnan(avg[b, oy, ox, c])
            else:
                assert avg[b, oy, ox, c] == np.clip(mean, -2, 2)
            assert mx[b, oy, ox, c] == np.clip(np.float32(np.nanmax(taps)), -2, 2)


def test_pooling_rejects_invalid_windows_and_shapes(lib) -> None:
    x = np.zeros((1, 3, 3, 2), np.int8)
    ok = _pool_params((1, 1), (2, 2), (0, 0), (-128, 127))
    run = lambda p, out=(1, 2, 2, 2): lib.run("avg_pool_s8", p, {"input": x}, {"output": out})
    run(ok)
    assert _status(lambda: run(dict(ok, stride_h=0))) == "E_PARAM"
    assert _status(lambda: run(dict(ok, filter_w=0))) == "E_PARAM"
    assert _status(lambda: run(dict(ok, pad_h=-1))) == "E_PARAM"
    assert _status(lambda: run(dict(ok, activation_min=1, activation_max=0))) == "E_PARAM"
    assert _status(lambda: run(ok, (1, 2, 2, 3))) == "E_SHAPE"
    # A window entirely in the padding has no taps: TFLite's AveragePool fails on it.
    assert _status(lambda: run(dict(ok, filter_h=1, filter_w=1, pad_h=2), (1, 2, 2, 2))) == "E_PARAM"


# ------------------------------------------------------------------- mean


def mean_prepare_model(in_scale: float, out_scale: float, count: int):
    m, s = calculate_multiplier_shift(float(np.float32(in_scale)) / float(np.float32(out_scale)))
    fold = min(count.bit_length() - 1, 32, 31 + s)
    return int(m * 2**fold // count), s - fold


@pytest.mark.parametrize("count", [1, 2, 3, 7, 64, 1000, 2**20 + 3])
@pytest.mark.parametrize("ratio", [0.37, 1.0, 3.5])
def test_mean_prepare_folds_the_count_as_tflm(lib, count, ratio) -> None:
    out = lib.prepare("mean_prepare", {"input_scale": 0.05, "output_scale": float(np.float32(0.05 / ratio)),
                                       "count": count})
    assert (out["multiplier"], out["shift"]) == mean_prepare_model(0.05, float(np.float32(0.05 / ratio)), count)


@pytest.mark.parametrize("fields", [{"count": 0}, {"input_scale": 0.0}, {"output_scale": -1.0},
                                    {"input_scale": float("nan")}])
def test_mean_prepare_rejects_invalid_quantization(lib, fields) -> None:
    base = {"input_scale": 0.05, "output_scale": 0.05, "count": 4}
    assert _status(lambda: lib.prepare("mean_prepare", dict(base, **fields))) == "E_PARAM"


@pytest.mark.parametrize("kind", ["s8", "s16"])
@pytest.mark.parametrize("axis_mask,out_shape", [(0b0110, (2, 1, 1, 3)), (0b1000, (2, 4, 5, 1)), (0b1111, (1, 1, 1, 1))])
def test_mean_matches_quantized_mean_or_sum(lib, kind, axis_mask, out_shape) -> None:
    rng = np.random.default_rng([axis_mask, len(kind)])
    info = np.iinfo(KINDS[kind])
    x = rng.integers(info.min, info.max + 1, (2, 4, 5, 3)).astype(KINDS[kind])
    axes = tuple(d for d in range(4) if axis_mask >> d & 1)
    count = int(np.prod([x.shape[d] for d in axes]))
    in_zp, out_zp = (-5, 7) if kind == "s8" else (0, 0)
    in_scale, out_scale = 0.04, 0.03
    m, s = mean_prepare_model(in_scale, out_scale, count)
    p = {"axis_mask": axis_mask, "input_zero_point": in_zp, "output_zero_point": out_zp, "multiplier": m, "shift": s}
    got = lib.run(f"mean_{kind}", p, {"input": x}, {"output": out_shape})["output"]
    sums = x.astype(np.int64).sum(axis=axes, keepdims=True) - in_zp * count
    want = np.vectorize(lambda v: min(max(mbqm(int(v), m, s) + out_zp, int(info.min)), int(info.max)))(sums)
    np.testing.assert_array_equal(got, want.reshape(out_shape))


def test_mean_rejects_invalid_params_and_shapes(lib) -> None:
    x = np.zeros((2, 3), np.int8)
    ok = {"axis_mask": 0b10, "input_zero_point": 0, "output_zero_point": 0, "multiplier": 1 << 30, "shift": -1}
    run = lambda p, out=(2, 1): lib.run("mean_s8", p, {"input": x}, {"output": out})
    run(ok)
    assert _status(lambda: run(dict(ok, input_zero_point=128))) == "E_PARAM"
    assert _status(lambda: run(dict(ok, multiplier=-1))) == "E_PARAM"
    assert _status(lambda: run(dict(ok, shift=31))) == "E_PARAM"
    assert _status(lambda: run(dict(ok, axis_mask=0b100))) == "E_PARAM"
    assert _status(lambda: run(ok, (3, 1))) == "E_SHAPE"


@pytest.mark.parametrize("dtype,entry", [(np.float32, "f32"), (np.float16, "f16")])
def test_float_mean_and_sum_round_the_exact_value_once(lib, dtype, entry) -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-10, 10, (3, 17)).astype(dtype)
    exact = [math.fsum(r.astype(np.float64).tolist()) for r in x]
    s = lib.run(f"reduce_sum_{entry}", {"axis_mask": 0b10}, {"input": x}, {"output": (3, 1)})["output"].ravel()
    m = lib.run(f"mean_{entry}", {"axis_mask": 0b10}, {"input": x}, {"output": (3, 1)})["output"].ravel()
    np.testing.assert_array_equal(s, np.array(exact, np.float64).astype(dtype))
    np.testing.assert_array_equal(m, (np.array(exact, np.float64) / 17).astype(dtype))


# -------------------------------------------------------------- BatchNorm


@pytest.mark.parametrize("dtype,entry", [(np.float32, "f32"), (np.float16, "f16")])
def test_batch_norm_is_one_fused_multiply_add_per_element(lib, dtype, entry) -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(-4, 4, (2, 3, 3, 5)).astype(dtype)
    scale = rng.uniform(0.25, 2, 5).astype(dtype)
    bias = rng.uniform(-1, 1, 5).astype(dtype)
    got = lib.run(f"batch_norm_{entry}", {"unused": 0}, {"input": x, "scale": scale, "bias": bias},
                  {"output": x.shape})["output"]
    flat = [Fraction(float(v)) * Fraction(float(scale[i % 5])) + Fraction(float(bias[i % 5]))
            for i, v in enumerate(x.ravel())]
    np.testing.assert_array_equal(got.ravel(), np.array([round_exact(f, dtype) for f in flat], dtype))


def round_exact(f: Fraction, dtype) -> float:
    """The nearest `dtype` value to the rational f, ties to even."""
    c = dtype(float(f))
    cands = [c, np.nextafter(c, dtype(np.inf)), np.nextafter(c, dtype(-np.inf))]
    def key(v):
        bits = int(np.array(v, dtype).view(np.uint16 if dtype == np.float16 else np.uint32))
        return abs(Fraction(float(v)) - f), bits & 1
    return min(cands, key=key)


def test_batch_norm_keeps_non_finite_values(lib) -> None:
    x = np.array([np.inf, -np.inf, np.nan, 1.0], np.float32).reshape(1, 1, 2, 2)
    got = lib.run("batch_norm_f32", {"unused": 0}, {"input": x, "scale": np.array([2.0, 0.0], np.float32),
                                                     "bias": np.array([1.0, 0.0], np.float32)},
                  {"output": x.shape})["output"].ravel()
    assert got[0] == np.inf and np.isnan(got[1]) and np.isnan(got[2]) and got[3] == 0.0


def test_batch_norm_rejects_mismatched_channels(lib) -> None:
    x = np.zeros((1, 2, 2, 3), np.float32)
    run = lambda s, b, out=(1, 2, 2, 3): lib.run("batch_norm_f32", {"unused": 0},
                                                 {"input": x, "scale": s, "bias": b}, {"output": out})
    c3 = np.ones(3, np.float32)
    run(c3, c3)
    assert _status(lambda: run(np.ones(2, np.float32), c3)) == "E_SHAPE"
    assert _status(lambda: run(c3, np.ones(4, np.float32))) == "E_SHAPE"
    assert _status(lambda: run(c3.reshape(1, 3), c3)) == "E_SHAPE"
    assert _status(lambda: run(c3, c3, (1, 2, 3, 2))) == "E_SHAPE"


def test_float_sums_survive_cancellation_and_ignore_order(lib) -> None:
    x = np.array([[1e30, 1.0, -1e30, 2.0**-30]], np.float32)
    got = lib.run("reduce_sum_f32", {"axis_mask": 0b10}, {"input": x}, {"output": (1, 1)})["output"]
    assert got[0, 0] == np.float32(1.0 + 2.0**-30)
    rng = np.random.default_rng(6)
    terms = (rng.uniform(-1, 1, 64) * 2.0 ** rng.integers(-40, 40, 64)).astype(np.float32)
    outs = {float(lib.run("reduce_sum_f32", {"axis_mask": 0b10}, {"input": rng.permutation(terms).reshape(1, -1)},
                          {"output": (1, 1)})["output"][0, 0]) for _ in range(8)}
    assert outs == {float(round_exact(sum(Fraction(float(t)) for t in terms), np.float32))}
    fc = lib.run("fully_connected_f32", {"activation_min": -np.inf, "activation_max": np.inf},
                 {"input": np.array([[1e20, 1.0, -1e20]], np.float32), "filter": np.ones((1, 3), np.float32),
                  "bias": np.zeros(1, np.float32)}, {"output": (1, 1)})["output"]
    assert fc[0, 0] == 1.0
