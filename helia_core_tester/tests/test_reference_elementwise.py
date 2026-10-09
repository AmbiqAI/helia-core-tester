"""Reference entries behind the Phase 4 operators: binary arithmetic, activations,
reductions, quantize and batch matmul, checked against independent numpy models and
for the rejection codes their argument validation promises."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference import params
from helia_core_tester.tests import tflm_numpy_model as model


@pytest.fixture(scope="module")
def lib() -> b.Bindings:
    return b.get_bindings()


def _code(fn) -> int:
    with pytest.raises(b.ReferenceKernelError) as info:
        fn()
    return info.value.code


def _binary(p: params.AddSubParams, act=(-128, 127)) -> b.HctBinaryParams:
    return b.HctBinaryParams(
        p.left_shift, p.input1_offset, p.input1_multiplier, p.input1_shift, p.input2_offset, p.input2_multiplier,
        p.input2_shift, p.output_offset, p.output_multiplier, p.output_shift, b.make_activation(*act),
    )


# ---- add / sub / mul ----


def _add_model(x1, x2, p: params.AddSubParams, sign=1, lo=-128, hi=127):
    def scaled(x, off, m, s):
        return model.mbqm32((x.astype(np.int64) + off) << p.left_shift, np.int64(m), np.int64(s))

    a = scaled(x1, p.input1_offset, p.input1_multiplier, p.input1_shift)
    c = scaled(x2, p.input2_offset, p.input2_multiplier, p.input2_shift)
    out = model.mbqm32(a + sign * c, np.int64(p.output_multiplier), np.int64(p.output_shift)) + p.output_offset
    return np.clip(out, lo, hi)


@pytest.mark.parametrize("s2", [(2, 3, 4), (1, 3, 1), (4,)])
def test_add_and_sub_s8_broadcast_match_model(lib, s2) -> None:
    rng = np.random.default_rng(3)
    x1 = rng.integers(-128, 128, (2, 3, 4)).astype(np.int8)
    x2 = rng.integers(-128, 128, s2).astype(np.int8)
    p = params.addsub_params("s8", 0.05, 3, 0.11, -7, 0.13, 2)
    out = lib.binary("add", "s8", _binary(p), x1, x2, (2, 3, 4))
    np.testing.assert_array_equal(out, _add_model(x1, np.broadcast_to(x2, x1.shape), p))
    ps = params.sub_params("s8", 0.05, 3, 0.11, -7, 0.13, 2)
    out = lib.binary("sub", "s8", _binary(ps), x1, x2, (2, 3, 4))
    np.testing.assert_array_equal(out, _add_model(x1, np.broadcast_to(x2, x1.shape), ps, sign=-1))


def test_mul_s8_matches_model_and_clamps_to_activation(lib) -> None:
    rng = np.random.default_rng(4)
    x1 = rng.integers(-128, 128, (3, 5)).astype(np.int8)
    x2 = rng.integers(-128, 128, (3, 5)).astype(np.int8)
    m, s = params.mul_params(0.05, 0.07, 0.02)
    p = b.HctBinaryParams(0, 4, 0, 0, -9, 0, 0, 1, m, s, b.make_activation(-20, 30))
    out = lib.binary("mul", "s8", p, x1, x2, (3, 5))
    prod = (x1.astype(np.int64) + 4) * (x2.astype(np.int64) - 9)
    expected = np.clip(model.mbqm32(prod, np.int64(m), np.int64(s)) + 1, -20, 30)
    np.testing.assert_array_equal(out, expected)


def test_binary_rejections(lib) -> None:
    x = np.zeros((2, 3), np.int8)
    p = params.addsub_params("s8", 0.1, 0, 0.1, 0, 0.1, 0)
    assert _code(lambda: lib.binary("add", "s8", _binary(p), x, np.zeros((2, 4), np.int8), (2, 3))) == b.E_DIMS
    assert _code(lambda: lib.binary("add", "s8", _binary(p), x, x, (2, 4))) == b.E_DIMS
    wide = params.AddSubParams(21, 0, p.input1_multiplier, p.input1_shift, 0, p.input2_multiplier, p.input2_shift, 0,
                               p.output_multiplier, p.output_shift)
    assert _code(lambda: lib.binary("add", "s8", _binary(wide), x, x, (2, 3))) == b.E_PARAM
    p16 = params.addsub_params("s16", 0.1, 0, 0.1, 0, 0.1, 0)
    off = params.AddSubParams(*([p16.left_shift, 1] + list(p16.__dict__.values())[2:]))
    x16 = np.zeros((2, 3), np.int16)
    assert _code(lambda: lib.binary("add", "s16", _binary(off, (-32768, 32767)), x16, x16, (2, 3))) == b.E_PARAM
    with pytest.raises(TypeError):
        lib.binary("add", "s8", _binary(p), x.astype(np.int16), x, (2, 3))


# ---- softmax / tanh / logistic ----


def test_softmax_s8_is_a_distribution_over_rows(lib) -> None:
    rng = np.random.default_rng(5)
    x = rng.integers(-128, 128, (4, 10)).astype(np.int8)
    sp = params.softmax_params_s8(1.0 / 16)
    out = lib.unary("softmax_s8", b.HctSoftmaxParams(sp.input_multiplier, sp.input_left_shift, sp.diff_min), x)
    probs = (out.astype(np.int64) + 128) / 256.0
    xf = x.astype(np.float64)
    real = np.exp((xf - xf.max(axis=1, keepdims=True)) / 16.0)
    real /= real.sum(axis=1, keepdims=True)
    assert np.abs(probs - real).max() <= 2 / 256
    tied = lib.unary("softmax_s8", b.HctSoftmaxParams(sp.input_multiplier, sp.input_left_shift, sp.diff_min),
                     np.full((1, 7), 5, np.int8))
    assert np.unique(tied).size == 1


def test_softmax_s16_uses_the_tflm_luts(lib) -> None:
    x = np.linspace(-32768, 32767, 24).astype(np.int16).reshape(2, 12)
    sp = params.softmax_params_s16(1.0 / 4096)
    out = lib.unary("softmax_s16", b.HctSoftmaxParams(sp.input_multiplier, sp.input_left_shift, 0), x)
    xf = x.astype(np.float64)
    real = np.exp((xf - xf.max(axis=1, keepdims=True)) / 4096.0)
    real /= real.sum(axis=1, keepdims=True)
    assert np.abs(out / 32768.0 - real).max() < 2e-3


def test_softmax_rejects_bad_parameters(lib) -> None:
    x = np.zeros((1, 4), np.int8)
    assert _code(lambda: lib.unary("softmax_s8", b.HctSoftmaxParams(0, 1, -10), x)) == b.E_PARAM
    assert _code(lambda: lib.unary("softmax_s8", b.HctSoftmaxParams(1 << 30, 1, 5), x)) == b.E_PARAM


@pytest.mark.parametrize("logistic, fn", [(False, np.tanh), (True, lambda v: 1.0 / (1.0 + np.exp(-v)))])
@pytest.mark.parametrize("scale", [1.0 / 32767, 1.0 / 4096, 2.0 ** -12])
def test_tanh_logistic_s16_track_float(lib, logistic, fn, scale) -> None:
    p = lib.tanh_logistic_s16_prepare(logistic, scale, 1.0 / 32768)
    x = np.linspace(-32768, 32767, 257).astype(np.int16)
    out = lib.unary("logistic_s16" if logistic else "tanh_s16", p, x)
    assert np.abs(out / 32768.0 - fn(x * scale)).max() < 2e-3


def test_tanh_logistic_prepare_rejections(lib) -> None:
    # The output must be exactly 2^-15; scales must be positive and finite.
    assert _code(lambda: lib.tanh_logistic_s16_prepare(False, 1.0 / 4096, 1.0 / 30000)) == b.E_PARAM
    assert _code(lambda: lib.tanh_logistic_s16_prepare(True, 0.0, 1.0 / 32768)) == b.E_PARAM
    assert _code(lambda: lib.tanh_logistic_s16_prepare(True, float("nan"), 1.0 / 32768)) == b.E_PARAM


# ---- leaky relu / relu / prelu / hard swish ----


@pytest.mark.parametrize("dtype, kind, zps", [(np.int8, "s8", (5, -3)), (np.int16, "s16", (0, 0))])
def test_leaky_relu_matches_model(lib, dtype, kind, zps) -> None:
    info = np.iinfo(dtype)
    x = np.linspace(info.min, info.max, 101).astype(dtype)
    p = lib.leaky_relu_prepare(0.02, zps[0], 0.2, 0.015, zps[1])
    out = lib.unary(f"leaky_relu_{kind}", p, x)
    v = x.astype(np.int64) - zps[0]
    alpha = model.mbqm32(v, np.int64(p.multiplier_alpha), np.int64(p.shift_alpha))
    ident = model.mbqm32(v, np.int64(p.multiplier_identity), np.int64(p.shift_identity))
    expected = np.clip(np.where(v >= 0, ident, alpha) + zps[1], info.min, info.max)
    np.testing.assert_array_equal(out, expected)


def test_leaky_relu_s16_rejects_zero_points(lib) -> None:
    p = lib.leaky_relu_prepare(0.02, 1, 0.2, 0.015, 0)
    assert _code(lambda: lib.unary("leaky_relu_s16", p, np.zeros(4, np.int16))) == b.E_PARAM


@pytest.mark.parametrize("act_max", [float("inf"), 6.0])
def test_relu_requantizes_then_clamps(lib, act_max) -> None:
    p = lib.relu_prepare(0.05, -10, 0.03, -128, 0.0, act_max, -128, 127)
    assert p.act_min == -128
    assert p.act_max == (127 if math.isinf(act_max) else min(127, -128 + round(6.0 / np.float32(0.03))))
    x = np.arange(-128, 128).astype(np.int8)
    out = lib.unary("relu_s8", p, x)
    expected = np.clip(model.mbqm32(x.astype(np.int64) + 10, np.int64(p.output_multiplier), np.int64(p.output_shift))
                       - 128, p.act_min, p.act_max)
    np.testing.assert_array_equal(out, expected)


def test_relu_rejections(lib) -> None:
    assert _code(lambda: lib.relu_prepare(0.05, 0, 0.05, 0, 0.0, 6.0, -100, 100)) == b.E_PARAM
    assert _code(lambda: lib.relu_prepare(0.05, 0, 0.05, 0, 1.0, 0.5, -128, 127)) == b.E_PARAM
    p = lib.relu_prepare(0.05, 0, 0.05, 0, 0.0, 6.0, -128, 127)
    p.act_min, p.act_max = 10, -10
    assert _code(lambda: lib.unary("relu_s8", p, np.zeros(3, np.int8))) == b.E_PARAM


def test_prelu_s8_broadcasts_alpha(lib) -> None:
    rng = np.random.default_rng(6)
    x = rng.integers(-128, 128, (1, 2, 3, 4)).astype(np.int8)
    alpha = rng.integers(-128, 128, (4,)).astype(np.int8)
    p = lib.prelu_prepare(0.05, 2, 0.01, -3, 0.04, 1)
    out = lib.prelu_s8(p, x, alpha)
    v = x.astype(np.int64) + p.input_offset
    a = alpha.astype(np.int64) + p.alpha_offset
    pos = model.mbqm32(v, np.int64(p.multiplier_1), np.int64(p.shift_1))
    neg = model.mbqm32(v * a, np.int64(p.multiplier_2), np.int64(p.shift_2))
    np.testing.assert_array_equal(out, np.clip(np.where(v >= 0, pos, neg) + p.output_offset, -128, 127))
    assert _code(lambda: lib.prelu_s8(p, x, np.zeros((3,), np.int8))) == b.E_DIMS


def test_hard_swish_s8_tracks_float(lib) -> None:
    p = lib.hard_swish_prepare(8.0 / 127, 0, 8.375 / 255, -117)
    x = np.arange(-128, 128).astype(np.int8)
    out = lib.unary("hard_swish_s8", p, x)
    real = x * (8.0 / 127)
    real = real * np.clip(real + 3, 0, 6) / 6
    assert np.abs((out.astype(np.int64) + 117) * (8.375 / 255) - real).max() <= 2 * (8.375 / 255)
    assert _code(lambda: lib.hard_swish_prepare(0.1, 200, 0.1, 0)) == b.E_PARAM


# ---- rsqrt / quantize ----


def test_rsqrt_s8_and_s16(lib) -> None:
    p = lib.rsqrt_prepare(1.0 / 64, -20, 1.0 / 32, -128)
    x = np.arange(-20, 128).astype(np.int8)
    out = lib.unary("rsqrt_s8", p, x)
    assert out[0] == 127  # input at the zero point saturates
    real = 1.0 / np.sqrt((x[1:].astype(float) + 20) / 64)
    assert np.abs((out[1:].astype(float) + 128) / 32 - real).max() <= 2 / 32
    assert _code(lambda: lib.unary("rsqrt_s8", p, np.array([-21], np.int8))) == b.E_PARAM
    p16 = lib.rsqrt_prepare(1.0 / 512, 0, 1.0 / 32768, 0)
    x16 = np.array([512, 2048, 8192, 32767], np.int16)
    out16 = lib.rsqrt_s16(p16, 1.0 / 512, 1.0 / 32768, x16)
    np.testing.assert_allclose(out16 / 32768.0, 1.0 / np.sqrt(x16 / 512.0), atol=2e-3)
    assert _code(lambda: lib.rsqrt_s16(p16, 1.0 / 512, 1.0 / 32768, np.array([-1], np.int16))) == b.E_PARAM


def test_quantize_rounds_half_away_and_saturates(lib) -> None:
    x = np.array([0.5, -0.5, 2.5, -2.5, 1000.0, -1000.0, 0.49], np.float32)
    np.testing.assert_array_equal(lib.quantize_f32("s8", 1.0, 0, x), [1, -1, 3, -3, 127, -128, 0])
    np.testing.assert_array_equal(lib.quantize_f32("s16", 0.5, 3, x[:4]), [4, 2, 8, -2])
    assert _code(lambda: lib.quantize_f32("s8", 1.0, 0, np.array([np.nan], np.float32))) == b.E_PARAM
    assert _code(lambda: lib.quantize_f32("s8", 0.0, 0, x)) == b.E_PARAM
    assert _code(lambda: lib.quantize_f32("s8", 1.0, 200, x)) == b.E_PARAM


# ---- mean ----


def test_mean_subtracts_the_input_zero_point(lib) -> None:
    rng = np.random.default_rng(7)
    x = rng.integers(-128, 128, (1, 5, 7, 3)).astype(np.int8)
    m, s = params.mean_params(0.05, 0.004)
    p = b.HctMeanParams(-64, 37, m, s, 1)
    out = lib.mean("s8", p, x, [1, 2], (1, 1, 1, 3))
    real = ((x.astype(float) + 64) * 0.05).mean(axis=(1, 2))
    assert np.abs(out.ravel() - np.clip(np.round(real / 0.004) + 37, -128, 127).ravel()).max() <= 1
    dropped = lib.mean("s8", b.HctMeanParams(-64, 37, m, s, 0), x, [1, 2], (1, 3))
    np.testing.assert_array_equal(dropped.ravel(), out.ravel())


def test_mean_fold_and_rejections(lib) -> None:
    m, s = lib.mean_fold(1 << 30, 0, 6)
    assert (m, s) == (((1 << 30) << 2) // 6, -2)
    assert _code(lambda: lib.mean_fold(1 << 30, 0, 0)) == b.E_PARAM
    x = np.zeros((1, 2, 2, 3), np.int8)
    p = b.HctMeanParams(0, 0, 1 << 30, 0, 1)
    assert _code(lambda: lib.mean("s8", p, x, [1, 2], (1, 2, 1, 3))) == b.E_DIMS
    assert _code(lambda: lib.mean("s8", p, x, [4], (1, 2, 2, 3))) == b.E_PARAM
    assert _code(lambda: lib.mean("s16", b.HctMeanParams(1, 0, 1 << 30, 0, 1), x.astype(np.int16), [1], (1, 1, 2, 3))) == b.E_PARAM


# ---- batch matmul ----


@pytest.mark.parametrize("kind, dtype, offsets", [("s8", np.int8, (3, -2, 5)), ("s16", np.int16, (0, 0, 0))])
def test_bmm_matches_einsum_with_batch_broadcast(lib, kind, dtype, offsets) -> None:
    rng = np.random.default_rng(8)
    info = np.iinfo(dtype)
    lo, hi = (-30, 30) if kind == "s8" else (-3000, 3000)
    lhs = rng.integers(lo, hi, (2, 3, 5)).astype(dtype)
    rhs = rng.integers(lo, hi, (1, 4, 5)).astype(dtype)
    m, s = params.quantize_multiplier(1.0 / (64 if kind == "s8" else 65536))
    p = b.HctBmmParams(offsets[0], offsets[1], offsets[2], m, s, b.make_activation(info.min, info.max))
    out = lib.bmm(kind, p, lhs, rhs, (2, 3, 4))
    acc = np.einsum("bmk,bnk->bmn", lhs.astype(np.int64) + offsets[0], np.broadcast_to(rhs, (2, 4, 5)).astype(np.int64) + offsets[1])
    mult = model.mbqm64 if kind == "s16" else model.mbqm32
    expected = np.clip(mult(acc, np.int64(m), np.int64(s)) + offsets[2], info.min, info.max)
    np.testing.assert_array_equal(out, expected)


def test_bmm_rejections(lib) -> None:
    lhs = np.zeros((1, 3, 5), np.int8)
    p = b.HctBmmParams(0, 0, 0, 1 << 30, 0, b.make_activation(-128, 127))
    assert _code(lambda: lib.bmm("s8", p, lhs, np.zeros((1, 4, 6), np.int8), (1, 3, 4))) == b.E_DIMS
    assert _code(lambda: lib.bmm("s8", p, lhs, np.zeros((2, 4, 5), np.int8), (1, 3, 4))) == b.E_DIMS
    p16 = b.HctBmmParams(1, 0, 0, 1 << 30, 0, b.make_activation(-32768, 32767))
    assert _code(lambda: lib.bmm("s16", p16, lhs.astype(np.int16), np.zeros((1, 4, 5), np.int16), (1, 3, 4))) == b.E_PARAM


def test_flat_struct_rejects_missing_and_unknown_fields() -> None:
    from helia_core_tester.generation.reference.run import flat_struct

    with pytest.raises(KeyError, match="missing"):
        flat_struct(b.HctReluParams, {"input_zero_point": 0})
    full = {name: 0 for name, _ in b.HctReluParams._fields_}
    with pytest.raises(KeyError, match="unknown"):
        flat_struct(b.HctReluParams, {**full, "extra": 1})


def test_bmm_f32_matches_matmul_with_broadcast_and_clamp(lib) -> None:
    rng = np.random.default_rng(9)
    for ls, rs in (((2, 3, 5), (1, 4, 5)), ((1, 2, 3, 5), (2, 1, 4, 5))):
        lhs = rng.standard_normal(ls).astype(np.float32)
        rhs = rng.standard_normal(rs).astype(np.float32)
        out_shape = np.broadcast_shapes(ls[:-2], rs[:-2]) + (ls[-2], rs[-2])
        expected = np.matmul(lhs.astype(np.float64), np.swapaxes(rhs, -1, -2).astype(np.float64))
        free = b.HctBmmParams(0, 0, 0, 0, 0, b.make_activation(0, 0, -math.inf, math.inf))
        np.testing.assert_allclose(lib.bmm("f32", free, lhs, rhs, out_shape), expected, rtol=1e-5, atol=1e-5)
        clamped = b.HctBmmParams(0, 0, 0, 0, 0, b.make_activation(0, 0, -0.5, 0.25))
        np.testing.assert_allclose(lib.bmm("f32", clamped, lhs, rhs, out_shape), np.clip(expected, -0.5, 0.25),
                                   rtol=1e-5, atol=1e-5)


def test_bmm_f32_propagates_nonfinite_when_unbounded(lib) -> None:
    lhs = np.ones((1, 2, 3), np.float32)
    lhs[0, 1, 0] = np.inf
    rhs = np.ones((1, 2, 3), np.float32)
    free = b.HctBmmParams(0, 0, 0, 0, 0, b.make_activation(0, 0, -math.inf, math.inf))
    out = lib.bmm("f32", free, lhs, rhs, (1, 2, 2))
    np.testing.assert_array_equal(out[0, 0], [3.0, 3.0])
    assert np.isposinf(out[0, 1]).all()


def test_bmm_f32_rejections(lib) -> None:
    lhs, rhs = np.zeros((1, 3, 5), np.float32), np.zeros((1, 4, 5), np.float32)
    act = b.make_activation(0, 0, -1.0, 1.0)
    for offsets in ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1 << 30, 0), (0, 0, 0, 0, 1)):
        assert _code(lambda: lib.bmm("f32", b.HctBmmParams(*offsets, act), lhs, rhs, (1, 3, 4))) == b.E_PARAM
    ok = b.HctBmmParams(0, 0, 0, 0, 0, act)
    for fmin, fmax in ((1.0, -1.0), (math.nan, 1.0)):
        bad = b.HctBmmParams(0, 0, 0, 0, 0, b.make_activation(0, 0, fmin, fmax))
        assert _code(lambda: lib.bmm("f32", bad, lhs, rhs, (1, 3, 4))) == b.E_PARAM
    assert _code(lambda: lib.bmm("f32", ok, lhs, np.zeros((1, 4, 6), np.float32), (1, 3, 4))) == b.E_DIMS
    assert _code(lambda: lib.bmm("f32", ok, lhs, rhs, (1, 4, 3))) == b.E_DIMS
    with pytest.raises(TypeError):
        lib.bmm("f32", ok, lhs.astype(np.float16), rhs, (1, 3, 4))
    with pytest.raises(ValueError, match="unknown bmm kind"):
        lib.bmm("f16", ok, lhs, rhs, (1, 3, 4))
