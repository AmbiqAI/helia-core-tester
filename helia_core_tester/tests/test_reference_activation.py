"""Relu, Relu6, LeakyRelu, PReLU and HardSwish on the C reference, against independent models.

Integer entries are checked bit-exactly against Python models of the TFLite (or, for
hard_swish_precise, CMSIS-NN) arithmetic and within one output step of the float64
math on the dequantized input. Float entries are checked against numpy in float64.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference.abi import dtype_code
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.tests.reference_models import broadcast_pairs, mbqm, rdbpot

KINDS = ["s8", "s16"]
_NP = {"s8": np.int8, "s16": np.int16}


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _info(kind):
    return np.iinfo(_NP[kind])


def _all_codes(kind, n=4096):
    info = _info(kind)
    if kind == "s8":
        return np.arange(info.min, info.max + 1, dtype=np.int8)
    return np.random.default_rng(0).integers(info.min, info.max + 1, n).astype(np.int16)


def _quant(rng, kind):
    s_in, s_out = (float(np.float32(x)) for x in rng.uniform(0.005, 0.05, 2) / (1 if kind == "s8" else 256))
    z_in, z_out = (int(z) for z in rng.integers(-30, 31, 2)) if kind == "s8" else (0, 0)
    return s_in, z_in, s_out, z_out


def _unary(lib, entry, params, x):
    return lib.run(entry, params, {"input": x}, {"output": x.shape})["output"]


# ------------------------------------------------------------------ Relu, Relu6


def _relu_prepare(lib, kind, s_in, z_in, s_out, z_out, act_max):
    return lib.prepare("relu_prepare", {
        "dtype": dtype_code("int8" if kind == "s8" else "int16"), "input_scale": s_in, "input_zero_point": z_in,
        "output_scale": s_out, "output_zero_point": z_out, "act_max": act_max})


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("act_max", [math.inf, 6.0])
@pytest.mark.parametrize("seed", range(3))
def test_relu_matches_the_model(lib, kind, act_max, seed) -> None:
    rng = np.random.default_rng(seed)
    s_in, z_in, s_out, z_out = _quant(rng, kind)
    p = _relu_prepare(lib, kind, s_in, z_in, s_out, z_out, act_max)
    info = _info(kind)
    assert p["activation_min"] == max(z_out, info.min)
    if math.isinf(act_max):
        assert p["activation_max"] == info.max
    else:
        assert p["activation_max"] == min(info.max, z_out + round(np.float32(act_max) / np.float32(s_out)))
    x = _all_codes(kind)
    got = _unary(lib, f"relu_{kind}", p, x)
    want = [min(max(p["output_offset"] + mbqm(int(v) - p["input_offset"], p["output_multiplier"], p["output_shift"]),
                    p["activation_min"]), p["activation_max"]) for v in x.tolist()]
    np.testing.assert_array_equal(got, want)
    real = np.clip((x.astype(np.float64) - z_in) * s_in, 0.0, act_max)
    ideal = np.clip(np.round(real / s_out) + z_out, info.min, info.max)
    assert np.abs(got.astype(np.float64) - ideal).max() <= 1


@pytest.mark.parametrize(
    "overrides",
    [{"input_scale": 0.0}, {"output_scale": math.nan}, {"input_zero_point": 128}, {"act_max": -1.0},
     {"act_max": math.nan}, {"dtype": dtype_code("float32")}],
)
def test_relu_prepare_rejects_invalid_quantization(lib, overrides) -> None:
    fields = {"dtype": dtype_code("int8"), "input_scale": 0.1, "input_zero_point": 0, "output_scale": 0.1,
              "output_zero_point": 0, "act_max": 6.0, **overrides}
    with pytest.raises(ReferenceKernelError, match="E_PARAM|E_DTYPE"):
        lib.prepare("relu_prepare", fields)


def test_relu_s16_prepare_rejects_zero_points(lib) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _relu_prepare(lib, "s16", 1e-4, 1, 1e-4, 0, 6.0)


@pytest.mark.parametrize("field, value", [("activation_min", 100), ("activation_max", 200), ("output_shift", 31),
                                          ("output_multiplier", -1), ("input_offset", 128)])
def test_relu_kernel_rejects_invalid_params(lib, field, value) -> None:
    p = {**_relu_prepare(lib, "s8", 0.1, 0, 0.1, 0, 6.0), field: value}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _unary(lib, "relu_s8", p, np.zeros(4, np.int8))


def test_relu6_upper_bound_saturates_at_the_dtype_limit(lib) -> None:
    # 6 / 0.01 = 600 codes above the zero point does not fit int8.
    p = _relu_prepare(lib, "s8", 0.01, 0, 0.01, 0, 6.0)
    assert p["activation_max"] == 127


# --------------------------------------------------------------------- LeakyRelu


def _leaky_prepare(lib, kind, alpha, s_in, z_in, s_out, z_out):
    return lib.prepare("leaky_relu_prepare", {
        "dtype": dtype_code("int8" if kind == "s8" else "int16"), "alpha": alpha, "input_scale": s_in,
        "input_zero_point": z_in, "output_scale": s_out, "output_zero_point": z_out})


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.2, 1.5])
def test_leaky_relu_matches_the_model(lib, kind, alpha) -> None:
    rng = np.random.default_rng(int(alpha * 100))
    s_in, z_in, s_out, z_out = _quant(rng, kind)
    p = _leaky_prepare(lib, kind, alpha, s_in, z_in, s_out, z_out)
    info = _info(kind)
    x = _all_codes(kind)
    got = _unary(lib, f"leaky_relu_{kind}", p, x)
    want = []
    for v in x.tolist():
        d = v - p["input_offset"]
        m, s = (p["identity_multiplier"], p["identity_shift"]) if d >= 0 else (p["alpha_multiplier"], p["alpha_shift"])
        want.append(min(max(p["output_offset"] + mbqm(d, m, s), info.min), info.max))
    np.testing.assert_array_equal(got, want)
    real = (x.astype(np.float64) - z_in) * s_in
    real = np.where(real >= 0, real, real * alpha)
    ideal = np.clip(np.round(real / s_out) + z_out, info.min, info.max)
    assert np.abs(got.astype(np.float64) - ideal).max() <= 1


@pytest.mark.parametrize("alpha", [-0.1, math.inf, math.nan])
def test_leaky_relu_prepare_rejects_alpha(lib, alpha) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _leaky_prepare(lib, "s8", alpha, 0.1, 0, 0.1, 0)


@pytest.mark.parametrize("field, value", [("alpha_shift", 31), ("identity_multiplier", -1), ("output_offset", -129)])
def test_leaky_relu_kernel_rejects_invalid_params(lib, field, value) -> None:
    p = {**_leaky_prepare(lib, "s8", 0.1, 0.1, 0, 0.1, 0), field: value}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _unary(lib, "leaky_relu_s8", p, np.zeros(4, np.int8))


# ------------------------------------------------------------------------- PReLU


def _prelu_prepare(lib, kind, s_in, z_in, s_a, z_a, s_out, z_out):
    return lib.prepare("prelu_prepare", {
        "dtype": dtype_code("int8" if kind == "s8" else "int16"), "input_scale": s_in, "input_zero_point": z_in,
        "alpha_scale": s_a, "alpha_zero_point": z_a, "output_scale": s_out, "output_zero_point": z_out})


def _prelu(lib, entry, params, x, a):
    shape = np.broadcast_shapes(x.shape, a.shape)
    return lib.run(entry, params, {"input": x, "alpha": a}, {"output": shape})["output"]


PRELU_SHAPES = [((2, 3, 4, 5), (1, 1, 1, 5)), ((2, 3, 4, 5), (2, 3, 4, 5)), ((1, 3, 1, 5), (2, 1, 4, 1)),
                ((7,), ())]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shapes", PRELU_SHAPES)
def test_prelu_matches_the_model(lib, kind, shapes) -> None:
    rng = np.random.default_rng(len(shapes[0]))
    info = _info(kind)
    s_in, z_in, s_out, z_out = _quant(rng, kind)
    s_a, z_a = (0.004, int(rng.integers(-20, 21))) if kind == "s8" else (1 / 32768, 0)
    p = _prelu_prepare(lib, kind, s_in, z_in, s_a, z_a, s_out, z_out)
    x = rng.integers(info.min, info.max + 1, shapes[0]).astype(_NP[kind])
    a = rng.integers(info.min, info.max + 1, shapes[1]).astype(_NP[kind])
    got = _prelu(lib, f"prelu_{kind}", p, x, a)
    xs, as_, shape = broadcast_pairs(x, a)
    want = []
    for xv, av in zip(xs, as_):
        d = xv + p["input_offset"]
        if d >= 0:
            v = mbqm(d, p["identity_multiplier"], p["identity_shift"])
        else:
            v = mbqm(d * (av + p["alpha_offset"]), p["alpha_multiplier"], p["alpha_shift"])
        want.append(min(max(v + p["output_offset"], info.min), info.max))
    np.testing.assert_array_equal(got, np.array(want).reshape(shape))
    real_x = (np.broadcast_to(x, shape).astype(np.float64) - z_in) * s_in
    real_a = (np.broadcast_to(a, shape).astype(np.float64) - z_a) * s_a
    ideal = np.clip(np.round(np.where(real_x >= 0, real_x, real_x * real_a) / s_out) + z_out, info.min, info.max)
    assert np.abs(got.astype(np.float64) - ideal).max() <= 1


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_float_prelu_is_numpy(lib, dtype) -> None:
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((2, 3, 4, 5)) * 100).astype(dtype)
    a = rng.standard_normal((1, 1, 1, 5)).astype(dtype)
    entry = "prelu_f16" if dtype == np.float16 else "prelu_f32"
    got = _prelu(lib, entry, {"unused": 0}, x, a)
    want = np.where(x >= 0, x, (x.astype(np.float32) * a.astype(np.float32))).astype(dtype)
    np.testing.assert_array_equal(got, want)


def test_prelu_rejects_incompatible_shapes(lib) -> None:
    p = _prelu_prepare(lib, "s8", 0.1, 0, 0.1, 0, 0.1, 0)
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        lib.run("prelu_s8", p, {"input": np.zeros((2, 3), np.int8), "alpha": np.zeros((4,), np.int8)},
                {"output": (2, 3)})


@pytest.mark.parametrize("field, value", [("input_offset", -128), ("alpha_offset", 129), ("alpha_shift", 31)])
def test_prelu_kernel_rejects_invalid_params(lib, field, value) -> None:
    p = {**_prelu_prepare(lib, "s8", 0.1, 0, 0.1, 0, 0.1, 0), field: value}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _prelu(lib, "prelu_s8", p, np.zeros(4, np.int8), np.zeros(4, np.int8))


# --------------------------------------------------------------------- HardSwish


def _hs_prepare(lib, entry, kind, s_in, z_in, s_out, z_out):
    return lib.prepare(entry, {"dtype": dtype_code("int8" if kind == "s8" else "int16"), "input_scale": s_in,
                               "input_zero_point": z_in, "output_scale": s_out, "output_zero_point": z_out})


def _sat16(v: int) -> int:
    return min(max(v, -32768), 32767)


def _srdhm16(a: int, b: int) -> int:
    if a == b == -32768:
        return 32767
    return (a * b + 2**14) >> 15


def _sdhm16(a: int, b: int) -> int:
    if a == b == -32768:
        return 32767
    q = abs(a * b) >> 15
    return q if a * b >= 0 else -q


def _model_hard_swish_compat(x: int, p: dict) -> int:
    """TFLite reference_ops::HardSwish<int8_t>."""
    hires = (x - p["input_zero_point"]) * 128
    preshift = _srdhm16(hires, p["output_multiplier_fixedpoint_int16"])
    e = p["reluish_multiplier_exponent"]
    r = hires
    if e > 0:
        r = _sat16(r << (e - 1))
    r = _srdhm16(r, p["reluish_multiplier_fixedpoint_int16"])
    if e > 0:
        r = _sat16(r << 1)
    if e < 0:
        r = rdbpot(r, -e)
    r = (r + 32768) >> 1
    y = rdbpot(_sdhm16(r, preshift), -p["output_multiplier_exponent"]) + p["output_zero_point"]
    return min(max(y, -128), 127)


def _model_hard_swish_precise(x: int, p: dict, kind: str) -> int:
    """CMSIS-NN arm_hard_swish_precise_*: arm_nn_requantize without single rounding."""
    info = _info(kind)
    d = x - p["input_offset"]
    xr = min(max(d + p["relu_q3"], 0), p["relu_q6"])
    if p["prescale"]:
        xr = (xr + (1 << (p["prescale"] - 1))) >> p["prescale"]
    y = d * xr
    shift = p["output_shift"]
    high = ((y << max(shift, 0)) * p["output_multiplier"] + 2**30) >> 31
    return min(max(rdbpot(high, max(-shift, 0)) + p["output_offset"], int(info.min)), int(info.max))


def _hard_swish_real(v: np.ndarray) -> np.ndarray:
    return v * np.clip(v + 3.0, 0.0, 6.0) / 6.0


def test_hard_swish_prepare_matches_the_cmsis_documented_example(lib) -> None:
    p = _hs_prepare(lib, "hard_swish_prepare", "s8", 0.125, 0, 0.125, 0)
    assert (p["output_multiplier_fixedpoint_int16"], p["output_multiplier_exponent"]) == (16384, -6)
    assert (p["reluish_multiplier_fixedpoint_int16"], p["reluish_multiplier_exponent"]) == (21845, 4)


@pytest.mark.parametrize("seed", range(6))
def test_hard_swish_compat_matches_the_tflite_model(lib, seed) -> None:
    rng = np.random.default_rng(seed)
    s_in = float(np.float32(rng.uniform(0.002, 0.2)))
    s_out = float(np.float32(rng.uniform(0.01, 0.1)))
    z_in, z_out = (int(z) for z in rng.integers(-40, 41, 2))
    p = _hs_prepare(lib, "hard_swish_prepare", "s8", s_in, z_in, s_out, z_out)
    x = _all_codes("s8")
    got = _unary(lib, "hard_swish_s8", p, x)
    np.testing.assert_array_equal(got, [_model_hard_swish_compat(v, p) for v in x.tolist()])
    ideal = np.clip(np.round(_hard_swish_real((x.astype(np.float64) - z_in) * s_in) / s_out) + z_out, -128, 127)
    assert np.abs(got.astype(np.float64) - ideal).max() <= 2


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(4))
def test_hard_swish_precise_matches_the_cmsis_model(lib, kind, seed) -> None:
    rng = np.random.default_rng(seed)
    s_in, z_in, s_out, z_out = _quant(rng, kind)
    p = _hs_prepare(lib, "hard_swish_precise_prepare", kind, s_in, z_in, s_out, z_out)
    assert p["relu_q3"] == math.floor(3.0 / s_in + 0.5) and p["relu_q6"] == math.floor(6.0 / s_in + 0.5)
    x = _all_codes(kind)
    got = _unary(lib, f"hard_swish_precise_{kind}", p, x)
    np.testing.assert_array_equal(got, [_model_hard_swish_precise(v, p, kind) for v in x.tolist()])
    info = _info(kind)
    ideal = np.clip(np.round(_hard_swish_real((x.astype(np.float64) - z_in) * s_in) / s_out) + z_out,
                    info.min, info.max)
    assert np.abs(got.astype(np.float64) - ideal).max() <= 1


def test_hard_swish_precise_prescale_keeps_the_product_in_int32(lib) -> None:
    # relu_q6 = 6 / 5e-5 = 120000: 32767 * 120000 needs two halvings to fit int32.
    p = _hs_prepare(lib, "hard_swish_precise_prepare", "s16", 5e-5, 0, 1e-3, 0)
    assert p["relu_q6"] == 120000 and p["prescale"] == 1
    assert 32767 * (p["relu_q6"] >> p["prescale"]) <= 2**31 - 1
    x = np.array([-32768, -1, 0, 1, 32767], np.int16)
    got = _unary(lib, "hard_swish_precise_s16", p, x)
    np.testing.assert_array_equal(got, [_model_hard_swish_precise(v, p, "s16") for v in x.tolist()])


@pytest.mark.parametrize(
    "entry, kind, overrides",
    [
        ("hard_swish_prepare", "s16", {}),
        ("hard_swish_prepare", "s8", {"input_scale": 0.0}),
        ("hard_swish_prepare", "s8", {"output_scale": math.inf}),
        ("hard_swish_prepare", "s8", {"output_zero_point": 128}),
        # in/out >= 128 makes the output exponent positive, which HardSwishPrepare refuses.
        ("hard_swish_prepare", "s8", {"input_scale": 1.0, "output_scale": 1e-3}),
        ("hard_swish_precise_prepare", "s16", {"input_zero_point": 3}),
        ("hard_swish_precise_prepare", "s8", {"input_scale": 1e-12}),
    ],
)
def test_hard_swish_prepare_rejects(lib, entry, kind, overrides) -> None:
    fields = {"input_scale": 0.05, "input_zero_point": 0, "output_scale": 0.05, "output_zero_point": 0, **overrides}
    with pytest.raises(ReferenceKernelError, match="E_PARAM|E_DTYPE"):
        _hs_prepare(lib, entry, kind, fields["input_scale"], fields["input_zero_point"], fields["output_scale"],
                    fields["output_zero_point"])


@pytest.mark.parametrize(
    "field, value",
    [("output_multiplier_exponent", 1), ("output_multiplier_exponent", -16), ("reluish_multiplier_exponent", 16),
     ("reluish_multiplier_fixedpoint_int16", 32768), ("output_multiplier_fixedpoint_int16", -1),
     ("input_zero_point", 128)],
)
def test_hard_swish_kernel_rejects_invalid_params(lib, field, value) -> None:
    p = {**_hs_prepare(lib, "hard_swish_prepare", "s8", 0.05, 0, 0.05, 0), field: value}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _unary(lib, "hard_swish_s8", p, np.zeros(4, np.int8))


@pytest.mark.parametrize(
    "field, value",
    [("relu_q3", -1), ("relu_q6", 1), ("prescale", 32), ("prescale", -1), ("output_shift", 31),
     ("output_multiplier", -1), ("output_offset", 1)],
)
def test_hard_swish_precise_kernel_rejects_invalid_params(lib, field, value) -> None:
    p = {**_hs_prepare(lib, "hard_swish_precise_prepare", "s16", 1e-3, 0, 1e-3, 0), field: value}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _unary(lib, "hard_swish_precise_s16", p, np.zeros(4, np.int16))


def test_hard_swish_entries_check_dtype_and_shape(lib) -> None:
    p = _hs_prepare(lib, "hard_swish_prepare", "s8", 0.05, 0, 0.05, 0)
    with pytest.raises(TypeError, match="must be int8"):
        _unary(lib, "hard_swish_s8", p, np.zeros(4, np.int16))
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        lib.run("hard_swish_s8", p, {"input": np.zeros(4, np.int8)}, {"output": (5,)})
    assert _unary(lib, "hard_swish_s8", p, np.zeros((0, 3), np.int8)).shape == (0, 3)


def test_float16_hard_swish_is_exact_then_rounded_once_for_every_half(lib) -> None:
    x = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    x = x[np.isfinite(x)]
    got = _unary(lib, "hard_swish_f16", {"unused": 0}, x)
    want = _hard_swish_real(x.astype(np.float64)).astype(np.float16)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


def test_float32_hard_swish_is_exact_then_rounded_once(lib) -> None:
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.uniform(-8, 8, 200_000), [-3.0, 3.0, 0.0, -0.0, 1e-30, -1e-30, 3e38]]).astype(np.float32)
    got = _unary(lib, "hard_swish_f32", {"unused": 0}, x)
    with np.errstate(over="ignore"):
        want = _hard_swish_real(x.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_float_hard_swish_nonfinite(lib, dtype) -> None:
    entry = "hard_swish_f16" if dtype == np.float16 else "hard_swish_f32"
    got = _unary(lib, entry, {"unused": 0}, np.array([np.nan, np.inf, -np.inf], dtype))
    assert np.isnan(got[0]) and np.isposinf(got[1]) and np.isnan(got[2])


# ---------------------------------------------------------------- Tanh, Logistic

from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_s16 import (  # noqa: E402
    nn_activation_reference_params,
)


def _reference_sigmoid_table() -> list:
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "reference" / "src" / "activation" / "tanh_logistic.c").read_text()
    body = src[src.index("sigmoid_table[256] = {"):]
    return [int(v) for v in re.findall(r"\b\d+\b", body[body.index("{"):body.index("};")])]


SIGMOID_TABLE = _reference_sigmoid_table()


def _model_tanh_logistic(x: int, multiplier: int, shift: int, is_tanh: bool) -> int:
    """reference_integer_ops::Tanh/Logistic(int16) on Python ints."""
    if multiplier == 0:
        multiplier, shift = 3 << shift, 0
    v = (x * multiplier + ((1 << (shift - 1)) if shift else 0)) >> shift
    a = abs(v)
    bits = 8 if is_tanh else 9
    uh = a >> bits
    if uh >= 255:
        r = (0xFFFF << 8) if is_tanh else (0x7FFF << 10)
    else:
        ua, ub = SIGMOID_TABLE[uh], SIGMOID_TABLE[uh + 1]
        r = (ua << bits) + (a & ((1 << bits) - 1)) * (ub - ua)
    if is_tanh:
        r = r - (1 << 23) + (1 << 7) if v >= 0 else -r + (1 << 23) + (1 << 7) - 1
        return r >> 8
    r = r + (1 << 9) if v >= 0 else (1 << 25) - r + (1 << 9) - 1
    return r >> 10


def test_sigmoid_table_tracks_the_sigmoid() -> None:
    t = np.array(SIGMOID_TABLE, dtype=np.float64)
    assert len(t) == 256 and t[0] == 32768 and t[-1] == 65535
    assert np.all(np.diff(t) >= 0)
    exact = 65536.0 / (1.0 + np.exp(-np.arange(256) * 32.0 / 768.0))
    assert np.abs(t - exact).max() < 1.2


def _tl_prepare(lib, entry, s_in, s_out=2.0 ** -15, z_in=0, z_out=0):
    return lib.prepare(f"{entry}_prepare", {"input_scale": s_in, "input_zero_point": z_in,
                                            "output_scale": s_out, "output_zero_point": z_out})


ALL_S16 = np.arange(-32768, 32768, dtype=np.int16)


@pytest.mark.parametrize("entry", ["tanh", "logistic"])
@pytest.mark.parametrize("s_in", [1 / 32767, 2.0 ** -12, 2.0 ** -11, 2.0 ** -13, 3e-4, 1e-5, 2.5])
def test_tanh_logistic_match_the_model_and_the_function(lib, entry, s_in) -> None:
    p = _tl_prepare(lib, entry, float(np.float32(s_in)))
    got = _unary(lib, f"{entry}_s16", p, ALL_S16)
    is_tanh = entry == "tanh"
    want = [_model_tanh_logistic(v, p["input_multiplier"], p["input_left_shift"], is_tanh) for v in ALL_S16.tolist()]
    np.testing.assert_array_equal(got, want)
    real = ALL_S16.astype(np.float64) * float(np.float32(s_in))
    with np.errstate(over="ignore"):
        fn = np.tanh(real) if is_tanh else 1.0 / (1.0 + np.exp(-real))
    # The table saturates past |x| = 10.67 (tanh) / 21.3 (logistic), where the function is within 2^-15 of 1.
    assert np.abs(got.astype(np.float64) / 32768.0 - fn).max() < 6e-4


@pytest.mark.parametrize("entry, s_in, want", [
    ("tanh", 2.0 ** -12, (0, 0)), ("tanh", 2.0 ** -11, (0, 1)), ("tanh", 2.0 ** -10, None),
    ("logistic", 2.0 ** -12, (0, 0)), ("logistic", 2.0 ** -11, None),
])
def test_power_of_two_input_scales_skip_the_multiplier(lib, entry, s_in, want) -> None:
    p = _tl_prepare(lib, entry, s_in)
    if want is None:
        assert p["input_multiplier"] > 16383
    else:
        assert (p["input_multiplier"], p["input_left_shift"]) == want


@pytest.mark.parametrize("entry", ["tanh", "logistic"])
@pytest.mark.parametrize("overrides", [{"z_in": 1}, {"z_out": -1}, {"s_out": 2.0 ** -14}, {"s_out": 1e-3},
                                       {"s_in": 0.0}, {"s_in": math.nan}, {"s_in": 3.0}])
def test_tanh_logistic_prepare_rejects(lib, entry, overrides) -> None:
    kw = {"s_in": 1 / 32767, **overrides}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _tl_prepare(lib, entry, **kw)


@pytest.mark.parametrize("entry", ["tanh_s16", "logistic_s16"])
@pytest.mark.parametrize("params", [{"input_multiplier": -1, "input_left_shift": 0},
                                    {"input_multiplier": 32768, "input_left_shift": 0},
                                    {"input_multiplier": 5, "input_left_shift": 32},
                                    {"input_multiplier": 0, "input_left_shift": 14}])
def test_tanh_logistic_kernels_reject_invalid_params(lib, entry, params) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _unary(lib, entry, params, np.zeros(4, np.int16))


def test_tanh_logistic_extremes_saturate(lib) -> None:
    p = _tl_prepare(lib, "tanh", 1e-2)
    x = np.array([-32768, -1, 0, 1, 32767], np.int16)
    # TFLite's tanh is symmetric about 0: it bottoms out at -32767, not -32768.
    np.testing.assert_array_equal(_unary(lib, "tanh_s16", p, x)[[0, 2, 4]], [-32767, 0, 32767])
    q = _tl_prepare(lib, "logistic", 1e-2)
    np.testing.assert_array_equal(_unary(lib, "logistic_s16", q, x)[[0, 2, 4]], [1, 16384, 32767])


@pytest.mark.parametrize("left_shift, want", [(0, (3, 0)), (2, (12, 0)), (-1, (3, 1)), (-5, (3, 5))])
def test_nn_activation_left_shift_maps_to_the_reference_params(left_shift, want) -> None:
    p = nn_activation_reference_params(left_shift)
    assert (p["input_multiplier"], p["input_left_shift"]) == want


@pytest.mark.parametrize("left_shift", [-3, 0, 2])
@pytest.mark.parametrize("entry", ["tanh_s16", "logistic_s16"])
def test_nn_activation_matches_the_cmsis_formula(lib, left_shift, entry) -> None:
    # arm_nn_activation_s16: multiplier 3 << left_shift, or 3 and a rounding shift of -left_shift.
    mult = 3 if left_shift < 0 else 3 << left_shift
    shift = -left_shift if left_shift < 0 else 0
    got = _unary(lib, entry, nn_activation_reference_params(left_shift), ALL_S16)
    want = [_model_tanh_logistic(v, mult, shift, entry == "tanh_s16") for v in ALL_S16.tolist()]
    np.testing.assert_array_equal(got, want)


# --------------------------------------------------------- NNActivationFloat

from helia_core_tester.generation.reference.abi import float_activation_code  # noqa: E402
from helia_core_tester.tests.reference_models import (  # noqa: E402
    TANH_LUT256_F16,
    tanh_reference_f16,
    tanh_reference_f16_mve,
)

FLOAT_ACTS = ["NONE", "RELU", "RELU6", "LEAKY_RELU", "SIGMOID", "TANH", "HARDSWISH"]


def _exact_activation(x: np.ndarray, kind: str, a: float) -> np.ndarray:
    v = x.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return {
            "NONE": lambda: v, "RELU": lambda: np.where(v < 0, 0.0, v), "RELU6": lambda: np.clip(v, 0.0, 6.0),
            "LEAKY_RELU": lambda: np.where(v >= 0, v, v * a), "SIGMOID": lambda: 1.0 / (1.0 + np.exp(-v)),
            "TANH": lambda: np.tanh(v), "HARDSWISH": lambda: v * np.clip(v + 3.0, 0.0, 6.0) / 6.0,
        }[kind]()


@pytest.mark.parametrize("kind", FLOAT_ACTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_float_activations_are_exact_then_rounded_once(lib, kind, dtype) -> None:
    rng = np.random.default_rng(FLOAT_ACTS.index(kind))
    x = (rng.uniform(-12, 12, 20_000)).astype(dtype)
    entry = "nn_activation_f16" if dtype == np.float16 else "nn_activation_f32"
    got = _unary(lib, entry, {"activation_type": float_activation_code(kind), "act_param": 0.125}, x)
    want = _exact_activation(x, kind, float(np.float32(0.125))).astype(dtype)
    if kind in ("SIGMOID", "TANH"):
        # libm's exp/tanh are faithful, not correctly rounded: an output can sit one ulp off at a near-tie.
        bits = np.int32 if dtype == np.float32 else np.int16
        assert np.abs(got.view(bits).astype(np.int64) - want.view(bits).astype(np.int64)).max() <= 1
    else:
        np.testing.assert_array_equal(got.view(np.uint32 if dtype == np.float32 else np.uint16),
                                      want.view(np.uint32 if dtype == np.float32 else np.uint16))


@pytest.mark.parametrize("kind", FLOAT_ACTS)
def test_float_activations_propagate_nan_and_clamp_infinities(lib, kind) -> None:
    x = np.array([np.nan, np.inf, -np.inf, -0.0], np.float32)
    got = _unary(lib, "nn_activation_f32", {"activation_type": float_activation_code(kind), "act_param": 0.125}, x)
    assert np.isnan(got[0])
    with np.errstate(all="ignore"):
        want = _exact_activation(x, kind, 0.125).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])


@pytest.mark.parametrize("params", [{"activation_type": 7, "act_param": 0.0}, {"activation_type": -1, "act_param": 0.0},
                                    {"activation_type": 3, "act_param": math.nan}])
def test_float_activation_rejects_invalid_params(lib, params) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _unary(lib, "nn_activation_f32", params, np.zeros(3, np.float32))


def test_float_activation_codes_accept_the_cmsis_names() -> None:
    assert float_activation_code("ARM_NN_FLT_ACT_SIGMOID") == float_activation_code("sigmoid")
    with pytest.raises(KeyError):
        float_activation_code("ARM_NN_FLT_ACT_GELU")


def test_tanh_lut_is_tanh_sampled_over_0_4() -> None:
    from pathlib import Path
    import re

    src = (Path(__file__).resolve().parents[1] / "reference" / "src" / "activation" / "nn_activation_float.c").read_text()
    body = src[src.index("tanh_lut_f16[257] = {"):]
    table = np.array([int(v, 16) for v in re.findall(r"0x([0-9A-F]{4})u", body[:body.index("};")])], np.uint16)
    np.testing.assert_array_equal(table, TANH_LUT256_F16.view(np.uint16))


@pytest.mark.parametrize("entry, model", [("tanh_lut_f16", tanh_reference_f16), ("tanh_lut_mve_f16", tanh_reference_f16_mve)])
def test_float16_tanh_lut_variants_match_their_models_on_every_pattern(lib, entry, model) -> None:
    x = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    got = _unary(lib, entry, {"unused": 0}, x)
    with np.errstate(all="ignore"):
        want = model(x)
    nan = np.isnan(x)
    assert np.all(np.isnan(got[nan]))
    np.testing.assert_array_equal(got[~nan].view(np.uint16), want[~nan].view(np.uint16))
    assert np.abs(got[~nan].astype(np.float64) - np.tanh(x[~nan].astype(np.float64))).max() < 2e-3
