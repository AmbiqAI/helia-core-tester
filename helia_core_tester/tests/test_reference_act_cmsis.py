"""Kernel-faithful float activations and softmax (nn_activation_f32_cmsis, nn_activation_f16_mve,
hard_swish_f32_cmsis, hard_swish_f16_mve, softmax_f32_cmsis) against independent restatements of the
ns-cmsis-nn arithmetic per target, hand-checkable cases, parameter and shape checks, the generators' route
selection and tolerance, the guard against the exact result, and the stored polynomial against the checkout's."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.reference.abi import float_activation_code
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.generation.reference.call import ReferenceCall

f32 = np.float32
SIGMOID, TANH, HARDSWISH = "ARM_NN_FLT_ACT_SIGMOID", "ARM_NN_FLT_ACT_TANH", "ARM_NN_FLT_ACT_HARDSWISH"
SRC = Path(__file__).resolve().parents[1] / "reference" / "src"


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _round(f: Fraction, dtype) -> float:
    c = dtype(float(f))
    cands = [c, np.nextafter(c, dtype(np.inf)), np.nextafter(c, dtype(-np.inf))]
    uint = np.uint16 if dtype == np.float16 else np.uint32
    return float(min(cands, key=lambda v: (abs(Fraction(float(v)) - f), int(np.array(v, dtype).view(uint)) & 1)))


def fma(a, b, c, dtype=np.float32) -> float:
    if not all(np.isfinite([float(a), float(b), float(c)])):
        with np.errstate(invalid="ignore"):
            return float(dtype(float(a) * float(b) + float(c)))
    return _round(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)), dtype)


def h(v) -> float:
    return float(np.float16(v))


def _hex_table(path: Path, name: str, width: int):
    body = path.read_text()
    body = body[body.index(name):]
    return [int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]{%d})u" % width, body[:body.index("};")])]


def _f32(bits):
    return [struct.unpack("<f", struct.pack("<I", b))[0] for b in bits]


EXP2 = _f32(_hex_table(SRC / "activation" / "cmsis_f32_tables.c", "hct_cmsis_exp2_lut_f32_bits", 8))
TANH32 = _f32(_hex_table(SRC / "activation" / "cmsis_f32_tables.c", "hct_cmsis_tanh_lut_f32_bits", 8))
EXP2_F16 = [float(np.array(b, np.uint16).view(np.float16))
            for b in _hex_table(SRC / "activation" / "cmsis_f16_tables.c", "hct_cmsis_exp2_lut_f16", 4)]
POLY = _f32(_hex_table(SRC / "softmax" / "softmax_f32_cmsis.c", "poly_bits[8]", 8))


def bits32(a) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def assert_bits(got, want) -> None:
    got, want = np.asarray(got), np.asarray(want, dtype=np.asarray(got).dtype)
    view = np.uint16 if got.dtype == np.float16 else np.uint32
    g, w = got.view(view), want.view(view)
    same = (g == w) | (np.isnan(got) & np.isnan(want))
    assert same.all(), f"first mismatch at {np.argmax(~same)}: got {got.ravel()[np.argmax(~same)]!r}, " \
                       f"want {want.ravel()[np.argmax(~same)]!r}"


# ------------------------------------------------------------- float32 restatements per target

def madd(fused, a, b, c):
    return f32(fma(a, b, c)) if fused else f32(f32(f32(a) * f32(b)) + f32(c))


def exp_lut32(fused, x):
    x = f32(min(max(x, f32(-80.0)), f32(80.0)))
    t = f32(x * f32(1.44269504088896341))
    n = int(t)
    n = n - 1 if t < f32(n) else n
    idx_f = f32(f32(t - f32(n)) * f32(256.0))
    idx = min(max(int(idx_f), 0), 255)
    frac = f32(idx_f - f32(idx))
    y = madd(fused, f32(f32(EXP2[idx + 1]) - f32(EXP2[idx])), frac, f32(EXP2[idx]))
    return f32(y * f32(2.0 ** min(max(n, -126), 127)))


def sigmoid32(target, x):
    fused = target != 2
    if x >= 0:
        return f32(f32(1.0) / f32(f32(1.0) + exp_lut32(fused, f32(-x))))
    e = exp_lut32(fused, x)
    return f32(e / f32(f32(1.0) + e))


def tanh32(target, x):
    if np.isnan(x):
        return f32(np.array(bits32(x) | 0x00400000, np.uint32).view(f32)) if target == 2 else f32(x)
    if target == 0 and x != 0 and abs(x) < 2.0 ** -126:
        return f32(np.copysign(0.0, x))  # Helium flushes the subnormal input
    ax = f32(abs(x))
    if not ax < 6.0:
        return f32(np.copysign(1.0, x))
    t = f32(ax * f32(64.0))
    idx = min(int(t), 383)
    frac = f32(t - f32(idx))
    return f32(np.copysign(madd(target != 2, f32(f32(TANH32[idx + 1]) - f32(TANH32[idx])), frac, f32(TANH32[idx])), x))


def hardswish_act32(target, x):
    sixth = f32(1.0 / 6.0)
    t = f32(fma(x, sixth, 0.5)) if target != 2 else f32(f32(x * sixth) + f32(0.5))
    t = f32(0.0) if np.isnan(t) and target == 0 else t
    t = f32(min(max(t, f32(0.0)), f32(1.0))) if not np.isnan(t) else t
    with np.errstate(invalid="ignore"):  # -inf * 0 is the NaN the kernel produces
        return f32(x * t)


def hard_swish32(x):
    t = f32(fma(x, f32(1.0 / 6.0), 0.5))
    t = f32(min(max(t, f32(0.0)), f32(1.0))) if not np.isnan(t) else f32(1.0)
    with np.errstate(invalid="ignore"):  # -inf * 0 is the NaN the kernel produces
        return f32(x * t)


def act32(name, target, xs):
    fn = {SIGMOID: sigmoid32, TANH: tanh32, HARDSWISH: hardswish_act32}[name]
    return np.array([fn(target, f32(v)) for v in xs], np.float32)


F32_EDGES = [0.0, -0.0, 0.5, -0.5, 1.0 / 64, 5.984375, 5.9999995, 6.0, -6.0, 6.5, -7.25, 3.0, -3.0, 2.9999998,
             -2.9999998, 79.9, 80.0, -80.0, -95.0, 1e-30, -1e-30, 1e-40, -1e-40, np.inf, -np.inf]


def f32_inputs(seed, n=160):
    rng = np.random.default_rng(seed)
    return np.concatenate([np.array(F32_EDGES, np.float32), rng.uniform(-9, 9, n).astype(np.float32)])


def run_act32(x, name, target, act_param=0.0):
    return kf.nn_activation_f32_cmsis_call(x, float_activation_code(name), act_param, target).output()


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("name", [SIGMOID, TANH, HARDSWISH])
def test_f32_activations_match_the_restatement(lib, name, target) -> None:
    x = f32_inputs(target * 7 + len(name))
    assert_bits(run_act32(x, name, target), act32(name, target, x))


def test_f32_tanh_nan_is_kept_on_helium_and_quieted_on_soft_float(lib) -> None:
    snan = np.array([0x7F800001, 0xFFA00000], np.uint32).view(np.float32)
    assert (bits32(run_act32(snan, TANH, 0)) == bits32(snan)).all()
    assert (bits32(run_act32(snan, TANH, 2)) == (bits32(snan) | 0x00400000)).all()


def test_fused_and_unfused_legs_differ_where_rounding_the_product_matters(lib) -> None:
    # Inputs where the soft-float leg's rounded product moves the result (found by a dense sweep): the FPU legs
    # (Helium, scalar FPU) agree with each other and differ from soft float there.
    for name, x in ((TANH, [-3.13382, -2.99556]), (SIGMOID, [-9.96753, -9.95684]), (HARDSWISH, [-2.99999])):
        x = np.array(x, np.float32)
        assert not np.array_equal(run_act32(x, name, 1), run_act32(x, name, 2))
        assert_bits(run_act32(x, name, 0), run_act32(x, name, 1))
        assert_bits(run_act32(x, name, 2), act32(name, 2, x))


def test_f32_activations_hand_checked(lib) -> None:
    # tanh(0.5): t = 32 exactly, so the result is table entry 32 with nothing interpolated.
    assert float(run_act32(np.array([0.5], f32), TANH, 1)[0]) == TANH32[32]
    assert run_act32(np.array([6.0, -7.0], f32), TANH, 0).tolist() == [1.0, -1.0]
    # sigmoid(0) = 1 / (1 + 2^0 table[0]) = 1/2 on every target.
    for target in (0, 1, 2):
        assert run_act32(np.array([0.0], f32), SIGMOID, target).tolist() == [0.5]
        assert run_act32(np.array([3.0, -3.0, 9.0], f32), HARDSWISH, target).tolist() == [3.0, 0.0, 9.0]
    # Helium's maxnm turns -0 into +0 where the scalar ternary keeps it.
    relu = float_activation_code("ARM_NN_FLT_ACT_RELU")
    neg0 = np.array([-0.0], f32)
    assert np.signbit(kf.nn_activation_f32_cmsis_call(neg0, relu, 0.0, 1).output()[0])
    assert not np.signbit(kf.nn_activation_f32_cmsis_call(neg0, relu, 0.0, 0).output()[0])


@pytest.mark.parametrize("name", ["ARM_NN_FLT_ACT_NONE", "ARM_NN_FLT_ACT_RELU", "ARM_NN_FLT_ACT_RELU6",
                                  "ARM_NN_FLT_ACT_LEAKY_RELU"])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_exact_activations_equal_the_exact_reference(lib, name, target) -> None:
    """Why the generator keeps the exact golden for these: the kernel rounds at most one product."""
    x = np.random.default_rng(5).uniform(-8, 8, 300).astype(np.float32)
    code = float_activation_code(name)
    got = kf.nn_activation_f32_cmsis_call(x, code, 0.125, target).output()
    exact = ReferenceCall("nn_activation_f32", {"activation_type": code, "act_param": 0.125}, {"input": x},
                          {"output": x.shape}).output()
    assert_bits(got, exact)


def test_hard_swish_f32_is_the_same_on_every_target(lib) -> None:
    x = f32_inputs(17)
    want = np.array([hard_swish32(f32(v)) for v in x], np.float32)
    for target in (0, 1, 2):
        got = ReferenceCall("hard_swish_f32_cmsis", {"target": target}, {"input": x}, {"output": x.shape}).output()
        finite = np.isfinite(x)
        assert_bits(got[finite], want[finite])
    assert np.isnan(ReferenceCall("hard_swish_f32_cmsis", {"target": 0}, {"input": np.array([-np.inf], f32)},
                                  {"output": (1,)}).output()[0])


# ------------------------------------------------------------- float16 restatements (Helium)

def exp_lut16(x):
    x = f32(max(min(float(x), 80.0), -80.0))
    t = f32(x * f32(1.44269504088896341))
    tq = h(t)
    n = int(tq)
    n = n - 1 if tq < n else n
    idx_f = f32(f32(t - f32(n)) * f32(256.0))
    idx = min(max(int(idx_f), 0), 255)
    frac = f32(idx_f - f32(idx))
    y = fma(frac, f32(f32(EXP2_F16[idx + 1]) - f32(EXP2_F16[idx])), EXP2_F16[idx])
    return h(f32(f32(y) * f32(2.0 ** min(max(n, -14), 15))))


def sigmoid16(x):
    if x >= 0:
        return h(f32(f32(1.0) / f32(f32(1.0) + f32(exp_lut16(-x)))))
    e = f32(exp_lut16(x))
    return h(f32(e / f32(f32(1.0) + e)))


def hardswish_act16(x):
    t = fma(x, h(1.0 / 6.0), 0.5, np.float16)
    return h(x * min(max(t, 0.0), 1.0))


def hard_swish16(x):
    t = min(max(h(x + 3.0), 0.0), 6.0)
    return h(x * h(t * h(1.0 / 6.0)))


def f16_inputs(seed, n=200):
    rng = np.random.default_rng(seed)
    edges = [0.0, -0.0, 3.0, -3.0, 2.998, -2.998, 4.0, -4.0, 8.0, -8.0, 11.5, 6.1e-5, -6.1e-5, 65504.0, -65504.0]
    return np.concatenate([np.array(edges, np.float16), rng.uniform(-10, 10, n).astype(np.float16)])


def run_act16(x, name, act_param=0.0):
    return kf.nn_activation_f16_mve_call(x, float_activation_code(name), act_param).output()


@pytest.mark.parametrize("name,fn", [(SIGMOID, sigmoid16), (HARDSWISH, hardswish_act16)])
def test_f16_activations_match_the_restatement(lib, name, fn) -> None:
    x = f16_inputs(len(name))
    assert_bits(run_act16(x, name), np.array([fn(float(v)) for v in x], np.float16))


def test_f16_tanh_is_the_helium_table_variant(lib) -> None:
    x = f16_inputs(3)
    want = ReferenceCall("tanh_lut_mve_f16", {"unused": 0}, {"input": x}, {"output": x.shape}).output()
    assert_bits(run_act16(x, TANH), want)


def test_hard_swish_f16_matches_the_restatement(lib) -> None:
    x = f16_inputs(9)
    got = ReferenceCall("hard_swish_f16_mve", {"unused": 0}, {"input": x}, {"output": x.shape}).output()
    assert_bits(got, np.array([hard_swish16(float(v)) for v in x], np.float16))


def test_f16_hand_checked(lib) -> None:
    assert run_act16(np.array([0.0], np.float16), SIGMOID).tolist() == [0.5]
    # 3 h(1/6) = 3 * 0x3155 is 2^-13 short of 1/2, so at x = -3 the fused gate is 2^-13, not 0.
    assert run_act16(np.array([3.0, -3.0, 8.0], np.float16), HARDSWISH).tolist() == [3.0, -3 * 2.0 ** -13, 8.0]
    # x + 3 = 4 exactly, then 4 * h(1/6) = 0.6665 rounds to 0x3955; 1 * that is the output.
    got = ReferenceCall("hard_swish_f16_mve", {"unused": 0}, {"input": np.array([1.0], np.float16)},
                        {"output": (1,)}).output()
    assert int(got.view(np.uint16)[0]) == 0x3955


def test_activation_budgets_cover_every_float16_value(lib) -> None:
    """The guard's budgets hold over the whole float16 domain the descriptors draw from (|x| <= 10)."""
    x = np.arange(65536, dtype=np.uint16).view(np.float16)
    x = x[np.isfinite(x) & (np.abs(x.astype(np.float64)) <= 10)]
    for name in (SIGMOID, HARDSWISH, TANH):
        exact = ReferenceCall("nn_activation_f16", {"activation_type": float_activation_code(name), "act_param": 0.0},
                              {"input": x}, {"output": x.shape})
        kf.check_close_to_exact(name, kf.nn_activation_f16_mve_call(x, float_activation_code(name), 0.0), exact,
                                kf.activation_atol(x, name))
    exact = ReferenceCall("hard_swish_f16", {"unused": 0}, {"input": x}, {"output": x.shape})
    kf.check_close_to_exact("hs", kf.hard_swish_cmsis_call(x, "cortex-m55"), exact, kf.hard_swish_atol(x))


@pytest.mark.parametrize("target", [0, 1, 2])
def test_activation_budgets_cover_a_dense_float32_sweep(lib, target) -> None:
    x = np.concatenate([np.linspace(-10, 10, 200001), [5.9999995, -5.9999995]]).astype(np.float32)
    for name in (SIGMOID, TANH, HARDSWISH):
        code = float_activation_code(name)
        exact = ReferenceCall("nn_activation_f32", {"activation_type": code, "act_param": 0.0}, {"input": x},
                              {"output": x.shape})
        kf.check_close_to_exact(name, kf.nn_activation_f32_cmsis_call(x, code, 0.0, target), exact,
                                kf.activation_atol(x, name))


# ------------------------------------------------------------- softmax float32

def cvt_s32(v) -> int:
    v = float(v)
    if np.isnan(v):
        return 0
    return int(max(min(v, 2.0 ** 31 - 1), -2.0 ** 31))


def exp_poly32(x):
    x = f32(x)
    m = cvt_s32(f32(x * f32(1.4426950408)))
    r = f32(fma(-f32(m), f32(0.6931471805), x))
    c = [f32(v) for v in POLY]
    a, b = fma(c[4], r, c[0]), fma(c[6], r, c[2])
    cc, d = fma(c[5], r, c[1]), fma(c[7], r, c[3])
    x2 = f32(r * r)
    x4 = f32(x2 * x2)
    y = fma(fma(d, x2, cc), x4, fma(b, x2, a))
    ybits = int(np.array(y, f32).view(np.int32)) + max(min(m << 23, 2 ** 31 - 1), -2 ** 31)
    y = f32(np.array(max(min(ybits, 2 ** 31 - 1), -2 ** 31), np.int32).view(f32))
    return f32(0.0) if m < -126 else y


def softmax32(target, x) -> np.ndarray:
    rows = x.reshape(-1, x.shape[-1])
    fused = target != 2
    out = np.zeros(rows.shape, np.float32)
    if rows.shape == (1, 2):
        a, b = rows[0]
        e0, e1 = (f32(1.0), exp_lut32(fused, f32(b - a))) if a >= b else (exp_lut32(fused, f32(a - b)), f32(1.0))
        s = f32(e0 + e1)
        out[0] = [f32(e0 / s), f32(e1 / s)]
        return out.reshape(x.shape)
    for r, row in enumerate(rows):
        total = f32(0.0)
        if target == 0:
            mx = max([f32(-3.4028235e38)] + [v for v in row if not np.isnan(v)])
            e = [exp_poly32(f32(v - mx)) for v in row]
            for c0 in range(0, len(e), 4):
                lane = (e[c0:c0 + 4] + [f32(0.0)] * 4)[:4]
                total = f32(total + f32(f32(f32(lane[0] + lane[1]) + lane[2]) + lane[3]))
        else:
            mx = row[0]
            for v in row[1:]:
                mx = v if v > mx else mx
            e = [exp_lut32(fused, f32(v - mx)) for v in row]
            for v in e:
                total = f32(total + v)
        inv = f32(f32(1.0) / total)
        out[r] = [f32(v * inv) for v in e]
    return out.reshape(x.shape)


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("shape", [(1, 3), (4, 7), (2, 8), (3, 9), (1, 17), (2, 2), (1, 1), (1, 2)])
def test_softmax_f32_matches_restatement(lib, target, shape) -> None:
    x = np.random.default_rng(sum(shape) + target).uniform(-4, 4, shape).astype(np.float32)
    call = kf.softmax_f32_cmsis_call(x, target)
    assert_bits(call.output(), softmax32(target, x))
    kf.check_close_to_exact("s", call, replace(call, entry="softmax_f32", params={"unused": 0}),
                            kf.softmax_f32_atol(x))


def test_softmax_f32_hand_checked(lib) -> None:
    for target in (0, 1, 2):
        assert kf.softmax_f32_cmsis_call(np.array([[0.75]], f32), target).output().tolist() == [[1.0]]
        assert kf.softmax_f32_cmsis_call(np.array([[2.0, 2.0]], f32), target).output().tolist() == [[0.5, 0.5]]
        four = kf.softmax_f32_cmsis_call(np.array([[1.0, 1.0, 1.0, 1.0]], f32), target).output()
        assert four.tolist() == [[0.25] * 4]


def test_softmax_f32_degenerate_row_is_nan_and_others_finite(lib) -> None:
    x = np.array([[-np.inf] * 3, [0.0, 1.0, 2.0]], np.float32)
    for target in (0, 1, 2):
        got = kf.softmax_f32_cmsis_call(x, target).output()
        assert np.isfinite(got[1]).all()
    assert np.isnan(kf.softmax_f32_cmsis_call(x, 0).output()[0]).all()


def test_softmax_f32_guard_rejects_a_wrong_emulation(lib) -> None:
    x = np.random.default_rng(3).uniform(-1, 1, (2, 5)).astype(np.float32)
    call = kf.softmax_f32_cmsis_call(x, 1)
    other = kf.softmax_f32_cmsis_call(np.flip(x, axis=1).copy(), 1)
    with pytest.raises(kf.KernelDeviationError):
        kf.check_close_to_exact("s", call, other, kf.softmax_f32_atol(x))


def test_stored_polynomial_is_the_checkouts(lib) -> None:
    root = os.environ.get("CMSIS_NN_ROOT")
    src = Path(root) / "Source" / "NNSupportFunctions" / "arm_nntables_flt.c" if root else None
    if src is None or not src.exists():
        pytest.skip("no ns-cmsis-nn checkout")
    text = src.read_text()
    body = text[text.index("arm_nn_exp_poly_coeffs_f32[8] = {"):]
    vals = [float(np.float32(float(v.strip().rstrip("fF"))))
            for v in body[body.index("{") + 1:body.index("};")].split(",") if v.strip()]
    assert vals == [float(v) for v in POLY]


# ------------------------------------------------------------- negative paths

def test_entries_reject_bad_params_dtypes_and_shapes(lib) -> None:
    x32 = np.zeros((2, 3), np.float32)
    x16 = np.zeros((2, 3), np.float16)
    sig = float_activation_code(SIGMOID)
    leaky = float_activation_code("ARM_NN_FLT_ACT_LEAKY_RELU")
    call = kf.nn_activation_f32_cmsis_call(x32, sig, 0.0, 0)
    assert _status(lambda: replace(call, params={**call.params, "target": 3}).output()) == "E_PARAM"
    assert _status(lambda: replace(call, params={**call.params, "target": -1}).output()) == "E_PARAM"
    assert _status(lambda: replace(call, params={**call.params, "activation_type": 7}).output()) == "E_PARAM"
    assert _status(lambda: replace(call, params={**call.params, "activation_type": leaky,
                                                 "act_param": float("inf")}).output()) == "E_PARAM"
    assert _status(lambda: replace(call, output_shapes={"output": (3, 2)}).output()) == "E_SHAPE"
    call16 = kf.nn_activation_f16_mve_call(x16, sig, 0.0)
    assert _status(lambda: replace(call16, params={**call16.params, "activation_type": -1}).output()) == "E_PARAM"
    assert _status(lambda: replace(call16, output_shapes={"output": (6,)}).output()) == "E_SHAPE"
    for entry in ("hard_swish_f32_cmsis", "softmax_f32_cmsis"):
        c = ReferenceCall(entry, {"target": 0}, {"input": x32}, {"output": x32.shape})
        assert _status(lambda: replace(c, params={"target": 5}).output()) == "E_PARAM"
        assert _status(lambda: replace(c, output_shapes={"output": (3, 2)}).output()) == "E_SHAPE"
    hs16 = ReferenceCall("hard_swish_f16_mve", {"unused": 0}, {"input": x16}, {"output": x16.shape})
    assert _status(lambda: replace(hs16, output_shapes={"output": (3, 2)}).output()) == "E_SHAPE"
    with pytest.raises(ValueError):
        kf.nn_activation_f32_cmsis_call(x32, sig, 0.0, 3)
    with pytest.raises(ValueError):
        kf.softmax_f32_cmsis_call(x32, 7)


def test_activation_guard_rejects_a_wrong_emulation(lib) -> None:
    x = np.linspace(-2, 2, 64).astype(np.float32)
    code = float_activation_code(TANH)
    exact = ReferenceCall("nn_activation_f32", {"activation_type": code, "act_param": 0.0}, {"input": x},
                          {"output": x.shape})
    kf.check_close_to_exact("ok", kf.nn_activation_f32_cmsis_call(x, code, 0.0, 1), exact, kf.activation_atol(x, TANH))
    wrong = kf.nn_activation_f32_cmsis_call(x, float_activation_code(SIGMOID), 0.0, 1)
    with pytest.raises(kf.KernelDeviationError):
        kf.check_close_to_exact("bad", wrong, exact, kf.activation_atol(x, TANH))


# ------------------------------------------------------------- route selection in the generators

def _generate(op_cls, desc, cpu, tmp_path):
    op = op_cls(desc, seed=7, target_cpu=cpu)
    op.generate_c_files(tmp_path)
    return op


def _act_desc(name, act, dtype, **extra):
    return {"operator": "NNActivationFloat", "name": name, "tensor_dtypes": {"input": dtype, "output": dtype},
            "input_shape": [1, 17], "activation_type": act, "act_param": 0.125, **extra}


@pytest.mark.parametrize("cpu,act,dtype,entry,target", [
    ("cortex-m55", SIGMOID, "FP32", "nn_activation_f32_cmsis", 0),
    ("cortex-m4", TANH, "FP32", "nn_activation_f32_cmsis", 1),
    ("cortex-m0", HARDSWISH, "FP32", "nn_activation_f32_cmsis", 2),
    ("cortex-m0", "ARM_NN_FLT_ACT_RELU", "FP32", "nn_activation_f32", None),
    ("cortex-m55", "ARM_NN_FLT_ACT_LEAKY_RELU", "FP16", "nn_activation_f16", None),
    ("cortex-m55", SIGMOID, "FP16", "nn_activation_f16_mve", None),
    ("cortex-m55", HARDSWISH, "FP16", "nn_activation_f16_mve", None),
    ("cortex-m55", TANH, "FP16", "tanh_lut_mve_f16", None),
])
def test_nn_activation_route_and_tolerance(tmp_path, cpu, act, dtype, entry, target) -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import OpNNActivationFloat

    op = _generate(OpNNActivationFloat, _act_desc("act_route_case", act, dtype), cpu, tmp_path)
    assert op._reference_call.entry == entry
    if target is not None:
        assert op._reference_call.params["target"] == target
    c_text = (tmp_path / "act_route_case_nn_activation_float.c").read_text()
    atol, rtol = kf.PYTORCH_TOLERANCE[np.dtype(np.float16 if dtype == "FP16" else np.float32)]
    assert f"{atol:g}" in c_text.replace("1e-05f", "1e-05") and f"{rtol:g}" in c_text


def test_nn_activation_keeps_a_pinned_exact_comparison(tmp_path) -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import OpNNActivationFloat

    desc = _act_desc("act_pinned_case", TANH, "FP16", comparison={"atol": 0.0, "rtol": 0.0})
    op = _generate(OpNNActivationFloat, desc, "cortex-m55", tmp_path)
    assert op._reference_call.entry == "tanh_lut_mve_f16"
    assert "1e-05" not in (tmp_path / "act_pinned_case_nn_activation_float.c").read_text()


@pytest.mark.parametrize("cpu,dtype,entry", [("cortex-m55", "FP32", "hard_swish_f32_cmsis"),
                                             ("cortex-m0", "FP32", "hard_swish_f32_cmsis"),
                                             ("cortex-m55", "FP16", "hard_swish_f16_mve"),
                                             ("cortex-m4", "FP16", "hard_swish_f16")])
def test_hard_swish_route(tmp_path, cpu, dtype, entry) -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_precise import OpHardSwishPrecise

    desc = {"operator": "HardSwishPrecise", "name": "hs_route_case", "tensor_dtypes": {"input": dtype, "output": dtype},
            "input_shape": [1, 1, 1, 13]}
    op = _generate(OpHardSwishPrecise, desc, cpu, tmp_path)
    assert op._reference_call.entry == entry


@pytest.mark.parametrize("cpu,target", [("cortex-m55", 0), ("cortex-m4", 1), ("cortex-m0", 2)])
def test_softmax_f32_route(tmp_path, cpu, target) -> None:
    from helia_core_tester.generation.ops.SoftmaxFunctions.softmax import OpSoftmax

    desc = {"operator": "Softmax", "name": "sm_route_case", "tensor_dtypes": {"input": "FP32", "output": "FP32"},
            "input_shape": [1, 2, 3, 7]}
    op = _generate(OpSoftmax, desc, cpu, tmp_path)
    assert op._reference_call.entry == "softmax_f32_cmsis"
    assert op._reference_call.params["target"] == target
