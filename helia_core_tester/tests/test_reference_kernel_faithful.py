"""Kernel-faithful float variants against independent restatements of the ns-cmsis-nn arithmetic they
model (matmul_nt_t_f16_mve, lstm_f32_cmsis), the stored activation tables against the checkout's own,
and the helpers that choose a target's leg and guard an emulated golden."""

from __future__ import annotations

import os
import re
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings

UNBOUNDED = (-1.0e30, 1.0e30)


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _round(f: Fraction, dtype) -> float:
    """The nearest `dtype` value to the rational f, ties to even."""
    c = dtype(float(f))
    cands = [c, np.nextafter(c, dtype(np.inf)), np.nextafter(c, dtype(-np.inf))]
    uint = np.uint16 if dtype == np.float16 else np.uint32
    return float(min(cands, key=lambda v: (abs(Fraction(float(v)) - f), int(np.array(v, dtype).view(uint)) & 1)))


def fma(a, b, c, dtype) -> float:
    return _round(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)), dtype)


def h(v) -> float:
    return float(np.float16(v))


# ------------------------------------------------------------- matmul f16

def lanes(lhs, rhs, k0, n):
    lane = [0.0] * 8
    for k in range(n):
        lane[k % 8] = fma(lhs[k0 + k], rhs[k0 + k], lane[k % 8], np.float16)
    return lane


def reduce16(l):
    return h(h(h(l[0] + l[1]) + h(l[2] + l[3])) + h(h(l[4] + l[5]) + h(l[6] + l[7])))


def fold(lhs, rhs, bias):
    acc = [np.float32(0)] * 4
    for k0 in range(0, len(lhs), 256):
        l = lanes(lhs, rhs, k0, min(256, len(lhs) - k0))
        for j in range(4):
            pair = np.float32(np.float32(l[2 * j]) + np.float32(l[2 * j + 1]))
            acc[j] = pair if k0 == 0 else np.float32(acc[j] + pair)
    total = np.float32(np.float32(acc[0] + acc[1]) + np.float32(acc[2] + acc[3]))
    return h(np.float32(np.float32(bias) + total))


def gather_columns(K, N):
    """Columns arm_nn_mat_mult_nt_t_f16 covers with 16/8/4-wide gathers (K < 32)."""
    c = N - N % 16
    c += (N - c) - (N - c) % 8
    c += (N - c) - (N - c) % 4
    return c


def matmul_model(lhs, rhs, bias, route, acc16):
    B, M, K = lhs.shape
    N = rhs.shape[1]
    out = np.zeros((B, M, N), np.float16)
    folds = route == 0 and not acc16 and K > 32
    gathered = gather_columns(K, N) if route == 0 and K < 32 else 0
    for b in range(B):
        for r in range(M):
            x = [float(v) for v in lhs[b, r]]
            for c in range(N):
                w = [float(v) for v in rhs[b, c]]
                bc = float(bias[c]) if bias.size else 0.0
                if route >= 2:
                    acc = 0.0
                    for k in range(K):
                        acc = fma(x[k], w[k], acc, np.float16) if route == 2 else h(acc + h(x[k] * w[k]))
                elif c < gathered:
                    acc = bc
                    for k in range(K):
                        acc = fma(w[k], x[k], acc, np.float16)
                elif folds:
                    acc = fold(x, w, bc)
                else:
                    acc = h(bc + reduce16(lanes(x, w, 0, K)))
                out[b, r, c] = acc
    return out


def _mm(lib, lhs, rhs, bias, route=0, acc16=0, activation=UNBOUNDED):
    return lib.run("matmul_nt_t_f16_mve", {"route": route, "acc16": acc16, "activation_min": activation[0],
                                           "activation_max": activation[1]},
                   {"lhs": lhs, "rhs": rhs, "bias": bias}, {"output": (lhs.shape[0], lhs.shape[1], rhs.shape[1])})[
        "output"]


@pytest.mark.parametrize("K,N", [(3, 5), (8, 8), (20, 20), (31, 21), (32, 4), (33, 6), (40, 12), (64, 3), (300, 5)])
@pytest.mark.parametrize("acc16", [0, 1])
def test_nt_t_route_matches_the_kernel_restatement(lib, K, N, acc16) -> None:
    rng = np.random.default_rng([K, N, acc16])
    lhs = rng.uniform(-1, 1, (2, 2, K)).astype(np.float16)
    rhs = rng.uniform(-1, 1, (2, N, K)).astype(np.float16)
    bias = rng.uniform(-0.5, 0.5, N).astype(np.float16)
    np.testing.assert_array_equal(_mm(lib, lhs, rhs, bias, 0, acc16), matmul_model(lhs, rhs, bias, 0, acc16))


@pytest.mark.parametrize("route", [1, 2, 3])
@pytest.mark.parametrize("K", [2, 9, 40])
def test_batch_matmul_routes_match_the_kernel_restatement(lib, route, K) -> None:
    rng = np.random.default_rng([route, K])
    lhs = rng.uniform(-1, 1, (1, 3, K)).astype(np.float16)
    rhs = rng.uniform(-1, 1, (1, 5, K)).astype(np.float16)
    none = np.zeros((0,), np.float16)
    np.testing.assert_array_equal(_mm(lib, lhs, rhs, none, route), matmul_model(lhs, rhs, none, route, 0))


def test_fold_keeps_what_float16_lanes_lose(lib) -> None:
    # K = 264: every lane opens at 1024 and then adds 0.25s, which a float16 lane at 1024 rounds away.
    # Folding restarts the lanes at element 256, so the last eight 0.25s survive in float32; with the
    # bias cancelling 8192 the fold leaves 2, float16 lanes throughout (acc16) leave 0.
    lhs = np.full((1, 1, 264), 0.25, np.float16)
    lhs[0, 0, :8] = 1024.0
    rhs = np.ones((1, 1, 264), np.float16)
    bias = np.array([-8192.0], np.float16)
    assert float(_mm(lib, lhs, rhs, bias, 0, 0)[0, 0, 0]) == 2.0
    assert float(_mm(lib, lhs, rhs, bias, 0, 1)[0, 0, 0]) == 0.0
    # K = 32 never folds: both entries agree.
    k32 = np.ascontiguousarray(lhs[:, :, :32])
    assert _mm(lib, k32, rhs[:, :, :32], bias, 0, 0)[0, 0, 0] == _mm(lib, k32, rhs[:, :, :32], bias, 0, 1)[0, 0, 0]


def test_scalar_route_fuses_exactly_where_rounding_the_product_differs(lib) -> None:
    # Values where a*b rounded to float16 first changes the sum: route 2 and route 3 differ.
    lhs = np.array([[[0.106201171875, 0.98046875]]], np.float16)
    rhs = np.array([[[1.0009765625, -0.10845947265625]]], np.float16)
    none = np.zeros((0,), np.float16)
    fused = _mm(lib, lhs, rhs, none, 2)[0, 0, 0]
    unfused = _mm(lib, lhs, rhs, none, 3)[0, 0, 0]
    assert fused == matmul_model(lhs, rhs, none, 2, 0)[0, 0, 0]
    assert unfused == matmul_model(lhs, rhs, none, 3, 0)[0, 0, 0]


def test_matmul_clamps_finite_values_and_passes_nan(lib) -> None:
    lhs = np.array([[[1.0, np.nan], [2.0, 0.0]]], np.float16)
    rhs = np.array([[[3.0, 0.0]]], np.float16)
    out = _mm(lib, lhs, rhs, np.zeros(0, np.float16), 1, 0, activation=(-1.0, 4.0))
    assert np.isnan(out[0, 0, 0]) and float(out[0, 1, 0]) == 4.0


def test_matmul_rejects_bad_params_and_shapes(lib) -> None:
    lhs = np.zeros((1, 2, 3), np.float16)
    rhs = np.zeros((1, 4, 3), np.float16)
    bias = np.zeros(4, np.float16)
    run = lambda **kw: _mm(lib, kw.get("lhs", lhs), kw.get("rhs", rhs), kw.get("bias", bias), kw.get("route", 0),
                           kw.get("acc16", 0), kw.get("activation", UNBOUNDED))
    run()
    assert _status(lambda: run(route=4)) == "E_PARAM"
    assert _status(lambda: run(acc16=2)) == "E_PARAM"
    assert _status(lambda: run(activation=(1.0, -1.0))) == "E_PARAM"
    assert _status(lambda: run(route=1)) == "E_PARAM"  # the batch matmul routes take no bias
    assert _status(lambda: run(rhs=np.zeros((1, 4, 2), np.float16))) == "E_SHAPE"
    assert _status(lambda: run(bias=np.zeros(3, np.float16))) == "E_SHAPE"
    assert _status(lambda: run(lhs=np.zeros((2, 2, 3), np.float16))) == "E_SHAPE"
    with pytest.raises(ValueError, match=r"\[B,M,K\]"):
        kf.matmul_f16_mve_call(np.zeros((2, 3)), np.zeros((1, 2, 3)), None, route=0, acc16=False,
                               activation=UNBOUNDED)


# ------------------------------------------------------------- lstm f32

def _tables():
    src = (Path(__file__).resolve().parents[1] / "reference" / "src" / "activation" / "cmsis_f32_tables.c").read_text()

    def table(name):
        body = src[src.index(name):]
        return [struct.unpack("<f", struct.pack("<I", int(v, 16)))[0]
                for v in re.findall(r"0x([0-9a-f]{8})u", body[:body.index("};")])]
    return table("hct_cmsis_exp2_lut_f32_bits"), table("hct_cmsis_tanh_lut_f32_bits")


EXP2, TANH = _tables()
f32 = np.float32


def madd(fused, a, b, c):
    return f32(fma(a, b, c, np.float32)) if fused else f32(f32(f32(a) * f32(b)) + f32(c))


def exp_lut(fused, x):
    x = f32(min(max(x, f32(-80.0)), f32(80.0)))
    t = f32(x * f32(1.44269504088896341))
    n = int(t)
    n = n - 1 if t < f32(n) else n
    frac_part = f32(t - f32(n))
    idx_f = f32(frac_part * f32(256.0))
    idx = min(max(int(idx_f), 0), 255)
    frac = f32(idx_f - f32(idx))
    y = madd(fused, f32(f32(EXP2[idx + 1]) - f32(EXP2[idx])), frac, f32(EXP2[idx]))
    n = min(max(n, -126), 127)
    return f32(y * f32(2.0 ** n))


def sigmoid(fused, x):
    if x >= 0:
        return f32(f32(1.0) / f32(f32(1.0) + exp_lut(fused, f32(-x))))
    e = exp_lut(fused, x)
    return f32(e / f32(f32(1.0) + e))


def tanh_lut(fused, x):
    ax = f32(abs(x))
    if not ax < 6.0:
        return f32(np.copysign(1.0, x))
    t = f32(ax * f32(64.0))
    idx = min(int(t), 383)
    frac = f32(t - f32(idx))
    return f32(np.copysign(madd(fused, f32(f32(TANH[idx + 1]) - f32(TANH[idx])), frac, f32(TANH[idx])), x))


def lstm_model(target, x, w, r, b, clip):
    """arm_nn_lstm_step_f32 on the FPU-scalar (1) or soft-float (2) leg, batch-major."""
    fused = target != 2
    B, T, I = x.shape
    H = w.shape[1]
    hid = np.zeros((B, H), np.float32)
    cell = np.zeros((B, H), np.float32)
    out = np.zeros((B, T, H), np.float32)
    for t in range(T):
        for bb in range(B):
            new = np.zeros(H, np.float32)
            for u in range(H):
                g = []
                for gate in range(4):
                    acc = f32(b[gate, u]) if b.size else f32(0)
                    d = f32(0)
                    for k in range(I):
                        d = madd(fused, x[bb, t, k], w[gate, u, k], d)
                    acc = f32(acc + d)
                    if t > 0:
                        d = f32(0)
                        for k in range(H):
                            d = madd(fused, hid[bb, k], r[gate, u, k], d)
                        acc = f32(acc + d)
                    g.append(tanh_lut(fused, acc) if gate == 2 else sigmoid(fused, acc))
                c = madd(True, g[0], cell[bb, u], f32(g[1] * g[2])) if fused else f32(
                    f32(g[0] * cell[bb, u]) + f32(g[1] * g[2]))
                if clip > 0:
                    c = f32(min(max(c, -clip), clip))
                cell[bb, u] = c
                new[u] = f32(g[3] * tanh_lut(fused, c))
                out[bb, t, u] = new[u]
            hid[bb] = new
    return out


@pytest.mark.parametrize("target", [1, 2])
@pytest.mark.parametrize("clip,bias", [(0.0, True), (0.5, True), (0.0, False)])
def test_lstm_scalar_legs_match_the_kernel_restatement(lib, target, clip, bias) -> None:
    rng = np.random.default_rng([target, int(clip * 10), int(bias)])
    x = rng.uniform(-2, 2, (2, 4, 3)).astype(np.float32)
    w = rng.uniform(-1.5, 1.5, (4, 3, 3)).astype(np.float32)
    r = rng.uniform(-1.5, 1.5, (4, 3, 3)).astype(np.float32)
    b = rng.uniform(-0.5, 0.5, (4, 3)).astype(np.float32) if bias else np.zeros((0,), np.float32)
    got = lib.run("lstm_f32_cmsis", {"time_major": 0, "cell_clip": clip, "target": target},
                  {"input": x, "input_weights": w, "recurrent_weights": r, "bias": b}, {"output": (2, 4, 3)})
    np.testing.assert_array_equal(got["output"], lstm_model(target, x, w, r, b, clip))


@pytest.mark.parametrize("target", [0, 1, 2])
def test_lstm_variant_stays_close_to_the_exact_lstm(lib, target) -> None:
    rng = np.random.default_rng(target)
    ins = {"input": rng.uniform(-1, 1, (1, 6, 4)).astype(np.float32),
           "input_weights": rng.uniform(-.5, .5, (4, 5, 4)).astype(np.float32),
           "recurrent_weights": rng.uniform(-.5, .5, (4, 5, 5)).astype(np.float32),
           "bias": rng.uniform(-.25, .25, (4, 5)).astype(np.float32)}
    cmsis = lib.run("lstm_f32_cmsis", {"time_major": 0, "cell_clip": 0.0, "target": target}, ins, {"output": (1, 6, 5)})
    exact = lib.run("lstm_f32", {"time_major": 0, "cell_clip": 0.0}, ins, {"output": (1, 6, 5)})
    diff = np.abs(cmsis["output"].astype(np.float64) - exact["output"])
    assert 0 < diff.max() < 1e-4  # the tables approximate; they never stray far


def test_lstm_variant_rejects_bad_params_and_shapes(lib) -> None:
    ins = {"input": np.zeros((1, 2, 3), np.float32), "input_weights": np.zeros((4, 2, 3), np.float32),
           "recurrent_weights": np.zeros((4, 2, 2), np.float32), "bias": np.zeros((0,), np.float32)}
    run = lambda p=None, i=ins: lib.run("lstm_f32_cmsis", {"time_major": 0, "cell_clip": 0.0, "target": 0, **(p or {})},
                                        i, {"output": (1, 2, 2)})
    run()
    assert _status(lambda: run({"target": 3})) == "E_PARAM"
    assert _status(lambda: run({"time_major": 2})) == "E_PARAM"
    assert _status(lambda: run({"cell_clip": -1.0})) == "E_PARAM"
    assert _status(lambda: run(i=dict(ins, recurrent_weights=np.zeros((4, 2, 3), np.float32)))) == "E_SHAPE"
    assert _status(lambda: run(i=dict(ins, bias=np.zeros((4, 3), np.float32)))) == "E_SHAPE"


def test_stored_tables_are_the_checkouts_tables() -> None:
    root = os.environ.get("CMSIS_NN_ROOT")
    src = Path(root) / "Source" / "NNSupportFunctions" / "arm_nntables_flt.c" if root else None
    if src is None or not src.exists():
        pytest.skip("no ns-cmsis-nn checkout")
    text = src.read_text()

    def table(name, n):
        body = text[text.index(f"{name}[{n}] = {{"):]
        vals = [v.strip().rstrip("fF") for v in body[body.index("{") + 1:body.index("};")].split(",") if v.strip()]
        return [float(np.float32(float(v))) for v in vals]
    assert table("arm_nn_exp2_lut_f32", 257) == [float(v) for v in EXP2]
    assert table("arm_nn_tanh_lut_f32", 385) == [float(v) for v in TANH]


# ------------------------------------------------------------- helpers

@pytest.mark.parametrize("cpu,mve16,leg", [("cortex-m55", True, 0), ("cortex-m4", False, 1), ("cortex-m0", False, 2)])
def test_targets_choose_their_leg(cpu, mve16, leg) -> None:
    assert kf.mve_float16_target(cpu) is mve16
    assert kf.float_target(cpu) == leg


def test_tolerances_are_pytorchs_assert_close_defaults() -> None:
    assert kf.tolerance_context(np.float16)["validation_atol"] == 1e-5
    assert kf.tolerance_context(np.float16)["validation_rtol"] == 1e-3
    assert kf.tolerance_context(np.float32)["validation_rtol"] == 1.3e-6


def test_bound_guard_rejects_an_emulation_outside_the_rounding_bound() -> None:
    lhs = np.full((1, 1, 4), 0.5, np.float16)
    rhs = np.full((1, 1, 4), 0.5, np.float16)
    kf.check_within_bound("ok", np.array([[[1.0]]], np.float16), lhs, rhs, None, np.float16)
    with pytest.raises(kf.KernelDeviationError, match="beyond the rounding bound"):
        kf.check_within_bound("bad", np.array([[[1.01]]], np.float16), lhs, rhs, None, np.float16)
