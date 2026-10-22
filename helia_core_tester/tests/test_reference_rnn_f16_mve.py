"""lstm_f16_mve, gru_f16_mve and svdf_f16_mve (arm_nn_lstm_step_f16, arm_nn_gru_step_f16 and arm_svdf_f16 on
Helium, as GCC -Ofast compiles them for cortex-m55) against an independent restatement of that arithmetic, bit for
bit; the stored float16 tables against the checkout's; the generators' route selection and tolerance."""

from __future__ import annotations

import json
import os
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops.LSTMFunctions.gru_unidirectional import OpGRUUnidirectional
from helia_core_tester.generation.ops.LSTMFunctions.lstm_unidirectional import OpLSTMUnidirectional
from helia_core_tester.generation.ops.SVDFunctions.svdf import OpSVDF
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.generation.reference.call import ReferenceCall

SRC = Path(__file__).resolve().parents[1] / "reference" / "src"
f16, f32 = np.float16, np.float32


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _stored(path: Path, name: str) -> list[float]:
    body = path.read_text()
    body = body[body.index(f"{name}[257] = {{"):]
    bits = [int(v, 16) for v in re.findall(r"0x([0-9A-Fa-f]{4})u", body[:body.index("};")])]
    assert len(bits) == 257
    return [float(v) for v in np.array(bits, np.uint16).view(f16)]


EXP2 = _stored(SRC / "activation" / "cmsis_f16_tables.c", "hct_cmsis_exp2_lut_f16")
TANH = _stored(SRC / "activation" / "nn_activation_float.c", "hct_cmsis_tanh_lut_f16")


# --------------------------------------------------------------- restatement

def _round(f: Fraction, dtype) -> float:
    c = dtype(float(f))
    cands = [c, np.nextafter(c, dtype(np.inf)), np.nextafter(c, dtype(-np.inf))]
    uint = np.uint16 if dtype == f16 else np.uint32
    return float(min(cands, key=lambda v: (abs(Fraction(float(v)) - f), int(np.array(v, dtype).view(uint)) & 1)))


def fma(a, b, c, dtype=f16) -> float:
    return _round(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)), dtype)


def h(v) -> float:
    """Round a float16 sum, difference or product (exact in binary64) to float16."""
    return float(f16(v))


def s(v) -> float:
    return float(f32(v))


def dot(a, b) -> float:
    lane = [0.0] * 8
    for k, (x, y) in enumerate(zip(a, b)):
        lane[k % 8] = fma(x, y, lane[k % 8])
    return h(h(h(lane[0] + lane[1]) + h(lane[2] + lane[3])) + h(h(lane[4] + lane[5]) + h(lane[6] + lane[7])))


def exp_lut(x) -> float:
    xf = min(max(float(x), -80.0), 80.0)
    t = s(f32(xf) * f32(1.44269504088896341))
    th = h(t)
    n = int(th)
    n = n - 1 if th < n else n
    idx_f = s(f32(s(f32(t) - f32(n))) * f32(256.0))
    idx = min(max(int(idx_f), 0), 255)
    frac = s(f32(idx_f) - f32(idx))
    y = fma(frac, s(f32(EXP2[idx + 1]) - f32(EXP2[idx])), EXP2[idx], f32)
    n = min(max(n, -14), 15)
    return h(s(f32(y) * f32(2.0 ** n)))


def sigmoid(x) -> float:
    if x >= 0:
        return h(s(f32(1.0) / f32(s(f32(1.0) + f32(exp_lut(-x))))))
    e = exp_lut(x)
    return h(s(f32(e) / f32(s(f32(1.0) + f32(e)))))


def tanh(x) -> float:
    if x == 0 or np.isnan(x):
        return x
    ax = abs(x)
    y = 1.0
    if not ax > 4.0:
        t = ax * 64.0
        idx = min(int(t), 255)
        y = fma(t - idx, h(TANH[idx + 1] - TANH[idx]), TANH[idx])
    return float(np.copysign(y, x))


def lstm_model(x, w, r, b, clip, time_major):
    seq = np.swapaxes(x, 0, 1) if time_major else x
    B, T, _ = seq.shape
    H = w.shape[1]
    clip = h(clip)
    hid = np.zeros((B, H))
    cell = np.zeros((B, H))
    out = np.zeros((B, T, H))
    for t in range(T):
        for bb in range(B):
            new = np.zeros(H)
            for u in range(H):
                g = []
                for gate in range(4):
                    acc = float(b[gate, u]) if b.size else 0.0
                    acc = h(acc + dot(seq[bb, t], w[gate, u]))
                    if t > 0:
                        acc = h(acc + dot(hid[bb], r[gate, u]))
                    g.append(tanh(acc) if gate == 2 else sigmoid(acc))
                c = fma(g[1], g[2], h(g[0] * cell[bb, u]))
                if clip > 0:
                    c = min(max(c, -clip), clip)
                cell[bb, u] = c
                new[u] = h(g[3] * tanh(c))
            hid[bb] = new
            out[bb, t] = new
    out = out.astype(f16)
    return np.swapaxes(out, 0, 1) if time_major else out


def prereset_sum(w, rh) -> float:
    H = len(w)
    total, k = 0.0, 0
    if H >= 8:
        lane = [0.0] * 8
        while k + 8 <= H:
            for j in range(8):
                lane[j] = h(lane[j] + h(float(w[k + j]) * rh[k + j]))
            k += 8
        total = lane[0]
        for j in range(1, 8):
            total = h(total + lane[j])
    for kk in range(k, H):
        total = fma(w[kk], rh[kk], total)
    return total


def gru_model(x, w, r, bi, bh, reset_after, time_major):
    seq = np.swapaxes(x, 0, 1) if time_major else x
    B, T, _ = seq.shape
    H = w.shape[1]
    out = np.zeros((B, T, H))

    def ip(g, xs, u):
        return h((float(bi[g, u]) if bi.size else 0.0) + dot(xs, w[g, u]))

    def hp(g, hs, u):
        acc = float(bh[g, u]) if bh.size else 0.0
        return acc if hs is None else h(acc + dot(hs, r[g, u]))

    for bb in range(B):
        prev = None
        for t in range(T):
            xs = seq[bb, t]
            stage = None
            if not reset_after and prev is not None:
                stage = [h(sigmoid(h(ip(1, xs, u) + hp(1, prev, u))) * prev[u]) for u in range(H)]
            new = []
            for u in range(H):
                z = sigmoid(h(ip(0, xs, u) + hp(0, prev, u)))
                if reset_after:
                    rg = sigmoid(h(ip(1, xs, u) + hp(1, prev, u)))
                    n_pre = fma(rg, hp(2, prev, u), ip(2, xs, u))
                else:
                    hh = float(bh[2, u]) if bh.size else 0.0
                    if stage is not None:
                        hh = h(hh + prereset_sum(r[2, u], stage))
                    n_pre = h(ip(2, xs, u) + hh)
                new.append(fma(z, 0.0 if prev is None else prev[u], h(h(1.0 - z) * tanh(n_pre))))
            prev = new
            out[bb, t] = new
    out = out.astype(f16)
    return np.swapaxes(out, 0, 1) if time_major else out


def svdf_model(x, wf, wt, bias, state, rank, lo, hi):
    S, B, _ = x.shape
    F, M = wt.shape
    U = F // rank
    st = state.astype(np.float64).copy()
    out = np.zeros((S, B, U))
    with np.errstate(over="ignore"):
        lo, hi = h(lo), h(hi)  # +-1e30 is +-inf in float16, as the kernel's bounds are
    for step in range(S):
        for bb in range(B):
            for f in range(F):
                st[bb, f, :-1] = st[bb, f, 1:].copy()
                st[bb, f, -1] = dot(x[step, bb], wf[f])
            a = [dot(wt[f], st[bb, f]) for f in range(F)]
            for u in range(U):
                if bias.size and rank == 1:
                    v = h(a[u] + float(bias[u]))
                else:
                    total = f32(bias[u]) if bias.size else f32(0.0)
                    for j in range(rank):
                        total = f32(total + f32(a[u * rank + j]))
                    v = h(total)
                out[step, bb, u] = min(max(v, lo), hi)
    return out.astype(f16), st.astype(f16)


def _lstm_ops(rng, B, T, I, H, bias, time_major):
    shape = (T, B, I) if time_major else (B, T, I)
    return {"input": rng.uniform(-1.5, 1.5, shape).astype(f16),
            "input_weights": rng.uniform(-1, 1, (4, H, I)).astype(f16),
            "recurrent_weights": rng.uniform(-1, 1, (4, H, H)).astype(f16),
            "bias": rng.uniform(-.5, .5, (4, H)).astype(f16) if bias else np.zeros((0,), f16)}


def _gru_ops(rng, B, T, I, H, bias, time_major):
    shape = (T, B, I) if time_major else (B, T, I)
    nb = (lambda: rng.uniform(-.5, .5, (3, H)).astype(f16)) if bias else (lambda: np.zeros((0,), f16))
    return {"input": rng.uniform(-1.5, 1.5, shape).astype(f16),
            "input_weights": rng.uniform(-1, 1, (3, H, I)).astype(f16),
            "recurrent_weights": rng.uniform(-1, 1, (3, H, H)).astype(f16),
            "input_bias": nb(), "recurrent_bias": nb()}


def _out(ops, H):
    return {"output": (*ops["input"].shape[:2], H)}


# ------------------------------------------------------------------- LSTM

@pytest.mark.parametrize("B,T,I,H,bias,clip,time_major", [
    (1, 3, 3, 4, True, 0.0, False),     # one partial lane block, short dots
    (2, 4, 11, 8, True, 0.75, False),   # dot tails past a full lane set, full hidden block, clip
    (2, 3, 5, 9, False, 0.0, True),     # hidden tail of one, no bias, time-major
    (1, 2, 17, 17, True, 0.3, False),   # two lane passes and a tail in both dots
])
def test_lstm_matches_the_restatement(lib, B, T, I, H, bias, clip, time_major) -> None:
    rng = np.random.default_rng([B, T, I, H])
    ops = _lstm_ops(rng, B, T, I, H, bias, time_major)
    got = lib.run("lstm_f16_mve", {"time_major": int(time_major), "cell_clip": clip}, ops, _out(ops, H))["output"]
    want = lstm_model(ops["input"], ops["input_weights"], ops["recurrent_weights"], ops["bias"], clip, time_major)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


def test_lstm_hand_checked_first_step(lib) -> None:
    """Zero weights: every gate is its bias. forget/input/output bias 0 -> sigmoid 0.5 (exp2 table entry 0 is 1);
    cell bias 1 -> tanh table entry 64 (tanh(1) = 0.76171875 in float16). cell = 0.5 * 0.76171875 = 0.380859375,
    hidden = 0.5 * tanh(0.380859375) by the table: t = 24.375, entries 24/25, so 0.5 * 0.3636... rounded."""
    ops = {"input": np.zeros((1, 1, 2), f16), "input_weights": np.zeros((4, 1, 2), f16),
           "recurrent_weights": np.zeros((4, 1, 1), f16), "bias": np.array([[0], [0], [1], [0]], f16)}
    got = lib.run("lstm_f16_mve", {"time_major": 0, "cell_clip": 0.0}, ops, {"output": (1, 1, 1)})["output"]
    y0, y1 = TANH[24], TANH[25]
    tanh_c = fma(0.375, h(y1 - y0), y0)
    assert TANH[64] == 0.76171875 and sigmoid(0.0) == 0.5
    assert float(got[0, 0, 0]) == h(0.5 * tanh_c)
    assert abs(float(got[0, 0, 0]) - 0.5 * np.tanh(0.5 * np.tanh(1.0))) < 1e-3


def test_lstm_stays_close_to_the_exact_lstm(lib) -> None:
    rng = np.random.default_rng(7)
    ops = _lstm_ops(rng, 2, 6, 4, 5, True, False)
    ops["input_weights"] = (ops["input_weights"] / 2).astype(f16)
    ops["recurrent_weights"] = (ops["recurrent_weights"] / 2).astype(f16)
    params = {"time_major": 0, "cell_clip": 0.0}
    emu = ReferenceCall("lstm_f16_mve", params, ops, _out(ops, 5))
    kf.check_close_to_exact("lstm", emu, ReferenceCall("lstm_f16", params, ops, _out(ops, 5)), atol=kf.RNN_F16_ATOL)


def test_lstm_rejects_bad_params_and_shapes(lib) -> None:
    ops = _lstm_ops(np.random.default_rng(0), 1, 2, 3, 2, True, False)

    def run(p=None, **swap):
        return lib.run("lstm_f16_mve", {"time_major": 0, "cell_clip": 0.0, **(p or {})}, {**ops, **swap},
                       {"output": (1, 2, 2)})
    run()
    assert _status(lambda: run({"time_major": 2})) == "E_PARAM"
    assert _status(lambda: run({"cell_clip": -1.0})) == "E_PARAM"
    assert _status(lambda: run({"cell_clip": float("inf")})) == "E_PARAM"
    assert _status(lambda: run(recurrent_weights=np.zeros((4, 2, 3), f16))) == "E_SHAPE"
    assert _status(lambda: run(input_weights=np.zeros((3, 2, 3), f16))) == "E_SHAPE"
    assert _status(lambda: run(bias=np.zeros((4, 3), f16))) == "E_SHAPE"


# -------------------------------------------------------------------- GRU

@pytest.mark.parametrize("reset_after", [True, False])
@pytest.mark.parametrize("B,T,I,H,bias,time_major", [
    (1, 3, 4, 3, True, False),      # scalar pre-reset chain (H < 8)
    (2, 3, 9, 8, True, True),       # vectorised pre-reset sum, no tail
    (1, 3, 5, 11, False, False),    # vectorised pre-reset sum with a fused tail, no bias
    (2, 2, 3, 17, True, False),     # two vector passes and a tail
])
def test_gru_matches_the_restatement(lib, reset_after, B, T, I, H, bias, time_major) -> None:
    rng = np.random.default_rng([B, T, I, H, int(reset_after)])
    ops = _gru_ops(rng, B, T, I, H, bias, time_major)
    params = {"time_major": int(time_major), "reset_after": int(reset_after)}
    got = lib.run("gru_f16_mve", params, ops, _out(ops, H))["output"]
    want = gru_model(ops["input"], ops["input_weights"], ops["recurrent_weights"], ops["input_bias"],
                     ops["recurrent_bias"], reset_after, time_major)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


def test_gru_prereset_sum_vectorises_only_from_eight_units() -> None:
    w = [0.1] * 7 + [0.2]
    rh = [1.0] * 8
    # Eight units: per-lane products rounded and summed left to right, not one fused chain.
    lanes = [h(0.0 + h(float(f16(v)) * 1.0)) for v in w]
    total = lanes[0]
    for v in lanes[1:]:
        total = h(total + v)
    assert prereset_sum(np.array(w, f16), rh) == total
    chain = 0.0
    for v in w[:7]:
        chain = fma(f16(v), 1.0, chain)
    assert prereset_sum(np.array(w[:7], f16), rh[:7]) == chain


def test_gru_hand_checked_first_step(lib) -> None:
    """Zero weights, candidate input bias 1: z = sigmoid(0) = 0.5, n = tanh(1) = 0.76171875 (table entry 64),
    h = (1 - 0.5) * 0.76171875 = 0.380859375 exactly."""
    ops = {"input": np.zeros((1, 1, 1), f16), "input_weights": np.zeros((3, 1, 1), f16),
           "recurrent_weights": np.zeros((3, 1, 1), f16), "input_bias": np.array([[0], [0], [1]], f16),
           "recurrent_bias": np.zeros((3, 1), f16)}
    for reset_after in (0, 1):
        got = lib.run("gru_f16_mve", {"time_major": 0, "reset_after": reset_after}, ops, {"output": (1, 1, 1)})
        assert float(got["output"][0, 0, 0]) == 0.380859375


def test_gru_stays_close_to_the_exact_gru(lib) -> None:
    rng = np.random.default_rng(3)
    ops = _gru_ops(rng, 2, 5, 4, 6, True, False)
    for key in ("input_weights", "recurrent_weights"):
        ops[key] = (ops[key] / 2).astype(f16)
    for reset_after in (0, 1):
        params = {"time_major": 0, "reset_after": reset_after}
        kf.check_close_to_exact("gru", ReferenceCall("gru_f16_mve", params, ops, _out(ops, 6)),
                                ReferenceCall("gru_f16", params, ops, _out(ops, 6)), atol=kf.RNN_F16_ATOL)


def test_gru_rejects_bad_params_and_shapes(lib) -> None:
    ops = _gru_ops(np.random.default_rng(0), 1, 2, 3, 2, True, False)

    def run(p=None, **swap):
        return lib.run("gru_f16_mve", {"time_major": 0, "reset_after": 1, **(p or {})}, {**ops, **swap},
                       {"output": (1, 2, 2)})
    run()
    assert _status(lambda: run({"reset_after": 2})) == "E_PARAM"
    assert _status(lambda: run({"time_major": -1})) == "E_PARAM"
    assert _status(lambda: run(recurrent_weights=np.zeros((3, 2, 3), f16))) == "E_SHAPE"
    assert _status(lambda: run(input_bias=np.zeros((4, 2), f16))) == "E_SHAPE"
    assert _status(lambda: run(recurrent_bias=np.zeros((3, 3), f16))) == "E_SHAPE"


# ------------------------------------------------------------------- SVDF

def _svdf_ops(rng, S, B, I, F, M, rank, bias):
    return {"input": rng.uniform(-1, 1, (S, B, I)).astype(f16), "weights_feature": rng.uniform(-1, 1, (F, I)).astype(f16),
            "weights_time": rng.uniform(-1, 1, (F, M)).astype(f16),
            "bias": rng.uniform(-.5, .5, (F // rank,)).astype(f16) if bias else np.zeros((0,), f16),
            "state": rng.uniform(-.5, .5, (B, F, M)).astype(f16)}


def _svdf_shapes(ops, rank):
    S, B, _ = ops["input"].shape
    return {"output": (S, B, ops["weights_feature"].shape[0] // rank), "state_out": ops["state"].shape}


@pytest.mark.parametrize("S,B,I,F,M,rank,bias,lo,hi", [
    (2, 1, 4, 4, 3, 1, True, -1e30, 1e30),    # rank 1 with bias: the float16 vector add
    (2, 1, 4, 4, 3, 1, False, -1e30, 1e30),   # no bias: the float32 sum from zero
    (3, 2, 9, 6, 10, 2, True, -1e30, 1e30),   # rank 2: float32 sum from the bias; dot tails past 8
    (2, 2, 17, 6, 1, 3, False, -0.5, 0.5),    # single-slot memory, clamped output
])
def test_svdf_matches_the_restatement(lib, S, B, I, F, M, rank, bias, lo, hi) -> None:
    rng = np.random.default_rng([S, B, I, F, M, rank])
    ops = _svdf_ops(rng, S, B, I, F, M, rank, bias)
    got = lib.run("svdf_f16_mve", {"rank": rank, "activation_min": lo, "activation_max": hi}, ops,
                  _svdf_shapes(ops, rank))
    out, state = svdf_model(ops["input"], ops["weights_feature"], ops["weights_time"], ops["bias"], ops["state"],
                            rank, lo, hi)
    np.testing.assert_array_equal(got["output"].view(np.uint16), out.view(np.uint16))
    np.testing.assert_array_equal(got["state_out"].view(np.uint16), state.view(np.uint16))


def test_svdf_hand_checked_step(lib) -> None:
    """features [1, 1] . x [1, 2] = 3 enters the newest slot; state [0.5, 0.25, 0] shifts to [0.25, 0, 3];
    time weights [1, 2, 4] give 0.25 + 0 + 12 = 12.25; bias 0.5 -> 12.75."""
    ops = {"input": np.array([[[1, 2]]], f16), "weights_feature": np.array([[1, 1]], f16),
           "weights_time": np.array([[1, 2, 4]], f16), "bias": np.array([0.5], f16),
           "state": np.array([[[0.5, 0.25, 0]]], f16)}
    got = lib.run("svdf_f16_mve", {"rank": 1, "activation_min": -1e30, "activation_max": 1e30}, ops,
                  _svdf_shapes(ops, 1))
    assert float(got["output"][0, 0, 0]) == 12.75
    np.testing.assert_array_equal(got["state_out"].reshape(-1), np.array([0.25, 0, 3], f16))


def test_svdf_bound_guard_accepts_the_variant_and_rejects_a_deviation(lib) -> None:
    ops = _svdf_ops(np.random.default_rng(1), 3, 2, 9, 6, 10, 2, True)
    call = ReferenceCall("svdf_f16_mve", {"rank": 2, "activation_min": -1e30, "activation_max": 1e30}, ops,
                         _svdf_shapes(ops, 2))
    out = call.run()["output"]
    kf.check_svdf_within_bound("svdf", out, call)
    bad = out.astype(np.float64)
    bad[1, 0, 0] += 1.0
    with pytest.raises(kf.KernelDeviationError, match="SVDF step 1"):
        kf.check_svdf_within_bound("svdf", bad, call)


def test_svdf_rejects_bad_params_and_shapes(lib) -> None:
    ops = _svdf_ops(np.random.default_rng(0), 1, 1, 4, 4, 3, 2, True)
    params = {"rank": 2, "activation_min": -1e30, "activation_max": 1e30}

    def run(p=None, shapes=None, **swap):
        return lib.run("svdf_f16_mve", {**params, **(p or {})}, {**ops, **swap}, shapes or _svdf_shapes(ops, 2))
    run()
    assert _status(lambda: run({"rank": 0})) == "E_PARAM"
    assert _status(lambda: run({"rank": 3})) == "E_PARAM"
    assert _status(lambda: run({"activation_min": 1.0, "activation_max": -1.0})) == "E_PARAM"
    assert _status(lambda: run(bias=np.zeros((3,), f16))) == "E_SHAPE"
    assert _status(lambda: run(weights_time=np.zeros((3, 3), f16))) == "E_SHAPE"
    assert _status(lambda: run(state=np.zeros((1, 4, 2), f16))) == "E_SHAPE"


# ----------------------------------------------------------------- tables

def test_stored_tables_are_float16_samples() -> None:
    i = np.arange(257, dtype=np.float64)
    assert EXP2 == [float(v) for v in (2.0 ** (i / 256)).astype(f16)]
    assert TANH == [float(v) for v in np.tanh(4 * i / 256).astype(f16)]


def test_stored_tables_are_the_checkouts_tables() -> None:
    root = os.environ.get("CMSIS_NN_ROOT")
    src = Path(root) / "Source" / "NNSupportFunctions" / "arm_nntables_flt.c" if root else None
    if src is None or not src.exists():
        pytest.skip("no ns-cmsis-nn checkout")
    assert _stored(src, "arm_nn_exp2_lut_f16") == EXP2
    assert _stored(src, "arm_nn_tanh_lut_f16") == TANH


# -------------------------------------------------------- route selection

def test_rnn_call_builder_maps_only_the_float16_recurrent_entries() -> None:
    ops = _gru_ops(np.random.default_rng(0), 1, 1, 2, 2, True, False)
    exact = ReferenceCall("gru_f16", {"time_major": 0, "reset_after": 1}, ops, _out(ops, 2))
    emu = kf.rnn_f16_mve_call(exact)
    assert emu.entry == "gru_f16_mve" and emu.params == exact.params and emu.output_shapes == exact.output_shapes
    gru32 = {k: v.astype(f32) for k, v in ops.items()}
    with pytest.raises(ValueError, match="no float16 Helium recurrent variant"):
        kf.rnn_f16_mve_call(ReferenceCall("gru_f32", {"time_major": 0, "reset_after": 1}, gru32, _out(ops, 2)))


_DESCS = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}
_ROUTED = [("lstm_unidirectional_float_clip_tail9_time_major_f16", OpLSTMUnidirectional, "lstm_f16"),
           ("lstm_unidirectional_float_stream_f16", OpLSTMUnidirectional, "lstm_f16"),
           ("gru_unidirectional_float_pre_reset_f16", OpGRUUnidirectional, "gru_f16"),
           ("gru_unidirectional_float_stream_f16", OpGRUUnidirectional, "gru_f16"),
           ("svdf_float_rank2_f16", OpSVDF, "svdf_f16")]


def _sidecar(out: Path) -> dict:
    return json.loads(next(out.rglob("*.sidecar.json")).read_text())


@pytest.mark.parametrize("name,op_cls,exact", _ROUTED)
def test_float16_cases_on_helium_take_the_kernel_route_at_pytorch_tolerance(name, op_cls, exact, tmp_path) -> None:
    op = op_cls(_DESCS[name], 1234, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    assert op.reference.entry == kf.RNN_F16_MVE_ENTRIES[exact]
    ctx = _sidecar(tmp_path)
    assert json.dumps(ctx).count('"validation_atol": 1e-05') == 1
    assert '"validation_rtol": 0.001' in json.dumps(ctx)


@pytest.mark.parametrize("name,op_cls,exact", _ROUTED)
def test_float16_cases_off_helium_keep_the_exact_golden(name, op_cls, exact, tmp_path) -> None:
    op = op_cls(_DESCS[name], 1234, target_cpu="cortex-m4")
    op.generate_c_files(tmp_path)
    assert op.reference.entry == exact
    assert '"validation_atol": 1e-05' not in json.dumps(_sidecar(tmp_path))


@pytest.mark.parametrize("name,op_cls", [("lstm_unidirectional_error_null_input_f16", OpLSTMUnidirectional),
                                         ("gru_unidirectional_error_null_input_f16", OpGRUUnidirectional),
                                         ("svdf_fault_zero_rank_f16", OpSVDF)])
def test_status_only_float16_cases_record_no_golden(name, op_cls, tmp_path) -> None:
    op = op_cls(_DESCS[name], 1234, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    assert op.status_only() and op.reference is None
