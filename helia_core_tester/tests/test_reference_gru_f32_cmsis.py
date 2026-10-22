"""gru_f32_cmsis (arm_nn_gru_step_f32 as GCC -Ofast compiles it for each float32 target) against an independent
restatement of that arithmetic, bit for bit; the shared float32 sigmoid/tanh helpers through it; the generator's
route selection and tolerance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops.LSTMFunctions.gru_unidirectional import OpGRUUnidirectional
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.generation.reference.call import ReferenceCall
from helia_core_tester.tests.test_reference_kernel_faithful import fma, madd, sigmoid, tanh_lut

f32 = np.float32
HELIUM, FPU, SOFT = 0, 1, 2


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def fma32(a, b, c):
    return f32(fma(a, b, c, np.float32))


# --------------------------------------------------------------- restatement

def proj(target, v, w, bias):
    """bias + v . w as the kernel's dot computes it on `target`."""
    if target == HELIUM:
        lane = [f32(0.0)] * 4
        for k in range(len(v)):
            lane[k % 4] = fma32(v[k], w[k], lane[k % 4])
        return f32(f32(f32(f32(lane[0] + lane[1]) + bias) + lane[2]) + lane[3])
    acc = f32(0.0)
    for k in range(len(v)):
        acc = madd(target != SOFT, v[k], w[k], acc)
    return f32(bias + acc)


def prereset_sum(target, w, rh):
    H = len(w)
    total, k = f32(0.0), 0
    if target == HELIUM and H >= 4:
        lane = [f32(0.0)] * 4
        while k + 4 <= H:
            for j in range(4):
                lane[j] = f32(lane[j] + f32(f32(w[k + j]) * f32(rh[k + j])))
            k += 4
        total = f32(f32(f32(lane[0] + lane[1]) + lane[2]) + lane[3])
    for k in range(k, H):
        total = madd(target != SOFT, w[k], rh[k], total)
    return total


def gru_model(target, x, w, r, bi, bh, reset_after):
    """Batch-major x [B,T,I]; w [3,H,I], r [3,H,H]; bi/bh [3,H] or empty; gates z, r, n."""
    B, T, _ = x.shape
    H = w.shape[1]
    fused = target != SOFT
    bias_i = (lambda g, u: f32(bi[g, u])) if bi.size else (lambda g, u: f32(0.0))
    bias_h = (lambda g, u: f32(bh[g, u])) if bh.size else (lambda g, u: f32(0.0))
    out = np.zeros((B, T, H), f32)
    for b in range(B):
        h = None
        for t in range(T):
            xt = x[b, t]
            ip = lambda g, u: proj(target, xt, w[g, u], bias_i(g, u))  # noqa: E731
            hp = lambda g, u: bias_h(g, u) if h is None else proj(target, h, r[g, u], bias_h(g, u))  # noqa: E731
            stage = None
            if not reset_after and h is not None:
                stage = [f32(sigmoid(fused, f32(hp(1, u) + ip(1, u))) * h[u]) for u in range(H)]
            new = []
            for u in range(H):
                z = sigmoid(fused, f32(hp(0, u) + ip(0, u)))
                xh = ip(2, u)
                if reset_after:
                    rr = sigmoid(fused, f32(hp(1, u) + ip(1, u)))
                    n_pre = f32(xh + f32(rr * hp(2, u))) if target == SOFT else fma32(rr, hp(2, u), xh)
                else:
                    hh = bias_h(2, u)
                    if stage is not None:
                        hh = f32(hh + prereset_sum(target, r[2, u], stage))
                    n_pre = f32(xh + hh)
                cand = tanh_lut(fused, n_pre)
                new.append(fma32(z, f32(0.0) if h is None else h[u], f32(f32(1.0 - z) * cand)))
            h = new
            out[b, t] = new
    return out


def _ops(rng, B, T, I, H, bias, time_major=False):
    shape = (T, B, I) if time_major else (B, T, I)
    return {"input": rng.uniform(-1, 1, shape).astype(f32),
            "input_weights": rng.uniform(-0.5, 0.5, (3, H, I)).astype(f32),
            "recurrent_weights": rng.uniform(-0.5, 0.5, (3, H, H)).astype(f32),
            "input_bias": rng.uniform(-0.25, 0.25, (3, H)).astype(f32) if bias else np.zeros((0,), f32),
            "recurrent_bias": rng.uniform(-0.25, 0.25, (3, H)).astype(f32) if bias else np.zeros((0,), f32)}


def _out(ops, H):
    return {"output": (*ops["input"].shape[:2], H)}


def _run(lib, ops, H, target, reset_after, time_major=False):
    params = {"time_major": int(time_major), "reset_after": int(reset_after), "target": target}
    return lib.run("gru_f32_cmsis", params, ops, _out(ops, H))["output"]


@pytest.mark.parametrize("target", [HELIUM, FPU, SOFT])
@pytest.mark.parametrize("reset_after", [True, False])
@pytest.mark.parametrize("B,T,I,H,bias", [
    (1, 3, 4, 3, True),    # pre-reset chain (H < 4), dot tails only
    (2, 3, 5, 4, True),    # one vector pass of the Helium pre-reset sum, no tail
    (1, 3, 3, 6, False),   # vector pass plus a fused tail, no bias
    (2, 2, 9, 9, True),    # two vector passes and a tail
])
def test_gru_matches_the_restatement(lib, target, reset_after, B, T, I, H, bias) -> None:
    ops = _ops(np.random.default_rng([B, T, I, H, int(reset_after), target]), B, T, I, H, bias)
    got = _run(lib, ops, H, target, reset_after)
    want = gru_model(target, ops["input"], ops["input_weights"], ops["recurrent_weights"], ops["input_bias"],
                     ops["recurrent_bias"], reset_after)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("target", [HELIUM, FPU, SOFT])
def test_time_major_is_a_layout_only(lib, target) -> None:
    ops = _ops(np.random.default_rng(7), 2, 3, 4, 5, True)
    batch_major = _run(lib, ops, 5, target, True)
    tm = dict(ops, input=np.ascontiguousarray(ops["input"].transpose(1, 0, 2)))
    np.testing.assert_array_equal(_run(lib, tm, 5, target, True, time_major=True).transpose(1, 0, 2), batch_major)


def test_targets_differ_where_the_kernel_legs_differ(lib) -> None:
    ops = _ops(np.random.default_rng(11), 2, 4, 9, 9, True)
    outs = [_run(lib, ops, 9, t, False) for t in (HELIUM, FPU, SOFT)]
    assert not np.array_equal(outs[0], outs[1]) and not np.array_equal(outs[1], outs[2])


def test_helium_prereset_sum_vectorises_only_from_four_units() -> None:
    w = [f32(0.1), f32(0.3), f32(0.7), f32(0.9)]
    rh = [f32(1.0 / 3.0)] * 4
    lanes = [f32(f32(0.0) + f32(a * b)) for a, b in zip(w, rh)]
    assert prereset_sum(HELIUM, w, rh) == f32(f32(f32(lanes[0] + lanes[1]) + lanes[2]) + lanes[3])
    chain = f32(0.0)
    for a, b in zip(w[:3], rh[:3]):
        chain = fma32(a, b, chain)
    assert prereset_sum(HELIUM, w[:3], rh[:3]) == chain == prereset_sum(FPU, w[:3], rh[:3])


def test_gru_hand_checked_first_step(lib) -> None:
    """Zero weights, candidate input bias 1: z = sigmoid(0) = 1 / (1 + 2^0) = 0.5 exactly, n = tanh table entry 64
    (t = 64, frac 0), h = (1 - 0.5) * n, on every target and both forms."""
    ops = {"input": np.zeros((1, 1, 1), f32), "input_weights": np.zeros((3, 1, 1), f32),
           "recurrent_weights": np.zeros((3, 1, 1), f32), "input_bias": np.array([[0], [0], [1]], f32),
           "recurrent_bias": np.zeros((3, 1), f32)}
    entry64 = tanh_lut(True, f32(1.0))
    assert abs(float(entry64) - np.tanh(1.0)) < 1e-7
    for target in (HELIUM, FPU, SOFT):
        for reset_after in (0, 1):
            assert _run(lib, ops, 1, target, reset_after)[0, 0, 0] == f32(0.5 * entry64)


def test_helium_nan_candidate_is_the_all_ones_nan(lib) -> None:
    ops = _ops(np.random.default_rng(5), 1, 2, 3, 3, True)
    ops["input"][0, 0, 1] = np.nan
    for target in (HELIUM, FPU, SOFT):
        got = _run(lib, ops, 3, target, True)
        assert np.isnan(got).all()
        if target == HELIUM:
            assert set(got[0, 0].view(np.uint32).tolist()) == {0xFFFFFFFF}


def test_saturating_inputs_stay_finite(lib) -> None:
    ops = _ops(np.random.default_rng(6), 1, 3, 3, 4, True)
    ops["input"][0, 1, 0] = np.inf
    for target in (HELIUM, FPU, SOFT):
        assert np.isfinite(_run(lib, ops, 4, target, True)).all()


@pytest.mark.parametrize("target", [HELIUM, FPU, SOFT])
def test_gru_stays_close_to_the_exact_gru(lib, target) -> None:
    ops = _ops(np.random.default_rng(3), 2, 5, 4, 6, True)
    for reset_after in (0, 1):
        exact = ReferenceCall("gru_f32", {"time_major": 0, "reset_after": reset_after}, ops, _out(ops, 6))
        emu = ReferenceCall("gru_f32_cmsis", {"time_major": 0, "reset_after": reset_after, "target": target}, ops,
                            _out(ops, 6))
        kf.check_close_to_exact("gru", emu, exact, atol=kf.GRU_F32_ATOL)
        assert not np.array_equal(emu.output(), exact.output())


def test_gru_rejects_bad_params_and_shapes(lib) -> None:
    ops = _ops(np.random.default_rng(0), 1, 2, 3, 2, True)

    def run(p=None, **swap):
        params = {"time_major": 0, "reset_after": 1, "target": 0, **(p or {})}
        return lib.run("gru_f32_cmsis", params, {**ops, **swap}, _out(ops, 2))

    assert _status(lambda: run({"target": 3})) == "E_PARAM"
    assert _status(lambda: run({"target": -1})) == "E_PARAM"
    assert _status(lambda: run({"reset_after": 2})) == "E_PARAM"
    assert _status(lambda: run({"time_major": -1})) == "E_PARAM"
    assert _status(lambda: run(recurrent_weights=np.zeros((3, 2, 3), f32))) == "E_SHAPE"
    assert _status(lambda: run(input_weights=np.zeros((2, 2, 3), f32))) == "E_SHAPE"
    assert _status(lambda: run(input_bias=np.zeros((4, 2), f32))) == "E_SHAPE"
    assert _status(lambda: run(recurrent_bias=np.zeros((3, 3), f32))) == "E_SHAPE"
    assert _status(lambda: run(input=np.zeros((1, 2, 4), f32))) == "E_SHAPE"
    with pytest.raises(TypeError, match="must be float32"):
        run(input=np.zeros((1, 2, 3), np.float16))


def test_zero_steps_write_nothing(lib) -> None:
    ops = _ops(np.random.default_rng(0), 1, 0, 3, 2, True)
    assert _run(lib, ops, 2, HELIUM, True).shape == (1, 0, 2)


# -------------------------------------------------------- route selection

def test_call_builder_maps_gru_f32_to_the_target_leg() -> None:
    ops = _ops(np.random.default_rng(0), 1, 1, 2, 2, True)
    exact = ReferenceCall("gru_f32", {"time_major": 0, "reset_after": 1}, ops, _out(ops, 2))
    for cpu, target in (("cortex-m55", HELIUM), ("cortex-m4", FPU), ("cortex-m0", SOFT)):
        emu = kf.gru_f32_cmsis_call(exact, cpu)
        assert emu.entry == "gru_f32_cmsis" and emu.params == {**exact.params, "target": target}
        assert emu.output_shapes == exact.output_shapes
    with pytest.raises(ValueError, match="emulates gru_f32"):
        kf.gru_f32_cmsis_call(ReferenceCall("gru_f16", exact.params, {k: v.astype(np.float16) for k, v in ops.items()},
                                            _out(ops, 2)), "cortex-m4")


_DESCS = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}
_F32 = ["gru_unidirectional_float_pre_reset_f32", "gru_unidirectional_float_time_major_stream_f32",
        "gru_unidirectional_float_no_bias_f32", "gru_unidirectional_float_nonfinite_nan_f32"]


def _sidecar(out):
    return json.loads(next(out.rglob("*.sidecar.json")).read_text())


@pytest.mark.parametrize("cpu,target", [("cortex-m55", HELIUM), ("cortex-m4", FPU), ("cortex-m0", SOFT)])
@pytest.mark.parametrize("name", _F32)
def test_float32_cases_take_the_kernel_leg_at_pytorch_tolerance(name, cpu, target, tmp_path) -> None:
    op = OpGRUUnidirectional(_DESCS[name], 1234, target_cpu=cpu)
    op.generate_c_files(tmp_path)
    assert op.reference.entry == "gru_f32_cmsis" and op.reference.params["target"] == target
    ctx = json.dumps(_sidecar(tmp_path))
    assert ctx.count('"validation_atol": 1e-05') == 1 and '"validation_rtol": 1.3e-06' in ctx


def test_status_only_float32_cases_record_no_golden(tmp_path) -> None:
    op = OpGRUUnidirectional(_DESCS["gru_unidirectional_error_null_input_f32"], 1234, target_cpu="cortex-m4")
    op.generate_c_files(tmp_path)
    assert op.status_only() and op.reference is None
