"""The recurrent reference entries: TFLM's integer LSTM and the TFLM SVDF port."""

from __future__ import annotations

import numpy as np
import pytest

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference import lstm
from helia_core_tester.generation.reference import params as P


@pytest.fixture(scope="module")
def lib() -> b.Bindings:
    return b.get_bindings()


def _code(fn) -> int:
    with pytest.raises(b.ReferenceKernelError) as info:
        fn()
    return info.value.code


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _float_lstm(case: lstm.LstmCase, time_major: bool) -> np.ndarray:
    """A float LSTM on the dequantized operands: the integer one must track it."""
    q = case.quant
    x = (case.input.astype(np.float64) - q.input_zero_point) * q.input_scale
    if time_major:
        x = x.transpose(1, 0, 2)
    batch, steps, _ = x.shape
    hidden = case.weights["input_gate_hidden"].shape[0]
    w = {k: v.astype(np.float64) * q.weight_scales[k] for k, v in case.weights.items()}
    bias = {g: case.biases[f"{g}_gate_bias"].astype(np.float64) * q.input_scale * q.weight_scales[f"{g}_gate_input"]
            for g in lstm.GATES}
    h = np.zeros((batch, hidden))
    c = np.zeros((batch, hidden))
    out = np.zeros((batch, steps, hidden))
    for t in range(steps):
        pre = {g: x[:, t] @ w[f"{g}_gate_input"].T + h @ w[f"{g}_gate_hidden"].T + bias[g] for g in lstm.GATES}
        c = _sigmoid(pre["forget"]) * c + _sigmoid(pre["input"]) * np.tanh(pre["cell"])
        h = _sigmoid(pre["output"]) * np.tanh(c)
        out[:, t] = h
    return out.transpose(1, 0, 2) if time_major else out


@pytest.mark.parametrize("kind, time_major", [("s8", False), ("s8", True), ("s16", False), ("s16", True)])
def test_lstm_tracks_a_float_lstm(kind, time_major) -> None:
    quant = lstm.default_quant(kind, 6, 5, input_zero_point=-7 if kind == "s8" else 0,
                               output_zero_point=4 if kind == "s8" else 0)
    case = lstm.build_lstm_case(np.random.default_rng(11), kind=kind, batch=2, time_steps=6, input_size=6,
                                hidden_size=5, time_major=time_major, quant=quant)
    real = (case.output.astype(np.float64) - quant.output_zero_point) * quant.output_scale
    tolerance = 0.03 if kind == "s8" else 0.01
    assert np.abs(real - _float_lstm(case, time_major)).max() < tolerance
    assert np.unique(case.output).size > 8


def test_lstm_rejections(lib) -> None:
    case = lstm.build_lstm_case(np.random.default_rng(1), kind="s8", batch=1, time_steps=2, input_size=3,
                                hidden_size=2, time_major=False)
    weights = {k: case.weights[k] for k in b.LSTM_WEIGHT_ORDER}

    def run(**changes):
        params = {**case.call.params, **changes}
        return lambda: lib.lstm("s8", lstm.lstm_struct(params), case.input, weights, case.biases)

    assert _code(run(cell_scale=0.001)) == b.E_PARAM  # not a power of two
    assert _code(run(input_zero_point=200)) == b.E_PARAM
    assert _code(run(input_scale=0.0)) == b.E_PARAM
    with pytest.raises(ValueError, match="input shape"):
        run(time_steps=3)()
    s16 = lstm.build_lstm_case(np.random.default_rng(2), kind="s16", batch=1, time_steps=2, input_size=3,
                               hidden_size=2, time_major=False)
    p16 = lstm.lstm_struct({**s16.call.params, "output_zero_point": 1})
    w16 = {k: s16.weights[k] for k in b.LSTM_WEIGHT_ORDER}
    assert _code(lambda: lib.lstm("s16", p16, s16.input, w16, s16.biases)) == b.E_PARAM


def test_lstm_quant_validation() -> None:
    with pytest.raises(ValueError, match="power of two"):
        lstm.LstmQuant("s8", 0.01, 0, 0.01, 0, 0.001, {k: 0.01 for k in lstm.WEIGHT_KEYS})
    with pytest.raises(ValueError, match="s16"):
        lstm.default_quant("s16", 3, 2, input_zero_point=4)
    with pytest.raises(ValueError, match="weight scales"):
        lstm.LstmQuant("s8", 0.01, 0, 0.01, 0, 2.0 ** -11, {})


def _svdf_numpy(x, state, wf, wt, bias, p, state_t):
    """Independent SVDF step: shift, feature matmul into the newest column, time dot, rank sum."""
    state = state.astype(np.int64).copy()
    state[:, :, :-1] = state[:, :, 1:]
    from helia_core_tester.tests import tflm_numpy_model as model

    info = np.iinfo(state_t)
    feat = (x.astype(np.int64) - p["input_zero_point"]) @ wf.T.astype(np.int64)
    state[:, :, -1] = np.clip(model.mbqm32(feat, np.int64(p["scale1_multiplier"]), np.int64(p["scale1_shift"])),
                              info.min, info.max)
    acc = np.einsum("bfm,fm->bf", state, wt.astype(np.int64))
    acc = acc.reshape(acc.shape[0], -1, p["rank"]).sum(axis=2) + (0 if bias is None else bias)
    out = model.mbqm32(acc, np.int64(p["scale2_multiplier"]), np.int64(p["scale2_shift"])) + p["output_zero_point"]
    return np.clip(out, -128, 127), state.astype(state_t)


@pytest.mark.parametrize("state_kind", ["s8", "s16"])
@pytest.mark.parametrize("use_bias", [True, False])
def test_svdf_matches_an_independent_step(lib, state_kind, use_bias) -> None:
    rng = np.random.default_rng(5)
    state_t = np.int8 if state_kind == "s8" else np.int16
    info = np.iinfo(state_t)
    for rank in (1, 2):
        n, i, f, m = 2, 7, 6, 4
        m1, s1 = P.quantize_multiplier(0.02)
        m2, s2 = P.quantize_multiplier(0.004)
        p = {"batch": n, "input_size": i, "num_filters": f, "memory_size": m, "rank": rank, "input_zero_point": -5,
             "output_zero_point": 3, "scale1_multiplier": m1, "scale1_shift": s1, "scale2_multiplier": m2,
             "scale2_shift": s2}
        x = rng.integers(-128, 128, (n, i)).astype(np.int8)
        wf = rng.integers(-127, 128, (f, i)).astype(np.int8)
        wt = rng.integers(-info.max, info.max + 1, (f, m)).astype(state_t)
        state = rng.integers(info.min // 2, info.max // 2, (n, f, m)).astype(state_t)
        bias = rng.integers(-3000, 3000, (f // rank,)).astype(np.int32) if use_bias else None
        out, new_state = lib.svdf(state_kind, b.HctSvdfParams(*p.values()), x, wf, wt, bias, state)
        ref_out, ref_state = _svdf_numpy(x, state, wf, wt, bias, p, state_t)
        np.testing.assert_array_equal(out, ref_out)
        np.testing.assert_array_equal(new_state, ref_state)


def test_svdf_rejections(lib) -> None:
    p = b.HctSvdfParams(1, 3, 4, 2, 3, 0, 0, 1 << 30, 0, 1 << 30, 0)  # rank 3 does not divide 4 filters
    x, wf = np.zeros((1, 3), np.int8), np.zeros((4, 3), np.int8)
    wt, st = np.zeros((4, 2), np.int8), np.zeros((1, 4, 2), np.int8)
    with pytest.raises(ValueError, match="rank"):
        lib.svdf("s8", p, x, wf, wt, None, st)
    ok = b.HctSvdfParams(1, 3, 4, 2, 2, 0, 0, 1 << 30, 0, 1 << 30, 0)
    bad_zp = b.HctSvdfParams(1, 3, 4, 2, 2, 300, 0, 1 << 30, 0, 1 << 30, 0)
    assert _code(lambda: lib.svdf("s8", bad_zp, x, wf, wt, None, st)) == b.E_PARAM
    bad_shift = b.HctSvdfParams(1, 3, 4, 2, 2, 0, 0, 1 << 30, 31, 1 << 30, 0)
    assert _code(lambda: lib.svdf("s8", bad_shift, x, wf, wt, None, st)) == b.E_PARAM
    with pytest.raises(ValueError, match="state shape"):
        lib.svdf("s8", ok, x, wf, wt, None, np.zeros((1, 4, 3), np.int8))
    with pytest.raises(TypeError):
        lib.svdf("s16", ok, x, wf, wt, None, st)
