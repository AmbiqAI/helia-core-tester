"""Float comparison policy for the operators outside the convolution / matmul family: on the shipped build a case
keeps its kernel-faithful golden at assert_close's defaults; on every other build its golden is the exact reference,
widened by a bound recomputed here independently from the case's operands (float64 numpy). Accumulating float32
operators that are never emulated (AvgPool, Mean, ReduceSum, SVDF, BatchNorm) carry their bound on every build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared import base as op_base
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.reference.call import ReferenceCall
from helia_core_tester.hardware import generated_test_bridge as bridge

U = {np.dtype(np.float32): 2.0 ** -24, np.dtype(np.float16): 2.0 ** -11}
DEFAULT = {np.dtype(np.float32): (1e-5, 1.3e-6), np.dtype(np.float16): (1e-5, 1e-3)}
DESCS = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}


class Case:
    def __init__(self, entry, call, golden, atol, rtol, op):
        self.entry, self.call, self.golden, self.atol, self.rtol, self.op = entry, call, golden, atol, rtol, op

    @property
    def dtype(self):
        return np.dtype(next(iter(self.call.inputs.values())).dtype)

    def widened_by(self):
        atol, rtol = DEFAULT[self.dtype]
        return self.atol - atol, self.rtol - rtol


@pytest.fixture
def generate(tmp_path, monkeypatch):
    """generate_test on a descriptor, capturing the recorded reference call and the emitted float comparison."""
    import helia_core_tester.generation.test_ops as generation_module

    recorded = []
    original = op_base.OperationBase.reference_golden

    def _record(self, call):
        out = original(self, call)
        recorded.append((self, call, out))
        return out

    monkeypatch.setattr(op_base.OperationBase, "reference_golden", _record)

    def _run(name, cpu="cortex-m55", numerics="shipped", desc=None):
        recorded.clear()
        desc = desc or DESCS[name]
        out = tmp_path / f"{cpu}_{numerics}"
        generation_module.generate_test(desc, str(out), seed=500, cpu=cpu, kernel_numerics=numerics)
        case_dir = next(p.parent for p in out.rglob(f"{desc['name']}.reference.json")) if recorded else None
        if case_dir is None:
            return None
        entry = json.loads((case_dir / f"{desc['name']}.reference.json").read_text())["entry"]
        atol, rtol = _emitted_comparison(next(case_dir.glob("*.c")).read_text())
        op, call, golden = recorded[-1]
        return Case(entry, call, golden, atol, rtol, op)

    return _run


def _emitted_comparison(source):
    """(atol, rtol) of the case's float validation call, plain or masked."""
    for macro, count, first in (("HELIA_VALIDATE_OUTPUTS", 9, 5), ("HELIA_VALIDATE_FLOATS_MASKED", 8, 4)):
        try:
            calls = bridge._extract_all_call_args(source, macro, expected_count=count)
        except bridge.UnsupportedGeneratedTestError:
            continue
        for args in calls:
            if macro.endswith("MASKED") or args[0].strip() == "FLOAT":
                return float(args[first].rstrip("fF")), float(args[first + 1].rstrip("fF"))
    raise AssertionError("no float validation call in the generated source")


def _close(a, b):
    return a == pytest.approx(b, rel=1e-6, abs=0.0)


def _assert_defaults(case):
    assert (case.atol, case.rtol) == DEFAULT[case.dtype]


def _assert_widened(case, atol_extra, rtol_extra=0.0):
    got_atol, got_rtol = case.widened_by()
    assert _close(got_atol, atol_extra), (got_atol, atol_extra)
    assert _close(got_rtol, rtol_extra) or (rtol_extra == 0.0 and abs(got_rtol) < 1e-12), (got_rtol, rtol_extra)


def _linear(mag, k, dtype):
    u = U[np.dtype(dtype)]
    finite = mag[np.isfinite(mag)]
    return (k + 2) * u * float(np.max(finite, initial=0.0)), 2 * u


# ------------------------------------------------------------------------------- helpers, hand-checkable

def test_abs_magnitude_reruns_on_absolute_values_without_the_clamp():
    x = np.array([[[[1.0], [-1.0]], [[-1.0], [1.0]]]], np.float32)
    window = {"stride_h": 2, "stride_w": 2, "filter_h": 2, "filter_w": 2, "pad_h": 0, "pad_w": 0}
    call = ReferenceCall("avg_pool_f32", {**window, "activation_min": -0.25, "activation_max": 0.25},
                         {"input": x}, {"output": (1, 1, 1, 1)})
    assert call.output().item() == 0.0
    assert kf.abs_magnitude(call).item() == 1.0
    assert call.params["activation_max"] == 0.25 and call.inputs["input"].min() == -1.0
    assert kf.linear_bound(kf.abs_magnitude(call), 4, np.float32) == (6 * 2.0 ** -24, 2.0 ** -23)


def test_abs_magnitude_of_a_two_term_batch_norm():
    call = ReferenceCall("batch_norm_f16", {"unused": 0},
                         {"input": np.array([[-1.0], [2.0]], np.float16), "scale": np.array([1.0], np.float16),
                          "bias": np.array([1.0], np.float16)}, {"output": (2, 1)})
    assert call.output().ravel().tolist() == [0.0, 3.0]
    assert kf.abs_magnitude(call).ravel().tolist() == [2.0, 3.0]
    assert call.inputs["input"].dtype == np.float16 and call.inputs["input"].min() == -1.0


def test_abs_magnitude_selects_a_named_output():
    x = np.ones((2, 1, 3), np.float32)
    call = ReferenceCall("svdf_f32", {"rank": 1, "activation_min": -1e30, "activation_max": 1e30},
                         {"input": -x, "weights_feature": np.ones((2, 3), np.float32),
                          "weights_time": np.ones((2, 2), np.float32), "bias": np.zeros((0,), np.float32),
                          "state": np.zeros((1, 2, 2), np.float32)},
                         {"output": (2, 1, 2), "state_out": (1, 2, 2)})
    # Step 0: projection 3 into the newest slot; step 1: 3 + 3 summed over the two time slots.
    assert kf.abs_magnitude(call, "output")[-1].tolist() == [[6.0, 6.0]]
    assert call.run()["output"][-1].tolist() == [[-6.0, -6.0]]


def test_float32_accumulation_bound_is_float32_summation_and_a_float16_store():
    assert kf.float32_accumulation_bound(np.full(3, 8.0), 8, np.float16) == (10 * 2.0 ** -24 * 8, 2.0 ** -10)
    assert kf.float32_accumulation_bound(np.full(3, 8.0), 8, np.float32) == (10 * 2.0 ** -24 * 8, 2.0 ** -23)
    assert kf.float32_accumulation_bound(np.zeros(0), 0, np.float32) == (0.0, 2.0 ** -23)
    with pytest.raises(ValueError, match="term count"):
        kf.float32_accumulation_bound(np.ones(1), -1, np.float32)


def test_recurrent_dot_bound_hand_checked():
    g, h, i = 3, 2, 3
    ones = np.ones
    # Every pre-activation: 3 input terms of 1, 2 recurrent terms of |h| <= 1, two biases of 1 -> M = 7, k = 7.
    got = kf.recurrent_dot_bound(ones((1, 4, i)), ones((g, h, i)), ones((g, h, h)), (ones((g, h)), ones((g, h))),
                                 np.float32)
    u = 2.0 ** -24
    assert got == pytest.approx((9 * u * 7 + 2 * u * 7) * (1.5 + 7 / 4), rel=1e-12)
    # No biases: k = 5, M = 5; an LSTM's cell bound replaces M in the carried term.
    got = kf.recurrent_dot_bound(ones((1, 4, i)), ones((4, h, i)), ones((4, h, h)), (np.zeros(0),), np.float16,
                                 state_bound=4.0)
    u = 2.0 ** -11
    assert got == pytest.approx((7 * u * 5 + 2 * u * 5) * (1.5 + 1.0), rel=1e-12)


def test_recurrent_dot_bound_ignores_nonfinite_rows_and_empty_sequences():
    w = np.ones((3, 2, 3))
    x = np.ones((2, 1, 3))
    x[0, 0, 1] = np.inf
    finite_only = kf.recurrent_dot_bound(x[1:], w, np.ones((3, 2, 2)), (), np.float32)
    assert kf.recurrent_dot_bound(x, w, np.ones((3, 2, 2)), (), np.float32) == finite_only
    # Zero time steps: the recurrent terms still bound M (h stays 0, so this is conservative).
    assert kf.recurrent_dot_bound(np.ones((1, 0, 3)), w, np.ones((3, 2, 2)), (), np.float32) == 0.0


@pytest.mark.parametrize("w_in,w_h,state", [
    (np.ones((3, 2)), np.ones((3, 2, 2)), None),
    (np.ones((3, 2, 3)), np.ones((3, 2, 3)), None),
    (np.ones((3, 2, 3)), np.ones((4, 2, 2)), None),
    (np.ones((3, 2, 3)), np.ones((3, 2, 2)), -1.0),
    (np.ones((3, 2, 3)), np.ones((3, 2, 2)), np.inf),
])
def test_recurrent_dot_bound_rejects_bad_shapes_and_state_bounds(w_in, w_h, state):
    with pytest.raises(ValueError):
        kf.recurrent_dot_bound(np.ones((1, 1, 3)), w_in, w_h, (), np.float32, state_bound=state)


# ------------------------------------------------------------------------------------------ AvgPool / MaxPool

def _avg_pool_bound(call):
    window = {k: call.params[k] for k in ("stride_h", "stride_w", "filter_h", "filter_w", "pad_h", "pad_w")}
    taps, weights = kf.avg_pool_windows(np.abs(call.inputs["input"].astype(np.float64)), window,
                                        call.output_shapes["output"])
    mag = np.sum(taps * weights, axis=(1, 2))
    return _linear(mag, window["filter_h"] * window["filter_w"], call.inputs["input"].dtype)


@pytest.mark.parametrize("name,cpu,numerics", [
    ("avg_pool_float_default_f32", "cortex-m55", "shipped"), ("avg_pool_float_default_f32", "cortex-m55", "bounded"),
    ("avg_pool_float_rect_kernel_f32", "cortex-m0", "shipped"), ("avg_pool_float_valid_stride1_f32", "cortex-m4",
                                                                  "bounded"),
])
def test_avg_pool_f32_carries_the_summation_bound_on_every_build(generate, name, cpu, numerics):
    case = generate(name, cpu, numerics)
    assert case.entry == "avg_pool_f32"
    _assert_widened(case, *_avg_pool_bound(case.call))


def test_avg_pool_f16_is_emulated_on_the_shipped_build_and_bounded_elsewhere(generate):
    shipped = generate("avg_pool_float_default_f16", "cortex-m55", "shipped")
    assert shipped.entry == "avg_pool_f16_mve"
    _assert_defaults(shipped)
    bounded = generate("avg_pool_float_default_f16", "cortex-m55", "bounded")
    assert bounded.entry == "avg_pool_f16"
    _assert_widened(bounded, *_avg_pool_bound(bounded.call))
    assert bounded.widened_by()[0] > 0


def test_max_pool_selects_an_input_and_needs_no_bound(generate):
    for numerics in ("shipped", "bounded"):
        case = generate("max_pool_float_default_f32", "cortex-m55", numerics)
        assert case.entry == "max_pool_f32"
        _assert_defaults(case)


# ------------------------------------------------------------------------------------------ Mean / ReduceSum

def _reduce_bound(call, mean):
    x = np.abs(call.inputs["input"].astype(np.float64))
    axes = tuple(a for a in range(4) if call.params["axis_mask"] >> a & 1)
    k = int(np.prod([x.shape[a] for a in axes]))
    mag = x.sum(axis=axes) / (k if mean else 1)
    atol, _ = _linear(mag, k, np.float32)
    return atol, 2 * U[call.inputs["input"].dtype]


@pytest.mark.parametrize("name,mean", [("mean_float_axis_hw_f32", True), ("reduce_sum_float_axis_c_f32", False),
                                       ("reduce_sum_float_axis_hwc_f32", False)])
@pytest.mark.parametrize("cpu,numerics", [("cortex-m55", "shipped"), ("cortex-m55", "bounded"),
                                          ("cortex-m0", "shipped")])
def test_float32_reductions_carry_the_summation_bound_on_every_build(generate, name, mean, cpu, numerics):
    case = generate(name, cpu, numerics)
    assert case.entry == ("mean_f32" if mean else "reduce_sum_f32")
    _assert_widened(case, *_reduce_bound(case.call, mean))


@pytest.mark.parametrize("name,mean", [("mean_float_axis_hwc_f16", True), ("reduce_sum_float_axis_hc_f16", False)])
def test_float16_reductions_are_emulated_on_the_shipped_build_and_bounded_elsewhere(generate, name, mean):
    shipped = generate(name, "cortex-m55", "shipped")
    assert shipped.entry == ("mean_f16_f32acc" if mean else "reduce_sum_f16_f32acc")
    _assert_defaults(shipped)
    bounded = generate(name, "cortex-m55", "bounded")
    assert bounded.entry == ("mean_f16" if mean else "reduce_sum_f16")
    _assert_widened(bounded, *_reduce_bound(bounded.call, mean))


@pytest.mark.parametrize("numerics", ["shipped", "bounded"])
def test_a_nonfinite_reduction_keeps_its_probe_and_mask(generate, numerics):
    case = generate("reduce_sum_float_nonfinite_nan_f32", "cortex-m55", numerics)
    assert case.entry == "reduce_sum_f32"
    np.testing.assert_array_equal(case.op.reference_probe(list(case.call.inputs.values())), case.golden)
    assert np.isnan(case.golden).any()
    assert np.isfinite(case.atol) and case.widened_by()[0] > 0


# ------------------------------------------------------------------------------------------------- SVDF

def _svdf_magnitude(call):
    ins = {k: np.abs(v.astype(np.float64)) for k, v in call.inputs.items()}
    x, wf, wt, bias, s = ins["input"], ins["weights_feature"], ins["weights_time"], ins["bias"], ins["state"].copy()
    rank = int(call.params["rank"])
    feats = wt.shape[0]
    for step in range(x.shape[0]):
        s = np.roll(s, -1, axis=2)
        s[:, :, -1] = x[step] @ wf.T
        out = np.einsum("fm,bfm->bf", wt, s).reshape(x.shape[1], feats // rank, rank).sum(-1)
        if bias.size:
            out = out + bias
    return out


@pytest.mark.parametrize("name", ["svdf_float_default_f32", "svdf_float_no_bias_f32", "svdf_float_rank2_f32"])
@pytest.mark.parametrize("numerics", ["shipped", "bounded"])
def test_svdf_f32_carries_the_nested_summation_bound_on_every_build(generate, name, numerics):
    case = generate(name, "cortex-m55", numerics)
    assert case.entry == "svdf_f32"
    hint = DESCS[name]["hint"]
    k = hint.get("input_height", 4) + hint.get("time_batches", 3) + hint.get("rank", 1) + 2
    _assert_widened(case, *_linear(_svdf_magnitude(case.call), k, np.float32))


def test_svdf_f16_is_emulated_on_the_shipped_build_and_bounded_elsewhere(generate):
    shipped = generate("svdf_float_rank2_f16", "cortex-m55", "shipped")
    assert shipped.entry == "svdf_f16_mve"
    _assert_defaults(shipped)
    bounded = generate("svdf_float_rank2_f16", "cortex-m55", "bounded")
    assert bounded.entry == "svdf_f16"
    hint = DESCS["svdf_float_rank2_f16"]["hint"]
    k = hint.get("input_height", 4) + hint.get("time_batches", 3) + hint.get("rank", 1) + 2
    _assert_widened(bounded, *_linear(_svdf_magnitude(bounded.call), k, np.float16))


# ---------------------------------------------------------------------------------------------- GRU / LSTM

def _dots(call, gates, state_bound=None):
    ins = {k: v.astype(np.float64) for k, v in call.inputs.items()}
    hidden = ins["recurrent_weights"].shape[1]
    x = np.abs(ins["input"]).reshape(-1, ins["input_weights"].shape[2])
    x = x[np.all(np.isfinite(x), axis=1)]
    mag = np.abs(ins["input_weights"]) @ x.T  # [G, H, rows]
    mag = mag + np.abs(ins["recurrent_weights"]).sum(axis=2)[:, :, None]
    k = ins["input_weights"].shape[2] + hidden
    for b in [ins[n] for n in ("input_bias", "recurrent_bias", "bias") if n in ins]:
        if b.size:
            mag = mag + np.abs(b).reshape(gates, hidden, 1)
            k += 1
    worst = float(np.max(mag, initial=0.0))
    u = U[call.inputs["input"].dtype]
    carried = worst if state_bound is None else state_bound
    return ((k + 2) * u * worst + 2 * u * worst) * (1.5 + carried / 4)


@pytest.mark.parametrize("name", ["gru_unidirectional_float_reset_after_f32", "gru_unidirectional_float_no_bias_f32",
                                  "gru_unidirectional_float_pre_reset_multi_batch_f32"])
def test_gru_f32_is_emulated_on_the_shipped_build_and_bounded_elsewhere(generate, name):
    for cpu, target in (("cortex-m55", 0), ("cortex-m4", 1), ("cortex-m0", 2)):
        shipped = generate(name, cpu, "shipped")
        assert shipped.entry == "gru_f32_cmsis" and shipped.call.params["target"] == target
        _assert_defaults(shipped)
    bounded = generate(name, "cortex-m4", "bounded")
    assert bounded.entry == "gru_f32"
    _assert_widened(bounded, kf.GRU_F32_ATOL + _dots(bounded.call, 3))


def test_gru_f16_is_emulated_on_helium_shipped_and_bounded_elsewhere(generate):
    name = "gru_unidirectional_float_pre_reset_f16"
    shipped = generate(name, "cortex-m55", "shipped")
    assert shipped.entry == "gru_f16_mve"
    _assert_defaults(shipped)
    for cpu, numerics in (("cortex-m55", "bounded"), ("cortex-m4", "shipped")):
        case = generate(name, cpu, numerics)
        assert case.entry == "gru_f16"
        _assert_widened(case, kf.RNN_F16_ATOL + _dots(case.call, 3))


@pytest.mark.parametrize("name", ["lstm_unidirectional_float_default_f32", "lstm_unidirectional_float_no_bias_f32",
                                  "lstm_unidirectional_float_clip_tail9_time_major_f32"])
def test_lstm_f32_is_emulated_on_the_shipped_build_and_bounded_elsewhere(generate, name):
    shipped = generate(name, "cortex-m4", "shipped")
    assert shipped.entry == "lstm_f32_cmsis" and shipped.call.params["target"] == 1
    _assert_defaults(shipped)
    bounded = generate(name, "cortex-m55", "bounded")
    assert bounded.entry == "lstm_f32"
    steps = float(DESCS[name].get("time_steps", 1))
    clip = float(DESCS[name].get("cell_clip", 0.0))
    cell = steps if clip <= 0 else min(steps, clip)
    _assert_widened(bounded, kf.LSTM_F32_ATOL + _dots(bounded.call, 4, cell))


def test_lstm_f16_is_emulated_on_helium_shipped_and_bounded_elsewhere(generate):
    name = "lstm_unidirectional_float_clip_vec8_batch2_f16"
    shipped = generate(name, "cortex-m55", "shipped")
    assert shipped.entry == "lstm_f16_mve"
    _assert_defaults(shipped)
    bounded = generate(name, "cortex-m55", "bounded")
    assert bounded.entry == "lstm_f16"
    steps, clip = float(DESCS[name]["time_steps"]), float(DESCS[name].get("cell_clip", 0.0))
    _assert_widened(bounded, kf.RNN_F16_ATOL + _dots(bounded.call, 4, steps if clip <= 0 else min(steps, clip)))


@pytest.mark.parametrize("name", ["gru_unidirectional_float_zero_time_steps_f32",
                                  "lstm_unidirectional_float_zero_time_steps_f16"])
def test_zero_time_steps_add_only_the_guard_budget(generate, name):
    case = generate(name, "cortex-m55", "bounded")
    budget = kf.GRU_F32_ATOL if name.startswith("gru") else kf.RNN_F16_ATOL
    _assert_widened(case, budget)


@pytest.mark.parametrize("name", ["gru_unidirectional_error_null_input_f32", "lstm_unidirectional_error_null_input_f16",
                                  "svdf_fault_zero_rank_f32"])
@pytest.mark.parametrize("numerics", ["shipped", "bounded"])
def test_status_only_cases_record_no_golden_on_either_build(generate, name, numerics):
    assert generate(name, "cortex-m55", numerics) is None


@pytest.mark.parametrize("name", ["gru_unidirectional_float_nonfinite_inf_f32", "lstm_unidirectional_float_nonfinite_nan_f16"])
def test_recurrent_nonfinite_cases_keep_a_finite_bound_on_the_bounded_build(generate, name):
    case = generate(name, "cortex-m55", "bounded")
    assert case.entry in ("gru_f32", "lstm_f16")
    assert np.isfinite(case.atol) and case.widened_by()[0] > 0


# ------------------------------------------------------------------------------------------------- Softmax

def _softmax_spread(x):
    a = x.astype(np.float64).reshape(-1, x.shape[-1])
    a = np.where(np.isfinite(a), a, np.nan)
    spread = np.nanmax(a, axis=1) - np.nanmin(a, axis=1)
    return float(np.nanmax(spread))


@pytest.mark.parametrize("cpu,target", [("cortex-m55", 0), ("cortex-m4", 1), ("cortex-m0", 2)])
def test_softmax_f32_is_emulated_on_the_shipped_build_and_bounded_elsewhere(generate, cpu, target):
    shipped = generate("softmax_float_wide_tail_f32", cpu, "shipped")
    assert shipped.entry == "softmax_f32_cmsis" and shipped.call.params["target"] == target
    _assert_defaults(shipped)
    bounded = generate("softmax_float_wide_tail_f32", cpu, "bounded")
    assert bounded.entry == "softmax_f32"
    x = bounded.call.inputs["input"]
    _assert_widened(bounded, (_softmax_spread(x) + x.shape[-1] + 8) * 2.0 ** -24 + 4e-6)


def test_softmax_f16_is_emulated_on_helium_shipped_and_bounded_elsewhere(generate):
    shipped = generate("softmax_float_vector8_f16", "cortex-m55", "shipped")
    assert shipped.entry == "softmax_f16_mve"
    _assert_defaults(shipped)
    bounded = generate("softmax_float_vector8_f16", "cortex-m55", "bounded")
    assert bounded.entry == "softmax_f16"
    _assert_widened(bounded, (_softmax_spread(bounded.call.inputs["input"]) + 8) * 2.0 ** -11)


def test_softmax_nonfinite_probe_returns_the_emitted_golden_on_the_bounded_build(generate):
    case = generate("softmax_float_nonfinite_inf_f32", "cortex-m55", "bounded")
    assert case.entry == "softmax_f32"
    np.testing.assert_array_equal(case.op.reference_probe(list(case.call.inputs.values())), case.golden)


# ------------------------------------------------------------------------------- NNActivationFloat / HardSwish

def _act(name, act, dtype, **extra):
    return {"operator": "NNActivationFloat", "name": name, "suite": "float", "_family": "ActivationFunctions",
            "tensor_dtypes": {"input": dtype, "output": dtype}, "input_shape": [1, 17], "activation_type": act,
            "act_param": 0.125, **extra}


@pytest.mark.parametrize("act,budget", [("ARM_NN_FLT_ACT_TANH", 2.5e-5), ("ARM_NN_FLT_ACT_SIGMOID", 1e-6)])
def test_nn_activation_f32_tables_are_emulated_shipped_and_budgeted_elsewhere(generate, act, budget):
    desc = _act("act_bound_case", act, "FP32", input_min=-8.0, input_max=8.0)
    shipped = generate(None, "cortex-m4", "shipped", desc=desc)
    assert shipped.entry == "nn_activation_f32_cmsis" and shipped.call.params["target"] == 1
    _assert_defaults(shipped)
    bounded = generate(None, "cortex-m4", "bounded", desc=desc)
    assert bounded.entry == "nn_activation_f32"
    _assert_widened(bounded, budget)


@pytest.mark.parametrize("cpu,numerics", [("cortex-m55", "bounded"), ("cortex-m4", "shipped")])
def test_nn_activation_f16_off_the_emulated_route_takes_the_exact_golden(generate, cpu, numerics):
    desc = _act("act_bound_f16", "ARM_NN_FLT_ACT_TANH", "FP16")
    case = generate(None, cpu, numerics, desc=desc)
    assert case.entry == "nn_activation_f16"
    _assert_widened(case, 2.0 ** -9)
    hs = generate(None, cpu, numerics, desc=_act("act_bound_hs16", "ARM_NN_FLT_ACT_HARDSWISH", "FP16",
                                                 input_min=-6.0, input_max=6.0))
    worst = max(1.0, float(np.max(np.abs(hs.call.inputs["input"].astype(np.float64)))))
    _assert_widened(hs, worst * (4 * 2.0 ** -11 + abs(float(np.float16(1 / 6)) * 6 - 1)))


@pytest.mark.parametrize("numerics", ["shipped", "bounded"])
def test_exact_activations_need_no_bound(generate, numerics):
    case = generate(None, "cortex-m0", numerics, desc=_act("act_relu_case", "ARM_NN_FLT_ACT_LEAKY_RELU", "FP32"))
    assert case.entry == "nn_activation_f32"
    _assert_defaults(case)


@pytest.mark.parametrize("name", ["nn_activation_float_tanh_lutedge_f32", "nn_activation_float_tanh_lutedge_neg_f32"])
def test_the_lutedge_tightening_wins_on_every_build(generate, name):
    for numerics, entry in (("shipped", "nn_activation_f32_cmsis"), ("bounded", "nn_activation_f32")):
        case = generate(name, "cortex-m55", numerics)
        assert case.entry == entry
        assert (case.atol, case.rtol) == (2e-6, 0.0)
        params = {k: case.call.params[k] for k in ("activation_type", "act_param")}
        exact = ReferenceCall("nn_activation_f32", params, dict(case.call.inputs), dict(case.call.output_shapes))
        assert np.max(np.abs(case.golden.astype(np.float64) - exact.output())) <= 2e-6


@pytest.mark.parametrize("name,bits", [("nn_activation_float_tanh_lut_cutoff_f16", 0x3BFA),
                                       ("nn_activation_float_tanh_lut_index_f16", 0xBB3E)])
def test_lut_grid_cases_keep_their_exact_comparison_and_bits_on_every_build(generate, name, bits):
    for numerics, entry in (("shipped", "tanh_lut_mve_f16"), ("bounded", "nn_activation_f16")):
        case = generate(name, "cortex-m55", numerics)
        assert case.entry == entry
        assert (case.atol, case.rtol) == (0.0, 0.0)
        assert set(case.golden.view(np.uint16).ravel().tolist()) == {bits}


@pytest.mark.parametrize("dtype,scale", [("FP32", 0.0), ("FP16", abs(float(np.float16(1 / 6)) * 6 - 1))])
def test_hard_swish_is_emulated_on_the_shipped_build_and_budgeted_elsewhere(generate, dtype, scale):
    name = f"hard_swish_float_default_{dtype.lower().replace('fp', 'f')}"
    shipped = generate(name, "cortex-m55", "shipped")
    assert shipped.entry == ("hard_swish_f32_cmsis" if dtype == "FP32" else "hard_swish_f16_mve")
    _assert_defaults(shipped)
    bounded = generate(name, "cortex-m55", "bounded")
    assert bounded.entry == ("hard_swish_f32" if dtype == "FP32" else "hard_swish_f16")
    x = bounded.call.inputs["input"]
    worst = max(1.0, float(np.max(np.abs(x.astype(np.float64)))))
    _assert_widened(bounded, worst * (4 * U[x.dtype] + scale))


# ------------------------------------------------------------------------------------------------ BatchNorm

@pytest.mark.parametrize("name,cpu,numerics", [
    ("batch_norm_default_f32", "cortex-m55", "shipped"), ("batch_norm_default_f32", "cortex-m55", "bounded"),
    ("batch_norm_default_f32", "cortex-m0", "shipped"), ("batch_norm_9_channel_f16", "cortex-m55", "shipped"),
    ("batch_norm_9_channel_f16", "cortex-m55", "bounded"),
])
def test_batch_norm_carries_the_two_term_bound_on_every_build(generate, name, cpu, numerics):
    case = generate(name, cpu, numerics)
    ins = {k: v.astype(np.float64) for k, v in case.call.inputs.items()}
    mag = np.abs(ins["input"]) * np.abs(ins["scale"]) + np.abs(ins["bias"])
    _assert_widened(case, *_linear(mag, 2, case.call.inputs["input"].dtype))


def test_batch_norm_nonfinite_probe_returns_the_emitted_golden(generate):
    case = generate("batch_norm_nonfinite_f32", "cortex-m55", "bounded")
    assert np.isfinite(case.atol)
    np.testing.assert_array_equal(case.op.reference_probe(list(case.call.inputs.values())), case.golden)
