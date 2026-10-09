"""Weighted operators on reference goldens: geometry, quantization policy,
descriptor knobs, determinism, and an independent check of the golden."""

from __future__ import annotations

import numpy as np
import pytest

from helia_core_tester.generation.reference import draw, params, policy
from helia_core_tester.generation.reference import weighted as wt
from helia_core_tester.tests import tflm_numpy_model as model


def _draw(spec, seed=7):
    return lambda: np.random.default_rng(seed).integers(-32, 32, size=spec.input_shape).astype(np.float32)


def _case(desc, spec_fn=wt.conv_spec, seed=3, **kwargs):
    spec = spec_fn(desc)
    return spec, wt.build_weighted_case(desc, spec, np.random.default_rng(seed), _draw(spec), **kwargs)


CONV = {"name": "c", "operator": "Convolve", "input_shape": [2, 7, 6, 4], "filter_shape": [3, 3, 4, 6],
        "strides": [2, 1], "padding": "same", "activation_dtype": "S8"}


# ---- geometry ----


def test_conv_spec_geometry_and_groups() -> None:
    spec = wt.conv_spec({**CONV, "groups": 2, "dilation": [2, 1]})
    assert spec.weight_shape == (6, 3, 3, 2)
    assert spec.output_shape == (2, 4, 6, 6)
    assert spec.params["dilation"] == [2, 1] and spec.fan_in == 18 and spec.fan_out == 54
    with pytest.raises(wt.ReferenceCaseError, match="groups"):
        wt.conv_spec({**CONV, "groups": 3})
    with pytest.raises(wt.ReferenceCaseError, match="padding"):
        wt.conv_spec({**CONV, "padding": "full"})
    with pytest.raises(wt.ReferenceCaseError, match="VALID window"):
        wt.conv_spec({**CONV, "padding": "valid", "filter_shape": [9, 3, 4, 6]})
    with pytest.raises(wt.ReferenceCaseError, match="rank-4"):
        wt.conv_spec({**CONV, "input_shape": [7, 6, 4]})


def test_dwconv_fc_tconv_specs() -> None:
    dw = wt.dwconv_spec({"name": "d", "input_shape": [1, 5, 5, 3], "filter_shape": [3, 3, 3, 2], "depth_multiplier": 2})
    assert dw.weight_shape == (1, 3, 3, 6) and dw.output_shape == (1, 3, 3, 6) and dw.params["depth_multiplier"] == 2
    fc = wt.fc_spec({"name": "f", "input_shape": [3, 2, 2, 5], "filter_shape": [7, 20]})
    assert fc.weight_shape == (7, 20) and fc.output_shape == (3, 7)
    same = wt.tconv_spec({"name": "t", "input_shape": [1, 4, 3, 2], "filter_shape": [3, 3, 5, 99], "strides": [2, 2], "padding": "same"})
    assert same.output_shape == (1, 8, 6, 5) and same.weight_shape == (5, 3, 3, 2)
    valid = wt.tconv_spec({"name": "t", "input_shape": [1, 4, 3, 2], "filter_shape": [3, 1, 5, 2], "strides": [2, 2], "padding": "valid"})
    assert valid.output_shape == (1, 9, 6, 5)
    with pytest.raises(wt.ReferenceCaseError, match="dilation"):
        wt.tconv_spec({"name": "t", "input_shape": [1, 4, 3, 2], "filter_shape": [3, 3, 5, 2], "dilation": 2})


# ---- the golden ----


def test_conv_golden_matches_the_numpy_model() -> None:
    spec, case = _case({**CONV, "activation": "RELU"})
    p = spec.params
    acc = model.conv_nhwc(case.input_q, case.weights_q, p["stride"], p["dilation"], p["pad"], spec.output_shape[1:3],
                          -case.input_quant.zero_point)
    expected = model.requantize(acc, case.bias_q, case.requant.multiplier, case.requant.shift,
                                case.output_quant.zero_point, case.act_min, case.act_max)
    np.testing.assert_array_equal(case.output_q, expected.astype(np.int8))
    assert case.call.kernel == "conv_s8" and case.call.tensors["filter"] is case.weights_q


def test_golden_uses_the_range_and_is_not_degenerate() -> None:
    _, case = _case(CONV)
    out = case.output_q.astype(np.int32)
    assert len(np.unique(out)) > 32
    # Calibrated, not saturated: few elements at the rails.
    rails = np.mean((out == -128) | (out == 127))
    assert rails < 0.05


def test_relu_moves_the_output_zero_point_to_the_floor() -> None:
    _, case = _case({**CONV, "activation": "RELU"})
    assert case.output_quant.zero_point == -128 and case.act_min == -128
    _, relu6 = _case({**CONV, "activation": "RELU6"})
    assert relu6.act_max <= 127 and relu6.output_quant.scale * (relu6.act_max - relu6.output_quant.zero_point) <= 6.0 + relu6.output_quant.scale


def test_bias_is_worth_three_to_eight_output_steps() -> None:
    _, case = _case(CONV)
    acc_scales = policy.bias_quant_scales(case.input_quant.scale, case.weight_quant, 6)
    steps = np.abs(case.bias_q * acc_scales / case.output_quant.scale)
    assert np.all(steps >= draw.BIAS_MIN_STEPS - 0.5) and np.all(steps <= draw.BIAS_MAX_STEPS + 0.5)
    _, no_bias = _case({**CONV, "use_bias": False})
    assert no_bias.bias_q is None and no_bias.call.tensors["bias"] is None


def test_s16_uses_int64_bias_symmetric_zero_points_and_int32_bias_variant() -> None:
    spec, case = _case({**CONV, "activation_dtype": "S16"})
    assert case.kernel == "conv_s16" and case.bias_q.dtype == np.int64
    assert case.input_quant.zero_point == 0 and case.output_quant.zero_point == 0
    assert case.output_q.dtype == np.int16
    _, b32 = _case({**CONV, "activation_dtype": "S16"}, bias_ctype="int32_t")
    assert b32.kernel == "conv_s16_b32" and b32.bias_q.dtype == np.int32
    with pytest.raises(wt.ReferenceCaseError, match="int32 bias"):
        _case({"name": "f", "input_shape": [2, 8], "filter_shape": [3, 8], "activation_dtype": "S16"}, wt.fc_spec, bias_ctype="int32_t")


def test_same_seed_same_case_and_streams_are_independent() -> None:
    _, a = _case(CONV, seed=11)
    _, b = _case(CONV, seed=11)
    np.testing.assert_array_equal(a.output_q, b.output_q)
    np.testing.assert_array_equal(a.weights_q, b.weights_q)
    _, c = _case(CONV, seed=12)
    assert not np.array_equal(a.weights_q, c.weights_q)


# ---- descriptor knobs ----


def test_explicit_quantization_block_wins() -> None:
    desc = {**CONV, "quantization": {"input": {"scale": 0.5, "zero_point": 4}, "output": {"scale": 0.75, "zero_point": -3}}}
    _, case = _case(desc)
    assert (case.input_quant.scale, case.input_quant.zero_point) == (0.5, 4)
    assert (case.output_quant.scale, case.output_quant.zero_point) == (0.75, -3)
    _, ranged = _case({**CONV, "quantization": {"input": {"range": [-8, 8]}}})
    assert ranged.input_quant.scale == pytest.approx(16 / 255, rel=1e-6)


def test_quantization_block_rejects_conflicts_and_unknown_keys() -> None:
    with pytest.raises(wt.ReferenceCaseError, match="unknown quantization keys"):
        _case({**CONV, "quantization": {"bias": {}}})
    with pytest.raises(ValueError, match="exactly one"):
        _case({**CONV, "quantization": {"input": {"scale": 0.5, "range": [-1, 1]}}})
    with pytest.raises(wt.ReferenceCaseError, match="both set it"):
        _case({**CONV, "calibration_range": [-1, 1], "quantization": {"input": {"range": [-2, 2]}}})
    with pytest.raises(wt.ReferenceCaseError, match="both set it"):
        _case({**CONV, "hint": {"force_per_tensor": True}, "quantization": {"weights": {"per_channel": True}}})
    with pytest.raises(wt.ReferenceCaseError, match="headroom"):
        _case({**CONV, "quantization": {"headroom": 0.5}})


def test_headroom_widens_the_output_range() -> None:
    _, base = _case(CONV)
    _, wide = _case({**CONV, "quantization": {"headroom": 2.0}})
    assert wide.output_quant.scale == pytest.approx(2 * base.output_quant.scale, rel=0.05)


def test_per_tensor_weights_on_request() -> None:
    fc = {"name": "f", "input_shape": [2, 9], "filter_shape": [4, 9], "activation_dtype": "S8"}
    _, per_channel = _case(fc, wt.fc_spec)
    assert per_channel.per_channel and per_channel.requant.multiplier.size == 4
    _, legacy = _case({**fc, "hint": {"force_per_tensor": True}}, wt.fc_spec)
    _, explicit = _case({**fc, "quantization": {"weights": {"per_channel": False}}}, wt.fc_spec)
    for case in (legacy, explicit):
        assert not case.per_channel and case.requant.multiplier.size == 1
    assert legacy.quant_context(type("B", (), {"format_array_as_c_literal": staticmethod(str)})) == {
        "multiplier": int(legacy.requant.multiplier[0]), "shift": int(legacy.requant.shift[0]), "per_channel": False}


def test_calibration_and_input_ranges_and_weight_gain() -> None:
    _, calibrated = _case({**CONV, "calibration_range": [-1.0, 1.0]})
    assert calibrated.input_quant.scale == pytest.approx(2 / 255, rel=1e-6)
    # Inputs drawn from [-32, 32) against a [-1, 1] calibration saturate, as the knob intends.
    assert np.mean(np.abs(calibrated.input_q.astype(int)) >= 127) > 0.5
    _, gained = _case({**CONV, "weight_gain": 4.0})
    _, plain = _case(CONV)
    w_gain = np.abs(np.asarray(gained.weight_quant.scales)).mean()
    w_plain = np.abs(np.asarray(plain.weight_quant.scales)).mean()
    assert w_gain == pytest.approx(2 * w_plain, rel=0.25)


def test_descriptor_activation_bounds_narrow_the_clamp() -> None:
    _, case = _case({**CONV, "activation_min": -20, "activation_max": 30})
    assert (case.act_min, case.act_max) == (-20, 30)
    assert case.output_q.min() >= -20 and case.output_q.max() <= 30
    with pytest.raises(wt.ReferenceCaseError, match="empty"):
        _case({**CONV, "activation_min": 40, "activation_max": 30})
    with pytest.raises(wt.ReferenceCaseError, match="no reference"):
        _case({**CONV, "activation": "TANH"})


def test_forced_filter_offset_reaches_the_golden() -> None:
    fc = {"name": "f", "input_shape": [2, 13], "filter_shape": [5, 13], "activation_dtype": "S8"}
    spec, case = _case(fc, wt.fc_spec, weights_offset=3)
    x = case.input_q.reshape(2, 13).astype(np.int64) - case.input_quant.zero_point
    acc = x @ (case.weights_q.astype(np.int64) + 3).T
    expected = model.requantize(acc, case.bias_q, case.requant.multiplier, case.requant.shift,
                                case.output_quant.zero_point, case.act_min, case.act_max)
    np.testing.assert_array_equal(case.output_q, expected.astype(np.int8))
    assert case.call.params["weights_offset"] == 3


# ---- s4 (legacy fixed quantization) ----


S4_CONV = {**CONV, "weight_dtype": "S4", "hint": {"extras": {"input_scale": 4.0, "input_zero_point": 3, "weight_scale": 1.0,
                                                         "output_scale": 4.0, "output_zero_point": 0, "per_channel": True}}}


def test_s4_keeps_the_extras_quantization_and_packs_weights() -> None:
    spec, case = _case(S4_CONV)
    assert case.kernel == "conv_s4"
    assert (case.input_quant.scale, case.input_quant.zero_point, case.output_quant.scale) == (4.0, 3, 4.0)
    assert case.weights_q.min() >= -8 and case.weights_q.max() <= 7
    assert case.weights_c.size == (case.weights_q.size + 1) // 2
    np.testing.assert_array_equal(case.weights_c, model.pack_int4(case.weights_q))
    assert case.requant.multiplier.size == spec.weight_shape[0]
    # Defaults apply when extras are absent.
    _, defaulted = _case({**CONV, "weight_dtype": "S4"})
    assert defaulted.input_quant.zero_point == 3


def test_s4_fc_is_per_tensor_with_output_scale_tied_to_input_times_weight() -> None:
    fc = {"name": "f", "input_shape": [1, 10], "filter_shape": [3, 10], "activation_dtype": "S8", "weight_dtype": "S4",
          "hint": {"extras": {"input_scale": 2.0, "weight_scale": 0.5, "output_scale": 9.0}}}
    _, case = _case(fc, wt.fc_spec)
    assert not case.per_channel and case.output_quant.scale == pytest.approx(1.0)


def test_s4_rejects_unsupported_combinations() -> None:
    with pytest.raises(wt.ReferenceCaseError, match="s8 activations"):
        _case({**S4_CONV, "activation_dtype": "S16"})
    with pytest.raises(wt.ReferenceCaseError, match="hint.extras"):
        _case({**S4_CONV, "quantization": {"input": {"scale": 1.0}}})
    with pytest.raises(wt.ReferenceCaseError, match="weight scales"):
        _case({**S4_CONV, "hint": {"extras": {"weight_scale": [1.0, 2.0]}}})


def test_unsupported_dtypes_are_refused() -> None:
    with pytest.raises(wt.ReferenceCaseError, match="s8/s16"):
        _case({**CONV, "activation_dtype": "FP32"})
    with pytest.raises(wt.ReferenceCaseError, match="weight dtype"):
        _case({**CONV, "weight_dtype": "S16"})


def test_pack_int4_layout_and_bounds() -> None:
    np.testing.assert_array_equal(policy.pack_int4(np.array([1, -1, 7], np.int8)), np.array([-15, 7], np.int8))
    with pytest.raises(ValueError):
        policy.pack_int4(np.array([8], np.int8))
    with pytest.raises(ValueError):
        policy.pack_int4(np.array([], np.int8))


def test_glorot_gain_validation() -> None:
    with pytest.raises(ValueError):
        draw.glorot_uniform(np.random.default_rng(0), (2, 2), 2, 2, gain=0.0)
    w = draw.glorot_uniform(np.random.default_rng(0), (1000,), 10, 10, gain=1.0)
    assert np.abs(w).max() <= np.sqrt(6 / 20)


def test_activation_bounds_use_tflm_rounding() -> None:
    _, case = _case({**CONV, "activation": "RELU6"})
    assert (case.act_min, case.act_max) == params.activation_range("RELU6", case.output_quant.scale,
                                                                   case.output_quant.zero_point, "s8")


# ---- the generation driver ----


def test_reference_case_writes_provenance_and_no_model(tmp_path) -> None:
    import json

    from helia_core_tester.generation.test_ops import _manifest_entry, generate_test

    desc = {**CONV, "name": "conv_ref_driver_s8", "use_bias": True, "hint": {"call_style": "baseline"}}
    generate_test(desc, str(tmp_path), seed=9, cpu="cortex-m4", run_seed=123)
    case = next(p.parent for p in tmp_path.rglob("descriptor.yaml") if p.parent.name == desc["name"])
    assert not list(case.glob("*.tflite"))
    record = json.loads((case / f"{desc['name']}.reference.json").read_text())
    assert record["kernel"] == "conv_s8" and record["seeds"] == {"run_seed": 123, "case_seed": 9}
    assert record["library_key"]
    # Harness-rendered operators write no sidecar; reference.json is their provenance.
    assert not list(case.glob("*.sidecar.json"))
    entry = _manifest_entry(desc, test_dir=case, generated_tests_dir=tmp_path, cpu="cortex-m4", reused=False)
    assert entry["reference"] == str(case / f"{desc['name']}.reference.json")


def test_float_cases_stay_on_the_converter_path() -> None:
    from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import OpConvolve
    from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import OpFullyConnected

    assert OpConvolve({**CONV, "activation_dtype": "S16"}, seed=1).uses_reference()
    assert not OpConvolve({**CONV, "activation_dtype": "FP32", "weight_dtype": "FP32"}, seed=1).uses_reference()
    assert not OpFullyConnected({"name": "f", "activation_dtype": "FP16", "weight_dtype": "FP16"}, seed=1).uses_reference()


def test_reference_rng_streams_are_independent_of_the_input_draw() -> None:
    from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import OpConvolve

    op = OpConvolve(CONV, seed=5)
    a = op.reference_rng("weights").integers(0, 1 << 30, size=8)
    assert not np.array_equal(a, op._seeded_rng().integers(0, 1 << 30, size=8))
    assert not np.array_equal(a, op.reference_rng("bias").integers(0, 1 << 30, size=8))
    np.testing.assert_array_equal(a, OpConvolve(CONV, seed=5).reference_rng("weights").integers(0, 1 << 30, size=8))
