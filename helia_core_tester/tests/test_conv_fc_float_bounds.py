"""Float comparisons of the accumulating operators (Convolve, DepthwiseConv, TransposeConv, FullyConnected,
BatchMatMul): on the shipped build a Helium float16 case keeps its emulated golden at assert_close's defaults;
every exact golden (float32 on every build, float16 off the shipped build) is widened by kf.linear_bound over
the terms each output sums. The bound is recomputed here in float64 from the recorded operands."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops import get_op_map
from helia_core_tester.generation.ops._shared import conv_reference as cr
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.reference.call import ReferenceCall

SEED = 500
U = {"f32": 2.0 ** -24, "f16": 2.0 ** -11}
DEFAULTS = {"f32": (1e-5, 1.3e-6), "f16": (1e-5, 1e-3)}
EXACT = {"Convolve": "conv", "DepthwiseConv": "depthwise_conv", "TransposeConv": "transpose_conv",
         "FullyConnected": "fully_connected", "BatchMatMul": "batch_matmul"}
_VALIDATION = re.compile(r"HELIA_VALIDATE_(?:OUTPUTS\(\s*FLOAT|FLOATS_MASKED\()(?:[^;]*?,)+?\s*"
                         r"([-0-9.e+]+)f,\s*([-0-9.e+]+)f,\s*\d+,\s*failures")


@pytest.fixture(scope="module")
def descriptors():
    return {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}


def _generate(desc, tmp_path: Path, numerics: str, cpu: str = "cortex-m55"):
    op = get_op_map()[desc["operator"]](copy.deepcopy(desc), SEED, target_cpu=cpu)
    op.kernel_numerics = numerics
    out = tmp_path / f"{desc['name']}_{cpu}_{numerics}"
    out.mkdir()
    op.generate_c_files(out)
    found = _VALIDATION.findall("\n".join(p.read_text() for p in out.glob("*.c")))
    assert len(found) == 1, found
    return op.reference, float(found[0][0]), float(found[0][1])


def _abs(call: ReferenceCall, name: str) -> np.ndarray:
    return np.abs(call.inputs[name].astype(np.float64))


def _conv_terms(call):
    x, w, b = _abs(call, "input"), _abs(call, "filter"), _abs(call, "bias")
    p = call.params
    n, h, wd, c = x.shape
    o, kh, kw, fc = w.shape
    per_group = o // (c // fc)
    _, oh, ow, _ = call.output_shapes["output"]
    mag = np.zeros((n, oh, ow, o))
    k = 0
    for oy in range(oh):
        for ox in range(ow):
            taps = [(ky, kx) for ky in range(kh) for kx in range(kw)
                    if 0 <= oy * p["stride_h"] - p["pad_h"] + ky * p["dilation_h"] < h
                    and 0 <= ox * p["stride_w"] - p["pad_w"] + kx * p["dilation_w"] < wd]
            k = max(k, len(taps) * fc + int(b.size > 0))
            for oc in range(o):
                g = oc // per_group
                for ky, kx in taps:
                    iy = oy * p["stride_h"] - p["pad_h"] + ky * p["dilation_h"]
                    ix = ox * p["stride_w"] - p["pad_w"] + kx * p["dilation_w"]
                    mag[:, oy, ox, oc] += x[:, iy, ix, g * fc:(g + 1) * fc] @ w[oc, ky, kx]
                mag[:, oy, ox, oc] += b[oc] if b.size else 0.0
    return mag, k


def _depthwise_terms(call):
    x, w, b = _abs(call, "input"), _abs(call, "filter"), _abs(call, "bias")
    p = call.params
    n, h, wd, c = x.shape
    _, kh, kw, oc_n = w.shape
    mult = oc_n // c
    _, oh, ow, _ = call.output_shapes["output"]
    mag = np.zeros((n, oh, ow, oc_n))
    k = 0
    for oy in range(oh):
        for ox in range(ow):
            taps = [(ky, kx) for ky in range(kh) for kx in range(kw)
                    if 0 <= oy * p["stride_h"] - p["pad_h"] + ky * p["dilation_h"] < h
                    and 0 <= ox * p["stride_w"] - p["pad_w"] + kx * p["dilation_w"] < wd]
            k = max(k, len(taps) + int(b.size > 0))
            for ky, kx in taps:
                iy = oy * p["stride_h"] - p["pad_h"] + ky * p["dilation_h"]
                ix = ox * p["stride_w"] - p["pad_w"] + kx * p["dilation_w"]
                mag[:, oy, ox] += x[:, iy, ix, np.arange(oc_n) // mult] * w[0, ky, kx]
    return mag + (b if b.size else 0.0), k


def _transpose_terms(call):
    x, w, b = _abs(call, "input"), _abs(call, "filter"), _abs(call, "bias")
    p = call.params
    n, h, wd, c = x.shape
    o, kh, kw, _ = w.shape
    out = call.output_shapes["output"]
    mag = np.zeros(out)
    count = np.zeros(out[1:3], dtype=int)
    for iy in range(h):
        for ix in range(wd):
            for ky in range(kh):
                for kx in range(kw):
                    oy, ox = iy * p["stride_h"] - p["pad_h"] + ky, ix * p["stride_w"] - p["pad_w"] + kx
                    if 0 <= oy < out[1] and 0 <= ox < out[2]:
                        mag[:, oy, ox] += x[:, iy, ix] @ w[:, ky, kx].T
                        count[oy, ox] += 1
    return mag + (b if b.size else 0.0), int(count.max()) * c + int(b.size > 0)


def _fc_terms(call):
    x, w, b = _abs(call, "input"), _abs(call, "filter"), _abs(call, "bias")
    x = x.reshape(call.output_shapes["output"][0], -1)
    return x @ w.T + (b if b.size else 0.0), w.shape[1] + int(b.size > 0)


def _bmm_terms(call):
    lhs, rhs = _abs(call, "lhs"), _abs(call, "rhs")
    a = np.swapaxes(lhs, -1, -2) if call.params["adj_x"] else lhs
    bm = np.swapaxes(rhs, -1, -2) if call.params["adj_y"] else rhs
    return np.matmul(a, bm), a.shape[-1]


TERMS = {"conv": _conv_terms, "depthwise_conv": _depthwise_terms, "transpose_conv": _transpose_terms,
         "fully_connected": _fc_terms, "batch_matmul": _bmm_terms}


def _expected(call, dtype: str):
    mag, k = TERMS[call.entry.rsplit("_", 1)[0]](call)
    finite = mag[np.isfinite(mag)]
    atol, rtol = DEFAULTS[dtype]
    return atol + (k + 2) * U[dtype] * float(finite.max(initial=0.0)), rtol + 2 * U[dtype], k


def _assert_bound(call, atol: float, rtol: float, dtype: str):
    want_atol, want_rtol, _ = _expected(call, dtype)
    # The kernel-side magnitude is the exact entry's own sum, rounded once to the case dtype.
    assert atol - DEFAULTS[dtype][0] == pytest.approx(want_atol - DEFAULTS[dtype][0],
                                                      rel=2 * U[dtype] + 1e-9)
    assert rtol == pytest.approx(want_rtol, rel=1e-6)


CASES = [
    "convolve_float_default_f16", "convolve_float_dilated_valid_f32", "convolve_float_dilated_valid_f16",
    "convolve_float_small_c1_k3_same_dilated_f16", "convolve_float_stride2_valid_f32",
    "depthwise_conv_float_depth_multiplier2_f16", "depthwise_conv_float_dilated_f32",
    "transpose_conv_float_default_f16", "transpose_conv_float_no_bias_f32", "transpose_conv_float_valid_f16",
    "fully_connected_float_no_bias_f16", "fully_connected_float_default_f32",
    "batch_matmul_float_adj_x_f16", "batch_matmul_float_adj_x_adj_y_f32", "batch_matmul_float_batched_f16",
]


@pytest.mark.parametrize("name", CASES)
def test_bounded_build_emits_the_exact_golden_and_its_linear_bound(descriptors, tmp_path, name):
    desc = descriptors[name]
    dtype = name.rsplit("_", 1)[1]
    call, atol, rtol = _generate(desc, tmp_path, "bounded")
    assert call.entry == f"{EXACT[desc['operator']]}_{dtype}"
    _assert_bound(call, atol, rtol, dtype)
    assert atol > DEFAULTS[dtype][0]


@pytest.mark.parametrize("name", [c for c in CASES if c.endswith("_f16")])
def test_shipped_helium_f16_keeps_the_emulated_golden_at_assert_close(descriptors, tmp_path, name):
    desc = descriptors[name]
    call, atol, rtol = _generate(desc, tmp_path, "shipped")
    assert call.entry.endswith("_mve")
    assert (atol, rtol) == DEFAULTS["f16"]


@pytest.mark.parametrize("cpu", ["cortex-m55", "cortex-m4", "cortex-m0"])
@pytest.mark.parametrize("name", [c for c in CASES if c.endswith("_f32")])
def test_f32_accumulations_carry_the_bound_on_every_build(descriptors, tmp_path, name, cpu):
    desc = descriptors[name]
    shipped = _generate(desc, tmp_path, "shipped", cpu)
    bounded = _generate(desc, tmp_path, "bounded", cpu)
    assert shipped[0].entry == bounded[0].entry == f"{EXACT[desc['operator']]}_f32"
    assert shipped[1:] == bounded[1:]
    _assert_bound(shipped[0], *shipped[1:], "f32")


def _variant(desc, **changes):
    d = copy.deepcopy(desc)
    d.update(changes)
    return d


def test_grouped_strided_dilated_conv_counts_in_range_taps_per_group(descriptors, tmp_path):
    base = descriptors["convolve_float_default_f32"]
    for dtype, cmp_dtype in (("f32", "FP32"), ("f16", "FP16")):
        desc = _variant(base, name=f"convolve_float_grouped_{dtype}", input_shape=[1, 7, 6, 4],
                        filter_shape=[3, 3, 2, 6], groups=2, strides=[2, 1], dilation=[2, 2], padding="SAME",
                        activation_dtype=cmp_dtype, weight_dtype=cmp_dtype,
                        tensor_dtypes={r: cmp_dtype for r in ("input", "output", "weights", "bias")})
        call, atol, rtol = _generate(desc, tmp_path, "bounded")
        assert call.entry == f"conv_{dtype}"
        _assert_bound(call, atol, rtol, dtype)
        # 3x3 taps of 2 channels per group, all in range somewhere, plus the bias.
        assert _expected(call, dtype)[2] == 3 * 3 * 2 + 1


def test_a_descriptor_tightening_wins_over_the_bound(descriptors, tmp_path):
    for name in ("convolve_float_default_f32", "fully_connected_float_default_f32", "batch_matmul_float_default_f32"):
        desc = _variant(descriptors[name], comparison={"atol": 5e-6, "rtol": 1e-6})
        _, atol, rtol = _generate(desc, tmp_path, "bounded")
        assert (atol, rtol) == (5e-6, 1e-6)


def test_bias_absent_drops_the_bias_term(descriptors, tmp_path):
    with_bias = _generate(descriptors["fully_connected_float_default_f32"], tmp_path, "bounded")[0]
    without = _generate(descriptors["fully_connected_float_no_bias_f32"], tmp_path, "bounded")[0]
    assert _expected(with_bias, "f32")[2] == with_bias.inputs["filter"].shape[1] + 1
    assert _expected(without, "f32")[2] == without.inputs["filter"].shape[1]


@pytest.mark.parametrize("name", ["convolve_float_nonfinite_nan_f16", "depthwise_conv_float_nonfinite_inf_f16",
                                  "batch_matmul_float_nonfinite_inf_f32", "transpose_conv_float_nonfinite_nan_f32"])
@pytest.mark.parametrize("numerics", ["shipped", "bounded"])
def test_nonfinite_cases_bound_only_the_finite_lanes(descriptors, tmp_path, name, numerics):
    desc = descriptors[name]
    dtype = name.rsplit("_", 1)[1]
    call, atol, rtol = _generate(desc, tmp_path, numerics)
    if numerics == "shipped" and dtype == "f16":
        assert call.entry.endswith("_mve") and (atol, rtol) == DEFAULTS[dtype]
        return
    assert call.entry == f"{EXACT[desc['operator']]}_{dtype}"
    assert np.isfinite(atol) and atol < 1.0
    _assert_bound(call, atol, rtol, dtype)


# ------------------------------------------------------------------ helpers

def test_all_ones_conv_bound_by_hand():
    x = np.ones((1, 3, 3, 2), np.float32)
    w = np.ones((1, 3, 3, 2), np.float32)
    window = {"stride_h": 1, "stride_w": 1, "dilation_h": 1, "dilation_w": 1, "pad_h": 1, "pad_w": 1}
    call = ReferenceCall("conv_f32", {**window, "activation_min": -1.0, "activation_max": 1.0},
                         {"input": x, "filter": w, "bias": np.ones(1, np.float32)}, {"output": (1, 3, 3, 1)})
    mag = cr.operand_magnitude(call)
    # Corner 2x2 taps, edge 2x3, centre 3x3, each over 2 channels, plus the bias: the clamp is off.
    np.testing.assert_array_equal(mag.reshape(3, 3), [[9, 13, 9], [13, 19, 13], [9, 13, 9]])
    k = cr.window_taps(3, 3, 1, 1, 1, 3) ** 2 * 2 + 1
    assert k == 19
    op = type("Op", (), {"desc": {}, "kernel_numerics": "bounded", "target_cpu": "cortex-m55"})()
    ctx = cr.exact_comparison(op, call, k, np.float32)
    assert ctx == {"validation_atol": 1e-5 + 21 * 19 * 2.0 ** -24, "validation_rtol": 1.3e-6 + 2 * 2.0 ** -24,
                   "kernel_faithful": False}


def test_magnitude_ignores_the_clamp_and_keeps_nonfinite_lanes_nonfinite():
    x = np.array([[1.0, -2.0, 3.0], [np.inf, 0.0, 0.0]], np.float32)
    w = np.array([[1.0, -1.0, 0.5], [2.0, 1.0, 1.0]], np.float32)
    call = ReferenceCall("fully_connected_f32", {"activation_min": -0.5, "activation_max": 0.5},
                         {"input": x, "filter": w, "bias": np.array([-1.0, 0.0], np.float32)},
                         {"output": (2, 2)})
    mag = cr.operand_magnitude(call)
    np.testing.assert_array_equal(mag[0], [1 + 2 + 1.5 + 1, 2 + 2 + 3])
    assert np.all(np.isinf(mag[1]))
    assert kf.linear_bound(mag, 4, np.float32)[0] == 6 * 2.0 ** -24 * 7.0
    assert call.params["activation_max"] == 0.5  # the recorded call is untouched


def test_magnitude_needs_an_entry_with_a_clamp():
    call = kf.softmax_f32_cmsis_call(np.zeros((1, 4), np.float32), 0)
    with pytest.raises(ValueError, match="activation clamp"):
        cr.operand_magnitude(call)


@pytest.mark.parametrize("args,expected", [
    ((5, 3, 1, 1, 1, 5), 3),   # SAME: interior positions see every tap
    ((5, 3, 1, 0, 1, 3), 3),   # VALID
    ((4, 3, 2, 1, 2, 2), 2),   # strided and dilated: (0,2,4)-1 -> 2 of 3 in range at most
    ((0, 3, 1, 0, 1, 2), 0),   # empty input: no tap is summed
    ((5, 3, 1, 0, 1, 0), 0),   # no output positions
    ((1, 3, 1, 5, 1, 1), 0),   # every tap in the padding
])
def test_window_taps(args, expected):
    assert cr.window_taps(*args) == expected


@pytest.mark.parametrize("args,expected", [
    ((3, 3, 2, 1, 6), 2),      # stride 2: at most 2 of 3 taps overlap
    ((3, 3, 1, 0, 5), 3),
    ((2, 1, 2, 0, 4), 1),      # 1-tap stride 2: no overlap
    ((0, 3, 1, 0, 2), 0),      # empty input
    ((3, 3, 1, 0, 0), 0),      # empty output
])
def test_transpose_window_taps(args, expected):
    assert cr.transpose_window_taps(*args) == expected


@pytest.mark.parametrize("fn,args", [
    (cr.window_taps, (5, 0, 1, 0, 1, 5)), (cr.window_taps, (5, 3, 0, 0, 1, 5)), (cr.window_taps, (5, 3, 1, 0, 0, 5)),
    (cr.window_taps, (-1, 3, 1, 0, 1, 5)), (cr.transpose_window_taps, (3, 0, 1, 0, 3)),
    (cr.transpose_window_taps, (3, 3, 0, 0, 3)),
])
def test_tap_counts_reject_bad_windows(fn, args):
    with pytest.raises(ValueError, match="window"):
        fn(*args)


def test_zero_term_reduction_bound_is_the_final_rounding_only():
    assert kf.linear_bound(np.zeros(0), 0, np.float16) == (0.0, 2 * 2.0 ** -11)
    op = type("Op", (), {"desc": {}, "kernel_numerics": "bounded", "target_cpu": "cortex-m55"})()
    ctx = kf.comparison_context(op, np.float16, *kf.linear_bound(np.zeros(3), 0, np.float16))
    assert (ctx["validation_atol"], ctx["kernel_faithful"]) == (1e-5, False)
