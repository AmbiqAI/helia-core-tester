"""Kernel-faithful float16 variants of the misc family (transpose_conv_f16_mve, avg_pool_f16_mve, softmax_f16_mve,
reduce_sum_f16_f32acc, mean_f16_f32acc) against independent restatements of the ns-cmsis-nn Helium arithmetic,
hand-checkable cases, their parameter and shape checks, the route selection the generators use, and the stored
softmax tables against the checkout's."""

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
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings

f32 = np.float32
UNBOUNDED = (-np.inf, np.inf)


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def with_params(call, **params):
    return replace(call, params={**call.params, **params})


def with_output(call, shape):
    return replace(call, output_shapes={"output": shape})


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _round(f: Fraction, dtype) -> float:
    c = dtype(float(f))
    cands = [c, np.nextafter(c, dtype(np.inf)), np.nextafter(c, dtype(-np.inf))]
    uint = np.uint16 if dtype == np.float16 else np.uint32
    return float(min(cands, key=lambda v: (abs(Fraction(float(v)) - f), int(np.array(v, dtype).view(uint)) & 1)))


def fma(a, b, c, dtype) -> float:
    return _round(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)), dtype)


def h(v) -> float:
    return float(np.float16(v))


def bits16(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float16).view(np.uint16)


def assert_bits(got, want) -> None:
    got, want = np.asarray(got, np.float16), np.asarray(want, np.float16)
    assert got.shape == want.shape
    same = (bits16(got) == bits16(want)) | (np.isnan(got) & np.isnan(want))
    assert same.all(), f"{np.argwhere(~same)[:4].tolist()}: got {got[~same][:4]} want {want[~same][:4]}"


def clamp(v, lo, hi):
    return v if np.isnan(v) else min(max(v, h(lo)), h(hi))


# ------------------------------------------------------------- transpose conv

def tc_restated(x, w, bias, window, out_shape, act=UNBOUNDED):
    n, ih, iw, ic = x.shape
    oc, kh, kw, _ = w.shape
    out = np.zeros(out_shape, np.float64)
    out[...] = 0.0 if bias is None else bias.astype(np.float64)
    for b in range(n):
        for iy in range(ih):
            for ix in range(iw):
                for c in range(ic):
                    v = float(x[b, iy, ix, c])
                    if v == 0.0:
                        continue
                    for ky in range(kh):
                        oy = iy * window["stride_h"] - window["pad_h"] + ky * window["dilation_h"]
                        if not 0 <= oy < out_shape[1]:
                            continue
                        for kx in range(kw):
                            ox = ix * window["stride_w"] - window["pad_w"] + kx * window["dilation_w"]
                            if not 0 <= ox < out_shape[2]:
                                continue
                            for o in range(oc):
                                out[b, oy, ox, o] = fma(v, w[o, ky, kx, c], out[b, oy, ox, o], np.float16)
    return np.vectorize(lambda v: clamp(v, *act))(out).astype(np.float16)


def _window(stride=2, pad=0, dilation=1):
    return {"stride_h": stride, "stride_w": stride, "dilation_h": dilation, "dilation_w": dilation,
            "pad_h": pad, "pad_w": pad}


@pytest.mark.parametrize("shape,filt,stride,pad,dilation,use_bias,act", [
    ((1, 3, 3, 2), (3, 2, 2, 2), 2, 0, 1, True, UNBOUNDED),
    ((1, 4, 4, 2), (3, 3, 3, 2), 2, 1, 1, True, (-0.5, 0.5)),
    ((2, 3, 2, 3), (9, 3, 3, 3), 1, 1, 1, False, UNBOUNDED),
    ((1, 3, 3, 1), (2, 2, 2, 1), 1, 0, 2, True, UNBOUNDED),
])
def test_transpose_conv_matches_restatement(lib, shape, filt, stride, pad, dilation, use_bias, act):
    rng = np.random.default_rng(sum(shape) + sum(filt))
    x = rng.uniform(-2, 2, shape).astype(np.float16)
    x.flat[::5] = 0.0
    w = rng.uniform(-1, 1, filt).astype(np.float16)
    bias = rng.uniform(-0.5, 0.5, filt[0]).astype(np.float16) if use_bias else None
    win = _window(stride, pad, dilation)
    ext = lambda i, k: (i - 1) * stride + (k - 1) * dilation + 1 - pad  # noqa: E731
    out_shape = (shape[0], ext(shape[1], filt[1]), ext(shape[2], filt[2]), filt[0])
    call = kf.transpose_conv_f16_mve_call(x, w, bias, win, out_shape, act)
    got = call.output()
    assert_bits(got, tc_restated(x, w, bias, win, out_shape, act))
    patches, rows = kf.transpose_conv_patches(x, w, win, out_shape)
    kf.check_within_bound("t", kf.unclamped_output(call).reshape(1, -1, filt[0]), patches, rows, bias, np.float16)


def test_transpose_conv_hand_checked(lib):
    x = np.array([[[[3.0]]]], np.float16)
    w = np.array([[[[0.1]]]], np.float16)
    got = kf.transpose_conv_f16_mve_call(x, w, np.array([1.0], np.float16), _window(1), (1, 1, 1, 1),
                                         UNBOUNDED).output()
    assert float(got.ravel()[0]) == fma(3.0, h(0.1), 1.0, np.float16)
    # The kernel skips zero inputs, so an infinite weight behind one leaves the bias.
    z = kf.transpose_conv_f16_mve_call(np.zeros((1, 1, 1, 1), np.float16), np.full((1, 1, 1, 1), np.inf, np.float16),
                                       np.array([0.25], np.float16), _window(1), (1, 1, 1, 1), UNBOUNDED).output()
    assert float(z.ravel()[0]) == 0.25


def test_transpose_conv_rejects_bad_params_and_shapes(lib):
    x = np.ones((1, 2, 2, 2), np.float16)
    w = np.ones((3, 2, 2, 2), np.float16)
    call = kf.transpose_conv_f16_mve_call(x, w, None, _window(), (1, 4, 4, 3), UNBOUNDED)
    assert _status(lambda: with_params(call, stride_h=0).output()) == "E_PARAM"
    assert _status(lambda: with_params(call, activation_min=1.0, activation_max=-1.0).output()) == "E_PARAM"
    assert _status(lambda: call.with_inputs(filter=np.ones((3, 2, 2, 1), np.float16)).output()) == "E_SHAPE"
    assert _status(lambda: call.with_inputs(bias=np.ones(2, np.float16)).output()) == "E_SHAPE"
    with pytest.raises(ValueError):
        kf.transpose_conv_f16_mve_call(x, np.ones((3, 2, 2, 1), np.float16), None, _window(), (1, 4, 4, 3),
                                       UNBOUNDED)


def test_transpose_conv_route_follows_the_gather_stride_limit():
    assert kf.transpose_conv_f16_mve_route(3, 3, 2)
    assert kf.transpose_conv_f16_mve_route(1, 1, 65535 // 7)
    assert not kf.transpose_conv_f16_mve_route(1, 1, 65535 // 7 + 1)


# ------------------------------------------------------------- avg pool

def pool_restated(x, win, out_shape, act=UNBOUNDED):
    n, ih, iw, c = x.shape
    out = np.zeros(out_shape, np.float16)
    for b in range(n):
        for oy in range(out_shape[1]):
            for ox in range(out_shape[2]):
                by, bx = oy * win["stride_h"] - win["pad_h"], ox * win["stride_w"] - win["pad_w"]
                cells = [(by + ky, bx + kx) for ky in range(win["filter_h"]) for kx in range(win["filter_w"])
                         if 0 <= by + ky < ih and 0 <= bx + kx < iw]
                inv = h(f32(1.0) / f32(len(cells)))
                for ch in range(c):
                    s = 0.0
                    for y, xx in cells:
                        s = h(s + float(x[b, y, xx, ch]))
                    out[b, oy, ox, ch] = clamp(h(s * inv), *act)
    return out


@pytest.mark.parametrize("shape,k,stride,pad,act", [
    ((1, 6, 6, 3), 2, 2, 0, UNBOUNDED),
    ((1, 5, 5, 9), 3, 2, 1, (-0.25, 0.25)),
    ((2, 4, 7, 17), 3, 1, 1, UNBOUNDED),
])
def test_avg_pool_matches_restatement(lib, shape, k, stride, pad, act):
    rng = np.random.default_rng(sum(shape))
    x = rng.uniform(-1, 1, shape).astype(np.float16)
    win = {"stride_h": stride, "stride_w": stride, "filter_h": k, "filter_w": k, "pad_h": pad, "pad_w": pad}
    out_shape = (shape[0], (shape[1] + 2 * pad - k) // stride + 1, (shape[2] + 2 * pad - k) // stride + 1, shape[3])
    call = kf.avg_pool_f16_mve_call(x, win, out_shape, act)
    got = call.output()
    assert_bits(got, pool_restated(x, win, out_shape, act))
    taps, weights = kf.avg_pool_windows(x, win, out_shape)
    kf.check_within_bound("p", kf.unclamped_output(call), taps, weights, None, np.float16)


def test_avg_pool_hand_checked(lib):
    x = np.array([0.5, 0.25, 0.125], np.float16).reshape(1, 1, 3, 1)
    win = {"stride_h": 1, "stride_w": 1, "filter_h": 1, "filter_w": 3, "pad_h": 0, "pad_w": 0}
    got = kf.avg_pool_f16_mve_call(x, win, (1, 1, 1, 1), UNBOUNDED).output()
    assert float(got.ravel()[0]) == h(0.875 * h(f32(1) / f32(3)))


def test_avg_pool_rejects_bad_params_and_shapes(lib):
    win = {"stride_h": 1, "stride_w": 1, "filter_h": 2, "filter_w": 2, "pad_h": 0, "pad_w": 0}
    call = kf.avg_pool_f16_mve_call(np.ones((1, 3, 3, 2), np.float16), win, (1, 2, 2, 2), UNBOUNDED)
    assert _status(lambda: with_params(call, filter_h=0).output()) == "E_PARAM"
    assert _status(lambda: with_params(call, pad_w=-1).output()) == "E_PARAM"
    assert _status(lambda: with_output(call, (1, 2, 2, 3)).output()) == "E_SHAPE"


# ------------------------------------------------------------- softmax

def _tables():
    src = (Path(__file__).resolve().parents[1] / "reference" / "src" / "softmax" / "softmax_f16_mve.c").read_text()
    poly_body = src[src.index("poly_bits[8]"):]
    poly = [struct.unpack("<f", struct.pack("<I", int(v, 16)))[0]
            for v in re.findall(r"0x([0-9a-f]{8})u", poly_body[:poly_body.index("};")])]
    lut_body = src[src.index("exp2_lut_f16[257]"):]
    lut = [int(v, 16) for v in re.findall(r"0x([0-9a-f]{4})u", lut_body[:lut_body.index("};")])]
    return poly, lut


POLY, LUT = _tables()
LOG2E, LN2 = f32(1.4426950408), f32(0.6931471805)


def cvt_s32(v) -> int:
    v = float(v)
    if np.isnan(v):
        return 0
    return int(max(min(v, 2.0 ** 31 - 1), -2.0 ** 31))


def exp_poly(x) -> float:
    x = f32(x)
    m = cvt_s32(f32(x * LOG2E))
    r = f32(fma(-f32(m), LN2, x, np.float32))
    c = [f32(v) for v in POLY]
    a, b = fma(c[4], r, c[0], np.float32), fma(c[6], r, c[2], np.float32)
    cc, d = fma(c[5], r, c[1], np.float32), fma(c[7], r, c[3], np.float32)
    x2 = f32(r * r)
    x4 = f32(x2 * x2)
    y = fma(fma(d, x2, cc, np.float32), x4, fma(b, x2, a, np.float32), np.float32)
    ybits = int(np.array(y, f32).view(np.int32)) + max(min(m << 23, 2 ** 31 - 1), -2 ** 31)
    y = float(np.array(max(min(ybits, 2 ** 31 - 1), -2 ** 31), np.int32).view(f32))
    return 0.0 if m < -126 else h(y)


def exp_lut(x) -> float:
    x = f32(max(min(float(x), 80.0), -80.0))
    t = f32(x * LOG2E)
    tq = h(t)
    n = cvt_s32(tq)
    n = n - 1 if tq < n else n
    idx_f = f32(f32(t - f32(n)) * f32(256.0))
    idx = min(max(cvt_s32(idx_f), 0), 255)
    frac = f32(idx_f - f32(idx))
    y0, y1 = (float(np.array(LUT[i], np.uint16).view(np.float16)) for i in (idx, idx + 1))
    y = fma(frac, f32(f32(y1) - f32(y0)), y0, np.float32)
    n = min(max(n, -14), 15)
    return h(f32(f32(y) * f32(2.0 ** n)))


def softmax_restated(x) -> np.ndarray:
    rows = x.reshape(-1, x.shape[-1]).astype(np.float64)
    out = np.zeros_like(rows)
    if rows.shape == (1, 2):
        a, b = rows[0]
        if a >= b:
            e = exp_lut(h(b - a))
            s = h(1.0 + e)
            out[0] = [h(1.0 / s), h(e / s)]
        else:
            e = exp_lut(h(a - b))
            s = h(e + 1.0)
            out[0] = [h(e / s), h(1.0 / s)]
        return out.reshape(x.shape).astype(np.float16)
    for r, row in enumerate(rows):
        mx = max([-65504.0] + [v for v in row if not np.isnan(v)])
        e = [exp_poly(h(v - mx)) for v in row]
        total = f32(0.0)
        for c0 in range(0, len(e), 8):
            lane = (e[c0:c0 + 8] + [0.0] * 8)[:8]
            part = h(h(h(lane[0] + lane[1]) + h(lane[2] + lane[3])) + h(h(lane[4] + lane[5]) + h(lane[6] + lane[7])))
            total = f32(total + f32(part))
        inv = h(f32(1.0) / total)
        out[r] = [h(v * inv) for v in e]
    return out.reshape(x.shape).astype(np.float16)


@pytest.mark.parametrize("shape", [(1, 3), (4, 7), (2, 8), (3, 9), (1, 17), (2, 2), (1, 1)])
def test_softmax_matches_restatement(lib, shape):
    rng = np.random.default_rng(sum(shape))
    x = rng.uniform(-3, 3, shape).astype(np.float16)
    call = kf.softmax_f16_mve_call(x)
    got = call.output()
    assert_bits(got, softmax_restated(x))
    kf.check_close_to_exact("s", call, replace(call, entry="softmax_f16"), kf.softmax_f16_atol(x))


@pytest.mark.parametrize("pair", [(0.5, -0.25), (-0.25, 0.5), (1.0, 1.0), (-7.0, 3.0)])
def test_softmax_1x2_specialisation(lib, pair):
    x = np.array([pair], np.float16)
    assert_bits(kf.softmax_f16_mve_call(x).output(), softmax_restated(x))


def test_softmax_hand_checked(lib):
    assert float(kf.softmax_f16_mve_call(np.array([[0.75]], np.float16)).output().ravel()[0]) == 1.0
    assert kf.softmax_f16_mve_call(np.array([[2.0, 2.0]], np.float16)).output().ravel().tolist() == [0.5, 0.5]
    # A 2-row tensor of width 2 takes the polynomial path, not the 1 x 2 specialisation.
    assert_bits(kf.softmax_f16_mve_call(np.array([[0.5, -0.25], [1.0, 0.0]], np.float16)).output(),
                softmax_restated(np.array([[0.5, -0.25], [1.0, 0.0]], np.float16)))


def test_softmax_degenerate_row_is_nan_and_others_finite(lib):
    x = np.array([[-np.inf] * 3, [0.0, 1.0, 2.0]], np.float16)
    got = kf.softmax_f16_mve_call(x).output()
    assert np.isnan(got[0]).all() and np.isfinite(got[1]).all()


def test_softmax_rejects_shape_mismatch(lib):
    call = kf.softmax_f16_mve_call(np.zeros((2, 3), np.float16))
    assert _status(lambda: with_output(call, (3, 2)).output()) == "E_SHAPE"


def test_softmax_guard_rejects_a_wrong_emulation(lib):
    x = np.random.default_rng(3).uniform(-1, 1, (2, 5)).astype(np.float16)
    call = kf.softmax_f16_mve_call(x)
    other = kf.softmax_f16_mve_call(np.flip(x, axis=1).copy())
    with pytest.raises(kf.KernelDeviationError):
        kf.check_close_to_exact("s", call, other, kf.softmax_f16_atol(x))


def test_stored_softmax_tables_are_the_checkouts(lib):
    root = os.environ.get("CMSIS_NN_ROOT")
    src = Path(root) / "Source" / "NNSupportFunctions" / "arm_nntables_flt.c" if root else None
    if src is None or not src.exists():
        pytest.skip("no ns-cmsis-nn checkout")
    text = src.read_text()
    poly = text[text.index("arm_nn_exp_poly_coeffs_f16[8] = {"):]
    vals = [v.strip().rstrip("fF") for v in poly[poly.index("{") + 1:poly.index("};")].split(",") if v.strip()]
    assert [float(f32(float(v))) for v in vals] == [float(v) for v in POLY]
    lut = text[text.index("arm_nn_exp2_lut_f16[257] = {"):]
    assert [int(v, 16) for v in re.findall(r"0x([0-9A-Fa-f]{4})u", lut[:lut.index("};")])] == LUT


# ------------------------------------------------------------- reduce_sum / mean

def reduce_restated(x, axes, *, mean, route) -> np.ndarray:
    rows, _ = kf.reduce_operands(x, axes, mean=False)
    k = rows.shape[-1]
    out = []
    for row in rows[:, 0, :]:
        if route == 1:
            lane = [f32(0.0)] * 8
            for j, v in enumerate(row):
                lane[j % 8] = f32(lane[j % 8] + f32(v))
            p = [f32(lane[2 * i] + lane[2 * i + 1]) for i in range(4)]
            s = f32(f32(f32(p[0] + p[1]) + p[2]) + p[3])
        else:
            s = f32(0.0)
            for v in row:
                s = f32(s + f32(v))
        out.append(h(f32(s * f32(f32(1.0) / f32(k)))) if mean else h(s))
    shape = tuple(1 if d in axes else s for d, s in enumerate(x.shape))
    return np.array(out, np.float16).reshape(shape)


@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("route", [0, 1])
@pytest.mark.parametrize("shape,axes", [((1, 3, 4, 5), [3]), ((1, 8, 8, 16), [1, 2, 3]), ((1, 3, 4, 5), [1, 3]),
                                        ((1, 8, 8, 16), [1, 2]), ((2, 3, 1, 33), [3])])
def test_reductions_match_restatement(lib, shape, axes, mean, route):
    x = np.random.default_rng(sum(shape) + len(axes)).uniform(-1, 1, shape).astype(np.float16)
    got = kf.reduce_f16_f32acc_call(x, axes, mean=mean, route=route).output()
    assert_bits(got, reduce_restated(x, axes, mean=mean, route=route))
    rows, weights = kf.reduce_operands(x, axes, mean=mean)
    kf.check_within_bound("r", got, rows, weights, None, np.float16)


def test_reduction_routes_round_differently_on_a_tie(lib):
    # Sequential: 1 + 2^-11 absorbs each 2^-24 (a float32 tie to even), then rounds to float16 1.0 (a tie).
    # Lanes: the two 2^-24 meet first, so the float32 sum sits above the float16 tie.
    x = np.array([1.0, 2.0 ** -11, 2.0 ** -24, 2.0 ** -24], np.float16).reshape(1, 1, 1, 4)
    seq = kf.reduce_f16_f32acc_call(x, [3], mean=False, route=0).output().ravel()[0]
    lanes = kf.reduce_f16_f32acc_call(x, [3], mean=False, route=1).output().ravel()[0]
    assert float(seq) == 1.0 and float(lanes) == 1.0 + 2.0 ** -10


def test_mean_hand_checked(lib):
    x = np.array([1, 2, 3, 4, 5], np.float16).reshape(1, 1, 1, 5)
    got = kf.reduce_f16_f32acc_call(x, [3], mean=True, route=1).output().ravel()[0]
    assert float(got) == h(f32(15.0) * f32(f32(1.0) / f32(5.0)))


def test_reductions_reject_bad_params_and_shapes(lib):
    call = kf.reduce_f16_f32acc_call(np.ones((1, 2, 2, 2), np.float16), [3], mean=True, route=0)
    assert _status(lambda: with_params(call, route=2).output()) == "E_PARAM"
    assert _status(lambda: with_params(call, axis_mask=1 << 4).output()) == "E_PARAM"
    assert _status(lambda: with_output(call, (1, 2, 2, 2)).output()) == "E_SHAPE"


@pytest.mark.parametrize("kind,shape,axes,route", [
    ("sum", (1, 8, 127, 1), [1], 0),     # spatial
    ("sum", (1, 3, 8, 33), [2], 0),      # spatial
    ("sum", (1, 4, 4, 1), [1, 2], 0),    # spatial is checked before the flatten suffix
    ("mean", (1, 4, 4, 1), [1, 2], 1),   # mean has no spatial path: flatten suffix (c is 1)
    ("sum", (1, 3, 4, 5), [3], 1),
    ("mean", (1, 8, 8, 16), [1, 2, 3], 1),
    ("mean", (1, 8, 8, 16), [1, 2], 0),  # middle block
    ("sum", (1, 3, 4, 5), [1, 3], 0),    # generic
    ("mean", (1, 3, 4, 5), [1], 0),      # middle block
    ("sum", (2, 3, 4, 5), [0], 0),       # n reduced: generic
])
def test_reduction_route_selection(kind, shape, axes, route):
    assert kf.reduce_f16_route(kind, shape, axes) == route


def test_reduction_route_rejects_bad_input():
    with pytest.raises(ValueError):
        kf.reduce_f16_route("sum", (3, 4, 5), [1])
    with pytest.raises(ValueError):
        kf.reduce_f16_route("max", (1, 3, 4, 5), [1, 3])
