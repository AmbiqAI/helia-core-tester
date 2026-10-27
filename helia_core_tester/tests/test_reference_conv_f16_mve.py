"""conv_f16_mve and matmul_nt_n_packed_f16_mve: ns-cmsis-nn's float16 convolution routes on Helium, bit for bit.

Each C route is checked against a NumPy restatement written from the kernels (arm_convolve_f16.c's patch GEMM,
direct and small-C legs, arm_convolve_1x1_f16.c, arm_nn_conv1d_k{3,5}{,_packed}_f16.c and
arm_nn_mat_mult_nt_n_packed_f16.c): fold and no fold, _acc16, channel tails, padded taps, bias and no bias. Then
hand-checkable cases, the entries' rejections, and the route selection that mirrors the routers.
"""

from __future__ import annotations

import numpy as np
import pytest

from helia_core_tester.generation.ops._shared import conv_reference as cr
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings

UNBOUNDED = (-1.0e30, 1.0e30)


def _h(v) -> float:
    return float(np.float16(v))


def _fma16(a, b, c) -> float:
    # A product of halves is exact in binary64 and so, at these magnitudes, is the sum: one rounding.
    return _h(np.float64(a) * np.float64(b) + np.float64(c))


def _fma32(a, b, c) -> float:
    return float(np.float32(np.float64(a) * np.float64(b) + np.float64(c)))


def _f32(v) -> float:
    return float(np.float32(v))


def _reduce8(l):
    a = _h(_h(l[0] + l[1]) + _h(l[2] + l[3]))
    b = _h(_h(l[4] + l[5]) + _h(l[6] + l[7]))
    return _h(a + b)


def _pairs(l, acc, first):
    p = [_f32(l[2 * j] + l[2 * j + 1]) for j in range(4)]
    return p if first else [_f32(acc[j] + p[j]) for j in range(4)]


def _pairs_sum(acc):
    return _f32(_f32(acc[0] + acc[1]) + _f32(acc[2] + acc[3]))


def _chain(a, b, start, fold):
    """Lane-per-column chain: 32-element blocks folded in float32 past 32 elements."""
    K = len(a)
    if not fold or K <= 32:
        acc = start
        for k in range(K):
            acc = _fma16(a[k], b[k], acc)
        return acc
    acc32 = None
    for k0 in range(0, K, 32):
        part = start if k0 == 0 else 0.0
        for k in range(k0, min(K, k0 + 32)):
            part = _fma16(a[k], b[k], part)
        acc32 = part if k0 == 0 else _f32(acc32 + part)
    return _h(acc32)


def _gather_cols(N):
    c = N - N % 16
    c += (N - c) - (N - c) % 8
    return c + (N - c) - (N - c) % 4


def _nt_t(row, wrow, c, N, bias, acc16):
    K = len(row)
    if K < 32 and c < _gather_cols(N):
        return _chain(row, wrow, bias, False)
    if not acc16 and K > 32:
        acc = [0.0] * 4
        for k0 in range(0, K, 256):
            lane = [0.0] * 8
            for k in range(k0, min(K, k0 + 256)):
                lane[(k - k0) % 8] = _fma16(row[k], wrow[k], lane[(k - k0) % 8])
            acc = _pairs(lane, acc, k0 == 0)
        return _h(_f32(bias + _pairs_sum(acc)))
    lane = [0.0] * 8
    for k in range(K):
        lane[k % 8] = _fma16(row[k], wrow[k], lane[k % 8])
    return _h(bias + _reduce8(lane))


def _packed_column(row, wrow, bias, policy):
    if policy == kf.NT_N_PACKED_PRECISE:
        acc32 = 0.0
        for k0 in range(0, len(row), 4):
            part = 0.0
            for k in range(k0, min(len(row), k0 + 4)):
                part = _fma16(row[k], wrow[k], part)
            acc32 = part if k0 == 0 and len(row) >= 4 else _f32(acc32 + part)
        return _h(_f32(acc32 + bias))
    return _chain(row, wrow, bias, policy == kf.NT_N_PACKED_FOLD)


def _taps(x, b, oy, ox, KH, KW, win):
    """In-range (ky, kx, iy, ix) in tap order."""
    H, W = x.shape[1:3]
    for ky in range(KH):
        iy = oy * win["stride_h"] - win["pad_h"] + ky * win["dilation_h"]
        for kx in range(KW):
            ix = ox * win["stride_w"] - win["pad_w"] + kx * win["dilation_w"]
            if 0 <= iy < H and 0 <= ix < W:
                yield ky, kx, iy, ix


def _direct_ohwi(x, w, bi, oy, ox, oc, bias, win, acc16):
    _, KH, KW, C = w.shape
    fold = not acc16 and KH * KW * C > 32
    lane, acc, first, n_vec, n_in = [0.0] * 8, [0.0] * 4, True, 0, 0
    for ky, kx, iy, ix in _taps(x, bi, oy, ox, KH, KW, win):
        n_in += C
        ic = 0
        while ic < C:
            end = C
            if fold:
                if n_vec == 32:
                    acc, first, lane, n_vec = _pairs(lane, acc, first), False, [0.0] * 8, 0
                if -(-(C - ic) // 8) > 32 - n_vec:
                    end = ic + (32 - n_vec) * 8
                n_vec += -(-(end - ic) // 8)
            for i in range(ic, end):
                lane[i % 8] = _fma16(x[bi, iy, ix, i], w[oc, ky, kx, i], lane[i % 8])
            ic = end
    if fold and n_in > 32:
        return _h(_f32(bias + _pairs_sum(_pairs(lane, acc, first))))
    return _h(bias + _reduce8(lane))


def _emulate(route, x, w, bias, win, out_shape, acc16):
    N, OH, OW, O = out_shape
    _, KH, KW, C = w.shape
    b = np.zeros(O) if bias is None else bias.astype(np.float64)
    xf, wf = x.astype(np.float64), w.astype(np.float64)
    out = np.zeros(out_shape, np.float16)
    patches = cr.conv_patches(x, KH, KW, win, (OH, OW)).astype(np.float64)
    for bi in range(N):
        for oy in range(OH):
            for ox in range(OW):
                row = patches[bi, oy * OW + ox]
                taps = list(_taps(xf, bi, oy, ox, KH, KW, win))
                for oc in range(O):
                    wrow = wf[oc].reshape(-1)
                    if route == kf.CONV_F16_PATCH_NT_T:
                        v = _nt_t(row, wrow, oc, O, b[oc], acc16)
                    elif route == kf.CONV_F16_PATCH_PACKED:
                        v = _packed_column(row, wrow, b[oc], kf.NT_N_PACKED_ACC16 if acc16 else kf.NT_N_PACKED_FOLD)
                    elif route == kf.CONV_F16_DIRECT_OHWI:
                        v = _direct_ohwi(xf, wf, bi, oy, ox, oc, b[oc], win, acc16)
                    elif route == kf.CONV_F16_DIRECT_PACKED:
                        a = [xf[bi, iy, ix, ic] for _, _, iy, ix in taps for ic in range(C)]
                        ww = [wf[oc, ky, kx, ic] for ky, kx, _, _ in taps for ic in range(C)]
                        v = _chain(a, ww, b[oc], not acc16 and KH * KW * C > 32)
                    elif route == kf.CONV_F16_SMALL_C:
                        acc = _f32(b[oc])
                        for ky, kx, iy, ix in taps:
                            for ic in range(C):
                                acc = _fma32(xf[bi, iy, ix, ic], wf[oc, ky, kx, ic], acc)
                        v = _h(acc)
                    elif route == kf.CONV_F16_CONV1D:
                        fold = not acc16 and KW * C > 32
                        span = (32 // KW) * 8 if fold else C
                        acc, lane = [0.0] * 4, [0.0] * 8
                        for ic0 in range(0, C, span):
                            lane = [0.0] * 8
                            for s in range(ic0, min(C, ic0 + span), 8):
                                for t in range(KW):
                                    for ic in range(s, min(s + 8, ic0 + span, C)):
                                        lane[ic % 8] = _fma16(xf[bi, 0, ox + t, ic], wf[oc, 0, t, ic], lane[ic % 8])
                            if fold:
                                acc = _pairs(lane, acc, ic0 == 0)
                        v = _h(_f32(b[oc] + _pairs_sum(acc))) if fold else _h(b[oc] + _reduce8(lane))
                    else:
                        fold = not acc16 and KW * C > 32
                        span = 32 // KW if fold else C
                        acc32 = 0.0
                        for ic0 in range(0, C, span):
                            part = b[oc] if ic0 == 0 else 0.0
                            for ic in range(ic0, min(C, ic0 + span)):
                                for t in range(KW):
                                    part = _fma16(wf[oc, 0, t, ic], xf[bi, 0, ox + t, ic], part)
                            acc32 = part if ic0 == 0 else _f32(acc32 + part)
                        v = _h(acc32) if fold else part
                    out[bi, oy, ox, oc] = v
    return out


def _out_hw(H, W, KH, KW, win):
    oh = (H + 2 * win["pad_h"] - win["dilation_h"] * (KH - 1) - 1) // win["stride_h"] + 1
    ow = (W + 2 * win["pad_w"] - win["dilation_w"] * (KW - 1) - 1) // win["stride_w"] + 1
    return oh, ow


def _win(s=(1, 1), p=(0, 0), d=(1, 1)):
    return {"stride_h": s[0], "stride_w": s[1], "pad_h": p[0], "pad_w": p[1], "dilation_h": d[0],
            "dilation_w": d[1]}


def _draw(seed, n, H, W, C, KH, KW, O, bias=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, H, W, C)).astype(np.float16)
    w = rng.uniform(-1, 1, (O, KH, KW, C)).astype(np.float16)
    b = rng.uniform(-0.5, 0.5, O).astype(np.float16) if bias else None
    return x, w, b


def _run(route, x, w, b, win, acc16=False, activation=UNBOUNDED, out_shape=None):
    if out_shape is None:
        oh, ow = _out_hw(x.shape[1], x.shape[2], w.shape[1], w.shape[2], win)
        out_shape = (x.shape[0], oh, ow, w.shape[0])
    return kf.conv_f16_mve_call(x, w, b, route=route, window=win, output_shape=out_shape, acc16=acc16,
                                activation=activation).output(), out_shape


GENERAL = [
    # (n, H, W, C, KH, KW, O, stride, pad, dilation)
    (1, 6, 6, 4, 3, 3, 12, (1, 1), (1, 1), (1, 1)),     # K 36: fold past 32, padded edges
    (1, 6, 6, 20, 5, 5, 5, (1, 1), (2, 2), (1, 1)),     # K 500: many fold blocks, channel tails of 4
    (2, 5, 7, 3, 3, 3, 9, (2, 1), (1, 1), (1, 1)),      # K 27 < 32: gather columns, batch 2
    (1, 7, 7, 8, 3, 3, 3, (1, 1), (0, 0), (1, 1)),      # K 72, three columns, no padding
    (1, 8, 8, 2, 3, 3, 3, (1, 1), (0, 0), (2, 2)),      # dilation 2
    (1, 4, 4, 16, 1, 1, 12, (1, 1), (0, 0), (1, 1)),    # 1x1, K 16
    (1, 8, 8, 16, 1, 1, 8, (2, 2), (0, 0), (1, 1)),     # 1x1 strided
    (1, 4, 4, 40, 1, 1, 21, (1, 1), (0, 0), (1, 1)),    # 1x1, K 40 folds; 16 + 4 gather never (K >= 32)
    (1, 3, 3, 300, 1, 1, 3, (1, 1), (0, 0), (1, 1)),    # K 300 > 256: two lane blocks in the nt_t fold
]


@pytest.mark.parametrize("shape", GENERAL)
@pytest.mark.parametrize("route", [kf.CONV_F16_PATCH_NT_T, kf.CONV_F16_PATCH_PACKED, kf.CONV_F16_DIRECT_OHWI,
                                   kf.CONV_F16_DIRECT_PACKED])
@pytest.mark.parametrize("acc16", [False, True])
def test_general_routes_match_kernel_restatement(shape, route, acc16):
    n, H, W, C, KH, KW, O, s, p, d = shape
    x, w, b = _draw(hash((shape, route)) & 0xFFFF, n, H, W, C, KH, KW, O)
    win = _win(s, p, d)
    got, out_shape = _run(route, x, w, b, win, acc16)
    want = _emulate(route, x, w, b, win, out_shape, acc16)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


@pytest.mark.parametrize("shape", [
    (1, 6, 30, 1, 3, 3, 8, (1, 1), (1, 1), (1, 1)),
    (2, 5, 61, 1, 3, 3, 5, (2, 2), (1, 1), (1, 1)),
    (1, 6, 29, 2, 3, 3, 8, (2, 2), (1, 1), (1, 1)),
    (1, 6, 30, 7, 3, 3, 3, (1, 1), (1, 1), (2, 2)),
    (1, 5, 9, 3, 5, 5, 4, (1, 1), (2, 2), (1, 1)),      # K 75: float32 accumulates past any block
])
@pytest.mark.parametrize("bias", [True, False])
def test_small_c_matches_kernel_restatement(shape, bias):
    n, H, W, C, KH, KW, O, s, p, d = shape
    x, w, b = _draw(hash(shape) & 0xFFFF, n, H, W, C, KH, KW, O, bias)
    win = _win(s, p, d)
    got, out_shape = _run(kf.CONV_F16_SMALL_C, x, w, b, win)
    want = _emulate(kf.CONV_F16_SMALL_C, x, w, b, win, out_shape, False)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


@pytest.mark.parametrize("shape", [
    (11, 4, 3, 8),     # K*C 12: no fold
    (6, 88, 3, 7),     # k3, 264 taps: fold every 80 channels (vector) / 10 channels (packed)
    (9, 16, 5, 23),    # k5, C 16: the kernel's OC16 block
    (7, 56, 5, 7),     # k5, fold every 48 / 6 channels, channel tail
    (12, 8, 5, 12),
    (9, 24, 3, 13),
])
@pytest.mark.parametrize("route", [kf.CONV_F16_CONV1D, kf.CONV_F16_CONV1D_PACKED])
@pytest.mark.parametrize("acc16", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_conv1d_matches_kernel_restatement(shape, route, acc16, bias):
    W, C, KW, O = shape
    x, w, b = _draw(hash((shape, route)) & 0xFFFF, 1, 1, W, C, 1, KW, O, bias)
    got, out_shape = _run(route, x, w, b, _win(), acc16)
    want = _emulate(route, x, w, b, _win(), out_shape, acc16)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


def test_hand_checkable_direct_and_small_c():
    # Ones over a 3x3 SAME window of a 3x3 single-channel image: every output counts its in-range taps.
    x = np.ones((1, 3, 3, 1), np.float16)
    w = np.ones((2, 3, 3, 1), np.float16)
    b = np.array([0.5, -1.0], np.float16)
    counts = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]], np.float64)
    want = (counts[..., None] + b.astype(np.float64)).astype(np.float16)[None]
    for route in (kf.CONV_F16_DIRECT_OHWI, kf.CONV_F16_DIRECT_PACKED, kf.CONV_F16_SMALL_C, kf.CONV_F16_PATCH_NT_T,
                  kf.CONV_F16_PATCH_PACKED):
        got, _ = _run(route, x, w, b, _win(p=(1, 1)))
        np.testing.assert_array_equal(got, want)


def test_fold_and_acc16_differ_where_float16_saturates():
    # 40 in-range taps of 1 * 1 on top of a bias of 2048 (float16 ulp 2): folding chains run the first 32 from
    # the bias (stuck at 2048), the last 8 from zero, and add in float32 (2056); _acc16 stays at 2048.
    x = np.ones((1, 1, 1, 40), np.float16)
    w = np.ones((8, 1, 1, 40), np.float16)
    b = np.full(8, 2048, np.float16)
    for route in (kf.CONV_F16_DIRECT_PACKED, kf.CONV_F16_PATCH_PACKED):
        assert (_run(route, x, w, b, _win())[0] == 2056).all()
        assert (_run(route, x, w, b, _win(), acc16=True)[0] == 2048).all()
    # The lane routes keep 5 taps per lane in float16 and add the bias in float32 when folding.
    assert (_run(kf.CONV_F16_DIRECT_OHWI, x, w, b, _win())[0] == 2088).all()
    assert (_run(kf.CONV_F16_DIRECT_OHWI, x, w, b, _win(), acc16=True)[0] == 2088).all()


def test_direct_folds_by_in_range_taps():
    # Patch length 36 > 32 folds, but a corner output of a SAME 3x3 has 4 * 4 = 16 in-range taps: it keeps the
    # float16 reduce; the centre (36 taps) folds. Both match the restatement.
    x, w, b = _draw(11, 1, 3, 3, 4, 3, 3, 2)
    got, out_shape = _run(kf.CONV_F16_DIRECT_OHWI, x, w, b, _win(p=(1, 1)))
    want = _emulate(kf.CONV_F16_DIRECT_OHWI, x, w, b, _win(p=(1, 1)), out_shape, False)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


def test_no_bias_matches_zero_bias_and_clamp():
    x, w, _ = _draw(3, 1, 6, 6, 4, 3, 3, 12, bias=False)
    zero = np.zeros(12, np.float16)
    for route in (kf.CONV_F16_PATCH_NT_T, kf.CONV_F16_DIRECT_OHWI, kf.CONV_F16_DIRECT_PACKED, kf.CONV_F16_SMALL_C):
        np.testing.assert_array_equal(_run(route, x, w, None, _win(p=(1, 1)))[0],
                                      _run(route, x, w, zero, _win(p=(1, 1)))[0])
    free = _run(kf.CONV_F16_PATCH_NT_T, x, w, None, _win(p=(1, 1)))[0]
    got = _run(kf.CONV_F16_PATCH_NT_T, x, w, None, _win(p=(1, 1)), activation=(-0.25, 0.5))[0]
    np.testing.assert_array_equal(got, np.clip(free, np.float16(-0.25), np.float16(0.5)))


def test_infinity_reaches_only_its_receptive_field():
    x, w, b = _draw(5, 1, 6, 6, 3, 3, 3, 4)
    x[0, 0, 0, 0] = np.inf
    got, _ = _run(kf.CONV_F16_SMALL_C, x, w, b, _win(p=(1, 1)))
    assert not np.isfinite(got[0, :2, :2]).all() and np.isfinite(got[0, 2:]).all()


@pytest.mark.parametrize("K", [1, 3, 4, 7, 32, 33, 70, 300])
@pytest.mark.parametrize("policy", [kf.NT_N_PACKED_FOLD, kf.NT_N_PACKED_ACC16, kf.NT_N_PACKED_PRECISE])
def test_nt_n_packed_matches_kernel_restatement(K, policy):
    rng = np.random.default_rng(K * 7 + policy)
    lhs = rng.uniform(-1, 1, (2, 5, K)).astype(np.float16)
    rhs = rng.uniform(-1, 1, (2, 11, K)).astype(np.float16)
    bias = rng.uniform(-0.5, 0.5, 11).astype(np.float16)
    got = kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, bias, policy=policy, activation=UNBOUNDED).output()
    want = np.array([[[_packed_column(lhs[b, m].astype(np.float64), rhs[b, n].astype(np.float64),
                                      float(bias[n]), policy) for n in range(11)] for m in range(5)]
                     for b in range(2)], np.float16)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


def test_nt_n_packed_equals_the_conv_patch_route():
    x, w, b = _draw(21, 1, 6, 6, 4, 3, 3, 12)
    win = _win(p=(1, 1))
    conv, out_shape = _run(kf.CONV_F16_PATCH_PACKED, x, w, b, win)
    patches = cr.conv_patches(x, 3, 3, win, out_shape[1:3])
    mm = kf.matmul_nt_n_packed_f16_mve_call(patches, w.reshape(1, 12, -1), b, policy=kf.NT_N_PACKED_FOLD,
                                            activation=UNBOUNDED).output()
    np.testing.assert_array_equal(conv.reshape(-1), mm.reshape(-1))


def _conv_raw(params=None, x=None, w=None, b=None, out=(1, 4, 4, 5)):
    x0, w0, b0 = _draw(1, 1, 6, 6, 3, 3, 3, 5)
    p = {"route": 0, "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0, "dilation_h": 1, "dilation_w": 1,
         "acc16": 0, "activation_min": -1.0, "activation_max": 1.0}
    p.update(params or {})
    return get_bindings().run("conv_f16_mve", p, {"input": x0 if x is None else x, "filter": w0 if w is None else w,
                                                  "bias": b0 if b is None else b}, {"output": out})


@pytest.mark.parametrize("params", [
    {"route": -1}, {"route": 7}, {"stride_h": 0}, {"stride_w": -1}, {"pad_h": -1}, {"dilation_w": 0},
    {"acc16": 2}, {"activation_min": 1.0, "activation_max": -1.0}, {"activation_max": float("nan")},
])
def test_conv_rejects_bad_params(params):
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _conv_raw(params)


@pytest.mark.parametrize("kw", [
    {"w": np.zeros((5, 3, 3, 2), np.float16)},                  # filter depth != input depth
    {"b": np.zeros(4, np.float16)},                             # bias length != filters
    {"out": (1, 4, 4, 4)},                                      # output channels != filters
    {"out": (2, 4, 4, 5)},                                      # output batch != input batch
    {"x": np.zeros((1, 6, 6, 8), np.float16), "w": np.zeros((5, 3, 3, 8), np.float16),
     "params": {"route": 4}},                                   # small C needs fewer than 8 channels
    {"params": {"route": 5}},                                   # conv1d needs height 1
    {"x": np.zeros((1, 1, 9, 3), np.float16), "w": np.zeros((5, 1, 4, 3), np.float16), "out": (1, 1, 6, 5),
     "params": {"route": 6}},                                   # conv1d needs k 3 or 5
    {"x": np.zeros((1, 1, 9, 3), np.float16), "w": np.zeros((5, 1, 3, 3), np.float16), "out": (1, 1, 6, 5),
     "params": {"route": 5}},                                   # conv1d output width W - K + 1
])
def test_conv_rejects_bad_shapes(kw):
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        _conv_raw(**kw)


def test_conv1d_rejects_padding_or_stride():
    args = {"x": np.zeros((1, 1, 9, 3), np.float16), "w": np.zeros((5, 1, 3, 3), np.float16), "out": (1, 1, 7, 5)}
    for params in ({"route": 5, "pad_w": 1}, {"route": 6, "stride_h": 2}):
        with pytest.raises(ReferenceKernelError, match="E_PARAM"):
            _conv_raw(params, **args)


def test_nt_n_packed_rejects_bad_inputs():
    lhs = np.zeros((1, 2, 5), np.float16)
    rhs = np.zeros((1, 3, 5), np.float16)
    run = get_bindings().run
    for policy in (-1, 3):
        with pytest.raises(ReferenceKernelError, match="E_PARAM"):
            run("matmul_nt_n_packed_f16_mve", {"policy": policy, "activation_min": -1.0, "activation_max": 1.0},
                {"lhs": lhs, "rhs": rhs, "bias": np.zeros(0, np.float16)}, {"output": (1, 2, 3)})
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        run("matmul_nt_n_packed_f16_mve", {"policy": 0, "activation_min": -1.0, "activation_max": 1.0},
            {"lhs": lhs, "rhs": np.zeros((1, 3, 4), np.float16), "bias": np.zeros(0, np.float16)},
            {"output": (1, 2, 3)})
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        run("matmul_nt_n_packed_f16_mve", {"policy": 0, "activation_min": -1.0, "activation_max": 1.0},
            {"lhs": lhs, "rhs": rhs, "bias": np.zeros(2, np.float16)}, {"output": (1, 2, 3)})


def test_call_builders_validate_operands():
    x, w, b = _draw(1, 1, 6, 6, 3, 3, 3, 5)
    with pytest.raises(ValueError):
        kf.conv_f16_mve_call(x[..., :2], w, b, route=0, window=_win(), output_shape=(1, 4, 4, 5), acc16=False,
                             activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.conv_f16_mve_call(x, w, b, route=9, window=_win(), output_shape=(1, 4, 4, 5), acc16=False,
                             activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.conv_f16_mve_call(x, w, b, route=0, window=_win(), output_shape=(1, 4, 4, 4), acc16=False,
                             activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.matmul_nt_n_packed_f16_mve_call(np.zeros((1, 2, 3)), np.zeros((1, 2, 4)), None, policy=0,
                                           activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.matmul_nt_n_packed_f16_mve_call(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), None, policy=5,
                                           activation=UNBOUNDED)


class _Op:
    def __init__(self, cpu="cortex-m55", hint=None):
        self.target_cpu = cpu
        self.desc = {"hint": hint or {}}


def _shapes(n, H, W, C, KH, KW, O, win):
    oh, ow = _out_hw(H, W, KH, KW, win)
    return (n, H, W, C), (O, KH, KW, C), (n, oh, ow, O)


R = kf
PACKED = {"weight_format": "NT_N_PACKED"}


@pytest.mark.parametrize("fn, hint, geom, win, expect", [
    # Router dispatch, in arm_convolve_nhwc_f16_body's order.
    ("arm_convolve_f16", None, (1, 5, 5, 4, 1, 1, 6), _win(), (R.CONV_F16_PATCH_NT_T, False)),           # 1x1
    ("arm_convolve_wrapper_f16_acc16", PACKED, (1, 8, 8, 16, 1, 1, 8), _win((2, 2)),
     (R.CONV_F16_PATCH_PACKED, True)),
    ("arm_convolve_f16", None, (1, 1, 10, 3, 1, 4, 5), _win(), None),                                    # 1xN kernel
    ("arm_convolve_f16", None, (1, 1, 9, 3, 1, 3, 4), _win(), (R.CONV_F16_CONV1D, False)),               # conv1d k3
    ("arm_convolve_nhwc_f16_acc16", PACKED, (1, 1, 13, 4, 1, 5, 8), _win(), (R.CONV_F16_CONV1D_PACKED, True)),
    ("arm_convolve_f16", None, (2, 1, 9, 3, 1, 3, 4), _win(), None),                                     # batch 2: 1xN
    ("arm_convolve_f16_acc16", None, (1, 6, 6, 3, 3, 3, 5), _win(p=(1, 1)), (R.CONV_F16_SMALL_C, False)),
    ("arm_convolve_f16", None, (1, 8, 8, 8, 3, 3, 8), _win(), (R.CONV_F16_PATCH_NT_T, False)),           # patch GEMM
    ("arm_convolve_f16", PACKED, (1, 4, 4, 8, 3, 3, 8), _win(), (R.CONV_F16_DIRECT_PACKED, False)),      # 4 positions
    ("arm_convolve_f16", None, (1, 7, 7, 8, 3, 3, 3), _win(), (R.CONV_F16_DIRECT_OHWI, False)),          # 3 channels
    ("arm_convolve_f16", None, (1, 6, 6, 20, 5, 5, 5), _win(p=(2, 2)), (R.CONV_F16_DIRECT_OHWI, False)),
    # Per-route entries and their declines.
    ("arm_convolve_1x1_nhwc_ohwi_f16", None, (1, 4, 4, 16, 1, 1, 12), _win(), (R.CONV_F16_PATCH_NT_T, False)),
    ("arm_convolve_1x1_nhwc_ohwi_f16", PACKED, (1, 4, 4, 16, 1, 1, 12), _win(), None),
    ("arm_convolve_1x1_nhwc_packed_f16_acc16", PACKED, (1, 4, 4, 16, 1, 1, 12), _win(),
     (R.CONV_F16_PATCH_PACKED, True)),
    ("arm_convolve_1x1_nhwc_ohwi_f16", None, (1, 6, 6, 4, 3, 3, 8), _win(), None),
    ("arm_convolve_1d_k5_nhwc_ohwi_f16", None, (1, 1, 12, 8, 1, 5, 12), _win(), (R.CONV_F16_CONV1D, False)),
    ("arm_convolve_1d_k5_nhwc_ohwi_f16", None, (1, 1, 12, 8, 1, 3, 12), _win(), None),
    ("arm_convolve_1d_k3_nhwc_packed_f16", None, (1, 1, 12, 8, 1, 3, 12), _win(), None),
    ("arm_convolve_small_c_nhwc_f16", PACKED, (1, 6, 6, 3, 3, 3, 8), _win(p=(1, 1)), (R.CONV_F16_SMALL_C, False)),
    ("arm_convolve_small_c_nhwc_f16", None, (1, 6, 6, 8, 3, 3, 8), _win(p=(1, 1)), None),
    ("arm_convolve_patch_gemm_nhwc_packed_f16", PACKED, (1, 6, 6, 4, 3, 3, 12), _win(p=(1, 1)),
     (R.CONV_F16_PATCH_PACKED, False)),
    ("arm_convolve_patch_gemm_nhwc_packed_f16", None, (1, 6, 6, 4, 3, 3, 12), _win(p=(1, 1)), None),
    ("arm_convolve_direct_nhwc_ohwi_f16_acc16", None, (1, 6, 6, 4, 3, 3, 5), _win(p=(1, 1)),
     (R.CONV_F16_DIRECT_OHWI, True)),
    ("arm_convolve_direct_nhwc_packed_f16", None, (1, 6, 6, 4, 3, 3, 5), _win(p=(1, 1)), None),
    ("arm_convolve_1_x_n_f16", None, (1, 1, 10, 3, 1, 4, 5), _win(), None),                              # pilot's
])
def test_route_selection(fn, hint, geom, win, expect):
    shapes = _shapes(*geom, win)
    got = cr.conv_f16_route(_Op(hint=hint), {"kernel_fn": fn}, *shapes, win, 1)
    assert got == (None if expect is None else {"route": expect[0], "acc16": expect[1]})


def test_route_selection_off_helium_and_grouped():
    shapes = _shapes(1, 6, 6, 3, 3, 3, 5, _win(p=(1, 1)))
    assert cr.conv_f16_route(_Op(cpu="cortex-m4"), {"kernel_fn": "arm_convolve_f16"}, *shapes, _win(p=(1, 1)), 1) \
        is None
    assert cr.conv_f16_route(_Op(), {"kernel_fn": "arm_convolve_f16"}, *shapes, _win(p=(1, 1)), 3) is None
    assert cr.conv_f16_route(_Op(), {"kernel_fn": "arm_convolve_s8"}, *shapes, _win(p=(1, 1)), 1) is None


def test_small_c_selection_respects_the_gather_offset_limit():
    win = _win((1, 1), (1, 1))
    wide = _shapes(1, 3, 20000, 7, 3, 3, 8, win)   # 20000 + ... > 65535 / 7
    assert cr.conv_f16_route(_Op(), {"kernel_fn": "arm_convolve_f16"}, *wide, win, 1) == \
        {"route": kf.CONV_F16_PATCH_NT_T, "acc16": False}


def test_patches_zero_fill_and_order():
    x = np.arange(1, 9, dtype=np.float16).reshape(1, 2, 2, 2)
    p = cr.conv_patches(x, 2, 2, _win(p=(1, 1)), (1, 1))
    np.testing.assert_array_equal(p[0, 0], [0, 0, 0, 0, 0, 0, 1, 2])


@pytest.mark.parametrize("case_name, entry, route", [
    ("convolve_float_nonfinite_inf_f16", "conv_f16_mve", kf.CONV_F16_SMALL_C),      # masked through the probe
    ("convolve_float_route_direct_packed_acc16_f16", "conv_f16_mve", kf.CONV_F16_DIRECT_PACKED),
    ("convolve_float_1d_k5_fold_c12_oc13_packed_f16", "conv_f16_mve", kf.CONV_F16_CONV1D_PACKED),
    ("convolve_float_route_small_c_declines_c8_f16", "conv_f16", None),              # status-only: exact golden
])
def test_generated_case_records_its_route_and_tolerance(tmp_path, case_name, entry, route):
    import json

    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.generation.io.descriptors import load_all_descriptors
    import helia_core_tester.generation.test_ops as generation_module

    desc = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}[case_name]
    generation_module.generate_test(desc, str(tmp_path), cpu="cortex-m55")
    case_dir = tmp_path / desc["_family"] / case_name
    record = json.loads((case_dir / f"{case_name}.reference.json").read_text())
    assert record["entry"] == entry
    source = (case_dir / f"{case_name}_convolve.c").read_text()
    if route is None:
        assert "1e-05f" not in source
        return
    assert record["params"]["route"] == route
    assert "        1e-05f,\n        0.001f," in source
