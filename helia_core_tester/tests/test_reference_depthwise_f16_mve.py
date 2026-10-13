"""depthwise_f16_mve and matmul_nt_n_packed_f16_mve: ns-cmsis-nn's float16 depthwise routes on Helium, bit for bit.

The C entries are checked against NumPy restatements written from arm_depthwise_conv_f16.c: the ch_mult == 1
direct kernel (interior runs [x_lo, x_hi) and per-pixel edge windows, lanes are channels, fold over the in-range
taps), the generic kernel (one fused scalar chain per output), the conv1d k3 / 2x5 specialisations, and the
one-input-channel route (zero-padded patch rows through arm_nn_mat_mult_nt_n_packed_f16, folding over every tap);
plus hand-checkable cases, the entries' rejections and the generator's route selection.
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
    # Products of halves are exact in binary64 and, for these magnitudes, so is the sum: one rounding.
    return float(np.float16(np.float64(a) * np.float64(b) + np.float64(c)))


def _chain(xs, ws, bias, fold):
    """arm_depthwise_direct_taps_mve_f16 / the generic loop / the packed matmul lane: from the bias, one fma per
    tap; with fold and more than 32 taps, a float32 sum of 32-tap float16 partials, one rounding."""
    acc, acc32, first, n_taps = float(bias), np.float32(0), True, 0
    fold = fold and len(xs) > 32
    for x, w in zip(xs, ws):
        if fold:
            if n_taps == 32:
                acc32 = np.float32(acc) if first else np.float32(acc32 + np.float32(acc))
                first, acc, n_taps = False, 0.0, 0
            n_taps += 1
        acc = _fma16(x, w, acc)
    if fold and not first:
        acc = _h(np.float32(acc32 + np.float32(acc)))
    return acc


def _start(base, d):
    return max(0, int((-base + d - 1) / d)) if d > 1 else max(0, -base)


def _end(base, d, k, size):
    return min(k, int((size - base + d - 1) / d)) if d > 1 else min(k, size - base)


def _direct(x, w, b, win, oh, ow, acc16):
    """arm_depthwise_conv_nhwc_direct_chmult1_f16 on MVE, as the kernel walks it."""
    n, H, W, C = x.shape
    _, KH, KW, _ = w.shape
    sh, sw, dh, dw, ph, pw = (win[k] for k in ("stride_h", "stride_w", "dilation_h", "dilation_w", "pad_h", "pad_w"))
    bias = np.zeros(C) if b is None else b.astype(np.float64)
    hi_num = W - 1 + pw - (KW - 1) * dw
    x_lo = min((pw + sw - 1) // sw if pw > 0 else 0, ow)
    x_hi = max(min(hi_num // sw + 1 if hi_num >= 0 else 0, ow), x_lo)
    out = np.zeros((n, oh, ow, C), np.float16)
    for bi in range(n):
        for oy in range(oh):
            by = oy * sh - ph
            y0, y1 = _start(by, dh), _end(by, dh, KH, H)
            for ox in range(ow):
                bx = ox * sw - pw
                if x_lo <= ox < x_hi:
                    x0, x1 = 0, KW
                else:
                    x0, x1 = _start(bx, dw), _end(bx, dw, KW, W)
                taps = [(ky, kx) for ky in range(y0, y1) for kx in range(x0, x1)]
                for c in range(C):
                    xs = [x[bi, by + dh * ky, bx + dw * kx, c] for ky, kx in taps]
                    ws = [w[0, ky, kx, c] for ky, kx in taps]
                    out[bi, oy, ox, c] = _chain(xs, ws, bias[c], not acc16)
    return out


def _generic(x, w, b, win, oh, ow, mult, acc16):
    """arm_depthwise_conv_f16_generic on MVE: fused scalar chain, fold over the in-range taps."""
    n, H, W, C = x.shape
    _, KH, KW, OC = w.shape
    sh, sw, dh, dw, ph, pw = (win[k] for k in ("stride_h", "stride_w", "dilation_h", "dilation_w", "pad_h", "pad_w"))
    bias = np.zeros(OC) if b is None else b.astype(np.float64)
    out = np.zeros((n, oh, ow, OC), np.float16)
    for bi in range(n):
        for oy in range(oh):
            for ox in range(ow):
                by, bx = oy * sh - ph, ox * sw - pw
                for ic in range(C):
                    for m in range(mult):
                        oc = m + ic * mult
                        xs, ws = [], []
                        for ky in range(_start(by, dh), _end(by, dh, KH, H)):
                            for kx in range(_start(bx, dw), _end(bx, dw, KW, W)):
                                xs.append(x[bi, by + dh * ky, bx + dw * kx, ic])
                                ws.append(w[0, ky, kx, oc])
                        out[bi, oy, ox, oc] = _chain(xs, ws, bias[oc], not acc16)
    return out


def _cin1(x, w, b, win, oh, ow, acc16):
    """The one-input-channel route: patch rows (padded taps zero) through arm_nn_mat_mult_nt_n_packed_f16."""
    n, H, W, _ = x.shape
    _, KH, KW, OC = w.shape
    sh, sw, dh, dw, ph, pw = (win[k] for k in ("stride_h", "stride_w", "dilation_h", "dilation_w", "pad_h", "pad_w"))
    bias = np.zeros(OC) if b is None else b.astype(np.float64)
    out = np.zeros((n, oh, ow, OC), np.float16)
    for bi in range(n):
        for oy in range(oh):
            for ox in range(ow):
                row = []
                for ky in range(KH):
                    for kx in range(KW):
                        iy, ix = oy * sh - ph + dh * ky, ox * sw - pw + dw * kx
                        row.append(x[bi, iy, ix, 0] if 0 <= iy < H and 0 <= ix < W else 0.0)
                for oc in range(OC):
                    ws = [w[0, ky, kx, oc] for ky in range(KH) for kx in range(KW)]
                    out[bi, oy, ox, oc] = _chain(row, ws, bias[oc], not acc16)
    return out


def _win(stride=(1, 1), dilation=(1, 1), pad=(0, 0)):
    return {"stride_h": stride[0], "stride_w": stride[1], "dilation_h": dilation[0], "dilation_w": dilation[1],
            "pad_h": pad[0], "pad_w": pad[1]}


def _out_hw(x, w, win):
    oh = (x.shape[1] + 2 * win["pad_h"] - (w.shape[1] - 1) * win["dilation_h"] - 1) // win["stride_h"] + 1
    ow = (x.shape[2] + 2 * win["pad_w"] - (w.shape[2] - 1) * win["dilation_w"] - 1) // win["stride_w"] + 1
    return oh, ow


def _draw(seed, n, H, W, C, KH, KW, mult=1, bias=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, H, W, C)).astype(np.float16)
    w = rng.uniform(-1, 1, (1, KH, KW, C * mult)).astype(np.float16)
    b = rng.uniform(-1, 1, C * mult).astype(np.float16) if bias else None
    return x, w, b


def _run(x, w, b, win, acc16=False, activation=UNBOUNDED):
    return kf.depthwise_f16_mve_call(x, w, b, window=win, out_hw=_out_hw(x, w, win), acc16=acc16,
                                     activation=activation).output()


def _run_cin1(x, w, b, win, acc16=False, activation=UNBOUNDED):
    oh, ow = _out_hw(x, w, win)
    lhs, rhs = cr.depthwise_cin1_operands(x, w, w.shape[1], w.shape[2], win, (oh, ow))
    out = kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, b, policy=kf.NT_N_PACKED_ACC16 if acc16 else kf.NT_N_PACKED_FOLD, activation=activation).output()
    return out.reshape(x.shape[0], oh, ow, w.shape[3])


def _same(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))


@pytest.mark.parametrize("shape", [
    # (n, H, W, C, KH, KW, stride, dilation, pad)
    (1, 6, 6, 3, 3, 3, (1, 1), (1, 1), (1, 1)),     # SAME 3x3: edge windows, channel tail 3
    (2, 7, 7, 20, 3, 3, (2, 2), (1, 1), (0, 0)),    # batch 2, stride 2, a 16-channel pair and a 4-channel tail
    (1, 8, 8, 9, 3, 3, (1, 1), (2, 2), (2, 2)),     # dilation 2: the dilated start/end formulas
    (1, 9, 9, 8, 7, 7, (1, 1), (1, 1), (3, 3)),     # 49 interior taps fold; edges 16..42 taps
    (1, 6, 10, 24, 6, 6, (1, 1), (1, 1), (2, 2)),   # 36 taps fold, no-fold edges
    (1, 5, 12, 4, 2, 9, (1, 2), (1, 1), (0, 4)),    # pad wider than half the kernel, stride 2 in x
    (1, 3, 3, 5, 5, 5, (1, 1), (1, 1), (4, 4)),     # padding wider than the kernel reach: bias-only outputs
])
@pytest.mark.parametrize("acc16", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_direct_route_matches_kernel_restatement(shape, acc16, bias):
    n, H, W, C, KH, KW, stride, dil, pad = shape
    x, w, b = _draw(hash(shape) & 0xFFFF, n, H, W, C, KH, KW, bias=bias)
    win = _win(stride, dil, pad)
    oh, ow = _out_hw(x, w, win)
    _same(_run(x, w, b, win, acc16), _direct(x, w, b, win, oh, ow, acc16))


@pytest.mark.parametrize("shape", [
    # (n, H, W, C, KH, KW, mult, stride, dilation, pad)
    (1, 5, 5, 2, 3, 3, 2, (1, 1), (1, 1), (1, 1)),
    (1, 7, 7, 3, 6, 6, 2, (1, 1), (1, 1), (2, 2)),  # 36 interior taps fold
    (2, 9, 8, 2, 7, 5, 3, (1, 2), (1, 1), (3, 2)),  # 35 taps fold, batch 2
    (1, 8, 8, 2, 3, 3, 4, (2, 1), (2, 3), (2, 3)),
])
@pytest.mark.parametrize("acc16", [False, True])
def test_generic_route_matches_kernel_restatement(shape, acc16):
    n, H, W, C, KH, KW, mult, stride, dil, pad = shape
    x, w, b = _draw(hash(shape) & 0xFFFF, n, H, W, C, KH, KW, mult)
    win = _win(stride, dil, pad)
    oh, ow = _out_hw(x, w, win)
    _same(_run(x, w, b, win, acc16), _generic(x, w, b, win, oh, ow, mult, acc16))


def test_specialisations_are_the_same_chain():
    # conv1d k3 (three fma from the bias) and 2x5 (ten, row by row) never fold: acc16 changes nothing.
    x, w, b = _draw(11, 1, 1, 12, 8, 1, 3)
    win = _win()
    _same(_run(x, w, b, win), _generic(x, w, b, win, 1, 10, 1, False))
    _same(_run(x, w, b, win), _run(x, w, b, win, acc16=True))
    x, w, b = _draw(12, 2, 2, 9, 3, 2, 5, mult=2)
    _same(_run(x, w, b, win), _generic(x, w, b, win, 1, 5, 2, False))
    _same(_run(x, w, b, win), _run(x, w, b, win, acc16=True))


@pytest.mark.parametrize("shape", [
    # (n, H, W, KH, KW, OC, stride, dilation, pad)
    (1, 6, 6, 3, 3, 8, (1, 1), (1, 1), (1, 1)),     # 9 taps, padded rows multiply zeros
    (1, 9, 9, 6, 6, 5, (1, 1), (1, 1), (2, 2)),     # 36 taps fold over every tap, padded ones included
    (2, 8, 7, 7, 5, 11, (2, 1), (1, 1), (3, 2)),    # 35 taps, batch 2, column tail 3
    (1, 10, 10, 3, 3, 4, (1, 1), (3, 3), (3, 3)),
])
@pytest.mark.parametrize("acc16", [False, True])
def test_cin1_route_matches_kernel_restatement(shape, acc16):
    n, H, W, KH, KW, OC, stride, dil, pad = shape
    x, w, b = _draw(hash(shape) & 0xFFFF, n, H, W, 1, KH, KW, mult=OC)
    win = _win(stride, dil, pad)
    oh, ow = _out_hw(x, w, win)
    _same(_run_cin1(x, w, b, win, acc16), _cin1(x, w, b, win, oh, ow, acc16))


def test_cin1_and_generic_fold_at_different_taps():
    # A corner output of a padded 6x6 window has 16 in-range taps (generic: no fold) but 36 packed taps (cin1:
    # folds after 32, the four in-range taps past the 32nd go to float32).
    x = np.ones((1, 6, 6, 1), np.float16)
    w = np.ones((1, 6, 6, 2), np.float16)
    b = np.full(2, 2048, np.float16)
    win = _win(pad=(2, 2))
    generic, cin1 = _run(x, w, b, win), _run_cin1(x, w, b, win)
    # Interior-most output (2, 2): 36 in-range taps -- both fold after 32: 2048 + 4.
    assert generic[0, 2, 2, 0] == cin1[0, 2, 2, 0] == 2052
    # Output (0, 0): in-range taps (ky, kx) in [2, 6) x [2, 6); the generic route never folds (2048 swallows +1);
    # the packed route's last block holds packed taps 32..35 = (5, 2..5), all in range: 2048 + 4.
    assert generic[0, 0, 0, 0] == 2048 and cin1[0, 0, 0, 0] == 2052


def test_hand_checkable_case():
    x = (np.arange(1, 17, dtype=np.float16) / 16).reshape(1, 4, 4, 1)
    w = np.ones((1, 3, 3, 1), np.float16)
    b = np.array([0.5], np.float16)
    got = _run(x, w, b, _win(pad=(1, 1)))
    xs = x[0, :, :, 0].astype(np.float64)
    want = np.array([[xs[max(0, i - 1):i + 2, max(0, j - 1):j + 2].sum() + 0.5 for j in range(4)] for i in range(4)])
    _same(got[0, :, :, 0], want.astype(np.float16))


def test_fold_and_acc16_differ_where_float16_saturates():
    # 49 taps of +1 on a bias of 2048 (ulp 2): float16 lanes stay at 2048; folding adds the last 17 in float32 and
    # rounds 2065 once, to 2064.
    x = np.ones((1, 7, 7, 4), np.float16)
    w = np.ones((1, 7, 7, 4), np.float16)
    b = np.full(4, 2048, np.float16)
    assert (_run(x, w, b, _win()) == 2064).all()
    assert (_run(x, w, b, _win(), acc16=True) == 2048).all()
    lhs = np.ones((1, 3, 49), np.float16)
    rhs = np.ones((1, 4, 49), np.float16)
    assert (kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, b, policy=kf.NT_N_PACKED_ACC16 if False else kf.NT_N_PACKED_FOLD, activation=UNBOUNDED).output() == 2064).all()
    assert (kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, b, policy=kf.NT_N_PACKED_ACC16 if True else kf.NT_N_PACKED_FOLD, activation=UNBOUNDED).output() == 2048).all()


def test_32_taps_never_fold():
    x = np.ones((1, 4, 8, 3), np.float16)
    w = np.ones((1, 4, 8, 3), np.float16)
    b = np.full(3, 2048, np.float16)
    _same(_run(x, w, b, _win()), _run(x, w, b, _win(), acc16=True))
    lhs, rhs = np.ones((1, 2, 32), np.float16), np.ones((1, 3, 32), np.float16)
    _same(kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, b, policy=kf.NT_N_PACKED_ACC16 if False else kf.NT_N_PACKED_FOLD, activation=UNBOUNDED).output(),
          kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, b, policy=kf.NT_N_PACKED_ACC16 if True else kf.NT_N_PACKED_FOLD, activation=UNBOUNDED).output())


def test_no_bias_matches_zero_bias():
    x, w, _ = _draw(3, 1, 9, 9, 8, 7, 7, bias=False)
    _same(_run(x, w, None, _win(pad=(3, 3))), _run(x, w, np.zeros(8, np.float16), _win(pad=(3, 3))))


def test_clamp():
    x, w, b = _draw(9, 1, 9, 9, 8, 7, 7)
    win = _win(pad=(3, 3))
    got = _run(x, w, b, win, activation=(-0.25, 0.5))
    _same(got, np.clip(_run(x, w, b, win), np.float16(-0.25), np.float16(0.5)))
    lhs, rhs = cr.depthwise_cin1_operands(x[..., :1], w, 7, 7, win, (9, 9))
    free = kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, b, policy=kf.NT_N_PACKED_ACC16 if False else kf.NT_N_PACKED_FOLD, activation=UNBOUNDED).output()
    got = kf.matmul_nt_n_packed_f16_mve_call(lhs, rhs, b, policy=kf.NT_N_PACKED_ACC16 if False else kf.NT_N_PACKED_FOLD, activation=(-0.25, 0.5)).output()
    _same(got, np.clip(free, np.float16(-0.25), np.float16(0.5)))


def test_infinity_reaches_only_its_receptive_field():
    x, w, b = _draw(5, 1, 6, 6, 2, 3, 3)
    x[0, 0, 0, 0] = np.inf
    got = _run(x, w, b, _win(pad=(1, 1)))
    assert not np.isfinite(got[0, :2, :2, 0]).all()
    assert np.isfinite(got[0, 2:, :, 0]).all() and np.isfinite(got[..., 1]).all()


def _dw_call(params=None, x=None, w=None, b=None, out=(1, 4, 4, 4)):
    x0, w0, b0 = _draw(1, 1, 6, 6, 2, 3, 3, mult=2)
    p = {**_win(), "acc16": 0, "activation_min": -1.0, "activation_max": 1.0}
    p.update(params or {})
    return get_bindings().run("depthwise_f16_mve", p,
                              {"input": x0 if x is None else x, "filter": w0 if w is None else w,
                               "bias": b0 if b is None else b}, {"output": out})


@pytest.mark.parametrize("params", [
    {"stride_h": 0}, {"stride_w": -1}, {"dilation_h": 0}, {"dilation_w": 0}, {"pad_h": -1}, {"pad_w": -2},
    {"acc16": 2}, {"acc16": -1}, {"activation_min": 1.0, "activation_max": -1.0}, {"activation_min": float("nan")},
])
def test_depthwise_rejects_bad_params(params):
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _dw_call(params)


@pytest.mark.parametrize("kw", [
    {"w": np.zeros((2, 3, 3, 4), np.float16)},     # filter batch != 1
    {"w": np.zeros((1, 3, 3, 3), np.float16)},     # output channels not a multiple of the input depth
    {"b": np.zeros(3, np.float16)},                # bias length != output channels
    {"out": (2, 4, 4, 4)},                         # output batch != input batch
    {"out": (1, 4, 4, 2)},                         # output channels != filter channels
    {"x": np.zeros((6, 6, 2), np.float16)},        # rank 3 input
])
def test_depthwise_rejects_bad_shapes(kw):
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        _dw_call(**kw)


def _mm_call(params=None, lhs=None, rhs=None, b=None, out=(1, 3, 5)):
    rng = np.random.default_rng(2)
    p = {"policy": 0, "activation_min": -1.0, "activation_max": 1.0}
    p.update(params or {})
    return get_bindings().run(
        "matmul_nt_n_packed_f16_mve", p,
        {"lhs": rng.uniform(-1, 1, (1, 3, 7)).astype(np.float16) if lhs is None else lhs,
         "rhs": rng.uniform(-1, 1, (1, 5, 7)).astype(np.float16) if rhs is None else rhs,
         "bias": np.zeros(5, np.float16) if b is None else b}, {"output": out})


@pytest.mark.parametrize("params", [
    {"policy": 3}, {"policy": -1}, {"activation_min": 1.0, "activation_max": -1.0}, {"activation_max": float("nan")},
])
def test_matmul_rejects_bad_params(params):
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _mm_call(params)


@pytest.mark.parametrize("kw", [
    {"rhs": np.zeros((1, 5, 6), np.float16)},      # K mismatch
    {"rhs": np.zeros((2, 5, 7), np.float16)},      # batch mismatch
    {"b": np.zeros(4, np.float16)},                # bias length != N
    {"out": (1, 3, 4)},                            # output columns != N
    {"lhs": np.zeros((3, 7), np.float16)},         # rank 2 lhs
])
def test_matmul_rejects_bad_shapes(kw):
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        _mm_call(**kw)


def test_call_builders_validate_operands():
    x, w, b = _draw(1, 1, 6, 6, 2, 3, 3, mult=2)
    with pytest.raises(ValueError):
        kf.depthwise_f16_mve_call(x, w[:, :, :, :3], b, window=_win(), out_hw=(4, 4), acc16=False,
                                  activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.depthwise_f16_mve_call(x[0], w, b, window=_win(), out_hw=(4, 4), acc16=False, activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.matmul_nt_n_packed_f16_mve_call(np.zeros((1, 2, 3)), np.zeros((1, 2, 4)), None, policy=kf.NT_N_PACKED_FOLD,
                                           activation=UNBOUNDED)


class _Op:
    def __init__(self, cpu="cortex-m55", hint=None):
        self.target_cpu = cpu
        self.desc = {"hint": hint or {}}


def _route(fn, in_shape, kh, kw, mult, win=None, out_shape=None, op=None):
    win = win or _win()
    if out_shape is None:
        oh = (in_shape[1] + 2 * win["pad_h"] - kh) // win["stride_h"] + 1
        ow = (in_shape[2] + 2 * win["pad_w"] - kw) // win["stride_w"] + 1
        out_shape = (in_shape[0], oh, ow, in_shape[3] * mult)
    return cr.depthwise_f16_route(op or _Op(), {"kernel_fn": fn}, in_shape, out_shape, kh, kw, mult, win)


@pytest.mark.parametrize("fn, in_shape, kh, kw, mult, expect", [
    ("arm_depthwise_conv_f16", (1, 6, 6, 3), 3, 3, 1, {"route": "chain", "acc16": False}),       # direct
    ("arm_depthwise_conv_f16_acc16", (1, 5, 5, 2), 3, 3, 2, {"route": "chain", "acc16": True}),  # generic
    ("arm_depthwise_nhwc_conv_f16", (1, 6, 6, 1), 3, 3, 8, {"route": "cin1", "acc16": False}),
    ("arm_depthwise_conv_wrapper_f16_acc16", (1, 6, 6, 1), 3, 3, 2, {"route": "cin1", "acc16": True}),
    ("arm_depthwise_conv_f16", (1, 6, 6, 1), 3, 3, 1, {"route": "chain", "acc16": False}),       # direct first
    ("arm_depthwise_conv_f16", (2, 2, 9, 1), 2, 5, 4, {"route": "chain", "acc16": False}),       # 2x5 first
    ("arm_depthwise_conv_f16", (1, 1, 9, 4), 1, 3, 1, {"route": "chain", "acc16": False}),       # conv1d k3
    ("arm_depthwise_conv_f16", (1, 6, 6, 1), 1, 1, 4, {"route": "chain", "acc16": False}),       # ctx 0: generic
    ("arm_depthwise_conv_direct_nhwc_f16_acc16", (1, 6, 6, 8), 3, 3, 1, {"route": "chain", "acc16": True}),
    ("arm_depthwise_conv_direct_nhwc_f16", (1, 5, 5, 4), 3, 3, 2, None),                         # declines
    ("arm_depthwise_conv_generic_nhwc_f16", (1, 6, 6, 1), 3, 3, 8, {"route": "chain", "acc16": False}),
    ("arm_depthwise_conv_cin1_nhwc_f16_acc16", (1, 6, 6, 1), 3, 3, 8, {"route": "cin1", "acc16": True}),
    ("arm_depthwise_conv_cin1_nhwc_f16", (1, 6, 6, 2), 3, 3, 4, None),                           # declines
    ("arm_depthwise_conv_1d_k3_nhwc_f16", (1, 1, 12, 8), 1, 3, 1, {"route": "chain", "acc16": False}),
    ("arm_depthwise_conv_1d_k3_nhwc_f16", (1, 6, 6, 8), 3, 3, 1, None),                          # declines
    ("arm_depthwise_conv_2x5_nhwc_f16", (1, 2, 9, 8), 2, 5, 1, {"route": "chain", "acc16": False}),
    ("arm_depthwise_conv_2x5_nhwc_f16", (1, 6, 6, 8), 3, 3, 1, None),                            # declines
    ("arm_depthwise_conv_s8", (1, 6, 6, 8), 3, 3, 1, None),
])
def test_route_selection(fn, in_shape, kh, kw, mult, expect):
    assert _route(fn, in_shape, kh, kw, mult) == expect


def test_route_selection_ctx_target_and_padding():
    # force_no_scratch: ctx 0, so the router falls back to the generic chain; the cin1 entry declines.
    no_scratch = _Op(hint={"force_no_scratch": True})
    assert _route("arm_depthwise_conv_f16", (1, 6, 6, 1), 3, 3, 8, op=no_scratch)["route"] == "chain"
    assert _route("arm_depthwise_conv_cin1_nhwc_f16", (1, 6, 6, 1), 3, 3, 8, op=no_scratch) is None
    # A 1x1 filter with stride 2 gets patch-gemm scratch: cin1.
    assert _route("arm_depthwise_conv_f16", (1, 6, 6, 1), 1, 1, 4, win=_win(stride=(2, 2)))["route"] == "cin1"
    # Non-Helium targets keep the exact golden.
    assert _route("arm_depthwise_conv_f16", (1, 6, 6, 3), 3, 3, 1, op=_Op(cpu="cortex-m4")) is None
    # Padding takes the conv1d k3 shape off its specialisation (the direct route still chains).
    assert _route("arm_depthwise_conv_1d_k3_nhwc_f16", (1, 1, 12, 8), 1, 3, 1, win=_win(pad=(0, 1))) is None


def test_taps_zero_fill_and_order():
    x = np.arange(1, 10, dtype=np.float16).reshape(1, 3, 3, 1)
    t = cr.depthwise_taps(x, 2, 2, _win(stride=(2, 2), pad=(1, 1)), (2, 2))
    np.testing.assert_array_equal(t[0, :, :, 0], [[[0, 0, 0, 1], [0, 0, 2, 3]], [[0, 4, 0, 7], [5, 6, 8, 9]]])
