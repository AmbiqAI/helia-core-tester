"""conv_1xn_f16_mve: ns-cmsis-nn's arm_convolve_1_x_n_f16 / _acc16 on Helium, bit for bit.

The C entry is checked against a NumPy restatement of the kernel's routes written from
arm_convolve_1_x_n_f16.c (strided gather blocks, remainder dots, padded and in-place rows through
arm_nn_mat_mult_nt_t_f16, packed filters through arm_nn_mat_mult_nt_n_packed_f16), plus hand-checkable
cases, the fold/no-fold split, and the entry's rejections.
"""

from __future__ import annotations

import numpy as np
import pytest

from helia_core_tester.generation.ops._shared import conv_reference as cr
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings

UNBOUNDED = (-1.0e30, 1.0e30)


def _h(v: float) -> float:
    return float(np.float16(v))


def _fma16(a, b, c) -> float:
    # Products of halves are exact in binary64 and, for these magnitudes, so is the sum: one rounding.
    return float(np.float16(np.float64(a) * np.float64(b) + np.float64(c)))


def _chain(lhs, rhs, bias, fold):
    K = len(lhs)
    first = 32 if fold and K > 32 else K
    acc = bias
    for k in range(first):
        acc = _fma16(rhs[k], lhs[k], acc)
    if first == K:
        return acc
    acc32 = np.float32(acc)
    for k0 in range(first, K, 32):
        part = 0.0
        for k in range(k0, min(K, k0 + 32)):
            part = _fma16(rhs[k], lhs[k], part)
        acc32 = np.float32(acc32 + np.float32(part))
    return _h(acc32)


def _lanes(lhs, rhs, k0, n):
    lane = [0.0] * 8
    for k in range(n):
        lane[k % 8] = _fma16(lhs[k0 + k], rhs[k0 + k], lane[k % 8])
    return lane


def _lanes16(lhs, rhs, bias):
    l = _lanes(lhs, rhs, 0, len(lhs))
    a = _h(_h(l[0] + l[1]) + _h(l[2] + l[3]))
    b = _h(_h(l[4] + l[5]) + _h(l[6] + l[7]))
    return _h(bias + _h(a + b))


def _fold(lhs, rhs, bias):
    acc = [np.float32(0)] * 4
    for k0 in range(0, len(lhs), 256):
        l = _lanes(lhs, rhs, k0, min(256, len(lhs) - k0))
        for j in range(4):
            pair = np.float32(np.float32(l[2 * j]) + np.float32(l[2 * j + 1]))
            acc[j] = pair if k0 == 0 else np.float32(acc[j] + pair)
    total = np.float32(np.float32(acc[0] + acc[1]) + np.float32(acc[2] + acc[3]))
    return _h(np.float32(np.float32(bias) + total))


def _gather_cols(K, N):
    c = N - N % 16
    c += (N - c) - (N - c) % 8
    c += (N - c) - (N - c) % 4
    return c


def _emulate(x, w, bias, stride, pad, out_w, acc16, packed):
    n, _, W, C = x.shape
    O, _, KW, _ = w.shape
    K = KW * C
    fold = not acc16 and K > 32
    patches = cr.conv_1xn_patches(x, KW, stride, pad, out_w).astype(np.float64)
    rhs = w.reshape(O, K).astype(np.float64)
    b = np.zeros(O) if bias is None else bias.astype(np.float64)
    valid = [ox for ox in range(out_w) if ox * stride - pad >= 0 and ox * stride - pad + KW <= W]
    first, last = (valid[0], valid[-1]) if valid else (out_w, out_w - 1)
    in_place = O >= 4 and ((O % 8 >= 4 and K >= 80) or K >= 224)
    out = np.zeros((n, 1, out_w, O), np.float16)
    for bi in range(n):
        for ox in range(out_w):
            lhs = patches[bi, ox]
            if packed:
                gather, matmul = O, False
            elif ox < first or ox > last or in_place:
                gather, matmul = (_gather_cols(K, O) if K < 32 else 0), True
            else:
                gather, matmul = (0 if fold and O < 4 else _gather_cols(K, O)), False
            for c in range(O):
                if c < gather:
                    v = _chain(lhs, rhs[c], b[c], fold and not matmul)
                elif fold:
                    v = _fold(lhs, rhs[c], b[c])
                else:
                    v = _lanes16(lhs, rhs[c], b[c])
                out[bi, 0, ox, c] = v
    return out


def _run(x, w, bias, *, stride=1, pad=0, out_w=None, acc16=False, packed=False, activation=UNBOUNDED):
    out_w = out_w if out_w is not None else (x.shape[2] + 2 * pad - w.shape[2]) // stride + 1
    return kf.conv_1xn_f16_mve_call(x, w, bias, stride_w=stride, pad_w=pad, out_w=out_w, acc16=acc16,
                                    packed=packed, activation=activation).output()


def _draw(seed, n, W, C, KW, O, bias=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 1, W, C)).astype(np.float16)
    w = rng.uniform(-1, 1, (O, 1, KW, C)).astype(np.float16)
    b = rng.uniform(-0.5, 0.5, O).astype(np.float16) if bias else None
    return x, w, b


def test_hand_checkable_case_with_padding():
    x = (np.arange(1, 13, dtype=np.float16) / 8).reshape(1, 1, 6, 2)
    w = np.ones((3, 1, 3, 2), np.float16)
    b = np.array([0.5, 0.0, -1.0], np.float16)
    got = _run(x, w, b, pad=1)
    window = [x[0, 0, max(0, i - 1):i + 2].astype(np.float64).sum() for i in range(6)]
    want = np.array([[s + bb for bb in b.astype(np.float64)] for s in window])
    np.testing.assert_array_equal(got.reshape(6, 3), want.astype(np.float16))


@pytest.mark.parametrize("shape", [
    # (n, W, C, KW, O, stride, pad): every route of the kernel
    (1, 22, 14, 7, 16, 1, 3),   # K 98: dual gather block, fold, padded rows through the matmul
    (1, 22, 4, 7, 16, 1, 3),    # K 28: gather without fold; padded rows take the matmul gather
    (1, 24, 8, 9, 12, 1, 4),    # K 72: 8-row block then the 4-row sub-block
    (1, 22, 10, 7, 21, 1, 3),   # K 70: 16 + 4 gather, one remainder dot (fold)
    (1, 22, 32, 7, 3, 1, 3),    # K 224, O < 4: every column through the long dot fold
    (1, 22, 40, 7, 3, 1, 0),    # K 280 > 256: dot fold over two blocks
    (1, 22, 8, 7, 5, 1, 0),     # K 56, O % 8 = 5 < 80: strided, 4-row block + 1 remainder
    (1, 12, 40, 3, 12, 1, 1),   # K 120, O % 8 = 4: in-place rows through the contiguous matmul
    (2, 11, 2, 4, 3, 2, 1),     # batch 2, stride 2, K 8 < 32 with three columns (remainder lanes)
    (1, 9, 3, 5, 5, 1, 0),      # K 15: one 4-row gather block and a remainder lane dot
])
@pytest.mark.parametrize("acc16", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_matches_kernel_restatement(shape, acc16, packed):
    n, W, C, KW, O, stride, pad = shape
    x, w, b = _draw(hash(shape) & 0xFFFF, n, W, C, KW, O)
    out_w = (W + 2 * pad - KW) // stride + 1
    got = _run(x, w, b, stride=stride, pad=pad, acc16=acc16, packed=packed)
    want = _emulate(x, w, b, stride, pad, out_w, acc16, packed)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))


def test_no_bias_matches_zero_bias():
    x, w, _ = _draw(3, 1, 22, 14, 7, 16, bias=False)
    zero = np.zeros(16, np.float16)
    np.testing.assert_array_equal(_run(x, w, None, pad=3), _run(x, w, zero, pad=3))


def test_fold_and_acc16_differ_where_float16_lanes_saturate():
    # Gather column: the bias 2048 swallows every +1 in float16 (ulp 2). Folding restarts at zero after 32
    # taps and adds the last 8 in float32: 2048 + 8. The _acc16 entry never folds: 2048.
    x = np.ones((1, 1, 2, 20), np.float16)
    w = np.ones((8, 1, 2, 20), np.float16)
    b = np.full(8, 2048, np.float16)
    assert (_run(x, w, b) == 2056).all()
    assert (_run(x, w, b, acc16=True) == 2048).all()


def test_k_of_32_never_folds():
    x = np.ones((1, 1, 2, 16), np.float16)
    w = np.ones((8, 1, 2, 16), np.float16)
    b = np.full(8, 2048, np.float16)
    np.testing.assert_array_equal(_run(x, w, b), _run(x, w, b, acc16=True))


def test_clamp_applies_to_every_route():
    x, w, b = _draw(9, 1, 22, 14, 7, 21)
    got = _run(x, w, b, pad=3, activation=(-0.25, 0.5))
    assert got.min() >= -0.25 and got.max() <= 0.5
    free = _run(x, w, b, pad=3)
    np.testing.assert_array_equal(got, np.clip(free, np.float16(-0.25), np.float16(0.5)))


def test_infinity_propagates_through_the_fused_chain():
    x, w, b = _draw(5, 1, 9, 4, 5, 4)
    x[0, 0, 0, 0] = np.inf
    got = _run(x, w, b)
    assert not np.isfinite(got[0, 0, 0]).all() and np.isfinite(got[0, 0, 1:]).all()


def _call(params=None, x=None, w=None, b=None, out=(1, 1, 5, 4)):
    x0, w0, b0 = _draw(1, 1, 9, 4, 5, 4)
    p = {"stride_w": 1, "pad_w": 0, "acc16": 0, "packed": 0, "activation_min": -1.0, "activation_max": 1.0}
    p.update(params or {})
    return get_bindings().run("conv_1xn_f16_mve", p,
                              {"input": x0 if x is None else x, "filter": w0 if w is None else w,
                               "bias": b0 if b is None else b}, {"output": out})


@pytest.mark.parametrize("params", [
    {"stride_w": 0}, {"stride_w": -1}, {"pad_w": -1}, {"acc16": 2}, {"packed": -1},
    {"activation_min": 1.0, "activation_max": -1.0}, {"activation_min": float("nan")},
])
def test_rejects_bad_params(params):
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _call(params)


@pytest.mark.parametrize("kw", [
    {"x": np.zeros((1, 2, 9, 4), np.float16)},              # input height 2
    {"w": np.zeros((4, 1, 1, 4), np.float16)},              # filter width 1 is not 1xN
    {"w": np.zeros((4, 1, 5, 3), np.float16)},              # filter depth != input depth
    {"w": np.zeros((4, 2, 5, 4), np.float16)},              # filter height 2
    {"b": np.zeros(5, np.float16)},                         # bias length != output channels
    {"out": (1, 1, 5, 3)},                                  # output channels != filters
    {"out": (2, 1, 5, 4)},                                  # output batch != input batch
])
def test_rejects_bad_shapes(kw):
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        _call(**kw)


def test_rejects_float32_operands():
    # The bindings refuse a mistyped operand before the entry sees it.
    x, _, _ = _draw(1, 1, 9, 4, 5, 4)
    with pytest.raises(TypeError, match="must be float16"):
        _call(x=x.astype(np.float32))


def test_call_builder_validates_operands():
    x, w, b = _draw(1, 1, 9, 4, 5, 4)
    with pytest.raises(ValueError):
        kf.conv_1xn_f16_mve_call(x[:, :, :, :3], w, b, stride_w=1, pad_w=0, out_w=5, acc16=False, packed=False,
                                 activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.conv_1xn_f16_mve_call(x, w, b, stride_w=0, pad_w=0, out_w=5, acc16=False, packed=False,
                                 activation=UNBOUNDED)
    with pytest.raises(ValueError):
        kf.conv_1xn_f16_mve_call(x, w, b, stride_w=1, pad_w=0, out_w=0, acc16=False, packed=False,
                                 activation=UNBOUNDED)


class _Op:
    def __init__(self, cpu="cortex-m55", hint=None):
        self.target_cpu = cpu
        self.desc = {"hint": hint or {}}


_WIN = {"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0, "dilation_h": 1, "dilation_w": 1}


@pytest.mark.parametrize("fn, kw, expect", [
    ("arm_convolve_1_x_n_f16", 7, {"acc16": False, "packed": False}),
    ("arm_convolve_1_x_n_f16_acc16", 7, {"acc16": True, "packed": False}),
    ("arm_convolve_1_x_n_nhwc_ohwi_f16_acc16", 3, {"acc16": True, "packed": False}),
    ("arm_convolve_f16", 4, {"acc16": False, "packed": False}),     # router: 1xN shape no conv1d claims
    ("arm_convolve_f16", 3, None),                                  # conv1d k3 specialization
    ("arm_convolve_wrapper_f16_acc16", 5, None),                    # conv1d k5 specialization
    ("arm_convolve_1x1_f16", 7, None),
])
def test_route_selection(fn, kw, expect):
    shape_in, shape_f, shape_out = (1, 1, 22, 8), (5, 1, kw, 8), (1, 1, 22 - kw + 1, 5)
    got = cr.conv_1xn_f16_route(_Op(), {"kernel_fn": fn}, shape_in, shape_f, shape_out, _WIN, 1)
    assert got == expect


def test_route_selection_packed_hint_and_non_mve_target():
    args = ({"kernel_fn": "arm_convolve_1_x_n_f16"}, (1, 1, 22, 8), (5, 1, 7, 8), (1, 1, 16, 5), _WIN, 1)
    assert cr.conv_1xn_f16_route(_Op(hint={"weight_format": "NT_N_PACKED"}), *args)["packed"] is True
    assert cr.conv_1xn_f16_route(_Op(cpu="cortex-m4"), *args) is None
    stride_h2 = dict(_WIN, stride_h=2)
    assert cr.conv_1xn_f16_route(_Op(), args[0], args[1], args[2], args[3], stride_h2, 1) is None
    assert cr.conv_1xn_f16_route(_Op(), *args[:-1], 2) is None


def test_patches_zero_fill_and_order():
    x = np.arange(1, 7, dtype=np.float16).reshape(1, 1, 3, 2)
    p = cr.conv_1xn_patches(x, 2, 2, 1, 2)
    np.testing.assert_array_equal(p[0], [[0, 0, 1, 2], [3, 4, 5, 6]])
