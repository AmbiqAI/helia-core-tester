"""The shim computes what the TFLM reference kernels compute, in the conventions
the harness uses, checked against an independent numpy model."""

from __future__ import annotations

import json

import numpy as np
import pytest

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference import params
from helia_core_tester.generation.reference.case import ReferenceCall
from helia_core_tester.generation.reference.run import run_reference
from helia_core_tester.tests import tflm_numpy_model as model


@pytest.fixture(scope="module")
def lib() -> b.Bindings:
    return b.get_bindings()


def _quant(rng, channels, lo=-12, hi=-6, wide=False):
    multiplier = rng.integers(1 << 30, (1 << 31) - 1, size=channels, dtype=np.int64).astype(np.int32)
    shift = rng.integers(lo, hi, size=channels).astype(np.int32)
    return b.PerChannel(multiplier, shift)


def _conv_params(stride, dilation, pad, offset, input_offset, output_offset, act):
    return b.HctConvParams(stride[0], stride[1], dilation[0], dilation[1], pad[0], pad[1], offset[0], offset[1], input_offset, output_offset, act)


CONV_GEOMETRIES = [
    # (in_hw, k, stride, dilation, padding, batch, in_ch, filter_in_ch, out_ch)
    ((7, 6), (3, 3), (1, 1), (1, 1), "SAME", 1, 5, 5, 7),
    ((9, 8), (3, 2), (2, 2), (1, 1), "SAME", 2, 3, 3, 4),  # stride + padding offset + real batch
    ((10, 9), (3, 3), (1, 2), (2, 2), "VALID", 1, 4, 4, 3),  # dilation (no SpaceToBatch lowering)
    ((6, 6), (3, 3), (1, 1), (2, 1), "SAME", 3, 6, 3, 6),  # groups = 2
    ((5, 5), (1, 1), (1, 1), (1, 1), "VALID", 1, 9, 9, 1),  # 1x1, single output channel
]


@pytest.mark.parametrize("geom", CONV_GEOMETRIES)
def test_conv_s8_matches_model(lib, geom) -> None:
    in_hw, k, stride, dilation, padding, batch, ci, fci, co = geom
    rng = np.random.default_rng(CONV_GEOMETRIES.index(geom))
    (oh, ow), ph, pw = params.conv_geometry(padding, in_hw, k, stride, dilation)
    x = rng.integers(-128, 128, size=(batch, *in_hw, ci), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(co, *k, fci), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-5000, 5000, size=co).astype(np.int32)
    quant = _quant(rng, co)
    input_offset, output_offset = 17, -9
    act = b.make_activation(-100, 110)
    p = _conv_params(stride, dilation, (ph.pad, pw.pad), (ph.offset, pw.offset), input_offset, output_offset, act)
    out = lib.conv("s8", p, quant, x, w, bias, (batch, oh, ow, co))
    acc = model.conv_nhwc(x, w, stride, dilation, (ph.pad, pw.pad), (oh, ow), input_offset)
    expected = model.requantize(acc, bias, quant.multiplier, quant.shift, output_offset, -100, 110)
    np.testing.assert_array_equal(out, expected.astype(np.int8))


def test_conv_s8_without_bias_equals_zero_bias(lib) -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(-128, 128, size=(1, 4, 4, 3), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(2, 3, 3, 3), dtype=np.int64).astype(np.int8)
    quant = _quant(rng, 2)
    p = _conv_params((1, 1), (1, 1), (1, 1), (0, 0), 5, 0, b.make_activation(-128, 127))
    none = lib.conv("s8", p, quant, x, w, None, (1, 4, 4, 2))
    zero = lib.conv("s8", p, quant, x, w, np.zeros(2, np.int32), (1, 4, 4, 2))
    np.testing.assert_array_equal(none, zero)


def test_conv_s16_uses_int64_bias_and_16_bit_multiplier(lib) -> None:
    rng = np.random.default_rng(16)
    x = rng.integers(-32768, 32768, size=(2, 5, 5, 4), dtype=np.int64).astype(np.int16)
    w = rng.integers(-127, 128, size=(3, 3, 3, 4), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-(1 << 34), 1 << 34, size=3, dtype=np.int64)
    quant = _quant(rng, 3, lo=-20, hi=-14)
    p = _conv_params((1, 1), (1, 1), (1, 1), (0, 0), 0, 0, b.make_activation(-32768, 32767))
    out = lib.conv("s16", p, quant, x, w, bias, (2, 5, 5, 3))
    acc = model.conv_nhwc(x, w, (1, 1), (1, 1), (1, 1), (5, 5))
    expected = model.requantize(acc, bias, quant.multiplier, quant.shift, 0, -32768, 32767, wide=True)
    np.testing.assert_array_equal(out, expected.astype(np.int16))
    # The 16-bit multiplier reduction is observable: the int32 path differs somewhere.
    narrow = model.requantize(acc, bias, quant.multiplier, quant.shift, 0, -32768, 32767, wide=False)
    assert not np.array_equal(narrow, expected)


def test_conv_s16_int32_bias_uses_the_int32_rescale(lib) -> None:
    rng = np.random.default_rng(17)
    x = rng.integers(-2000, 2000, size=(1, 4, 4, 2), dtype=np.int64).astype(np.int16)
    w = rng.integers(-127, 128, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-1000, 1000, size=2).astype(np.int32)
    quant = _quant(rng, 2, lo=-14, hi=-10)
    p = _conv_params((1, 1), (1, 1), (0, 0), (1, 1), 0, 0, b.make_activation(-32768, 32767))
    out = lib.conv("s16_b32", p, quant, x, w, bias, (1, 4, 4, 2))
    acc = model.conv_nhwc(x, w, (1, 1), (1, 1), (0, 0), (4, 4))
    expected = model.requantize(acc, bias, quant.multiplier, quant.shift, 0, -32768, 32767, wide=False)
    np.testing.assert_array_equal(out, expected.astype(np.int16))


@pytest.mark.parametrize("filter_shape", [(3, 3, 3, 5), (2, 1, 3, 3)])  # 135 (odd) and 18 elements
def test_conv_s4_equals_s8_on_unpacked_weights(lib, filter_shape) -> None:
    rng = np.random.default_rng(4)
    x = rng.integers(-128, 128, size=(1, 6, 6, filter_shape[3]), dtype=np.int64).astype(np.int8)
    w = rng.integers(-7, 8, size=filter_shape, dtype=np.int64).astype(np.int8)
    quant = _quant(rng, filter_shape[0])
    (oh, ow), ph, pw = params.conv_geometry("SAME", (6, 6), filter_shape[1:3], (1, 1))
    p = _conv_params((1, 1), (1, 1), (ph.pad, pw.pad), (ph.offset, pw.offset), -3, 4, b.make_activation(-128, 127))
    shape = (1, oh, ow, filter_shape[0])
    s8 = lib.conv("s8", p, quant, x, w, None, shape)
    s4 = lib.conv("s4", p, quant, x, model.pack_int4(w), None, shape, filter_shape=filter_shape)
    np.testing.assert_array_equal(s4, s8)


def test_conv_f32_matches_model_with_relu6(lib) -> None:
    rng = np.random.default_rng(32)
    x = rng.uniform(-2, 2, size=(2, 6, 5, 3)).astype(np.float32)
    w = rng.uniform(-1, 1, size=(4, 3, 3, 3)).astype(np.float32)
    bias = rng.uniform(-1, 1, size=4).astype(np.float32)
    (oh, ow), ph, pw = params.conv_geometry("SAME", (6, 5), (3, 3), (2, 1), (1, 2))
    p = _conv_params((2, 1), (1, 2), (ph.pad, pw.pad), (ph.offset, pw.offset), 0, 0, b.make_activation(0, 0, 0.0, 6.0))
    out = lib.conv("f32", p, None, x, w, bias, (2, oh, ow, 4))
    acc = model.conv_nhwc(x, w, (2, 1), (1, 2), (ph.pad, pw.pad), (oh, ow))
    np.testing.assert_allclose(out, np.clip(acc + bias, 0.0, 6.0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("multiplier, dilation, stride", [(1, (1, 1), (1, 1)), (2, (2, 2), (1, 1)), (3, (1, 1), (2, 2))])
def test_dwconv_s8_matches_model(lib, multiplier, dilation, stride) -> None:
    rng = np.random.default_rng(multiplier * 10 + dilation[0])
    c = 3
    x = rng.integers(-128, 128, size=(2, 8, 7, c), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(1, 3, 3, c * multiplier), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-3000, 3000, size=c * multiplier).astype(np.int32)
    quant = _quant(rng, c * multiplier)
    (oh, ow), ph, pw = params.conv_geometry("SAME", (8, 7), (3, 3), stride, dilation)
    p = b.HctDwConvParams(_conv_params(stride, dilation, (ph.pad, pw.pad), (ph.offset, pw.offset), 11, -4, b.make_activation(-128, 127)), multiplier)
    out = lib.dwconv("s8", p, quant, x, w, bias, (2, oh, ow, c * multiplier))
    acc = model.dwconv_nhwc(x, w, multiplier, stride, dilation, (ph.pad, pw.pad), (oh, ow), 11)
    expected = model.requantize(acc, bias, quant.multiplier, quant.shift, -4, -128, 127)
    np.testing.assert_array_equal(out, expected.astype(np.int8))


def test_dwconv_s16_and_s4(lib) -> None:
    rng = np.random.default_rng(5)
    x16 = rng.integers(-32768, 32768, size=(1, 5, 5, 4), dtype=np.int64).astype(np.int16)
    w = rng.integers(-7, 8, size=(1, 3, 3, 4), dtype=np.int64).astype(np.int8)
    bias64 = rng.integers(-(1 << 30), 1 << 30, size=4, dtype=np.int64)
    quant = _quant(rng, 4, lo=-18, hi=-12)
    p = b.HctDwConvParams(_conv_params((1, 1), (1, 1), (1, 1), (0, 0), 0, 0, b.make_activation(-32768, 32767)), 1)
    out = lib.dwconv("s16", p, quant, x16, w, bias64, (1, 5, 5, 4))
    acc = model.dwconv_nhwc(x16, w, 1, (1, 1), (1, 1), (1, 1), (5, 5))
    np.testing.assert_array_equal(out, model.requantize(acc, bias64, quant.multiplier, quant.shift, 0, -32768, 32767, wide=True).astype(np.int16))

    x8 = rng.integers(-128, 128, size=(1, 5, 5, 4), dtype=np.int64).astype(np.int8)
    p8 = b.HctDwConvParams(_conv_params((1, 1), (1, 1), (1, 1), (0, 0), 2, 1, b.make_activation(-128, 127)), 1)
    quant8 = _quant(rng, 4)
    s8 = lib.dwconv("s8", p8, quant8, x8, w, None, (1, 5, 5, 4))
    s4 = lib.dwconv("s4", p8, quant8, x8, model.pack_int4(w), None, (1, 5, 5, 4), filter_shape=w.shape)
    np.testing.assert_array_equal(s4, s8)


def test_dwconv_f32(lib) -> None:
    rng = np.random.default_rng(6)
    x = rng.uniform(-1, 1, size=(1, 6, 6, 2)).astype(np.float32)
    w = rng.uniform(-1, 1, size=(1, 3, 3, 4)).astype(np.float32)
    p = b.HctDwConvParams(_conv_params((1, 1), (1, 1), (1, 1), (0, 0), 0, 0, b.make_activation(0, 0)), 2)
    out = lib.dwconv("f32", p, None, x, w, None, (1, 6, 6, 4))
    np.testing.assert_allclose(out, model.dwconv_nhwc(x, w, 2, (1, 1), (1, 1), (1, 1), (6, 6)), rtol=1e-5, atol=1e-6)


def _fc_model(x, w, bias, mult, shift, input_offset, weights_offset, output_offset, lo, hi, wide=False):
    acc = (x.reshape(-1, w.shape[1]).astype(np.int64) + input_offset) @ (w.astype(np.int64) + weights_offset).T
    return model.requantize(acc, bias, mult, shift, output_offset, lo, hi, wide=wide)


def test_fc_s8_per_channel_with_real_batch_and_rank3_input(lib) -> None:
    rng = np.random.default_rng(8)
    x = rng.integers(-128, 128, size=(2, 3, 10), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(5, 10), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-2000, 2000, size=5).astype(np.int32)
    quant = _quant(rng, 5)
    p = b.HctFcParams(-7, 0, 3, b.make_activation(-128, 127))
    out = lib.fc("s8", p, quant, x, w, bias, (6, 5))
    np.testing.assert_array_equal(out, _fc_model(x, w, bias, quant.multiplier, quant.shift, -7, 0, 3, -128, 127).astype(np.int8))


def test_fc_s8_per_tensor_honours_weights_offset(lib) -> None:
    # The force_filter_offset path: a per-tensor kernel with a nonzero filter zero point.
    rng = np.random.default_rng(9)
    x = rng.integers(-128, 128, size=(3, 7), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(4, 7), dtype=np.int64).astype(np.int8)
    quant = b.PerChannel(np.array([1 << 30], np.int32), np.array([-9], np.int32))
    p = b.HctFcParams(5, -3, -2, b.make_activation(-128, 127))
    out = lib.fc("s8", p, quant, x, w, None, (3, 4))
    np.testing.assert_array_equal(out, _fc_model(x, w, None, quant.multiplier, quant.shift, 5, -3, -2, -128, 127).astype(np.int8))
    with_zero = lib.fc("s8", b.HctFcParams(5, 0, -2, b.make_activation(-128, 127)), quant, x, w, None, (3, 4))
    assert not np.array_equal(out, with_zero)


def test_fc_s16_s4_f32(lib) -> None:
    rng = np.random.default_rng(10)
    x16 = rng.integers(-32768, 32768, size=(2, 9), dtype=np.int64).astype(np.int16)
    w = rng.integers(-7, 8, size=(3, 9), dtype=np.int64).astype(np.int8)
    bias64 = rng.integers(-(1 << 32), 1 << 32, size=3, dtype=np.int64)
    quant = _quant(rng, 3, lo=-16, hi=-10)
    out = lib.fc("s16", b.HctFcParams(0, 0, 0, b.make_activation(-32768, 32767)), quant, x16, w, bias64, (2, 3))
    np.testing.assert_array_equal(out, _fc_model(x16, w, bias64, quant.multiplier, quant.shift, 0, 0, 0, -32768, 32767, wide=True).astype(np.int16))

    x8 = rng.integers(-128, 128, size=(2, 9), dtype=np.int64).astype(np.int8)
    quant8 = _quant(rng, 3)
    p8 = b.HctFcParams(1, 0, 0, b.make_activation(-128, 127))
    s8 = lib.fc("s8", p8, quant8, x8, w, None, (2, 3))
    s4 = lib.fc("s4", p8, quant8, x8, model.pack_int4(w), None, (2, 3), filter_shape=w.shape)
    np.testing.assert_array_equal(s4, s8)

    xf = rng.uniform(-1, 1, size=(4, 9)).astype(np.float32)
    wf = rng.uniform(-1, 1, size=(3, 9)).astype(np.float32)
    bf = rng.uniform(-1, 1, size=3).astype(np.float32)
    out = lib.fc("f32", b.HctFcParams(0, 0, 0, b.make_activation(0, 0, -0.5, 0.5)), None, xf, wf, bf, (4, 3))
    np.testing.assert_allclose(out, np.clip(xf.astype(np.float64) @ wf.T + bf, -0.5, 0.5), rtol=1e-5, atol=1e-6)


def _tconv_model(x, w, stride, pad, out_shape, input_offset):
    n, h, wd, c = x.shape
    o, kh, kw, _ = w.shape
    out = np.zeros(out_shape, dtype=np.int64 if x.dtype != np.float32 else np.float64)
    xs = x.astype(out.dtype) + (input_offset if x.dtype != np.float32 else 0)
    for iy in range(h):
        for ix in range(wd):
            for fy in range(kh):
                for fx in range(kw):
                    oy, ox = iy * stride[0] - pad[0] + fy, ix * stride[1] - pad[1] + fx
                    if 0 <= oy < out_shape[1] and 0 <= ox < out_shape[2]:
                        out[:, oy, ox, :] += xs[:, iy, ix, :] @ w[:, fy, fx, :].astype(out.dtype).T
    return out


def test_tconv_s8_and_f32(lib) -> None:
    rng = np.random.default_rng(11)
    x = rng.integers(-128, 128, size=(2, 4, 3, 3), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(5, 3, 3, 3), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-500, 500, size=5).astype(np.int32)
    quant = _quant(rng, 5)
    ph, pw = params.tconv_padding("SAME", (8, 6), (3, 3), (2, 2))
    p = _conv_params((2, 2), (1, 1), (ph.pad, pw.pad), (ph.offset, pw.offset), 6, -1, b.make_activation(-128, 127))
    out = lib.tconv("s8", p, quant, x, w, bias, (2, 8, 6, 5))
    acc = _tconv_model(x, w, (2, 2), (ph.pad, pw.pad), (2, 8, 6, 5), 6)
    np.testing.assert_array_equal(out, model.requantize(acc, bias, quant.multiplier, quant.shift, -1, -128, 127).astype(np.int8))

    xf = rng.uniform(-1, 1, size=(1, 3, 3, 2)).astype(np.float32)
    wf = rng.uniform(-1, 1, size=(2, 3, 3, 2)).astype(np.float32)
    pf = _conv_params((2, 2), (1, 1), (0, 0), (0, 0), 0, 0, b.make_activation(0, 0))
    out = lib.tconv("f32", pf, None, xf, wf, None, (1, 7, 7, 2))
    np.testing.assert_allclose(out, _tconv_model(xf, wf, (2, 2), (0, 0), (1, 7, 7, 2), 0), rtol=1e-5, atol=1e-6)


def test_tconv_s16(lib) -> None:
    rng = np.random.default_rng(12)
    x = rng.integers(-32768, 32768, size=(1, 3, 3, 2), dtype=np.int64).astype(np.int16)
    w = rng.integers(-127, 128, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-(1 << 30), 1 << 30, size=2, dtype=np.int64)
    quant = _quant(rng, 2, lo=-16, hi=-12)
    p = _conv_params((2, 2), (1, 1), (0, 0), (0, 0), 0, 0, b.make_activation(-32768, 32767))
    out = lib.tconv("s16", p, quant, x, w, bias, (1, 6, 6, 2))
    acc = _tconv_model(x, w, (2, 2), (0, 0), (1, 6, 6, 2), 0)
    np.testing.assert_array_equal(out, model.requantize(acc, bias, quant.multiplier, quant.shift, 0, -32768, 32767, wide=True).astype(np.int16))


def _pool_model(x, k, stride, pad, out_hw, op):
    n, h, w, c = x.shape
    out = np.zeros((n, *out_hw, c), dtype=np.float64)
    for oy in range(out_hw[0]):
        for ox in range(out_hw[1]):
            ys = slice(max(oy * stride[0] - pad[0], 0), min(oy * stride[0] - pad[0] + k[0], h))
            xs = slice(max(ox * stride[1] - pad[1], 0), min(ox * stride[1] - pad[1] + k[1], w))
            window = x[:, ys, xs, :].astype(np.int64 if op != "favg" else np.float64)
            if op == "max":
                out[:, oy, ox, :] = window.max(axis=(1, 2))
            elif op == "favg":
                out[:, oy, ox, :] = window.mean(axis=(1, 2))
            else:
                total = window.sum(axis=(1, 2))
                count = window.shape[1] * window.shape[2]
                out[:, oy, ox, :] = np.where(total > 0, (total + count // 2) // count, -((-total + count // 2) // count))
    return out


@pytest.mark.parametrize("dtype, kind", [(np.int8, "s8"), (np.int16, "s16")])
def test_pools_quantized(lib, dtype, kind) -> None:
    rng = np.random.default_rng(13)
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max + 1, size=(2, 7, 6, 3), dtype=np.int64).astype(dtype)
    (oh, ow), ph, pw = params.conv_geometry("SAME", (7, 6), (3, 2), (2, 2))
    p = b.HctPoolParams(2, 2, 3, 2, ph.pad, pw.pad, ph.offset, pw.offset, b.make_activation(info.min, info.max))
    avg = lib.pool("avgpool", kind, p, x, (2, oh, ow, 3))
    mx = lib.pool("maxpool", kind, p, x, (2, oh, ow, 3))
    np.testing.assert_array_equal(mx, _pool_model(x, (3, 2), (2, 2), (ph.pad, pw.pad), (oh, ow), "max").astype(dtype))
    np.testing.assert_array_equal(avg, _pool_model(x, (3, 2), (2, 2), (ph.pad, pw.pad), (oh, ow), "avg").astype(dtype))


def test_pool_activation_clamps(lib) -> None:
    x = np.arange(-8, 8, dtype=np.int8).reshape(1, 4, 4, 1)
    p = b.HctPoolParams(1, 1, 1, 1, 0, 0, 0, 0, b.make_activation(-2, 3))
    out = lib.pool("maxpool", "s8", p, x, (1, 4, 4, 1))
    np.testing.assert_array_equal(out, np.clip(x, -2, 3))


def test_pools_f32(lib) -> None:
    rng = np.random.default_rng(14)
    x = rng.uniform(-3, 3, size=(1, 5, 5, 2)).astype(np.float32)
    p = b.HctPoolParams(2, 2, 2, 2, 0, 0, 1, 1, b.make_activation(0, 0))
    avg = lib.pool("avgpool", "f32", p, x, (1, 3, 3, 2))
    np.testing.assert_allclose(avg, _pool_model(x, (2, 2), (2, 2), (0, 0), (3, 3), "favg"), rtol=1e-5, atol=1e-6)


# ---- rejection: the shim validates instead of touching memory ----


def _small_conv():
    x = np.zeros((1, 4, 4, 2), np.int8)
    w = np.zeros((3, 3, 3, 2), np.int8)
    quant = b.PerChannel(np.full(3, 1 << 30, np.int32), np.full(3, -1, np.int32))
    return x, w, quant


def _code(fn) -> int:
    with pytest.raises(b.ReferenceKernelError) as info:
        fn()
    return info.value.code


def test_rejects_inconsistent_output_shape(lib) -> None:
    x, w, quant = _small_conv()
    p = _conv_params((1, 1), (1, 1), (1, 1), (0, 0), 0, 0, b.make_activation(-128, 127))
    assert _code(lambda: lib.conv("s8", p, quant, x, w, None, (1, 3, 4, 3))) == b.E_DIMS
    assert _code(lambda: lib.conv("s8", p, quant, x, w, None, (1, 4, 4, 2))) == b.E_DIMS
    assert _code(lambda: lib.conv("s8", p, quant, x, w, None, (2, 4, 4, 3))) == b.E_DIMS


def test_rejects_bad_parameters(lib) -> None:
    x, w, quant = _small_conv()
    good = dict(stride=(1, 1), dilation=(1, 1), pad=(1, 1), offset=(0, 0), input_offset=0, output_offset=0, act=b.make_activation(-128, 127))

    def conv(**changes):
        kw = {**good, **changes}
        p = _conv_params(kw["stride"], kw["dilation"], kw["pad"], kw["offset"], kw["input_offset"], kw["output_offset"], kw["act"])
        return lambda: lib.conv("s8", p, quant, x, w, None, (1, 4, 4, 3))

    assert _code(conv(act=b.make_activation(10, -10))) == b.E_PARAM
    assert _code(conv(act=b.make_activation(-129, 127))) == b.E_PARAM
    assert _code(conv(input_offset=129)) == b.E_PARAM
    assert _code(conv(output_offset=128)) == b.E_PARAM
    assert _code(conv(stride=(0, 1))) == b.E_PARAM
    assert _code(conv(offset=(2, 0))) == b.E_PARAM
    assert _code(conv(pad=(-1, 1))) == b.E_PARAM


def test_rejects_wrong_quant_and_bias_lengths(lib) -> None:
    x, w, _ = _small_conv()
    p = _conv_params((1, 1), (1, 1), (1, 1), (0, 0), 0, 0, b.make_activation(-128, 127))
    short = b.PerChannel(np.full(2, 1 << 30, np.int32), np.full(2, -1, np.int32))
    assert _code(lambda: lib.conv("s8", p, short, x, w, None, (1, 4, 4, 3))) == b.E_PARAM
    negative = b.PerChannel(np.full(3, -1, np.int32), np.full(3, -1, np.int32))
    assert _code(lambda: lib.conv("s8", p, negative, x, w, None, (1, 4, 4, 3))) == b.E_PARAM
    _, _, quant = _small_conv()
    assert _code(lambda: lib.conv("s8", p, quant, x, w, np.zeros(2, np.int32), (1, 4, 4, 3))) == b.E_PARAM


def test_rejects_s16_shift_the_int64_rescale_cannot_take(lib) -> None:
    x = np.zeros((1, 3, 3, 1), np.int16)
    w = np.zeros((1, 1, 1, 1), np.int8)
    p = _conv_params((1, 1), (1, 1), (0, 0), (0, 0), 0, 0, b.make_activation(-32768, 32767))
    quant = b.PerChannel(np.array([1 << 30], np.int32), np.array([8], np.int32))
    assert _code(lambda: lib.conv("s16", p, quant, x, w, None, (1, 3, 3, 1))) == b.E_PARAM
    # s16 activations are symmetric: a zero point is refused.
    p_off = _conv_params((1, 1), (1, 1), (0, 0), (0, 0), 3, 0, b.make_activation(-32768, 32767))
    ok = b.PerChannel(np.array([1 << 30], np.int32), np.array([-1], np.int32))
    assert _code(lambda: lib.conv("s16", p_off, ok, x, w, None, (1, 3, 3, 1))) == b.E_PARAM


def test_per_channel_fc_honours_weights_offset(lib) -> None:
    # CMSIS-NN's per-channel FC still applies a filter offset (force_filter_offset cases).
    rng = np.random.default_rng(21)
    x = rng.integers(-128, 128, size=(2, 13), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(5, 13), dtype=np.int64).astype(np.int8)
    bias = rng.integers(-500, 500, size=5).astype(np.int32)
    quant = _quant(rng, 5)
    for offset in (3, -4, 128, -127):
        out = lib.fc("s8", b.HctFcParams(2, offset, -1, b.make_activation(-128, 127)), quant, x, w, bias, (2, 5))
        expected = _fc_model(x, w, bias, quant.multiplier, quant.shift, 2, offset, -1, -128, 127)
        np.testing.assert_array_equal(out, expected.astype(np.int8))


def test_rejects_bad_fc_offsets_and_shapes(lib) -> None:
    x = np.zeros((1, 4), np.int8)
    w = np.zeros((2, 4), np.int8)
    quant = b.PerChannel(np.full(2, 1 << 30, np.int32), np.full(2, -1, np.int32))
    assert _code(lambda: lib.fc("s8", b.HctFcParams(0, 129, 0, b.make_activation(-128, 127)), quant, x, w, None, (1, 2))) == b.E_PARAM
    assert _code(lambda: lib.fc("s8", b.HctFcParams(0, 0, 0, b.make_activation(-128, 127)), quant, np.zeros((1, 5), np.int8), w, None, (1, 2))) == b.E_DIMS


def test_rejects_dilated_tconv_and_bad_dw(lib) -> None:
    x, _, _ = _small_conv()
    w = np.zeros((3, 2, 2, 2), np.int8)
    quant = b.PerChannel(np.full(3, 1 << 30, np.int32), np.full(3, -1, np.int32))
    p = _conv_params((2, 2), (2, 2), (0, 0), (0, 0), 0, 0, b.make_activation(-128, 127))
    assert _code(lambda: lib.tconv("s8", p, quant, x, w, None, (1, 8, 8, 3))) == b.E_UNSUPPORTED
    dw = b.HctDwConvParams(_conv_params((1, 1), (1, 1), (0, 0), (0, 0), 0, 0, b.make_activation(-128, 127)), 2)
    wdw = np.zeros((1, 1, 1, 3), np.int8)  # 3 != 2 channels * multiplier 2
    assert _code(lambda: lib.dwconv("s8", dw, quant, x, wdw, None, (1, 4, 4, 3))) == b.E_DIMS


def test_null_entry_arguments_are_rejected(lib) -> None:
    # Called below the bindings: a NULL bias with a nonzero length must not be read.
    x, w, quant = _small_conv()
    p = _conv_params((1, 1), (1, 1), (1, 1), (0, 0), 0, 0, b.make_activation(-128, 127))
    out = np.zeros((1, 4, 4, 3), np.int8)
    shapes = [b.make_shape(a.shape) for a in (x, w, out)]
    import ctypes

    code = lib._lib.hct_ref_conv_s8(
        ctypes.byref(p), ctypes.byref(quant.as_struct()), ctypes.byref(shapes[0]), x.ctypes.data,
        ctypes.byref(shapes[1]), w.ctypes.data, None, 3, ctypes.byref(shapes[2]), out.ctypes.data,
    )
    assert code == b.E_NULL
    code = lib._lib.hct_ref_conv_s8(
        ctypes.byref(p), None, ctypes.byref(shapes[0]), x.ctypes.data,
        ctypes.byref(shapes[1]), w.ctypes.data, None, 0, ctypes.byref(shapes[2]), out.ctypes.data,
    )
    assert code == b.E_NULL
    rank7 = b.HctShape()
    rank7.rank = 7
    code = lib._lib.hct_ref_maxpool_s8(ctypes.byref(b.HctPoolParams()), ctypes.byref(rank7), x.ctypes.data, ctypes.byref(shapes[2]), out.ctypes.data)
    assert code == b.E_DIMS


def test_bindings_refuse_silent_casts(lib) -> None:
    x, w, quant = _small_conv()
    p = _conv_params((1, 1), (1, 1), (1, 1), (0, 0), 0, 0, b.make_activation(-128, 127))
    with pytest.raises(TypeError, match="int8"):
        lib.conv("s8", p, quant, x.astype(np.int16), w, None, (1, 4, 4, 3))
    with pytest.raises(TypeError, match="contiguous"):
        lib.conv("s8", p, quant, np.asfortranarray(np.zeros((1, 4, 4, 2), np.int8))[:, :, ::-1, :], w, None, (1, 4, 4, 3))
    with pytest.raises(ValueError, match="packed s4"):
        lib.conv("s4", p, quant, x, w, None, (1, 4, 4, 3), filter_shape=w.shape)
    with pytest.raises(ValueError, match="unknown kernel kind"):
        lib.conv("s32", p, quant, x, w, None, (1, 4, 4, 3))
    with pytest.raises(ValueError):
        b.PerChannel(np.zeros(2, np.int32), np.zeros(3, np.int32))


# ---- ReferenceCall / run_reference ----


def test_reference_call_runs_and_records_provenance(tmp_path) -> None:
    rng = np.random.default_rng(15)
    x = rng.integers(-128, 128, size=(1, 5, 5, 2), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(3, 3, 3, 2), dtype=np.int64).astype(np.int8)
    quant = params.per_channel(0.05, [0.01, 0.02, 0.005], 0.1)
    call = ReferenceCall(
        kernel="conv_s8",
        params={
            "stride": [1, 1], "dilation": [1, 1], "pad": [1, 1], "pad_offset": [0, 0],
            "input_offset": 4, "output_offset": -2, "act": {"min": -128, "max": 127},
            "multiplier": quant.multiplier, "shift": quant.shift,
        },
        tensors={"input": x, "filter": w, "bias": None},
        output_shape=(1, 5, 5, 3),
        output_dtype="int8",
        quant={"input": {"scale": 0.05, "zero_point": -4}},
    )
    out = run_reference(call)
    assert out.shape == (1, 5, 5, 3) and out.dtype == np.int8
    path = call.to_json(tmp_path / "case.reference.json", seeds={"run_seed": 7, "case_seed": 9}, library_key="abc")
    record = json.loads(path.read_text())
    assert record["kernel"] == "conv_s8" and record["library_key"] == "abc"
    assert record["seeds"] == {"run_seed": 7, "case_seed": 9}
    assert record["tensors"]["bias"] is None
    assert record["tensors"]["input"]["shape"] == [1, 5, 5, 2]
    assert record["params"]["multiplier"] == [int(m) for m in quant.multiplier]
    assert call.family == "conv"


def test_reference_call_validates_itself() -> None:
    with pytest.raises(ValueError):
        ReferenceCall("conv", {}, {}, (1,), "int8")
    with pytest.raises(ValueError):
        ReferenceCall("conv_s8", {}, {}, (1, 0), "int8")
    with pytest.raises(TypeError):
        ReferenceCall("conv_s8", {}, {"input": [1, 2]}, (1,), "int8")
    with pytest.raises(TypeError):
        ReferenceCall("conv_s8", {}, {}, (1,), "not-a-dtype")


def test_run_reference_rejects_unknown_kernels_and_missing_tensors() -> None:
    with pytest.raises(ValueError, match="no reference entry"):
        run_reference(ReferenceCall("softmax_s8", {}, {}, (1,), "int8"))
    with pytest.raises(KeyError, match="input"):
        run_reference(ReferenceCall("maxpool_s8", {"stride": [1, 1], "filter": [1, 1], "pad": [0, 0], "act": {"min": -128, "max": 127}}, {}, (1, 1, 1, 1), "int8"))


# ---- OperationBase hooks ----


def _op(call):
    from helia_core_tester.generation.ops._shared.base import OperationBase

    class Op(OperationBase):
        calls = 0

        def build_keras_model(self):
            return None

        def build_reference(self):
            type(self).calls += 1
            return call

    return Op({"name": "hook_case_s8", "operator": "Relu", "activation_dtype": "S8"}, seed=3)


def test_operation_without_reference_keeps_its_old_path() -> None:
    op = _op(None)
    assert op.reference is None
    with pytest.raises(NotImplementedError):
        op.golden()
    sidecar = op._build_generation_sidecar("test", {"name": "hook_case_s8"})
    assert "reference" not in sidecar


def test_operation_reference_golden_is_cached_and_recorded() -> None:
    x = np.arange(16, dtype=np.int8).reshape(1, 4, 4, 1)
    call = ReferenceCall(
        "maxpool_s8",
        {"stride": [2, 2], "filter": [2, 2], "pad": [0, 0], "act": {"min": -128, "max": 127}},
        {"input": x},
        (1, 2, 2, 1),
        "int8",
    )
    op = _op(call)
    first = op.golden()
    np.testing.assert_array_equal(first.ravel(), [5, 7, 13, 15])
    assert op.golden() is first and type(op).calls == 1
    sidecar = op._build_generation_sidecar("test", {"name": "hook_case_s8"})
    assert sidecar["reference"]["kernel"] == "maxpool_s8"
    assert sidecar["reference"]["library_key"] == b.loaded_library_key()
