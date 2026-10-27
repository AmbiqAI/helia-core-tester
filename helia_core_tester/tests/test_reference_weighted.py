"""Conv, DepthwiseConv, TransposeConv, FullyConnected and BatchMatMul on the C reference, against
independent Python models (unbounded ints, per-tap loops written from the TFLite definitions)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.generation.reference.weighted import pack_int4, quantize_bias, quantize_filter, same_or_valid
from helia_core_tester.tests.reference_models import mbqm

KINDS = {"s8": np.int8, "s16": np.int16}


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def mbqm64(x: int, m: int, shift: int) -> int:
    """MultiplyByQuantizedMultiplier(int64): the multiplier reduced to 16 bits, one rounding shift."""
    reduced = (m + (1 << 15)) >> 16 if m < 0x7FFF0000 else 0x7FFF
    total = 15 - shift
    return (x * reduced + (1 << (total - 1))) >> total


def _requant(kind, acc, m, s, out_offset, lo, hi):
    v = (mbqm64(acc, m, s) if kind == "s16" else mbqm(acc, m, s)) + out_offset
    return min(max(v, lo), hi)


def _channel_quant(lib, rng, out_c, in_scale=0.05, out_scale=0.2):
    filter_scales = rng.uniform(0.002, 0.02, out_c).astype(np.float32)
    q = lib.run("per_channel_quant", {"input_scale": in_scale, "output_scale": out_scale},
                {"filter_scale": filter_scales}, {"multiplier": (out_c,), "shift": (out_c,)})
    return q["multiplier"], q["shift"]


def _case(rng, kind, in_shape, filter_shape, out_c):
    info = np.iinfo(KINDS[kind])
    x = rng.integers(info.min, info.max + 1, in_shape).astype(KINDS[kind])
    w = rng.integers(-127, 128, filter_shape).astype(np.int8)
    bias = rng.integers(-2000, 2001, out_c).astype(np.int64 if kind == "s16" else np.int32)
    return x, w, bias


def _conv_params(kind, stride=(1, 1), dilation=(1, 1), pad=(0, 0), zp=(0, 0), act=None):
    info = np.iinfo(KINDS[kind])
    lo, hi = act if act else (int(info.min), int(info.max))
    return {"stride_h": stride[0], "stride_w": stride[1], "dilation_h": dilation[0], "dilation_w": dilation[1],
            "pad_h": pad[0], "pad_w": pad[1], "input_offset": zp[0], "output_offset": zp[1],
            "activation_min": lo, "activation_max": hi}


# ------------------------------------------------------------------- Conv


def conv_model(kind, p, x, w, bias, mult, shift, out_hw):
    n, ih, iw, ic = x.shape
    oc, fh, fw, fc = w.shape
    groups = ic // fc
    per_group = oc // groups
    out = np.zeros((n, *out_hw, oc), dtype=np.int64)
    for b in range(n):
        for oy in range(out_hw[0]):
            for ox in range(out_hw[1]):
                for o in range(oc):
                    g = o // per_group
                    acc = 0
                    for ky in range(fh):
                        for kx in range(fw):
                            y = oy * p["stride_h"] - p["pad_h"] + ky * p["dilation_h"]
                            xx = ox * p["stride_w"] - p["pad_w"] + kx * p["dilation_w"]
                            if 0 <= y < ih and 0 <= xx < iw:
                                for c in range(fc):
                                    acc += int(w[o, ky, kx, c]) * (int(x[b, y, xx, g * fc + c]) + p["input_offset"])
                    acc += int(bias[o]) if bias.size else 0
                    out[b, oy, ox, o] = _requant(kind, acc, int(mult[o]), int(shift[o]), p["output_offset"],
                                                 p["activation_min"], p["activation_max"])
    return out


@pytest.mark.parametrize("kind", ["s8", "s16"])
@pytest.mark.parametrize("groups,stride,dilation,padding", [
    (1, (1, 1), (1, 1), "valid"), (1, (2, 1), (1, 1), "same"), (2, (1, 2), (2, 1), "same"), (3, (1, 1), (1, 2), "valid"),
])
def test_conv_matches_the_model(lib, kind, groups, stride, dilation, padding) -> None:
    rng = np.random.default_rng([groups, stride[0], dilation[1], len(kind)])
    in_shape, kh, kw, oc = (2, 6, 7, 3 * groups), 3, 2, 2 * groups
    oh, ph = same_or_valid(padding, in_shape[1], kh, stride[0], dilation[0])
    ow, pw = same_or_valid(padding, in_shape[2], kw, stride[1], dilation[1])
    x, w, bias = _case(rng, kind, in_shape, (oc, kh, kw, in_shape[3] // groups), oc)
    mult, shift = _channel_quant(lib, rng, oc)
    p = _conv_params(kind, stride, dilation, (ph, pw), (7, -3) if kind == "s8" else (0, 0), act=(-100, 90))
    got = lib.run(f"conv_{kind}", p, {"input": x, "filter": w, "bias": bias, "multiplier": mult, "shift": shift},
                  {"output": (2, oh, ow, oc)})["output"]
    np.testing.assert_array_equal(got, conv_model(kind, p, x, w, bias, mult, shift, (oh, ow)))
    assert got.min() >= -100 and got.max() <= 90


def test_conv_without_a_bias_is_a_zero_bias(lib) -> None:
    rng = np.random.default_rng(3)
    x, w, bias = _case(rng, "s8", (1, 4, 4, 2), (3, 2, 2, 2), 3)
    mult, shift = _channel_quant(lib, rng, 3)
    p = _conv_params("s8")
    run = lambda b: lib.run("conv_s8", p, {"input": x, "filter": w, "bias": b, "multiplier": mult, "shift": shift},
                            {"output": (1, 3, 3, 3)})["output"]
    np.testing.assert_array_equal(run(np.zeros(0, np.int32)), run(np.zeros(3, np.int32)))


def test_conv_float_is_the_exact_sum_rounded_once(lib) -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, (1, 5, 5, 4)).astype(np.float32)
    w = rng.uniform(-1, 1, (3, 3, 3, 4)).astype(np.float32)
    b = rng.uniform(-1, 1, 3).astype(np.float32)
    p = {"stride_h": 1, "stride_w": 1, "dilation_h": 1, "dilation_w": 1, "pad_h": 1, "pad_w": 1,
         "activation_min": -1.5, "activation_max": 1.5}
    got = lib.run("conv_f32", p, {"input": x, "filter": w, "bias": b}, {"output": (1, 5, 5, 3)})["output"]
    xp = np.pad(x.astype(np.float64), ((0, 0), (1, 1), (1, 1), (0, 0)))
    want = np.empty((1, 5, 5, 3))
    for oy in range(5):
        for ox in range(5):
            win = xp[0, oy:oy + 3, ox:ox + 3, :]
            for o in range(3):
                # Products of binary32 values are exact in binary64; fsum sums them exactly.
                want[0, oy, ox, o] = math.fsum((win * w[o].astype(np.float64)).ravel().tolist() + [float(b[o])])
    np.testing.assert_array_equal(got, np.clip(want.astype(np.float32), -1.5, 1.5))

    h = lib.run("conv_f16", p, {"input": x.astype(np.float16), "filter": w.astype(np.float16),
                                "bias": b.astype(np.float16)}, {"output": (1, 5, 5, 3)})["output"]
    assert h.dtype == np.float16 and np.all(np.abs(h.astype(np.float32)) <= 1.5)


def test_conv_rejects_invalid_shapes_and_params(lib) -> None:
    rng = np.random.default_rng(1)
    x, w, bias = _case(rng, "s8", (1, 4, 4, 4), (2, 2, 2, 3), 2)  # 4 % 3 != 0: no grouping
    mult, shift = _channel_quant(lib, rng, 2)
    inputs = {"input": x, "filter": w, "bias": bias, "multiplier": mult, "shift": shift}
    run = lambda p, ins=inputs, out=(1, 3, 3, 2): lib.run("conv_s8", p, ins, {"output": out})
    assert _status(lambda: run(_conv_params("s8"))) == "E_SHAPE"
    w = w[..., :2]
    ok = dict(inputs, filter=np.ascontiguousarray(w))
    run(_conv_params("s8"), ok)
    assert _status(lambda: run(_conv_params("s8"), ok, (1, 3, 3, 3))) == "E_SHAPE"
    assert _status(lambda: run(_conv_params("s8"), dict(ok, bias=np.zeros(3, np.int32)))) == "E_SHAPE"
    assert _status(lambda: run(_conv_params("s8"), dict(ok, multiplier=mult[:1]))) == "E_SHAPE"
    assert _status(lambda: run(_conv_params("s8"), dict(ok, shift=np.array([31, 0], np.int32)))) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s8"), dict(ok, multiplier=np.array([-1, 1], np.int32)))) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s8", stride=(0, 1)), ok)) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s8", dilation=(1, 0)), ok)) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s8", pad=(-1, 0)), ok)) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s8", zp=(129, 0)), ok)) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s8", act=(5, 4)), ok)) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s8", act=(-129, 0)), ok)) == "E_PARAM"
    with pytest.raises(TypeError, match="must be int8"):
        lib.run("conv_s8", _conv_params("s8"), dict(ok, input=x.astype(np.int16)), {"output": (1, 3, 3, 2)})


def test_conv_s16_rejects_offsets_and_wide_shifts(lib) -> None:
    rng = np.random.default_rng(2)
    x, w, bias = _case(rng, "s16", (1, 3, 3, 1), (1, 1, 1, 1), 1)
    mult, shift = _channel_quant(lib, rng, 1)
    ins = {"input": x, "filter": w, "bias": bias, "multiplier": mult, "shift": shift}
    run = lambda p, i=ins: lib.run("conv_s16", p, i, {"output": (1, 3, 3, 1)})
    assert _status(lambda: run(_conv_params("s16", zp=(1, 0)))) == "E_PARAM"
    assert _status(lambda: run(_conv_params("s16"), dict(ins, shift=np.array([8], np.int32)))) == "E_PARAM"
    run(_conv_params("s16"), dict(ins, shift=np.array([7], np.int32)))


def test_conv_s8_rejects_an_accumulator_past_int32(lib) -> None:
    x = np.full((1, 1, 1, 1), 127, np.int8)
    w = np.full((1, 1, 1, 1), 127, np.int8)
    ins = {"input": x, "filter": w, "bias": np.array([2**31 - 16129], np.int32),
           "multiplier": np.array([2**30], np.int32), "shift": np.array([0], np.int32)}
    run = lambda b: lib.run("conv_s8", _conv_params("s8"), dict(ins, bias=np.array([b], np.int32)),
                            {"output": (1, 1, 1, 1)})
    run(2**31 - 1 - 127 * 127)
    assert _status(lambda: run(2**31 - 127 * 127)) == "E_PARAM"


# -------------------------------------------------------------- Depthwise


@pytest.mark.parametrize("kind", ["s8", "s16"])
@pytest.mark.parametrize("multiplier", [1, 3])
def test_depthwise_is_conv_with_one_group_per_input_channel(lib, kind, multiplier) -> None:
    rng = np.random.default_rng([multiplier, len(kind)])
    ic = 3
    oc = ic * multiplier
    x, dw, bias = _case(rng, kind, (2, 5, 6, ic), (1, 3, 3, oc), oc)
    mult, shift = _channel_quant(lib, rng, oc)
    p = _conv_params(kind, (2, 1), (1, 2), (1, 2), (-4, 6) if kind == "s8" else (0, 0))
    ins = {"input": x, "filter": dw, "bias": bias, "multiplier": mult, "shift": shift}
    out = (2, 3, 4, oc)
    got = lib.run(f"depthwise_conv_{kind}", p, ins, {"output": out})["output"]
    as_conv = np.ascontiguousarray(np.transpose(dw, (3, 1, 2, 0)))  # 1HWO -> OHW1
    want = lib.run(f"conv_{kind}", p, dict(ins, filter=as_conv), {"output": out})["output"]
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, conv_model(kind, p, x, as_conv, bias, mult, shift, out[1:3]))


def test_depthwise_rejects_a_filter_depth_not_a_multiple(lib) -> None:
    rng = np.random.default_rng(4)
    x, dw, bias = _case(rng, "s8", (1, 3, 3, 2), (1, 1, 1, 3), 3)
    mult, shift = _channel_quant(lib, rng, 3)
    assert _status(lambda: lib.run("depthwise_conv_s8", _conv_params("s8"),
                                   {"input": x, "filter": dw, "bias": bias, "multiplier": mult, "shift": shift},
                                   {"output": (1, 3, 3, 3)})) == "E_SHAPE"


# ---------------------------------------------------------- TransposeConv


def transpose_model(kind, p, x, w, bias, mult, shift, out_hw):
    n, ih, iw, ic = x.shape
    oc, fh, fw, _ = w.shape
    acc = np.zeros((n, *out_hw, oc), dtype=object)
    acc[...] = 0
    for b in range(n):
        for y in range(ih):
            for xx in range(iw):
                for ky in range(fh):
                    for kx in range(fw):
                        oy = y * p["stride_h"] + ky - p["pad_h"]
                        ox = xx * p["stride_w"] + kx - p["pad_w"]
                        if 0 <= oy < out_hw[0] and 0 <= ox < out_hw[1]:
                            for o in range(oc):
                                for c in range(ic):
                                    acc[b, oy, ox, o] += (int(x[b, y, xx, c]) + p["input_offset"]) * int(w[o, ky, kx, c])
    out = np.zeros(acc.shape, np.int64)
    for idx in np.ndindex(acc.shape):
        a = acc[idx] + (int(bias[idx[3]]) if bias.size else 0)
        out[idx] = _requant(kind, a, int(mult[idx[3]]), int(shift[idx[3]]), p["output_offset"],
                            p["activation_min"], p["activation_max"])
    return out


@pytest.mark.parametrize("kind", ["s8", "s16"])
@pytest.mark.parametrize("stride,pad,out_hw", [((1, 1), (0, 0), (5, 6)), ((2, 2), (1, 0), (6, 9)), ((2, 1), (0, 1), (8, 4))])
def test_transpose_conv_matches_the_scatter_model(lib, kind, stride, pad, out_hw) -> None:
    rng = np.random.default_rng([stride[0], pad[1], len(kind)])
    x, w, bias = _case(rng, kind, (1, 3, 4, 2), (3, 3, 3, 2), 3)
    mult, shift = _channel_quant(lib, rng, 3)
    p = _conv_params(kind, stride, (1, 1), pad, (5, -2) if kind == "s8" else (0, 0))
    got = lib.run(f"transpose_conv_{kind}", p, {"input": x, "filter": w, "bias": bias, "multiplier": mult,
                                                "shift": shift}, {"output": (1, *out_hw, 3)})["output"]
    np.testing.assert_array_equal(got, transpose_model(kind, p, x, w, bias, mult, shift, out_hw))


def test_transpose_conv_rejects_dilation(lib) -> None:
    rng = np.random.default_rng(6)
    x, w, bias = _case(rng, "s8", (1, 2, 2, 1), (1, 2, 2, 1), 1)
    mult, shift = _channel_quant(lib, rng, 1)
    assert _status(lambda: lib.run("transpose_conv_s8", _conv_params("s8", dilation=(2, 1)),
                                   {"input": x, "filter": w, "bias": bias, "multiplier": mult, "shift": shift},
                                   {"output": (1, 3, 3, 1)})) == "E_PARAM"


# --------------------------------------------------------- FullyConnected


@pytest.mark.parametrize("kind,filter_offset", [("s8", 0), ("s8", 3), ("s16", 0)])
def test_fully_connected_matches_the_model(lib, kind, filter_offset) -> None:
    rng = np.random.default_rng([filter_offset, len(kind)])
    x, w, bias = _case(rng, kind, (3, 10), (4, 10), 4)
    mult, shift = _channel_quant(lib, rng, 4)
    in_off, out_off = (-6, 4) if kind == "s8" else (0, 0)
    info = np.iinfo(KINDS[kind])
    p = {"input_offset": in_off, "filter_offset": filter_offset, "output_offset": out_off,
         "activation_min": int(info.min), "activation_max": int(info.max)}
    got = lib.run(f"fully_connected_{kind}", p, {"input": x, "filter": w, "bias": bias, "multiplier": mult,
                                                  "shift": shift}, {"output": (3, 4)})["output"]
    for b in range(3):
        for o in range(4):
            acc = sum((int(x[b, k]) + in_off) * (int(w[o, k]) + filter_offset) for k in range(10)) + int(bias[o])
            assert got[b, o] == _requant(kind, acc, int(mult[o]), int(shift[o]), out_off, info.min, info.max)


def test_fully_connected_per_tensor_is_one_repeated_multiplier(lib) -> None:
    rng = np.random.default_rng(8)
    x, w, bias = _case(rng, "s8", (2, 6), (5, 6), 5)
    p = {"input_offset": 0, "filter_offset": 0, "output_offset": 0, "activation_min": -128, "activation_max": 127}
    m, s = np.full(5, 1 << 30, np.int32), np.full(5, -8, np.int32)
    got = lib.run("fully_connected_s8", p, {"input": x, "filter": w, "bias": bias, "multiplier": m, "shift": s},
                  {"output": (2, 5)})["output"]
    acc = x.astype(np.int64) @ w.T.astype(np.int64) + bias
    np.testing.assert_array_equal(got, np.clip([[mbqm(int(a), 1 << 30, -8) for a in r] for r in acc], -128, 127))


def test_fully_connected_rejects_mismatched_features(lib) -> None:
    rng = np.random.default_rng(9)
    x, w, bias = _case(rng, "s8", (2, 6), (5, 7), 5)
    mult, shift = _channel_quant(lib, rng, 5)
    p = {"input_offset": 0, "filter_offset": 0, "output_offset": 0, "activation_min": -128, "activation_max": 127}
    assert _status(lambda: lib.run("fully_connected_s8", p, {"input": x, "filter": w, "bias": bias,
                                                             "multiplier": mult, "shift": shift},
                                   {"output": (2, 5)})) == "E_SHAPE"


def test_fully_connected_float_rounds_once(lib) -> None:
    rng = np.random.default_rng(10)
    x = rng.uniform(-1, 1, (2, 33)).astype(np.float32)
    w = rng.uniform(-1, 1, (3, 33)).astype(np.float32)
    b = rng.uniform(-1, 1, 3).astype(np.float32)
    got = lib.run("fully_connected_f32", {"activation_min": -np.inf, "activation_max": np.inf},
                  {"input": x, "filter": w, "bias": b}, {"output": (2, 3)})["output"]
    want = [[np.float32(math.fsum((x[i].astype(np.float64) * w[o]).tolist() + [float(b[o])])) for o in range(3)]
            for i in range(2)]
    np.testing.assert_array_equal(got, np.array(want, np.float32))


# ------------------------------------------------------------ BatchMatMul


@pytest.mark.parametrize("kind", ["s8", "s16"])
@pytest.mark.parametrize("adj_x,adj_y", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_batch_matmul_matches_numpy_and_broadcasts(lib, kind, adj_x, adj_y) -> None:
    rng = np.random.default_rng([adj_x, adj_y, len(kind)])
    info = np.iinfo(KINDS[kind])
    lhs = rng.integers(info.min, info.max + 1, (2, 1, 3, 5)).astype(KINDS[kind])
    rhs = rng.integers(info.min, info.max + 1, (1, 3, 5, 4)).astype(KINDS[kind])
    lhs_s = np.ascontiguousarray(np.swapaxes(lhs, -1, -2)) if adj_x else lhs
    rhs_s = np.ascontiguousarray(np.swapaxes(rhs, -1, -2)) if adj_y else rhs
    lo, ro, oo = (3, -2, 5) if kind == "s8" else (0, 0, 0)
    m, s = 1 << 30, -14 if kind == "s16" else -9
    p = {"adj_x": adj_x, "adj_y": adj_y, "lhs_offset": lo, "rhs_offset": ro, "output_offset": oo, "multiplier": m,
         "shift": s, "activation_min": int(info.min), "activation_max": int(info.max)}
    got = lib.run(f"batch_matmul_{kind}", p, {"lhs": lhs_s, "rhs": rhs_s}, {"output": (2, 3, 3, 4)})["output"]
    acc = (lhs.astype(np.int64) + lo) @ (rhs.astype(np.int64) + ro)
    want = np.vectorize(lambda a: _requant(kind, int(a), m, s, oo, int(info.min), int(info.max)))(acc)
    np.testing.assert_array_equal(got, want)


def test_batch_matmul_rejects_mismatched_inner_dims_and_batches(lib) -> None:
    p = {"adj_x": 0, "adj_y": 0, "lhs_offset": 0, "rhs_offset": 0, "output_offset": 0, "multiplier": 1 << 30,
         "shift": -8, "activation_min": -128, "activation_max": 127}
    a, b = np.zeros((1, 2, 3), np.int8), np.zeros((1, 4, 2), np.int8)
    assert _status(lambda: lib.run("batch_matmul_s8", p, {"lhs": a, "rhs": b}, {"output": (1, 2, 2)})) == "E_SHAPE"
    b = np.zeros((3, 3, 2), np.int8)
    assert _status(lambda: lib.run("batch_matmul_s8", dict(p), {"lhs": np.zeros((2, 2, 3), np.int8), "rhs": b},
                                   {"output": (2, 2, 2)})) == "E_SHAPE"
    assert _status(lambda: lib.run("batch_matmul_s8", dict(p, adj_x=2), {"lhs": a, "rhs": np.zeros((1, 3, 2), np.int8)},
                                   {"output": (1, 2, 2)})) == "E_PARAM"


# ---------------------------------------------------------------- weights


def test_quantize_filter_is_symmetric_per_channel() -> None:
    w = np.array([[[[1.0, -2.0]]], [[[0.5, 0.25]]], [[[0.0, 0.0]]]], np.float32)
    q = quantize_filter(w)
    np.testing.assert_allclose(q.scales, [2 / 127, 0.5 / 127, 1.0], rtol=1e-7)
    assert q.values[0].ravel().tolist() == [64, -127] and q.values[1].ravel().tolist() == [127, 64]
    assert q.values[2].ravel().tolist() == [0, 0]
    assert quantize_filter(w, "S4").values.max() <= 7
    with pytest.raises(ValueError):
        quantize_filter(np.array([np.nan], np.float32))
    with pytest.raises(ValueError):
        quantize_filter(w, "S16")
    with pytest.raises(ValueError):
        quantize_filter(w, scales=[1.0, 2.0])


def test_quantize_bias_rounds_half_away_and_refuses_overflow() -> None:
    assert quantize_bias(np.array([0.25, -0.25]), 0.5, np.array([1.0, 1.0]), wide=False).tolist() == [1, -1]
    assert quantize_bias(np.array([1.0]), 1.0, np.array([1e-12]), wide=True).dtype == np.int64
    with pytest.raises(ValueError, match="does not fit"):
        quantize_bias(np.array([1.0]), 1.0, np.array([1e-12]), wide=False)
    with pytest.raises(ValueError):
        quantize_bias(np.array([1.0, 2.0]), 1.0, np.array([0.0, 1.0]), wide=False)


def test_pack_int4_is_low_nibble_first() -> None:
    assert pack_int4(np.array([1, -1, 7])).view(np.uint8).tolist() == [0xF1, 0x07]
    with pytest.raises(ValueError):
        pack_int4(np.array([8]))


@pytest.mark.parametrize("padding,size,k,stride,dil,want", [
    ("same", 7, 3, 2, 1, (4, 1)), ("valid", 7, 3, 2, 1, (3, 0)), ("same", 5, 2, 1, 2, (5, 1)), ("same", 1, 1, 1, 1, (1, 0)),
])
def test_same_or_valid_is_tflite_padding(padding, size, k, stride, dil, want) -> None:
    assert same_or_valid(padding, size, k, stride, dil) == want


def test_same_or_valid_rejects_a_window_larger_than_the_input() -> None:
    with pytest.raises(ValueError, match="window larger"):
        same_or_valid("valid", 2, 3, 1, 1)
    with pytest.raises(ValueError):
        same_or_valid("causal", 4, 1, 1, 1)
