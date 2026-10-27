"""Sqrt, Rsqrt, Quantize, Dequantize and Requantize on the C reference, against independent models.

Quantized sqrt and quantize/dequantize are modelled in numpy float32 step for step as TFLite
evaluates them; int16 rsqrt with the LUT regenerated in Python; float (r)sqrt by the bit-level
ns-cmsis-nn#295 model; requantize by a Python model of CMSIS-NN's arm_nn_requantize.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.tests.reference_models import (
    lut_lookup_int16,
    lut_populate_int16,
    rdbpot,
    sqrt_float_reference,
)

F32 = np.float32


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _run(lib, entry, params, x, out_shape=None):
    return lib.run(entry, params, {"input": x}, {"output": x.shape if out_shape is None else out_shape})["output"]


def _uq(s_in, z_in, s_out, z_out):
    return {"input_scale": s_in, "input_zero_point": z_in, "output_scale": s_out, "output_zero_point": z_out}


# ------------------------------------------------------------ quantized sqrt


def _model_sqrt(x, s_in, z_in, s_out, z_out, info):
    out = []
    for v in x.tolist():
        deq = F32(s_in) * F32(v - z_in)
        q = int(np.trunc(np.sqrt(deq) / F32(s_out))) + z_out
        out.append(min(max(q, info.min), info.max))
    return out


@pytest.mark.parametrize("dtype, quant", [
    (np.int8, (0.125, 0, 0.125, 0)), (np.int8, (0.03, -100, 0.02, -128)), (np.int8, (0.5, 10, 0.05, 3)),
    (np.int16, (1 / 32768, 0, 1 / 32768, 0)), (np.int16, (1e-3, 0, 2e-4, 0)),
])
def test_quantized_sqrt_matches_tflite(lib, dtype, quant) -> None:
    info = np.iinfo(dtype)
    s_in, z_in, s_out, z_out = float(F32(quant[0])), quant[1], float(F32(quant[2])), quant[3]
    x = np.arange(max(z_in, info.min), info.max + 1, dtype=np.int64)
    if dtype == np.int16:
        x = x[:: 7]
    x = x.astype(dtype)
    got = _run(lib, "sqrt_s8" if dtype == np.int8 else "sqrt_s16", _uq(s_in, z_in, s_out, z_out), x)
    np.testing.assert_array_equal(got, _model_sqrt(x, s_in, z_in, s_out, z_out, info))


def test_quantized_sqrt_refuses_a_negative_value_before_writing(lib) -> None:
    x = np.array([4, 9, -1, 16], np.int8)
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, "sqrt_s8", _uq(0.125, 0, 0.125, 0), x)


@pytest.mark.parametrize("entry, params", [
    ("sqrt_s8", _uq(0.0, 0, 0.1, 0)), ("sqrt_s8", _uq(0.1, 0, math.inf, 0)), ("sqrt_s8", _uq(0.1, 128, 0.1, 0)),
    ("sqrt_s16", _uq(0.1, 1, 0.1, 0)), ("rsqrt_s16", _uq(0.1, 0, 0.1, -1)), ("rsqrt_s16", _uq(math.nan, 0, 0.1, 0)),
])
def test_quantized_entries_reject_invalid_quantization(lib, entry, params) -> None:
    x = np.zeros(4, np.int8 if entry.endswith("s8") else np.int16)
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, entry, params, x)


def test_quantized_sqrt_saturates_huge_ratios(lib) -> None:
    x = np.array([0, 1, 127], np.int8)
    got = _run(lib, "sqrt_s8", _uq(1e30, 0, 1e-30, 0), x)
    np.testing.assert_array_equal(got, [0, 127, 127])


# ----------------------------------------------------------- quantized rsqrt


@pytest.mark.parametrize("s_in, s_out", [(1 / 512, 1 / 32768), (1 / 32768, 1 / 32768), (1e-3, 1 / 4096)])
def test_int16_rsqrt_is_the_tflite_lut(lib, s_in, s_out) -> None:
    s_in, s_out = float(F32(s_in)), float(F32(s_out))

    def fn(v):
        return F32(32767) * F32(s_out) if v <= 0 else F32(1) / np.sqrt(F32(v))

    lut = lut_populate_int16(s_in, 0, s_out, 0, fn)
    x = np.arange(0, 32768, dtype=np.int16)
    got = _run(lib, "rsqrt_s16", _uq(s_in, 0, s_out, 0), x)
    np.testing.assert_array_equal(got, [lut_lookup_int16(v, lut) for v in x.tolist()])


def test_int16_rsqrt_refuses_negative_inputs(lib) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, "rsqrt_s16", _uq(1 / 512, 0, 1 / 32768, 0), np.array([5, -1], np.int16))


# --------------------------------------------------------------- float sqrt


@pytest.mark.parametrize("reciprocal", [False, True])
def test_float16_sqrt_every_pattern_matches_the_contract_model(lib, reciprocal) -> None:
    bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    got = _run(lib, "rsqrt_f16" if reciprocal else "sqrt_f16", {"unused": 0}, bits.view(np.float16))
    with np.errstate(all="ignore"):
        want = sqrt_float_reference(bits, reciprocal)
    np.testing.assert_array_equal(got.view(np.uint16), want)


@pytest.mark.parametrize("reciprocal", [False, True])
def test_float32_sqrt_matches_the_contract_model(lib, reciprocal) -> None:
    rng = np.random.default_rng(3)
    specials = [0, 0x80000000, 0x7F800000, 0xFF800000, 0xBF800000, 0x7F800123, 0xFFC00567, 0x7FC00000, 1,
                0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x80000001, 0x3F800000]
    bits = np.concatenate([np.array(specials, np.uint32), rng.integers(0, 1 << 32, 200_000, dtype=np.uint32)])
    got = _run(lib, "rsqrt_f32" if reciprocal else "sqrt_f32", {"unused": 0}, bits.view(np.float32))
    with np.errstate(all="ignore"):
        want = sqrt_float_reference(bits, reciprocal)
    np.testing.assert_array_equal(got.view(np.uint32), want)


# ----------------------------------------------------------------- quantize


def _qparams(scale, zp, lo=-math.inf, hi=math.inf):
    return {"scale": scale, "zero_point": zp, "activation_min": lo, "activation_max": hi}


def _model_quantize(x, scale, zp, lo, hi, info):
    v = np.clip(x.astype(F32), F32(lo), F32(hi))
    r = (v / F32(scale)).astype(F32)
    q = np.sign(r) * np.floor(np.abs(r) + F32(0.5))
    return np.clip(q.astype(np.int64) + zp, info.min, info.max)


@pytest.mark.parametrize("entry, dtype, scale, zp", [
    ("quantize_f32_s8", np.int8, 2 / 255, -1), ("quantize_f32_s8", np.int8, 1 / 255, -128),
    ("quantize_f32_s16", np.int16, 1 / 32767, 0), ("quantize_f32_s16", np.int16, 3e-3, 0),
])
@pytest.mark.parametrize("act", [(-math.inf, math.inf), (0.0, math.inf), (0.0, 6.0)])
def test_quantize_is_affine_quantize_after_the_activation(lib, entry, dtype, scale, zp, act) -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(-2, 8, 5000), np.arange(-40, 41) * F32(scale) / 2]).astype(F32)
    scale = float(F32(scale))
    got = _run(lib, entry, _qparams(scale, zp, *act), x)
    np.testing.assert_array_equal(got, _model_quantize(x, scale, zp, *act, np.iinfo(dtype)))


def test_quantize_rounds_halves_away_from_zero(lib) -> None:
    x = np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5], F32)
    np.testing.assert_array_equal(_run(lib, "quantize_f32_s8", _qparams(1.0, 0), x), [1, -1, 2, -2, 3, -3])


def test_quantize_saturates_and_refuses_nonfinite(lib) -> None:
    x = np.array([1e30, -1e30, 3e9], F32)
    np.testing.assert_array_equal(_run(lib, "quantize_f32_s16", _qparams(1e-3, 0), x), [32767, -32768, 32767])
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ReferenceKernelError, match="E_PARAM"):
            _run(lib, "quantize_f32_s8", _qparams(0.1, 0), np.array([0.0, bad], F32))
    # An activation that clamps the infinity away leaves a finite value to quantize.
    np.testing.assert_array_equal(_run(lib, "quantize_f32_s8", _qparams(0.1, 0, 0.0, 6.0), np.array([np.inf], F32)),
                                  [60])


@pytest.mark.parametrize("params", [_qparams(0.0, 0), _qparams(0.1, 200), _qparams(0.1, 0, 1.0, -1.0),
                                    _qparams(0.1, 0, math.nan, 1.0)])
def test_quantize_rejects_invalid_params(lib, params) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, "quantize_f32_s8", params, np.zeros(3, F32))


def test_quantize_s16_rejects_a_zero_point(lib) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, "quantize_f32_s16", _qparams(0.1, 3), np.zeros(3, F32))


# --------------------------------------------------------------- dequantize


@pytest.mark.parametrize("entry, dtype, scale, zp", [
    ("dequantize_s8_f32", np.int8, 2 / 255, -1), ("dequantize_s8_f32", np.int8, 0.1, 50),
    ("dequantize_s16_f32", np.int16, 1 / 32767, 0), ("dequantize_s16_f32", np.int16, 7e-3, 0),
])
@pytest.mark.parametrize("act", [(-math.inf, math.inf), (0.0, math.inf), (0.0, 6.0)])
def test_dequantize_is_tflite_dequantize_then_the_activation(lib, entry, dtype, scale, zp, act) -> None:
    info = np.iinfo(dtype)
    x = np.arange(info.min, info.max + 1, dtype=np.int64)[:: (1 if dtype == np.int8 else 5)].astype(dtype)
    scale = float(F32(scale))
    got = _run(lib, entry, _qparams(scale, zp, *act), x, x.shape)
    want = np.clip((x.astype(np.float64) - zp) * scale, act[0], act[1]).astype(F32)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, np.clip((x.astype(F32) - F32(zp)) * F32(scale), F32(act[0]), F32(act[1])))


def test_float16_widening_is_exact_and_quiets_nans(lib) -> None:
    bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    got = _run(lib, "dequantize_f16_f32", {"activation_min": -math.inf, "activation_max": math.inf},
               bits.view(np.float16)).view(np.uint32)
    frac = (bits & 0x3FF).astype(np.uint32)
    nan = ((bits >> 10) & 0x1F == 0x1F) & (frac != 0)
    widened = bits.view(np.float16).astype(np.float32).view(np.uint32)
    want = np.where(nan, ((bits >> 15).astype(np.uint32) << 31) | 0x7FC00000 | (frac << 13), widened)
    np.testing.assert_array_equal(got, want)


def test_float16_widening_applies_the_activation(lib) -> None:
    x = np.array([-3.0, 0.5, 7.0, np.inf, -np.inf], np.float16)
    got = _run(lib, "dequantize_f16_f32", {"activation_min": 0.0, "activation_max": 6.0}, x)
    np.testing.assert_array_equal(got, [0.0, 0.5, 6.0, 6.0, 0.0])


# --------------------------------------------------------------- requantize


def _model_cmsis_requantize(x: int, m: int, shift: int) -> int:
    left, right = max(shift, 0), max(-shift, 0)
    scaled = ((x << left) + 2**31) % 2**32 - 2**31
    return rdbpot((scaled * m + (1 << 30)) >> 31, right)


@pytest.mark.parametrize("entry, dtype", [("requantize_s8", np.int8), ("requantize_s16", np.int16)])
@pytest.mark.parametrize("m, shift, zi, zo", [(1 << 30, 0, 0, 0), (1518500250, -3, 5, -7), (1073741824, 2, -20, 9),
                                               (2147483647, -31, 0, 0), (1342177280, 1, 0, 0)])
def test_requantize_matches_the_cmsis_model(lib, entry, dtype, m, shift, zi, zo) -> None:
    info = np.iinfo(dtype)
    x = np.arange(info.min, info.max + 1, dtype=np.int64)[:: (1 if dtype == np.int8 else 3)].astype(dtype)
    p = {"multiplier": m, "shift": shift, "input_zero_point": zi, "output_zero_point": zo}
    got = _run(lib, entry, p, x)
    want = [min(max(_model_cmsis_requantize(v - zi, m, shift) + zo, info.min), info.max) for v in x.tolist()]
    np.testing.assert_array_equal(got, want)


def test_requantize_differs_from_tflite_rounding_on_negative_ties(lib) -> None:
    # 0.5 * -3 = -1.5: CMSIS's high multiply rounds the tie up (-1), TFLite's SRDHM away from zero (-2).
    p = {"multiplier": 1 << 30, "shift": 0, "input_zero_point": 0, "output_zero_point": 0}
    np.testing.assert_array_equal(_run(lib, "requantize_s8", p, np.array([-3, 3], np.int8)), [-1, 2])


@pytest.mark.parametrize("field, value", [("multiplier", -1), ("shift", 31), ("shift", -32),
                                          ("input_zero_point", 128), ("output_zero_point", -129)])
def test_requantize_rejects_invalid_params(lib, field, value) -> None:
    p = {"multiplier": 1 << 30, "shift": 0, "input_zero_point": 0, "output_zero_point": 0, field: value}
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _run(lib, "requantize_s8", p, np.zeros(3, np.int8))
