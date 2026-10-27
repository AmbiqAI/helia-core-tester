"""Softmax on the C reference, against independent models.

int8 input: a Python model of TFLite's gemmlowp Softmax<int8> on unbounded ints. int16: a
model of SoftmaxInt16, with the LUTs regenerated here by TFLM's float LUTPopulate. Float: numpy
in float64, rounded once.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.reference.abi import dtype_code
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.tests.reference_models import lut_lookup_int16, lut_populate_int16, mbqm, rdbpot, srdhm

I32_MAX, I32_MIN = 2**31 - 1, -(2**31)


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _c_tables() -> dict:
    src = (Path(__file__).resolve().parents[1] / "reference" / "src" / "softmax" / "softmax.c").read_text()
    tables = {}
    for name in ("exp_lut", "one_over_one_plus_x_lut"):
        body = src[src.index(f"{name}[HCT_LUT_S16_SIZE] = {{"):]
        tables[name] = [int(v) for v in re.findall(r"-?\d+", body[body.index("{"):body.index("};")])]
    return tables


TABLES = _c_tables()


def test_luts_are_tflm_lut_populate() -> None:
    rng = np.float32(65535)
    with np.errstate(over="ignore"):
        exp_lut = lut_populate_int16(np.float32(10.0) / rng, 32767, np.float32(2.0) / rng, 0,
                                      lambda v: np.exp(np.float32(v)).astype(np.float32))
        obo = lut_populate_int16(np.float32(1.0) / rng, -32768, np.float32(2.0) / rng, 0,
                                  lambda v: np.float32(1) / (np.float32(1) + np.float32(v)))
    assert TABLES["exp_lut"] == exp_lut
    assert TABLES["one_over_one_plus_x_lut"] == obo


def test_luts_match_the_tables_the_s16_kernel_is_handed() -> None:
    from helia_core_tester.generation.ops.SoftmaxFunctions.softmax_luts import EXP_LUT, ONE_BY_ONE_LUT

    assert TABLES["exp_lut"] == [int(v) for v in re.findall(r"-?\d+", EXP_LUT)]
    assert TABLES["one_over_one_plus_x_lut"] == [int(v) for v in re.findall(r"-?\d+", ONE_BY_ONE_LUT)]


# ------------------------------------------------------------------ models


def _sat_pot(x: int, e: int) -> int:
    t = (1 << (31 - e)) - 1
    return I32_MAX if x > t else (I32_MIN if x < -t else x << e)


def _exp_on_negative_values(a: int) -> int:
    x = (((a & ((1 << 24) - 1)) - (1 << 24)) << 5) + (1 << 28)
    rem = ((a & ((1 << 24) - 1)) - (1 << 24)) - a
    x2 = srdhm(x, x)
    poly = rdbpot(srdhm(rdbpot(srdhm(x2, x2), 2) + srdhm(x2, x), 715827883) + x2, 1)
    r = 1895147668 + srdhm(1895147668, x + poly)
    for k, m in enumerate((1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242)):
        if rem & (1 << (24 + k)):
            r = srdhm(r, m)
    return I32_MAX if a == 0 else r


def _one_over_one_plus_x(a: int) -> int:
    s = a + I32_MAX
    half = (s + (1 if s >= 0 else -1)) // 2 if s >= 0 else -((-s + 1) // 2)
    x = 1515870810 + srdhm(half, -1010580540)
    for _ in range(3):
        x = x + _sat_pot(srdhm(x, (1 << 29) - srdhm(half, x)), 2)
    return _sat_pot(x, 1)


def _model_softmax_s8(row, p, out_bits):
    qmin = -(1 << (out_bits - 1))
    mx = max(row)

    def e(v):
        return _exp_on_negative_values(srdhm((v - mx) << p["input_left_shift"], p["input_multiplier"]))

    total = sum(rdbpot(e(v), 12) for v in row if v - mx >= p["diff_min"])
    h = 32 - total.bit_length()
    scale = _one_over_one_plus_x(((total << h) & 0xFFFFFFFF) - (1 << 31))
    out = []
    for v in row:
        if v - mx < p["diff_min"]:
            out.append(qmin)
            continue
        shift = 12 - h + 31 - out_bits
        prod = srdhm(scale, e(v))
        y = rdbpot(prod, shift) if shift <= 31 else 0
        out.append(min(max(y + qmin, qmin), -qmin - 1))
    return out


def _model_softmax_s16(row, p):
    mx = max(row)
    ex = [lut_lookup_int16(min(max(mbqm(v - mx, p["input_multiplier"], p["input_left_shift"]) + 32767, -32768), 32767),
                  TABLES["exp_lut"]) for v in row]
    total = sum(ex)
    h = 32 - total.bit_length()
    shifted = ((total << (h - 1)) + (1 << 13)) >> 14
    rec = lut_lookup_int16(min(max(shifted - (1 << 15) - (1 << 16), -32768), 32767), TABLES["one_over_one_plus_x_lut"])
    rs = 31 - h
    return [min(max((e * rec + (1 << (rs - 1))) >> rs, 0), 32767) for e in ex]


# ----------------------------------------------------------------- prepare


def _prepare(lib, in_dt, out_dt, s_in, z_in=0, s_out=None, z_out=None, beta=1.0):
    defaults = {("int8", "int8"): (1 / 256, -128), ("int8", "int16"): (1 / 65536, -32768),
                ("int16", "int16"): (1 / 32768, 0)}
    d_scale, d_zp = defaults.get((in_dt, out_dt), (1 / 256, -128))
    return lib.prepare("softmax_prepare", {
        "input_dtype": dtype_code(in_dt), "output_dtype": dtype_code(out_dt), "beta": beta, "input_scale": s_in,
        "input_zero_point": z_in, "output_scale": d_scale if s_out is None else s_out,
        "output_zero_point": d_zp if z_out is None else z_out})


def test_prepare_matches_tflite_softmax_params(lib) -> None:
    p = _prepare(lib, "int8", "int8", 1 / 128)
    # 2^-7 * 2^26 = 2^19: multiplier 2^30 with left shift 20; radius floor(31 * 2^26 / 2^20).
    assert p == {"input_multiplier": 1 << 30, "input_left_shift": 20, "diff_min": -(31 * 64)}
    q = _prepare(lib, "int16", "int16", 1 / 32767)
    m, s = q["input_multiplier"], q["input_left_shift"]
    assert abs(m * 2.0 ** (s - 31) - (1 / 32767) * 65535 / 10) < 1e-9 and q["diff_min"] == 0


@pytest.mark.parametrize("args", [
    ("int8", "int8", 1 / 128, 0, 1 / 256, -127),       # output zero point must be -128
    ("int8", "int8", 1 / 128, 0, 1 / 255, -128),       # output scale must be exactly 1/256
    ("int8", "int16", 1 / 128, 0, 1 / 65536, 0),       # int16 output from int8 sits at -32768
    ("int16", "int16", 1 / 32767, 1, None, None),      # int16 input is symmetric
    ("int16", "int16", 1 / 32767, 0, 1 / 16384, 0),    # int16 output scale is 2^-15
    ("int8", "int8", 1e-9, 0, None, None),             # multiplier must exceed one
    ("int8", "int8", math.nan, 0, None, None),
    ("int8", "int8", 1 / 128, 200, None, None),
])
def test_prepare_rejects(lib, args) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _prepare(lib, *args)


@pytest.mark.parametrize("in_dt, out_dt", [("int16", "int8"), ("float32", "float32"), ("int8", "float32")])
def test_prepare_rejects_dtype_pairs(lib, in_dt, out_dt) -> None:
    with pytest.raises(ReferenceKernelError, match="E_DTYPE"):
        _prepare(lib, in_dt, out_dt, 1 / 128, s_out=1 / 256, z_out=-128)


def test_prepare_rejects_nonpositive_beta(lib) -> None:
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        _prepare(lib, "int8", "int8", 1 / 128, beta=0.0)


# ----------------------------------------------------------------- kernels


def _rows(rng, info, n_rows, depth):
    rows = []
    for k in range(n_rows):
        span = [info.max, 40, 3, 0][k % 4]
        c = int(rng.integers(info.min + span, info.max - span + 1))
        rows.append(np.clip(rng.integers(c - span, c + span + 1, size=depth), info.min, info.max))
    return np.array(rows, dtype=info.dtype)


@pytest.mark.parametrize("entry, out_bits", [("softmax_s8", 8), ("softmax_s8_s16", 16)])
@pytest.mark.parametrize("s_in", [1 / 128, 2 / 255, 0.05, 0.3])
@pytest.mark.parametrize("depth", [1, 3, 8, 37])
def test_int8_softmax_matches_the_gemmlowp_model(lib, entry, out_bits, s_in, depth) -> None:
    rng = np.random.default_rng(depth)
    p = _prepare(lib, "int8", "int8" if out_bits == 8 else "int16", float(np.float32(s_in)))
    x = _rows(rng, np.iinfo(np.int8), 24, depth)
    got = lib.run(entry, p, {"input": x}, {"output": x.shape})["output"]
    want = [_model_softmax_s8(row.tolist(), p, out_bits) for row in x]
    np.testing.assert_array_equal(got, want)
    real = (x.astype(np.float64) - x.max(axis=1, keepdims=True)) * float(np.float32(s_in))
    prob = np.exp(real) / np.exp(real).sum(axis=1, keepdims=True)
    ideal = np.round(prob * 2.0**out_bits) - 2.0 ** (out_bits - 1)
    assert np.abs(got.astype(np.float64) - np.clip(ideal, -(2 ** (out_bits - 1)), 2 ** (out_bits - 1) - 1)).max() <= 2


def test_int8_softmax_drops_differences_below_diff_min(lib) -> None:
    p = {**_prepare(lib, "int8", "int8", 1 / 128), "diff_min": -10}
    x = np.array([[100, 95, 89, 80]], np.int8)
    got = lib.run("softmax_s8", p, {"input": x}, {"output": x.shape})["output"]
    assert got[0, 2] == -128 and got[0, 3] == -128
    np.testing.assert_array_equal(got[0], _model_softmax_s8(x[0].tolist(), p, 8))


@pytest.mark.parametrize("s_in", [1 / 32767, 1e-4, 3e-3, 2 ** -10])
@pytest.mark.parametrize("depth", [1, 3, 8, 37])
def test_int16_softmax_matches_the_model(lib, s_in, depth) -> None:
    rng = np.random.default_rng(depth + 100)
    p = _prepare(lib, "int16", "int16", float(np.float32(s_in)))
    x = _rows(rng, np.iinfo(np.int16), 24, depth)
    got = lib.run("softmax_s16", p, {"input": x}, {"output": x.shape})["output"]
    np.testing.assert_array_equal(got, [_model_softmax_s16(row.tolist(), p) for row in x])
    real = (x.astype(np.float64) - x.max(axis=1, keepdims=True)) * float(np.float32(s_in))
    prob = np.exp(real) / np.exp(real).sum(axis=1, keepdims=True)
    # exp of -10 and below reads as the LUT floor (2 codes), which a long row of tiny exps adds up.
    assert np.abs(got / 32768.0 - prob).max() < 3e-3


@pytest.mark.parametrize("entry, dtype, params", [
    ("softmax_s8", np.int8, {"input_multiplier": -1, "input_left_shift": 20, "diff_min": -1984}),
    ("softmax_s8", np.int8, {"input_multiplier": 1 << 30, "input_left_shift": 31, "diff_min": 0}),
    ("softmax_s8", np.int8, {"input_multiplier": 1 << 30, "input_left_shift": 20, "diff_min": 1}),
    ("softmax_s8", np.int8, {"input_multiplier": 1 << 30, "input_left_shift": 20, "diff_min": -1985}),
    ("softmax_s16", np.int16, {"input_multiplier": 1 << 30, "input_left_shift": 31, "diff_min": 0}),
    ("softmax_s16", np.int16, {"input_multiplier": 1 << 30, "input_left_shift": 0, "diff_min": -1}),
])
def test_kernels_reject_invalid_params(lib, entry, dtype, params) -> None:
    x = np.zeros((2, 4), dtype)
    with pytest.raises(ReferenceKernelError, match="E_PARAM"):
        lib.run(entry, params, {"input": x}, {"output": x.shape})


def test_kernels_reject_scalars_and_overlong_rows(lib) -> None:
    p = _prepare(lib, "int8", "int8", 1 / 128)
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        lib.run("softmax_s8", p, {"input": np.zeros((), np.int8)}, {"output": ()})
    x = np.zeros((1, 4096), np.int8)
    with pytest.raises(ReferenceKernelError, match="E_SHAPE"):
        lib.run("softmax_s8", p, {"input": x}, {"output": x.shape})
    # The longest accepted row keeps its sum of exps inside int32.
    x = np.zeros((1, 4095), np.int8)
    got = lib.run("softmax_s8", p, {"input": x}, {"output": x.shape})["output"]
    assert np.all(got == -128)  # 1/4095 of 256 codes rounds to 0


def test_empty_tensors_are_accepted(lib) -> None:
    p = _prepare(lib, "int8", "int8", 1 / 128)
    assert lib.run("softmax_s8", p, {"input": np.zeros((0, 4), np.int8)}, {"output": (0, 4)})["output"].shape == (0, 4)


# ------------------------------------------------------------------- float


def _float_model(x: np.ndarray) -> np.ndarray:
    v = x.astype(np.float64)
    with np.errstate(invalid="ignore"):
        e = np.exp(v - np.max(v, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("dtype, entry", [(np.float32, "softmax_f32"), (np.float16, "softmax_f16")])
def test_float_softmax_is_float64_rounded_once(lib, dtype, entry) -> None:
    rng = np.random.default_rng(7)
    for shape in [(4, 1), (3, 2), (5, 8), (2, 3, 37)]:
        x = (rng.standard_normal(shape) * 4).astype(dtype)
        got = lib.run(entry, {"unused": 0}, {"input": x}, {"output": x.shape})["output"]
        np.testing.assert_array_equal(got, _float_model(x).astype(dtype))
        np.testing.assert_allclose(got.astype(np.float64).sum(axis=-1), 1.0, rtol=0, atol=4e-3)


@pytest.mark.parametrize("dtype, entry", [(np.float32, "softmax_f32"), (np.float16, "softmax_f16")])
def test_float_softmax_nonfinite_rows(lib, dtype, entry) -> None:
    x = np.array([[np.nan, 1, 2, 3], [np.inf, 1, 2, 3], [-np.inf, 1, 2, 3], [-np.inf] * 4, [0, 1, 2, 3]], dtype)
    got = lib.run(entry, {"unused": 0}, {"input": x}, {"output": x.shape})["output"]
    want = _float_model(x).astype(dtype)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
    assert np.all(np.isnan(got[0])) and got[2, 0] == 0 and np.all(np.isnan(got[3]))
