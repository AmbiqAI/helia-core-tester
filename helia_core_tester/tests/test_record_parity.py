"""Parity of a recorded output against the golden compiled into its harness."""

from __future__ import annotations

import numpy as np
import pytest

from helia_core_tester.record import parity
from helia_core_tester.record.capture import TensorRecord


def _record(actual, golden, dtype, mode="exact", tolerance=0, atol=0.0, rtol=0.0, mask=None) -> TensorRecord:
    np_dtype = np.dtype(dtype if dtype != "bool" else np.bool_)
    a = np.asarray(actual, dtype=np_dtype)
    g = np.asarray(golden, dtype=np_dtype)
    return TensorRecord("x_expected_output", dtype, mode, a.shape, tolerance, atol, rtol, a, g,
                        None if mask is None else np.asarray(mask, np.uint8))


def test_identical_integers() -> None:
    got = parity.compare(_record([1, -128, 127], [1, -128, 127], "int8"))
    assert (got.verdict, got.differing, got.failing, got.max_abs_diff) == (parity.IDENTICAL, 0, 0, 0.0)


def test_exact_integer_difference_is_a_mismatch() -> None:
    got = parity.compare(_record([1, 2, 4], [1, 2, 3], "int8"))
    assert (got.verdict, got.failing, got.first_failing_index, got.max_abs_diff) == (parity.MISMATCH, 1, 2, 1.0)


@pytest.mark.parametrize("tolerance, verdict", [(0, parity.MISMATCH), (1, parity.WITHIN_TOLERANCE), (2, parity.WITHIN_TOLERANCE)])
def test_tolerant_integer_boundary(tolerance, verdict) -> None:
    got = parity.compare(_record([-32768, 32767], [-32767, 32767], "int16", mode="tolerant", tolerance=tolerance))
    assert got.verdict == verdict and got.differing == 1


def test_integer_extremes_do_not_overflow() -> None:
    got = parity.compare(_record([-(2**31)], [2**31 - 1], "int32", mode="tolerant", tolerance=1))
    assert got.verdict == parity.MISMATCH and got.max_abs_diff == float(2**32 - 1)


def test_booleans() -> None:
    assert parity.compare(_record([True, False], [True, False], "bool")).verdict == parity.IDENTICAL
    assert parity.compare(_record([True, True], [True, False], "bool")).verdict == parity.MISMATCH


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_float_nan_pairs_and_signed_zero(dtype) -> None:
    nan = np.array(np.nan, dtype)
    neg_nan = -nan
    # The check mode guarantees NaN-ness, not sign or payload (ns-cmsis-nn#333).
    assert parity.compare(_record([nan, 1.0], [neg_nan, 1.0], dtype, mode="float")).verdict == parity.IDENTICAL
    assert parity.compare(_record([nan, 1.0], [nan, 1.0], dtype, mode="float")).verdict == parity.IDENTICAL
    zero = parity.compare(_record([0.0], [-0.0], dtype, mode="float"))
    assert (zero.verdict, zero.differing, zero.max_abs_diff) == (parity.WITHIN_TOLERANCE, 1, 0.0)


@pytest.mark.parametrize(
    "actual, golden",
    [([np.inf], [-np.inf]), ([np.nan], [1.0]), ([1.0], [np.inf]), ([np.inf], [np.nan])],
)
def test_float_nonfinite_disagreements_are_mismatches(actual, golden) -> None:
    got = parity.compare(_record(actual, golden, "float32", mode="float", atol=1e9, rtol=1.0))
    assert got.verdict == parity.MISMATCH and got.max_abs_diff is None


def test_float_tolerance_boundary() -> None:
    inside = parity.compare(_record([1.0 + 1.5e-6], [1.0], "float64", mode="float", atol=1e-6, rtol=1e-6))
    outside = parity.compare(_record([1.0 + 2.5e-6], [1.0], "float64", mode="float", atol=1e-6, rtol=1e-6))
    assert inside.verdict == parity.WITHIN_TOLERANCE
    assert outside.verdict == parity.MISMATCH and outside.first_failing_index == 0


def test_masked_lanes_are_skipped() -> None:
    got = parity.compare(_record([9.0, 2.0], [1.0, 2.0], "float32", mode="float", mask=[1, 0]))
    assert got.verdict == parity.IDENTICAL
    unmasked = parity.compare(_record([9.0, 2.0], [1.0, 2.0], "float32", mode="float", mask=[0, 0]))
    assert unmasked.verdict == parity.MISMATCH


def test_zero_length_tensor_is_identical() -> None:
    got = parity.compare(_record(np.zeros(0), np.zeros(0), "int8"))
    assert (got.verdict, got.count, got.max_abs_diff) == (parity.IDENTICAL, 0, None)


def test_worst_verdict() -> None:
    assert parity.worst([]) == parity.IDENTICAL
    assert parity.worst([parity.IDENTICAL, parity.WITHIN_TOLERANCE]) == parity.WITHIN_TOLERANCE
    assert parity.worst([parity.MISMATCH, parity.IDENTICAL]) == parity.MISMATCH
