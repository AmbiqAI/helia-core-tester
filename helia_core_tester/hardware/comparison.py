"""Runtime output comparison using helia-core-tester's resolved comparison modes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from helia_core_tester.generation.io.dtypes import resolve_comparison


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    mismatch_count: int
    max_abs_diff: float
    mode: str
    # Elements off the golden, tolerance aside.
    diff_count: int | None = 0


def finite_or_none(value: float) -> float | None:
    """The value, or None if not finite."""
    return value if math.isfinite(value) else None


def strict_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    """Drop the integer tolerance: exact match."""
    if comparison.get("mode") == "tolerant_int":
        return {"mode": "exact_int"}
    return dict(comparison)


def int_abs_diff(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """|actual - expected| as uint64, overflow-free."""
    a = actual.astype(np.int64)
    b = expected.astype(np.int64)
    # Wrapping uint64 subtraction gives the exact gap.
    hi = np.maximum(a, b).view(np.uint64)
    lo = np.minimum(a, b).view(np.uint64)
    return hi - lo


def compare_output(actual: np.ndarray, expected: np.ndarray, descriptor_or_comparison: dict[str, Any]) -> ComparisonResult:
    comparison = descriptor_or_comparison
    if "mode" not in comparison:
        comparison = resolve_comparison(descriptor_or_comparison)

    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    if actual_np.shape != expected_np.shape:
        raise ValueError(f"Shape mismatch: actual {actual_np.shape}, expected {expected_np.shape}")

    mode = str(comparison["mode"])
    diff_count = None
    if mode in ("exact_int", "tolerant_int"):
        abs_diff = int_abs_diff(actual_np, expected_np)
        if mode == "exact_int":
            diffs = actual_np != expected_np
        else:
            diffs = abs_diff > np.uint64(int(comparison.get("tolerance", 0)))
        diff_count = int(np.count_nonzero(abs_diff))
        max_abs_diff = float(np.max(abs_diff)) if actual_np.size else 0.0
    elif mode == "float":
        atol = float(comparison.get("atol", 0.0))
        rtol = float(comparison.get("rtol", 0.0))
        actual_float = actual_np.astype(np.float64).reshape(-1)
        expected_float = expected_np.astype(np.float64).reshape(-1)
        if "nonfinite_mask" in comparison:
            mask = np.asarray(comparison["nonfinite_mask"])
            if (mask.ndim != 1 or mask.size != actual_float.size
                    or (mask.size and mask.dtype.kind not in "biu") or not np.all((mask == 0) | (mask == 1))):
                raise ValueError("nonfinite_mask must be a flat binary mask with one entry per output element")
            actual_float = actual_float[mask == 0]
            expected_float = expected_float[mask == 0]
        finite = np.isfinite(actual_float) & np.isfinite(expected_float)
        matching_nonfinite = (np.isnan(actual_float) & np.isnan(expected_float)) | (
            np.isinf(actual_float) & (actual_float == expected_float)
        )
        diffs = ~(finite | matching_nonfinite)
        # Subtract only finite pairs: NaN > tolerance and Inf > Inf are both false.
        abs_diff = np.abs(actual_float[finite] - expected_float[finite])
        tol = atol + rtol * np.abs(expected_float[finite])
        max_abs_diff = float("inf") if np.any(diffs) else (float(np.max(abs_diff)) if abs_diff.size else 0.0)
        diff_count = int(np.count_nonzero(diffs) + np.count_nonzero(abs_diff))
        diffs[finite] = abs_diff > tol
    elif mode == "bool":
        diffs = actual_np.astype(bool) != expected_np.astype(bool)
        max_abs_diff = float(np.max(diffs.astype(np.int32))) if actual_np.size else 0.0
    elif mode == "none":
        # Unvalidated: pass, metrics unknown.
        return ComparisonResult(passed=True, mismatch_count=0, max_abs_diff=float("nan"), mode=mode, diff_count=None)
    else:
        raise ValueError(f"Unsupported comparison mode: {mode}")

    mismatch_count = int(np.count_nonzero(diffs))
    return ComparisonResult(
        passed=mismatch_count == 0,
        mismatch_count=mismatch_count,
        max_abs_diff=max_abs_diff,
        mode=mode,
        diff_count=mismatch_count if diff_count is None else diff_count,
    )


def compare_status(actual_status: int, descriptor_or_comparison: dict[str, Any]) -> ComparisonResult:
    comparison = descriptor_or_comparison
    mode = str(comparison["mode"])
    if mode != "exact_status":
        raise ValueError(f"Unsupported status comparison mode: {mode}")

    expected_status = int(comparison["expected_status"])
    mismatch_count = 0 if int(actual_status) == expected_status else 1
    return ComparisonResult(
        passed=mismatch_count == 0,
        mismatch_count=mismatch_count,
        # No output elements were compared.
        max_abs_diff=float("nan"),
        mode=mode,
        diff_count=None,
    )
