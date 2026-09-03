"""Runtime output comparison using helia-core-tester's resolved comparison modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from helia_core_tester.generation.io.dtypes import resolve_comparison


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    mismatch_count: int
    max_abs_diff: float
    mode: str


def compare_output(actual: np.ndarray, expected: np.ndarray, descriptor_or_comparison: dict[str, Any]) -> ComparisonResult:
    comparison = descriptor_or_comparison
    if "mode" not in comparison:
        comparison = resolve_comparison(descriptor_or_comparison)

    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    if actual_np.shape != expected_np.shape:
        raise ValueError(f"Shape mismatch: actual {actual_np.shape}, expected {expected_np.shape}")

    mode = str(comparison["mode"])
    if mode == "exact_int":
        diffs = actual_np != expected_np
        max_abs_diff = float(np.max(np.abs(actual_np.astype(np.int64) - expected_np.astype(np.int64)))) if actual_np.size else 0.0
    elif mode == "tolerant_int":
        tolerance = int(comparison.get("tolerance", 0))
        abs_diff = np.abs(actual_np.astype(np.int64) - expected_np.astype(np.int64))
        diffs = abs_diff > tolerance
        max_abs_diff = float(np.max(abs_diff)) if actual_np.size else 0.0
    elif mode == "float":
        atol = float(comparison.get("atol", 0.0))
        rtol = float(comparison.get("rtol", 0.0))
        abs_diff = np.abs(actual_np.astype(np.float64) - expected_np.astype(np.float64))
        tol = atol + (rtol * np.abs(expected_np.astype(np.float64)))
        diffs = abs_diff > tol
        max_abs_diff = float(np.max(abs_diff)) if actual_np.size else 0.0
    elif mode == "bool":
        diffs = actual_np.astype(bool) != expected_np.astype(bool)
        max_abs_diff = float(np.max(diffs.astype(np.int32))) if actual_np.size else 0.0
    elif mode == "none":
        # Intentionally unvalidated (HELIA_VALIDATE_OUTPUTS=NONE). Report a pass
        # without diffing rather than crashing the session.
        return ComparisonResult(passed=True, mismatch_count=0, max_abs_diff=0.0, mode=mode)
    else:
        raise ValueError(f"Unsupported comparison mode: {mode}")

    mismatch_count = int(np.count_nonzero(diffs))
    return ComparisonResult(
        passed=mismatch_count == 0,
        mismatch_count=mismatch_count,
        max_abs_diff=max_abs_diff,
        mode=mode,
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
        max_abs_diff=float(abs(int(actual_status) - expected_status)),
        mode=mode,
    )
