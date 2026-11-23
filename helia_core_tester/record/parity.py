"""Compare a recorded output with the golden the case was generated with.

Each tensor record holds both: what the library's host scalar build produced
(`actual`) and the golden compiled into the harness (`golden`). The verdicts:

- identical: every element has the same bits (for floats: the same bits, or both NaN).
- within_tolerance: not identical, but every element passes the case's own check
  (integer: |diff| <= tolerance in tolerant mode; float: |diff| <= atol + rtol*|golden|
  with matching non-finite classes). Integer goldens are meant to be bit-exact, so this
  is a finding to triage, not a pass.
- mismatch: at least one element fails the case's own check.

Masked lanes (non-finite don't-care, issue #74) are skipped, as the check mode skips them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from helia_core_tester.record.capture import FLOAT_DTYPES, TensorRecord

IDENTICAL = "identical"
WITHIN_TOLERANCE = "within_tolerance"
MISMATCH = "mismatch"
_RANK = {IDENTICAL: 0, WITHIN_TOLERANCE: 1, MISMATCH: 2}


@dataclass(frozen=True)
class TensorParity:
    symbol: str
    dtype: str
    mode: str
    count: int
    verdict: str
    differing: int
    failing: int
    max_abs_diff: Optional[float]
    first_failing_index: Optional[int] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "dtype": self.dtype,
            "mode": self.mode,
            "count": self.count,
            "verdict": self.verdict,
            "differing": self.differing,
            "failing": self.failing,
            "max_abs_diff": self.max_abs_diff,
            "first_failing_index": self.first_failing_index,
        }


def worst(verdicts) -> str:
    verdicts = list(verdicts)
    return max(verdicts, key=_RANK.__getitem__) if verdicts else IDENTICAL


def compare(record: TensorRecord) -> TensorParity:
    actual = record.actual.reshape(-1)
    golden = record.golden.reshape(-1)
    keep = np.ones(actual.shape, dtype=bool) if record.mask is None else record.mask.reshape(-1) == 0
    if record.dtype in FLOAT_DTYPES:
        differing, failing, max_diff = _compare_float(actual, golden, keep, record.atol, record.rtol)
    else:
        differing, failing, max_diff = _compare_int(actual, golden, keep, record)
    n_differing = int(differing.sum())
    n_failing = int(failing.sum())
    verdict = MISMATCH if n_failing else (WITHIN_TOLERANCE if n_differing else IDENTICAL)
    return TensorParity(
        symbol=record.symbol,
        dtype=record.dtype,
        mode=record.mode,
        count=int(actual.size),
        verdict=verdict,
        differing=n_differing,
        failing=n_failing,
        max_abs_diff=max_diff,
        first_failing_index=int(np.flatnonzero(failing)[0]) if n_failing else None,
    )


def _compare_int(actual, golden, keep, record: TensorRecord):
    diff = np.abs(actual.astype(np.int64) - golden.astype(np.int64)) if actual.size else np.zeros(0, np.int64)
    differing = (diff != 0) & keep
    allowed = record.tolerance if record.mode == "tolerant" else 0
    failing = (diff > allowed) & keep
    max_diff = float(diff[keep].max()) if keep.any() else None
    return differing, failing, max_diff


def _compare_float(actual, golden, keep, atol: float, rtol: float):
    a = actual.astype(np.float64)
    g = golden.astype(np.float64)
    same_bits = actual.view(_uint_view(actual.dtype)) == golden.view(_uint_view(golden.dtype))
    both_nan = np.isnan(a) & np.isnan(g)
    differing = ~(same_bits | both_nan) & keep
    finite = np.isfinite(a) & np.isfinite(g)
    # A non-finite pair passes only as both NaN or as infinities of one sign.
    nonfinite_ok = both_nan | (np.isinf(a) & np.isinf(g) & (np.sign(a) == np.sign(g)))
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.where(finite, np.abs(a - g), 0.0)
        within = diff <= atol + rtol * np.abs(np.where(finite, g, 0.0))
    failing = np.where(finite, ~within, ~nonfinite_ok) & keep
    measured = finite & keep
    max_diff = float(diff[measured].max()) if measured.any() else None
    return differing, failing, max_diff


def _uint_view(dtype: np.dtype) -> np.dtype:
    return np.dtype(f"<u{np.dtype(dtype).itemsize}")
