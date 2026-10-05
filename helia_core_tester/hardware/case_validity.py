"""Timing status: why cycles may not gate perf.

Measurement statuses (error_path, overflow, zero_cycles, below_floor) drop
a case from the perf gate. degenerate_output is informational: it judges the
golden, not the cycles, so the case still gates. A descriptor's
degenerate_golden_reason marks a constant golden as intended. The floor is an empty
call through the same timed window as every kernel; it is recorded, never subtracted.
FLOOR_FACTOR = 3 keeps the fixed window cost under a third of a valid reading.
valid_for_regression also needs a passing comparison: cycles from wrong
output never gate, while timing_status still describes the cycles alone.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from helia_core_tester.generation.golden_check import MIN_ELEMENTS, golden_problem

from .case_bundle import DEGENERATE_REASON_KEY, FLOOR_CASE_ID, blob_numpy
from .measurement import SampleStatistics

FLOOR_FACTOR = 3
# Statuses whose cycles still gate perf.
GATING_STATUSES = frozenset({"valid", "degenerate_output"})


def golden_degenerate(golden: np.ndarray) -> bool:
    """True when the golden cannot catch a broken kernel."""
    values = np.asarray(golden)
    if values.dtype == np.bool_:
        return golden_problem(values, "bool") is not None
    if values.dtype.kind in "iu":
        return golden_problem(values, f"{values.dtype.name}_t") is not None
    # Non-integer: spread check only.
    return values.size >= MIN_ELEMENTS and np.unique(values).size <= 2


def timing_status(case, floor_cycles: float | None) -> str:
    """The first reason a case's cycles cannot gate perf."""
    stats: SampleStatistics = case.statistics
    # Expected success is a normal case.
    if case.case_bundle.expected_status_code not in (None, 0):
        return "error_path"
    if stats.overflow_detected:
        return "overflow"
    if stats.sample_count == 0 or stats.median_cycles <= 0:
        return "zero_cycles"
    if floor_cycles and stats.median_cycles < FLOOR_FACTOR * floor_cycles:
        return "below_floor"
    intended = case.case_bundle.manifest.get(DEGENERATE_REASON_KEY)
    if not intended and golden_degenerate(blob_numpy(case.case_bundle.expected_output)):
        return "degenerate_output"
    return "valid"


def classify_case(case, floor_cycles: float | None):
    """The case with timing_status and validity set."""
    status = timing_status(case, floor_cycles)
    valid = status in GATING_STATUSES and bool(case.comparison.passed)
    stats = replace(case.statistics, timing_status=status, valid_for_regression=valid)
    return replace(case, statistics=stats)


def apply_floor(cases) -> tuple[dict | None, list]:
    """Pull out the floor case; classify the rest."""
    floor = next((case for case in cases if case.case_bundle.case_id == FLOOR_CASE_ID), None)
    rest = [case for case in cases if case is not floor]
    if floor is None or floor.rejection is not None or floor.statistics.sample_count == 0:
        return None, [classify_case(case, None) for case in rest]
    median = floor.statistics.median_cycles
    record = {
        "case_id": FLOOR_CASE_ID,
        "median_cycles": median,
        "mad_cycles": floor.statistics.mad_cycles,
        "below_floor_factor": FLOOR_FACTOR,
        "below_floor_cycles": FLOOR_FACTOR * median,
    }
    return record, [classify_case(case, median) for case in rest]
