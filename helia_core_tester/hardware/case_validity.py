"""Timing status: why cycles may not gate perf.

Only "valid" cases gate perf; all stay correctness-checked. A descriptor's
degenerate_golden_reason marks a constant golden as intended, so it stays valid. The floor is an empty
call through the same timed window as every kernel; it is recorded, never subtracted.
FLOOR_FACTOR = 3 keeps the fixed window cost under a third of a valid reading.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .case_bundle import DEGENERATE_REASON_KEY, FLOOR_CASE_ID, blob_numpy
from .measurement import SampleStatistics

FLOOR_FACTOR = 3
# Smaller goldens are too short to judge.
MIN_JUDGED_ELEMENTS = 8
SATURATED_FRACTION = 0.9


def golden_degenerate(golden: np.ndarray) -> bool:
    """True when the golden cannot catch a broken kernel."""
    values = np.asarray(golden).reshape(-1)
    if values.size < MIN_JUDGED_ELEMENTS:
        return False
    distinct = np.unique(values).size
    if values.dtype == np.bool_:
        return distinct == 1
    if distinct <= 2:
        return True
    if values.dtype.kind != "i":
        return False
    bounds = np.iinfo(values.dtype)
    saturated = np.count_nonzero((values == bounds.min) | (values == bounds.max))
    return saturated >= SATURATED_FRACTION * values.size


def timing_status(case, floor_cycles: float | None) -> str:
    """The first reason a case's cycles cannot gate perf."""
    stats: SampleStatistics = case.statistics
    if case.case_bundle.expected_status_code is not None:
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
    stats = replace(case.statistics, timing_status=status, valid_for_regression=status == "valid")
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
