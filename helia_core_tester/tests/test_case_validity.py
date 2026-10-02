from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helia_core_tester.hardware.case_bundle import (
    EMPTY_CALL_KERNEL_ID, FLOOR_CASE_ID, build_abs_s8_case_bundle, build_floor_bundle, load_case_bundle,
)
from helia_core_tester.hardware.case_validity import apply_floor, golden_degenerate, timing_status
from helia_core_tester.hardware.kernel_registry import lookup_kernel_id
from helia_core_tester.hardware.comparison import ComparisonResult
from helia_core_tester.hardware.measurement import SampleStatistics
from helia_core_tester.hardware.session import CaseRunResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS = SampleStatistics(5, 900.0, 1000.0, 1000.0, 1000.0, 1.0, True, False, ())


def _case(bundle, *, median=1000.0, overflow=False, samples=5, status=None):
    if status is not None:
        bundle = SimpleNamespace(case_id="err", expected_status_code=status)
    stats = replace(STATS, median_cycles=median, overflow_detected=overflow, sample_count=samples)
    return CaseRunResult(bundle, ComparisonResult(True, 0, 0.0, "exact_int"), b"", (), (), stats)


@pytest.fixture()
def abs_bundle(tmp_path: Path):
    return build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_v", input_shape=(1, 4, 4, 2))


@pytest.mark.parametrize(
    ("golden", "degenerate"),
    [
        (np.arange(-16, 16, dtype=np.int8), False),
        (np.zeros(16, dtype=np.int8), True),
        (np.array([0, 5] * 8, dtype=np.int8), True),
        (np.array([127] * 17 + [-128, 3, 4], dtype=np.int8), True),
        (np.array([-32768] * 5 + list(range(10)), dtype=np.int16), False),
        (np.array([True, False] * 8), False),
        (np.ones(16, dtype=np.bool_), True),
        (np.linspace(-1, 1, 16, dtype=np.float32), False),
        (np.zeros(3, dtype=np.int8), False),
    ],
)
def test_golden_degenerate(golden, degenerate) -> None:
    assert golden_degenerate(golden) is degenerate


def test_timing_status_order(abs_bundle) -> None:
    assert timing_status(_case(abs_bundle), 30.0) == "valid"
    assert timing_status(_case(abs_bundle, status=-1), 30.0) == "error_path"
    assert timing_status(_case(abs_bundle, overflow=True, median=0.0), 30.0) == "overflow"
    assert timing_status(_case(abs_bundle, median=0.0), 30.0) == "zero_cycles"
    assert timing_status(_case(abs_bundle, samples=0), 30.0) == "zero_cycles"
    assert timing_status(_case(abs_bundle, median=89.0), 30.0) == "below_floor"
    assert timing_status(_case(abs_bundle, median=90.0), 30.0) == "valid"
    assert timing_status(_case(abs_bundle, median=89.0), None) == "valid"


def test_apply_floor_records_and_classifies(abs_bundle, tmp_path: Path) -> None:
    floor = build_floor_bundle(tmp_path, board_id="apollo510_evb")
    cases = [_case(floor, median=40.0), _case(abs_bundle, median=100.0), _case(abs_bundle, median=200.0)]
    record, rest = apply_floor(cases)
    assert record["median_cycles"] == 40.0 and record["below_floor_cycles"] == 120.0
    assert [c.statistics.timing_status for c in rest] == ["below_floor", "valid"]
    assert [c.statistics.valid_for_regression for c in rest] == [False, True]
    record, rest = apply_floor(cases[1:])
    assert record is None and [c.statistics.timing_status for c in rest] == ["valid", "valid"]


def test_floor_bundle_loads_and_matches_registry(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_floor_bundle(tmp_path, board_id="apollo510_evb").manifest_path)
    assert bundle.case_id == FLOOR_CASE_ID
    assert bundle.expected_output.byte_length == 0
    assert bundle.kernel_id == EMPTY_CALL_KERNEL_ID
    assert lookup_kernel_id(PROJECT_ROOT, family="Timing", operator="EmptyCall") == EMPTY_CALL_KERNEL_ID
