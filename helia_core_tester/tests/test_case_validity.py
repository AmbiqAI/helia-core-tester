from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helia_core_tester.hardware.case_bundle import (
    DEGENERATE_REASON_KEY, EMPTY_CALL_KERNEL_ID, FLOOR_CASE_ID, build_abs_s8_case_bundle, build_floor_bundle,
    load_case_bundle,
)
from helia_core_tester.hardware.case_validity import apply_floor, classify_case, golden_degenerate, timing_status
from helia_core_tester.hardware.kernel_registry import lookup_kernel_id
from helia_core_tester.hardware.comparison import ComparisonResult
from helia_core_tester.hardware.measurement import SampleStatistics
from helia_core_tester.hardware.session import CaseRunResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS = SampleStatistics(5, 900.0, 1000.0, 1000.0, 1000.0, 1.0, True, False, ())


def _case(bundle, *, median=1000.0, overflow=False, samples=5, status=None, passed=True):
    if status is not None:
        bundle = SimpleNamespace(case_id="err", expected_status_code=status)
    stats = replace(STATS, median_cycles=median, overflow_detected=overflow, sample_count=samples)
    return CaseRunResult(bundle, ComparisonResult(passed, 0 if passed else 3, 0.0, "exact_int"), b"", (), (), stats)


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


def test_expected_success_is_not_error_path(abs_bundle) -> None:
    manifest = {**abs_bundle.manifest, "correctness_comparison": {"mode": "exact_status", "expected_status": 0}}
    assert timing_status(_case(replace(abs_bundle, manifest=manifest)), 30.0) == "valid"


def test_unclassified_stats_agree_with_validity() -> None:
    from helia_core_tester.hardware.measurement import compute_sample_statistics

    sample = SimpleNamespace(cycles_per_invocation=10.0, overflow=True, unsupported_counters=())
    stats = compute_sample_statistics([sample])
    assert (stats.timing_status, stats.valid_for_regression) == ("overflow", False)
    assert compute_sample_statistics([]).timing_status == "zero_cycles"


def test_apply_floor_records_and_classifies(abs_bundle, tmp_path: Path) -> None:
    floor = build_floor_bundle(tmp_path, board_id="apollo510_evb")
    cases = [_case(floor, median=40.0), _case(abs_bundle, median=100.0), _case(abs_bundle, median=200.0)]
    record, rest = apply_floor(cases)
    assert record["median_cycles"] == 40.0 and record["below_floor_cycles"] == 120.0
    assert [c.statistics.timing_status for c in rest] == ["below_floor", "valid"]
    assert [c.statistics.valid_for_regression for c in rest] == [False, True]
    record, rest = apply_floor(cases[1:])
    assert record is None and [c.statistics.timing_status for c in rest] == ["valid", "valid"]


def test_wrong_output_never_gates(abs_bundle) -> None:
    stats = classify_case(_case(abs_bundle, passed=False), 30.0).statistics
    # Timing stays clean; validity drops.
    assert (stats.timing_status, stats.valid_for_regression) == ("valid", False)


def test_floor_bundle_loads_and_matches_registry(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_floor_bundle(tmp_path, board_id="apollo510_evb").manifest_path)
    assert bundle.case_id == FLOOR_CASE_ID
    assert bundle.expected_output.byte_length == 0
    assert bundle.kernel_id == EMPTY_CALL_KERNEL_ID
    assert lookup_kernel_id(PROJECT_ROOT, family="Timing", operator="EmptyCall") == EMPTY_CALL_KERNEL_ID


def test_degenerate_reason_keeps_case_valid(tmp_path: Path) -> None:
    flat = build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_flat", input_shape=(1, 4, 4, 2))
    golden_path = flat.root_dir / "blobs" / "expected_output.bin"
    golden_path.write_bytes(bytes(golden_path.stat().st_size))
    flat = load_case_bundle(flat.manifest_path)
    assert timing_status(_case(flat), None) == "degenerate_output"
    # A boring golden still gates perf.
    assert classify_case(_case(flat), None).statistics.valid_for_regression is True
    flat.manifest[DEGENERATE_REASON_KEY] = "constant by design"
    assert timing_status(_case(flat), None) == "valid"


def test_floor_bundle_records_board_cpu(tmp_path: Path) -> None:
    bundle = build_floor_bundle(tmp_path, board_id="apollo3p_evb", cpu="cortex-m4")
    assert bundle.manifest["target_cpu"] == "cortex-m4"


def test_tanh_selector_rejects_s8() -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.tanh import OpTanh

    assert OpTanh({"activation_dtype": "S16"}).kernel_fn() == "arm_tanh_s16"
    with pytest.raises(NotImplementedError, match="S8"):
        OpTanh({"activation_dtype": "S8"}).kernel_fn()
