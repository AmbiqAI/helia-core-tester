from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from helia_core_tester.scripts.ab_bundles import delta_pct, load_bundle, main, select_counters

FIXTURES = Path(__file__).parent / "fixtures" / "ab_bundles"
A = str(FIXTURES / "a")
B = str(FIXTURES / "b")


def _run(capsys: pytest.CaptureFixture[str], *extra: str, a: str = A, b: str = B) -> tuple[int, str]:
    status = main([a, b, *extra])
    return status, capsys.readouterr().out


def _copy_b(tmp_path: Path, replace: tuple[str, str]) -> str:
    """Copy fixture B with one line edited."""
    root = tmp_path / "b"
    shutil.copytree(FIXTURES / "b", root)
    csv_path = root / "case_summary.csv"
    csv_path.write_text(csv_path.read_text().replace(*replace))
    return str(root)


def test_load_bundle_counters_and_session_id() -> None:
    bundle = load_bundle(FIXTURES / "a")
    assert bundle.session_id == "fixture-a"
    assert bundle.counters == [
        "median_cycles",
        "ARM_PMU_CPU_CYCLES",
        "ARM_PMU_INST_RETIRED",
        "ARM_PMU_MVE_INST_RETIRED",
        "ARM_PMU_STALL_FRONTEND",
    ]
    assert bundle.value("identical_case", "median_cycles") == 1000.0
    assert load_bundle(FIXTURES / "b").value("partial_case", "ARM_PMU_MVE_INST_RETIRED") is None
    assert load_bundle(FIXTURES / "b").session_id == "b"


def test_delta_pct_handles_zero_baseline() -> None:
    assert delta_pct(100.0, 110.0) == pytest.approx(10.0)
    assert delta_pct(0.0, 0.0) == 0.0
    assert math.isinf(delta_pct(0.0, 1.0))


def test_identical_counter_passes(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = _run(capsys, "--counter", "median_cycles", "--counter", "ARM_PMU_STALL_FRONTEND")
    assert status == 0
    assert "== PASS: 4 shared cases within limits" in out
    assert "1000.000       1000.000    +0.000%" in out


def test_drifted_retired_counter_fails(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = _run(capsys)
    assert status == 1
    assert "drifted_case ARM_PMU_INST_RETIRED 1500.000 -> 1515.000 +1.000% > 0%" in out
    assert "drifted_case ARM_PMU_MVE_INST_RETIRED 100.000 -> 90.000 -10.000% > 0%" in out
    # Cycles and stalls default to no limit.
    assert "drifted_case median_cycles" not in out
    assert "drifted_case ARM_PMU_STALL_FRONTEND" not in out
    assert "identical_case ARM_PMU" not in out


def test_limits_relax_and_gate_cycles(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = _run(capsys, "--max-delta-pct", "10", "--max-cycle-delta-pct", "5")
    assert status == 1
    assert "drifted_case ARM_PMU_INST_RETIRED" not in out
    assert "drifted_case median_cycles 2000.000 -> 2200.000 +10.000% > 5%" in out
    assert "drifted_case ARM_PMU_CPU_CYCLES 2010.000 -> 2210.000 +9.950% > 5%" in out


def test_missing_cases_listed_not_fatal(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = _run(capsys, "--counter", "median_cycles")
    assert status == 0
    assert "== cases only in A (1)\n  only_in_a" in out
    assert "== cases only in B (1)\n  only_in_b" in out


def test_overflow_case_flagged_not_gated(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = _run(capsys)
    assert status == 1
    assert "== overflow_case [invalid, overflow]" in out
    assert "== flagged, not gated (1)\n  overflow_case (invalid, overflow)" in out
    assert "overflow_case ARM_PMU_INST_RETIRED" not in out


def test_mismatch_case_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    b = _copy_b(tmp_path, ("identical_case,1,true,0", "identical_case,1,false,7"))
    _, out = _run(capsys, "--counter", "median_cycles", b=b)
    assert "== identical_case [mismatch]" in out
    assert "  identical_case (mismatch)" in out


def test_one_sided_empty_cell_violates(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = _run(capsys, "--counter", "ARM_PMU_MVE_INST_RETIRED")
    assert status == 1
    assert "50.000              -          -" in out
    assert "  partial_case ARM_PMU_MVE_INST_RETIRED missing on one side" in out
    # Ungated counters skip the missing check.
    status, out = _run(capsys, "--counter", "ARM_PMU_MVE_INST_RETIRED", "--max-delta-pct", "inf")
    assert "partial_case ARM_PMU_MVE_INST_RETIRED missing" in out
    status, out = _run(capsys, "--counter", "ARM_PMU_STALL_FRONTEND")
    assert status == 0


def test_counter_only_on_one_side_listed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    b = _copy_b(tmp_path, (",ARM_PMU_INST_RETIRED,", ",ARM_PMU_OTHER,"))
    _, out = _run(capsys, "--counter", "median_cycles", b=b)
    assert "== counters only in A (1)\n  ARM_PMU_INST_RETIRED" in out
    assert "== counters only in B (1)\n  ARM_PMU_OTHER" in out


def test_no_shared_cases_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    b = _copy_b(tmp_path, ("_case,", "_renamed,"))
    status, out = _run(capsys, b=b)
    assert status == 1
    assert "== FAIL: no shared cases" in out


def test_median_delta_summary(capsys: pytest.CaptureFixture[str]) -> None:
    _, out = _run(capsys, "--counter", "ARM_PMU_INST_RETIRED")
    # Overflow row excluded from median.
    assert "== median delta per counter\nARM_PMU_INST_RETIRED    +0.000%" in out


def test_select_counters_dedupes_and_validates() -> None:
    a, b = load_bundle(FIXTURES / "a"), load_bundle(FIXTURES / "b")
    assert select_counters(a, b, ["median_cycles", "median_cycles"]) == ["median_cycles"]
    with pytest.raises(SystemExit, match="ARM_PMU_NOPE"):
        select_counters(a, b, ["median_cycles", "ARM_PMU_NOPE"])
