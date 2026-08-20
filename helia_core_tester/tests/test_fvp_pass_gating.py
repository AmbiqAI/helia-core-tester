"""Phase 2 of the generation/bridge unification plan: FVP-pass gating.

Verifies helia_core_tester.perf_stream.fvp_gate correctly consults the most
recently recorded FVP test_report_<cpu>_*.json before the hardware bridge
converts a generated test case into a CaseBundle -- refusing to bridge any
case FVP itself did not record as PASS, while remaining a no-op when no FVP
report is available at all (so host-only environments without FVP access,
like this sandbox, don't spuriously fail).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helia_core_tester.perf_stream.fvp_gate import (
    FvpCaseFailedGateError,
    find_latest_fvp_report,
    lookup_fvp_case_status,
    require_fvp_pass,
)
from helia_core_tester.perf_stream.generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_report(root: Path, cpu: str, suite: str, descriptor_results: dict) -> Path:
    report_dir = root / "artifacts" / "reports" / "tests" / suite / cpu
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"test_report_{cpu}_20260101_000000.json"
    report_path.write_text(
        json.dumps({"descriptor_results": descriptor_results}, indent=2)
    )
    return report_path


def test_no_report_available_is_a_no_op(tmp_path: Path) -> None:
    # tmp_path has no artifacts/reports/tests/... at all.
    assert find_latest_fvp_report(tmp_path, cpu="cortex-m55", suite="int") is None
    assert lookup_fvp_case_status(tmp_path, "some_case", cpu="cortex-m55", suite="int") is None
    # Should not raise -- allow_missing_report defaults to True.
    require_fvp_pass(tmp_path, "some_case", cpu="cortex-m55", suite="int")


def test_missing_report_raises_when_required_explicitly(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.fvp_gate import FvpReportUnavailableError

    with pytest.raises(FvpReportUnavailableError):
        require_fvp_pass(
            tmp_path, "some_case", cpu="cortex-m55", suite="int", allow_missing_report=False
        )


def test_case_not_in_report_is_a_no_op(tmp_path: Path) -> None:
    _write_fake_report(tmp_path, "cortex-m55", "int", {"other_case": {"test_result": {"status": "PASS"}}})
    assert lookup_fvp_case_status(tmp_path, "missing_case", cpu="cortex-m55", suite="int") is None
    require_fvp_pass(tmp_path, "missing_case", cpu="cortex-m55", suite="int")


def test_recorded_pass_does_not_raise(tmp_path: Path) -> None:
    _write_fake_report(
        tmp_path, "cortex-m55", "int", {"my_case": {"test_result": {"status": "PASS"}}}
    )
    status = lookup_fvp_case_status(tmp_path, "my_case", cpu="cortex-m55", suite="int")
    assert status is not None
    assert status.passed
    require_fvp_pass(tmp_path, "my_case", cpu="cortex-m55", suite="int")


def test_recorded_fail_raises_gate_error(tmp_path: Path) -> None:
    _write_fake_report(
        tmp_path, "cortex-m55", "int", {"my_case": {"test_result": {"status": "FAIL"}}}
    )
    status = lookup_fvp_case_status(tmp_path, "my_case", cpu="cortex-m55", suite="int")
    assert status is not None
    assert not status.passed
    with pytest.raises(FvpCaseFailedGateError):
        require_fvp_pass(tmp_path, "my_case", cpu="cortex-m55", suite="int")


def test_latest_report_picked_when_multiple_exist(tmp_path: Path) -> None:
    report_dir = tmp_path / "artifacts" / "reports" / "tests" / "int" / "cortex-m55"
    report_dir.mkdir(parents=True)
    (report_dir / "test_report_cortex-m55_20260101_000000.json").write_text(
        json.dumps({"descriptor_results": {"my_case": {"test_result": {"status": "FAIL"}}}})
    )
    (report_dir / "test_report_cortex-m55_20260102_000000.json").write_text(
        json.dumps({"descriptor_results": {"my_case": {"test_result": {"status": "PASS"}}}})
    )
    latest = find_latest_fvp_report(tmp_path, cpu="cortex-m55", suite="int")
    assert latest is not None
    assert "20260102" in latest.name
    status = lookup_fvp_case_status(tmp_path, "my_case", cpu="cortex-m55", suite="int")
    assert status is not None
    assert status.passed


def test_bridge_skips_case_with_recorded_fvp_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a real discovered generated test case is refused by the
    bridge (as UnsupportedGeneratedTestError, the same type existing callers
    already catch) when a fake FVP report records it as FAIL -- confirming the
    gate is actually wired into build_case_bundle_from_generated_test.
    """
    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter="add_default_s8")
    assert cases, "expected a discoverable BasicMathFunctions add_default_s8 case"
    case = cases[0]

    fake_report_path = _write_fake_report(
        tmp_path, case.cpu, "int", {case.name: {"test_result": {"status": "FAIL"}}}
    )
    import helia_core_tester.perf_stream.fvp_gate as fvp_gate_module

    monkeypatch.setattr(
        fvp_gate_module, "find_latest_fvp_report", lambda *a, **k: fake_report_path
    )

    with pytest.raises(UnsupportedGeneratedTestError, match="recorded FVP status"):
        build_case_bundle_from_generated_test(
            PROJECT_ROOT, case, output_root=tmp_path / "out"
        )


def test_bridge_ignores_gate_when_require_fvp_pass_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate can be explicitly disabled (e.g. for host-only bridge unit
    tests that don't have a real FVP report), in which case bridging proceeds
    even if a FAIL would otherwise have been recorded.
    """
    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter="add_default_s8")
    assert cases
    case = cases[0]

    fake_report_path = _write_fake_report(
        tmp_path, case.cpu, "int", {case.name: {"test_result": {"status": "FAIL"}}}
    )
    import helia_core_tester.perf_stream.fvp_gate as fvp_gate_module

    monkeypatch.setattr(
        fvp_gate_module, "find_latest_fvp_report", lambda *a, **k: fake_report_path
    )

    # Should not raise the FVP-gate error (may still fail/succeed on its own
    # merits, but should get past the gate at least).
    try:
        build_case_bundle_from_generated_test(
            PROJECT_ROOT, case, output_root=tmp_path / "out2", require_fvp_pass=False
        )
    except UnsupportedGeneratedTestError as exc:
        assert "recorded FVP status" not in str(exc)


def test_real_stale_fvp_report_recognizes_grouped_conv_case_pass() -> None:
    """Sanity check against the real (if stale) FVP report already present in
    this repo's artifacts/ directory: convolve_grouped_conv_case_01_s8 passed
    under FVP, confirming lookup_fvp_case_status() correctly parses a real
    report file, not just the synthetic fixtures above.
    """
    status = lookup_fvp_case_status(
        PROJECT_ROOT, "convolve_grouped_conv_case_01_s8", cpu="cortex-m55", suite="int"
    )
    if status is None:
        pytest.skip("no FVP report present for cortex-m55/int in this environment")
    assert status.passed
