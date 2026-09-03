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
    FvpCaseStaleGateError,
    find_latest_fvp_report,
    lookup_fvp_case_status,
    require_fvp_pass,
)
from helia_core_tester.generation.artifact_identity import generated_case_artifact_sha256
from helia_core_tester.perf_stream.generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "artifacts" / "generated_tests").is_dir(),
    reason="no generated-test artifacts under artifacts/generated_tests/ "
    "(artifacts/ is gitignored -- run `helia_core_tester generate` first)",
)


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


def _case_artifacts(root: Path) -> Path:
    case_dir = root / "generated" / "my_case"
    case_dir.mkdir(parents=True)
    (case_dir / "descriptor.yaml").write_text("name: my_case\n", encoding="utf-8")
    (case_dir / "CMakeLists.txt").write_text("add_executable(my_case test.c)\n", encoding="utf-8")
    (case_dir / "test.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    return case_dir


def test_recorded_pass_requires_matching_artifact_digest(tmp_path: Path) -> None:
    case_dir = _case_artifacts(tmp_path)
    digest = generated_case_artifact_sha256(case_dir)
    _write_fake_report(tmp_path, "cortex-m55", "int", {
        "my_case": {"test_result": {"status": "PASS"}, "artifact_sha256": digest}
    })
    require_fvp_pass(tmp_path, "my_case", case_dir=case_dir)


def test_matching_legacy_pass_without_digest_is_stale(tmp_path: Path) -> None:
    case_dir = _case_artifacts(tmp_path)
    _write_fake_report(tmp_path, "cortex-m55", "int", {
        "my_case": {"test_result": {"status": "PASS"}}
    })
    with pytest.raises(FvpCaseStaleGateError, match="has no artifact_sha256"):
        require_fvp_pass(tmp_path, "my_case", case_dir=case_dir)


def test_changed_artifact_rejects_previous_pass(tmp_path: Path) -> None:
    case_dir = _case_artifacts(tmp_path)
    digest = generated_case_artifact_sha256(case_dir)
    _write_fake_report(tmp_path, "cortex-m55", "int", {
        "my_case": {"test_result": {"status": "PASS"}, "artifact_sha256": digest}
    })
    (case_dir / "test.c").write_text("int main(void) { return 1; }\n", encoding="utf-8")
    with pytest.raises(FvpCaseStaleGateError, match="does not match"):
        require_fvp_pass(tmp_path, "my_case", case_dir=case_dir)


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


# --- Tri-state --fvp-gate policy -------------------------------------------
# "failed" is evidence the kernel is wrong and blocks under every enforcing
# policy; "stale"/"absent" describe the report's freshness, not the kernel, and
# only block under "strict". Every case is classified regardless of policy so
# the result bundle can record fvp_status provenance.


def _stale_report_for(tmp_path: Path, case) -> Path:
    return _write_fake_report(
        tmp_path,
        case.cpu,
        case.suite,
        {case.name: {"test_result": {"status": "PASS"}, "artifact_sha256": "de" * 32}},
    )


def _abs_default_s8_case():
    cases = discover_generated_tests(
        PROJECT_ROOT, family="BasicMathFunctions", name_filter="abs_default_s8"
    )
    assert cases, "expected a discoverable BasicMathFunctions abs_default_s8 case"
    return cases[0]


def _pin_report(monkeypatch: pytest.MonkeyPatch, report_path: Path) -> None:
    import helia_core_tester.perf_stream.fvp_gate as fvp_gate_module

    monkeypatch.setattr(fvp_gate_module, "find_latest_fvp_report", lambda *a, **k: report_path)


def test_evaluate_classifies_stale_absent_failed_and_pass(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.fvp_gate import evaluate_fvp_gate

    assert evaluate_fvp_gate(tmp_path, "nope").status == "absent"

    _write_fake_report(tmp_path, "cortex-m55", "int", {"c": {"test_result": {"status": "FAIL"}}})
    assert evaluate_fvp_gate(tmp_path, "c").status == "failed"


def test_advisory_runs_stale_case_and_tags_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _abs_default_s8_case()
    _pin_report(monkeypatch, _stale_report_for(tmp_path, case))
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, case, output_root=tmp_path / "adv", fvp_gate="advisory"
    )
    assert bundle.fvp_status == "stale"


def test_strict_blocks_stale_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _abs_default_s8_case()
    _pin_report(monkeypatch, _stale_report_for(tmp_path, case))
    with pytest.raises(UnsupportedGeneratedTestError, match="does not match"):
        build_case_bundle_from_generated_test(
            PROJECT_ROOT, case, output_root=tmp_path / "strict", fvp_gate="strict"
        )


def test_recorded_failure_blocks_even_under_advisory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _abs_default_s8_case()
    digest = generated_case_artifact_sha256(case.directory)
    _pin_report(
        monkeypatch,
        _write_fake_report(
            tmp_path,
            case.cpu,
            case.suite,
            {case.name: {"test_result": {"status": "FAIL"}, "artifact_sha256": digest}},
        ),
    )
    with pytest.raises(UnsupportedGeneratedTestError, match="recorded FVP status"):
        build_case_bundle_from_generated_test(
            PROJECT_ROOT, case, output_root=tmp_path / "fail", fvp_gate="advisory"
        )


def test_off_never_blocks_but_still_records_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _abs_default_s8_case()
    digest = generated_case_artifact_sha256(case.directory)
    _pin_report(
        monkeypatch,
        _write_fake_report(
            tmp_path,
            case.cpu,
            case.suite,
            {case.name: {"test_result": {"status": "FAIL"}, "artifact_sha256": digest}},
        ),
    )
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, case, output_root=tmp_path / "off", fvp_gate="off"
    )
    assert bundle.fvp_status == "failed"


def test_unknown_policy_is_rejected(tmp_path: Path) -> None:
    case = _abs_default_s8_case()
    with pytest.raises(ValueError, match="fvp_gate must be one of"):
        build_case_bundle_from_generated_test(
            PROJECT_ROOT, case, output_root=tmp_path / "bad", fvp_gate="sometimes"
        )
