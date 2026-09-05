from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

from helia_core_tester.core.config import DEFAULT_TIMEOUT_SECONDS, Config
from helia_core_tester.reporting.generator import TIMEOUT_FAILURE_PREFIX, ReportGenerator
from helia_core_tester.reporting import models as reporting_models
from helia_core_tester.reporting.models import DescriptorResult

status_enum = reporting_models.TestStatus


def _descriptor_result(name: str, status, reason: str | None) -> DescriptorResult:
    return DescriptorResult(
        descriptor_name=name,
        descriptor_path=Path(f"{name}.yaml"),
        descriptor_content={"name": name, "operator": "Mean"},
        status=status,
        test_result=reporting_models.TestResult(
            test_name=name,
            status=status,
            duration=180.0,
            cpu="cortex-m55",
            elf_path=f"{name}.elf",
            failure_reason=reason,
            error_type="timeout" if status == status_enum.TIMEOUT else None,
            exit_code=124 if status == status_enum.TIMEOUT else 0,
        ),
        failure_reason=reason,
    )


def _report() -> reporting_models.TestReport:
    start = datetime(2026, 1, 1, 0, 0, 0)
    return reporting_models.TestReport(
        run_id="timeout-render",
        start_time=start,
        end_time=start + timedelta(seconds=200),
        cpu="cortex-m55",
        descriptor_results={
            "mean_f16_nonfinite": _descriptor_result(
                "mean_f16_nonfinite", status_enum.TIMEOUT, "Test execution timed out"
            ),
            "add_s8_basic": _descriptor_result("add_s8_basic", status_enum.PASS, None),
        },
    )


def test_timeout_renders_as_a_junit_failure_with_a_distinct_message(tmp_path: Path) -> None:
    generator = ReportGenerator(output_dir=tmp_path)
    junit_path = generator.generate_reports(_report(), formats=["junit"])["junit"]

    testsuite = ET.parse(junit_path).getroot()
    assert testsuite.get("failures") == "1"
    assert testsuite.get("errors") == "0"
    assert testsuite.get("skipped") == "0"

    cases = {case.get("name"): case for case in testsuite.findall("testcase")}

    timed_out = cases["mean_f16_nonfinite"]
    failure = timed_out.find("failure")
    assert failure is not None, "a timed-out case must be a JUnit failure"
    assert timed_out.find("skipped") is None
    assert timed_out.find("error") is None
    assert failure.get("type") == "timeout"
    assert TIMEOUT_FAILURE_PREFIX in failure.get("message")

    passed = cases["add_s8_basic"]
    assert list(passed) == []


def test_timeout_is_not_a_pass_or_a_skip_in_the_json_report(tmp_path: Path) -> None:
    report = _report()
    generator = ReportGenerator(output_dir=tmp_path)
    json_path = generator.generate_reports(report, formats=["json"])["json"]

    payload = json.loads(json_path.read_text())
    assert payload["timed_out"] == 1
    assert payload["passed"] == 1
    assert payload["skipped"] == 0
    assert payload["descriptor_results"]["mean_f16_nonfinite"]["test_result"]["status"] == "TIMEOUT"

    failed_names = {result.descriptor_name for result in report.get_failed_tests()}
    assert "mean_f16_nonfinite" in failed_names
    assert "mean_f16_nonfinite" not in {r.descriptor_name for r in report.get_passed_tests()}
    assert "mean_f16_nonfinite" not in {r.descriptor_name for r in report.get_skipped_tests()}


def _config_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_timeout_defaults_to_a_nonzero_per_case_budget(tmp_path: Path) -> None:
    root = _config_root(tmp_path)
    assert Config(project_root=root).timeout == DEFAULT_TIMEOUT_SECONDS
    assert DEFAULT_TIMEOUT_SECONDS > 0


def test_timeout_zero_stays_an_explicit_opt_out(tmp_path: Path) -> None:
    root = _config_root(tmp_path)
    cfg = Config(project_root=root, timeout=0.0, _explicit_overrides={"project_root", "timeout"})
    assert cfg.timeout == 0.0


def test_fvp_cli_timeout_run_default_matches_the_config_default(tmp_path: Path) -> None:
    from helia_core_tester.fvp.cli import build_arg_parser

    args = build_arg_parser(tmp_path, tmp_path).parse_args([])
    assert args.timeout_run == DEFAULT_TIMEOUT_SECONDS


def _timeout_run_value(cfg) -> str:
    from helia_core_tester.core.steps.run import RunStep

    commands = RunStep(cfg)._run_commands()
    assert commands, "expected at least one FVP run command"
    cmd = commands[0]
    return cmd[cmd.index("--timeout-run") + 1]


def test_the_default_timeout_reaches_the_child_process(tmp_path: Path) -> None:
    cfg = Config(project_root=_config_root(tmp_path))
    assert float(_timeout_run_value(cfg)) == DEFAULT_TIMEOUT_SECONDS


def test_the_timeout_opt_out_reaches_the_child_process(tmp_path: Path) -> None:
    # The child's own default is non-zero, so an unforwarded 0 would silently
    # become the default instead of the opt-out the caller asked for.
    cfg = Config(
        project_root=_config_root(tmp_path),
        timeout=0.0,
        _explicit_overrides={"project_root", "timeout"},
    )
    assert float(_timeout_run_value(cfg)) == 0.0
