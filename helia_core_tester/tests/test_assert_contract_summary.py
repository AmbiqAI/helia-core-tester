"""scripts/assert_contract_summary.py: the self-validate contract legs cannot pass by
omission."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "assert_contract_summary.py"

spec = importlib.util.spec_from_file_location("assert_contract_summary", SCRIPT)
acs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acs)

AUDIT = "test_real_tree_symbols_are_declared_or_known_drift"
PARITY = "test_parity_assert_against_the_real_headers"


def _report(covered: int = 3, cases: int = 12, schema: str = acs.REPORT_SCHEMA) -> str:
    return json.dumps({
        "schema": schema,
        "cases_scanned": cases,
        "covered": {f"arm_fx_{i}": ["case"] for i in range(covered)},
        "uncovered": {},
        "unknown_symbols": {},
    })


def _junit(**outcomes: str) -> str:
    cases = []
    for name, outcome in outcomes.items():
        inner = {"passed": "", "skipped": '<skipped message="no checkout"/>',
                 "failed": '<failure message="boom"/>', "error": '<error message="boom"/>'}[outcome]
        cases.append(f'<testcase classname="t" name="{name}" time="0.1">{inner}</testcase>')
    return f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="{len(cases)}">{"".join(cases)}</testsuite></testsuites>'


@pytest.fixture
def files(tmp_path: Path):
    def write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text)
        return path
    return write


def _run(*argv: str) -> tuple[int, str, str]:
    import io
    from contextlib import redirect_stderr, redirect_stdout
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = acs.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_present_leg_passes_with_inventory_tests_and_doctor(files) -> None:
    report = files("inventory.json", _report())
    junit = files("junit.xml", _junit(**{AUDIT: "passed", PARITY: "passed"}))
    doctor = files("doctor.log", "✓ .../kernel_contracts.json: 475 public functions (303 kernels)")
    code, out, err = _run("--expect", "present", "--inventory-exit", "0", "--report", str(report),
                          "--junit", str(junit), "--test", AUDIT, "--test", PARITY, "--doctor-log", str(doctor))
    assert code == 0, err
    assert "12 cases, 3 covered functions" in out and f"{AUDIT}: passed" in out and "doctor: contract present" in out


def test_absent_leg_passes_when_everything_refused(files) -> None:
    junit = files("junit.xml", _junit(**{AUDIT: "skipped"}))
    doctor = files("doctor.log", "⚠ kernel contract: absent (/x; contract-driven commands are unavailable)")
    generate = files("generate.log", "GenerationError: arm_convolve_s8: ... needs an ns-cmsis-nn that carries the export (AmbiqAI/ns-cmsis-nn#549 or later)")
    code, out, err = _run("--expect", "absent", "--inventory-exit", "2", "--junit", str(junit), "--test", AUDIT,
                          "--doctor-log", str(doctor), "--generate-log", str(generate))
    assert code == 0, err
    assert f"{AUDIT}: skipped" in out and "failed closed" in out


@pytest.mark.parametrize(
    ("argv_extra", "report_text", "junit_text", "message"),
    [
        (["--inventory-exit", "2"], _report(), _junit(**{AUDIT: "passed"}), "inventory exited 2, expected 0"),
        (["--inventory-exit", "0"], _report(covered=0), _junit(**{AUDIT: "passed"}), "lists no covered function"),
        (["--inventory-exit", "0"], _report(cases=0), _junit(**{AUDIT: "passed"}), "scanned no generated cases"),
        (["--inventory-exit", "0"], _report(schema="hct.other/9"), _junit(**{AUDIT: "passed"}), "does not carry schema"),
        (["--inventory-exit", "0"], "{not json", _junit(**{AUDIT: "passed"}), "is not valid JSON"),
        (["--inventory-exit", "0"], _report(), _junit(**{AUDIT: "skipped"}), f"{AUDIT} skipped; a present leg must run it"),
        (["--inventory-exit", "0"], _report(), _junit(**{AUDIT: "failed"}), f"{AUDIT} failed"),
        (["--inventory-exit", "0"], _report(), _junit(**{AUDIT: "error"}), f"{AUDIT} failed"),
        (["--inventory-exit", "0"], _report(), _junit(other="passed"), "has no testcase for"),
        (["--inventory-exit", "0"], _report(), "<testsuites><testcase", "is not valid XML"),
    ],
)
def test_present_leg_fails_closed(files, argv_extra, report_text, junit_text, message) -> None:
    report = files("inventory.json", report_text)
    junit = files("junit.xml", junit_text)
    code, _, err = _run("--expect", "present", *argv_extra, "--report", str(report), "--junit", str(junit), "--test", AUDIT)
    assert code == 1 and message in err, err


@pytest.mark.parametrize(
    ("inventory_exit", "junit_text", "doctor_text", "generate_text", "message"),
    [
        (0, _junit(**{AUDIT: "skipped"}), "kernel contract: absent", acs.GENERATE_ABSENT_MARK, "inventory exited 0, expected 2"),
        (2, _junit(**{AUDIT: "passed"}), "kernel contract: absent", acs.GENERATE_ABSENT_MARK, f"{AUDIT} passed; an absent leg must skip it"),
        (2, _junit(**{AUDIT: "failed"}), "kernel contract: absent", acs.GENERATE_ABSENT_MARK, f"{AUDIT} failed; an absent leg must skip it"),
        (2, _junit(**{AUDIT: "skipped"}), "475 public functions", acs.GENERATE_ABSENT_MARK, "doctor log"),
        (2, _junit(**{AUDIT: "skipped"}), "kernel contract: absent", "generated 185 cases", "failing closed"),
    ],
)
def test_absent_leg_fails_closed(files, inventory_exit, junit_text, doctor_text, generate_text, message) -> None:
    junit = files("junit.xml", junit_text)
    doctor = files("doctor.log", doctor_text)
    generate = files("generate.log", generate_text)
    code, _, err = _run("--expect", "absent", "--inventory-exit", str(inventory_exit), "--junit", str(junit),
                        "--test", AUDIT, "--doctor-log", str(doctor), "--generate-log", str(generate))
    assert code == 1 and message in err, err


def test_missing_inputs_are_errors(tmp_path: Path, files) -> None:
    code, _, err = _run("--expect", "present", "--inventory-exit", "0")
    assert code == 1 and "needs --report" in err
    code, _, err = _run("--expect", "present", "--inventory-exit", "0", "--report", str(tmp_path / "nope.json"))
    assert code == 1 and "cannot be read" in err
    report = files("inventory.json", _report())
    code, _, err = _run("--expect", "present", "--inventory-exit", "0", "--report", str(report), "--test", AUDIT)
    assert code == 1 and "needs --junit" in err
    code, _, err = _run("--expect", "absent", "--inventory-exit", "2", "--test", AUDIT)
    assert code == 1 and "needs --junit" in err
    code, _, err = _run("--expect", "absent", "--inventory-exit", "2", "--doctor-log", str(tmp_path / "nope.log"))
    assert code == 1 and "cannot be read" in err


def test_parametrized_testcase_names_match_and_any_failure_wins(files) -> None:
    junit = files("junit.xml", f'<testsuites><testsuite><testcase name="{AUDIT}[m55]"/>'
                               f'<testcase name="{AUDIT}[m4]"><failure/></testcase></testsuite></testsuites>')
    assert acs.junit_outcomes(junit, [AUDIT]) == {AUDIT: "failed"}


def test_script_runs_as_a_process(files) -> None:
    import subprocess
    import sys
    report = files("inventory.json", _report())
    result = subprocess.run([sys.executable, str(SCRIPT), "--expect", "present", "--inventory-exit", "0",
                             "--report", str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(SCRIPT), "--expect", "absent", "--inventory-exit", "0"],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "expected 2" in result.stderr
