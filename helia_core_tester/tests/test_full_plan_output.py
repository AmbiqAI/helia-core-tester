"""`full --plan` prints the plan to stdout at every verbosity and never runs a step."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester import cli as cli_module
from helia_core_tester.core.config import Config
from helia_core_tester.core.pipeline import FullTestPipeline
from helia_core_tester.core.steps.base import StepPlan

PLANS = [
    StepPlan(
        name="generate",
        will_run=True,
        reason="ready",
        commands=[["pytest", "test_ops.py"]],
        outputs={"generated_tests_root": "/gen", "empty": ""},
        details={"cwd": "/work", "unset": None},
    ),
    StepPlan(name="host-check", will_run=False, reason="skipped by config"),
    StepPlan(
        name="run",
        will_run=True,
        reason="ready",
        commands=[["fvp", "--report-formats", "json", "html"], ["fvp", "--cpu", "cortex-m4"]],
        outputs={"empty": ""},
        details={"unset": None},
    ),
]

EXPECTED = [
    "Plan:",
    "1. generate: will run (ready)",
    "   cmd: pytest test_ops.py",
    "   outputs: generated_tests_root=/gen",
    "   details: cwd=/work",
    "2. host-check: skipped (skipped by config)",
    "3. run: will run (ready)",
    "   cmd: fvp --report-formats json html",
    "   cmd: fvp --cpu cortex-m4",
]


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "tester"
    (root / "helia_core_tester" / "generation").mkdir(parents=True)
    return root


@pytest.fixture
def fixed_plan(monkeypatch) -> list[str]:
    calls: list[str] = []

    def _build_plan(self):
        calls.append("build_plan")
        return list(PLANS)

    def _no_step(*args, **kwargs):
        raise AssertionError("no step may execute under --plan")

    monkeypatch.setattr(FullTestPipeline, "build_plan", _build_plan)
    monkeypatch.setattr("helia_core_tester.core.pipeline._run_step", _no_step)
    monkeypatch.setattr(FullTestPipeline, "_ensure_runtime_env", _no_step)
    return calls


def test_plan_lines_render_every_field_and_skip_empty_values(tmp_path, fixed_plan) -> None:
    assert FullTestPipeline(Config(project_root=_repo(tmp_path))).plan_lines() == EXPECTED


def test_empty_plan_is_just_the_header(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(FullTestPipeline, "build_plan", lambda self: [])
    assert FullTestPipeline(Config(project_root=_repo(tmp_path))).plan_lines() == ["Plan:"]


@pytest.mark.parametrize("verbosity", [None, "0", "1", "2", "3"])
def test_full_plan_prints_to_stdout_at_every_verbosity(tmp_path, fixed_plan, verbosity) -> None:
    argv = ["full", "--plan", "--repo-root", str(_repo(tmp_path))]
    if verbosity is not None:
        argv += ["-v", verbosity]
    result = CliRunner().invoke(cli_module.app, argv)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == EXPECTED
    assert fixed_plan == ["build_plan"]


def test_run_with_plan_config_prints_once_and_executes_nothing(tmp_path, fixed_plan, capsys) -> None:
    cfg = Config(project_root=_repo(tmp_path), plan=True, _explicit_overrides={"project_root", "plan"})
    assert FullTestPipeline(cfg).run() is True
    assert capsys.readouterr().out.splitlines() == EXPECTED
    assert fixed_plan == ["build_plan"]


def test_full_plan_with_rejected_option_prints_no_plan(tmp_path, fixed_plan) -> None:
    result = CliRunner().invoke(
        cli_module.app, ["full", "--plan", "--report-formats", "json,,md", "--repo-root", str(_repo(tmp_path))]
    )
    assert result.exit_code == 2
    assert "Plan:" not in result.stdout
    assert fixed_plan == []
