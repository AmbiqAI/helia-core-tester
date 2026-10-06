from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware import candidate_eval
from helia_core_tester.hardware.candidate_check import CheckError
from helia_core_tester.tests import test_score_bundles as sb
from helia_core_tester.tests.test_harness_lock import _git, _repo

runner = CliRunner()


@pytest.fixture
def kernels(tmp_path: Path) -> Path:
    return _repo(tmp_path / "nn", {"Source/Conv/a.c": "int a;\n", "Include/arm_nnfunctions.h": "int f;\n", "nsx/CMakeLists.txt": "x\n"})


class FakeRun:
    """Writes a bundle per `hardware run` call."""

    def __init__(self, root: Path, base: dict | None = None, **cand) -> None:
        self.root, self.base, self.cand, self.calls = root, base or {}, cand, []

    def __call__(self, args: list[str]):
        self.calls.append(args)
        session = args[args.index("--session-id") + 1]
        options = self.cand if session.startswith("eval-") else self.base
        return 0, {"bundle": str(sb._bundle(self.root, session, **options)), "totals": {"failed": 0}}


def _baseline(tmp_path: Path, kernels: Path, repeats: int = 2, **base) -> tuple[Path, FakeRun]:
    spec = candidate_eval.RunSpec("apollo510_evb", kernels, pmu=("cpu:ARM_PMU_INST_RETIRED",))
    out, run = tmp_path / "base", FakeRun(tmp_path / "reports", base)
    out.mkdir()
    candidate_eval.write_baseline(spec, out, repeats, run=run)
    return out, run


def test_baseline_pins_options_and_reuses_build(tmp_path, kernels) -> None:
    out, run = _baseline(tmp_path, kernels, repeats=3)
    meta = candidate_eval.read_baseline(out)
    assert meta["base_commit"] == _git(kernels, "rev-parse", "HEAD").strip() and len(meta["sessions"]) == 3
    assert all((out / "bundles" / s / "case_summary.csv").is_file() for s in meta["sessions"])
    first, repeat = run.calls[0], run.calls[1]
    assert "--inline-asm" in first and "--placement" in first and "--skip-flash" not in first
    assert "--skip-generate" in repeat and "--skip-flash" in repeat


def test_baseline_refuses_edited_tree(tmp_path, kernels) -> None:
    (kernels / "Source/Conv/a.c").write_text("int b;\n")
    with pytest.raises(CheckError, match="has changes"):
        _baseline(tmp_path, kernels)


def test_faster_candidate_passes(tmp_path, kernels) -> None:
    out, _ = _baseline(tmp_path, kernels)
    run = FakeRun(tmp_path / "reports", cycles={"conv_a": 800.0})
    (kernels / "Source/Conv/a.c").write_text("int a; /* faster */\n")
    verdict = candidate_eval.evaluate(kernels, out, 0.005, run=run)
    assert verdict["verdict"] == "pass" and verdict["hidden"] is None
    args = run.calls[0]
    assert args[args.index("--golden-from") + 1].endswith("-1") and "--skip-generate" in args and "--skip-flash" not in args
    assert {c["case_id"] for c in verdict["cases"]} == set(sb.CASES)


def test_out_of_bounds_edit_is_rejected_before_run(tmp_path, kernels) -> None:
    out, _ = _baseline(tmp_path, kernels)
    (kernels / "nsx/CMakeLists.txt").write_text("target_compile_options(x -O0)\n")
    run = FakeRun(tmp_path / "reports")
    verdict = candidate_eval.evaluate(kernels, out, 0.005, run=run)
    assert verdict["verdict"] == "rejected" and verdict["stage"] == "check" and not run.calls


@pytest.mark.parametrize(("rc", "expected"), [(3, "refused"), (5, "error"), (1, "error")])
def test_run_without_bundle_maps_exit(tmp_path, kernels, rc, expected) -> None:
    out, _ = _baseline(tmp_path, kernels)
    verdict = candidate_eval.evaluate(kernels, out, 0.005, run=lambda args: (rc, None))
    assert verdict["verdict"] == expected and verdict["stage"] == "run"


def test_hidden_ids_never_print(tmp_path, kernels, monkeypatch) -> None:
    monkeypatch.setattr(sb, "FIELDS", sb.FIELDS + ["hidden"])
    out, _ = _baseline(tmp_path, kernels, rows={"dw_a": {"hidden": "true"}})
    run = FakeRun(tmp_path / "reports", rows={"dw_a": {"hidden": "true", "comparison_passed": "false"}})
    verdict = candidate_eval.evaluate(kernels, out, 0.005, run=run)
    assert verdict["verdict"] == "fail" and verdict["hidden"]["failures"] == {"comparison_failed": 1}
    assert "dw_a" not in json.dumps(verdict)


def test_eval_cli_prints_one_json_and_exits(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)
    monkeypatch.setattr(candidate_eval, "hardware_run", FakeRun(tmp_path / "reports"))
    result = runner.invoke(app, ["candidate", "eval", "--kernels", str(kernels), "--baseline", str(out)])
    verdict = json.loads(result.stdout)
    assert verdict["verdict"] == "no_gain" and result.exit_code == verdict["exit_code"] == 4
    result = runner.invoke(app, ["candidate", "eval", "--kernels", str(kernels), "--baseline", str(out), "--board", "apollo3p_evb"])
    assert result.exit_code == 2
