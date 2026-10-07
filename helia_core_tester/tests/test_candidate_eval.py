from __future__ import annotations

import json
import os
import shutil
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
    return _repo(tmp_path / "nn", {
        "Source/Conv/a.c": "int a;\n", "Include/arm_nnfunctions.h": "int f;\n", "nsx/CMakeLists.txt": "x\n",
        "nsx/nsx-module.yaml": "name: nsx-cmsis-nn\n",
    })


class FakeRun:
    """Writes a bundle per `hardware run` call."""

    def __init__(self, root: Path, base: dict | None = None, **cand) -> None:
        self.root, self.base, self.cand, self.calls = root, base or {}, cand, []

    def __call__(self, args: list[str], log: Path):
        self.calls.append(args)
        session = args[args.index("--session-id") + 1]
        options = self.cand if session.startswith("eval-") else self.base
        path = sb._bundle(self.root, session, **options)
        _built(path, Path(args[args.index("--cmsis-nn-root") + 1]), args)
        return 0, {"bundle": str(path), "totals": {"failed": 0}}


def _built(bundle: Path, root: Path, args: list[str]) -> None:
    """Record provenance like a real run."""
    manifest = json.loads((bundle / "session_manifest.json").read_text())
    head = _git(root, "rev-parse", "HEAD").strip()
    dirty = bool(_git(root, "status", "--porcelain").strip())
    manifest["build"]["kernels"] = {
        "root": str(root), "root_head": head, "root_dirty": dirty, "tree_hash": candidate_eval.snapshot_hash(root),
    }
    golden = Path(args[args.index("--golden-from") + 1]) if "--golden-from" in args else None
    manifest["compare"] = {"strict": golden is not None, "golden_session_id": golden.name if golden else None}
    (bundle / "session_manifest.json").write_text(json.dumps(manifest))


def _baseline(tmp_path: Path, kernels: Path, repeats: int = 2, **base) -> tuple[Path, FakeRun]:
    spec = candidate_eval.RunSpec("apollo510_evb", kernels, pmu=("cpu:ARM_PMU_INST_RETIRED",))
    out, run = tmp_path / "base", FakeRun(tmp_path / "reports", base)
    out.mkdir()
    candidate_eval.write_baseline(spec, out, repeats, run=run)
    return out, run


def _eval(kernels: Path, out: Path, run) -> dict:
    return candidate_eval.evaluate(kernels, out, candidate_eval.read_baseline(out), 0.005, run=run)


def test_baseline_pins_options_and_reuses_build(tmp_path, kernels) -> None:
    out, run = _baseline(tmp_path, kernels, repeats=3)
    meta = candidate_eval.read_baseline(out)
    assert meta["base_commit"] == _git(kernels, "rev-parse", "HEAD").strip() and len(meta["sessions"]) == 3
    assert candidate_eval.RunSpec.from_json(meta["run"], kernels).pmu == ("cpu:ARM_PMU_INST_RETIRED",)
    assert all((out / "bundles" / s / "case_summary.csv").is_file() for s in meta["sessions"])
    first, repeat = run.calls[0], run.calls[1]
    assert "--inline-asm" in first and "--placement" in first and "--skip-flash" not in first
    assert "--skip-generate" in repeat and "--skip-flash" in repeat


def test_baseline_refuses_edited_tree(tmp_path, kernels) -> None:
    (kernels / "Source/Conv/a.c").write_text("int b;\n")
    with pytest.raises(CheckError, match="has changes"):
        _baseline(tmp_path, kernels)


def test_faster_candidate_passes_from_a_snapshot(tmp_path, kernels) -> None:
    out, _ = _baseline(tmp_path, kernels)
    run = FakeRun(tmp_path / "reports", cycles={"conv_a": 800.0})
    (kernels / "Source/Conv/a.c").write_text("int a; /* faster */\n")
    verdict = _eval(kernels, out, run)
    assert verdict["verdict"] == "pass" and verdict["hidden"] is None
    args = run.calls[0]
    assert args[args.index("--golden-from") + 1].endswith("-1") and "--skip-generate" in args and "--skip-flash" not in args
    # The build reads the copy, not the agent's tree.
    root = Path(args[args.index("--cmsis-nn-root") + 1])
    assert root == out / "snapshot" and (root / "Source/Conv/a.c").read_text() == "int a; /* faster */\n"
    assert {c["case_id"] for c in verdict["cases"]} == set(sb.CASES)


def test_candidate_git_config_never_runs(tmp_path, kernels) -> None:
    out, _ = _baseline(tmp_path, kernels)
    marker = tmp_path / "pwned"
    _git(kernels, "config", "core.fsmonitor", f"touch {marker}; false #")
    _git(kernels, "config", "core.hooksPath", str(tmp_path))
    assert _eval(kernels, out, FakeRun(tmp_path / "reports"))["verdict"] == "no_gain"
    assert not marker.exists()


@pytest.mark.parametrize("edit", ["nsx", "symlink"])
def test_out_of_bounds_edit_is_rejected_before_run(tmp_path, kernels, edit) -> None:
    out, _ = _baseline(tmp_path, kernels)
    if edit == "nsx":
        (kernels / "nsx/CMakeLists.txt").write_text("target_compile_options(x -O0)\n")
    else:
        (kernels / "Source/Conv/b.c").symlink_to(tmp_path / "elsewhere.c")
    run = FakeRun(tmp_path / "reports")
    verdict = _eval(kernels, out, run)
    assert verdict["verdict"] == "rejected" and verdict["stage"] == "check" and not run.calls


def test_missing_base_copy_refuses(tmp_path, kernels) -> None:
    out, _ = _baseline(tmp_path, kernels)
    shutil.rmtree(out / "kernels.git")
    verdict = _eval(kernels, out, FakeRun(tmp_path / "reports"))
    assert verdict["verdict"] == "refused" and verdict["stage"] == "check"


@pytest.mark.parametrize(("rc", "expected"), [(3, "refused"), (5, "error"), (1, "error")])
def test_run_without_bundle_maps_exit(tmp_path, kernels, rc, expected) -> None:
    out, _ = _baseline(tmp_path, kernels)
    verdict = _eval(kernels, out, lambda args, log: (rc, None))
    assert verdict["verdict"] == expected and verdict["stage"] == "run"


def test_lost_cases_refuse_not_fail(tmp_path, kernels) -> None:
    out, _ = _baseline(tmp_path, kernels)
    verdict = _eval(kernels, out, FakeRun(tmp_path / "reports", drop=("fc_a",)))
    assert verdict["verdict"] == "refused" and {f["kind"] for f in verdict["failures"]} == {"missing_case"}


def test_hidden_ids_never_print(tmp_path, kernels, monkeypatch) -> None:
    monkeypatch.setattr(sb, "FIELDS", sb.FIELDS + ["hidden"])
    out, _ = _baseline(tmp_path, kernels, rows={"dw_a": {"hidden": "true"}})
    run = FakeRun(tmp_path / "reports", rows={"dw_a": {"hidden": "true", "comparison_passed": "false"}})
    verdict = _eval(kernels, out, run)
    assert verdict["verdict"] == "fail" and verdict["hidden"]["failures"] == {"comparison_failed": 1}
    assert verdict["hidden"]["cases"] == 1 and "dw_a" not in json.dumps(verdict)


def _cli_eval(kernels: Path, out: Path, *extra: str):
    result = runner.invoke(app, ["candidate", "eval", "--kernels", str(kernels), "--baseline", str(out), *extra])
    return result, json.loads(result.stdout) if result.stdout.strip() else None


def test_eval_cli_prints_one_json_and_exits(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)
    monkeypatch.setattr(candidate_eval, "hardware_run", FakeRun(tmp_path / "reports"))
    result, verdict = _cli_eval(kernels, out)
    assert verdict["verdict"] == "no_gain" and result.exit_code == verdict["exit_code"] == 4
    (kernels / "nsx/CMakeLists.txt").write_text("y\n")
    result, verdict = _cli_eval(kernels, out)
    assert verdict["verdict"] == "rejected" and result.exit_code == 3
    result, _ = _cli_eval(kernels, out, "--board", "apollo3p_evb")
    assert result.exit_code == 2


def test_eval_cli_crash_is_an_error_verdict(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)
    monkeypatch.setattr(candidate_eval, "evaluate", lambda *a, **k: 1 / 0)
    result, verdict = _cli_eval(kernels, out)
    assert verdict["verdict"] == "error" and result.exit_code == 5
    assert list((out / "logs").glob("eval-error-*.log"))


def test_baseline_cli_refuses_used_out(tmp_path, kernels) -> None:
    out = tmp_path / "used"
    out.mkdir()
    (out / "x").write_text("x")
    result = runner.invoke(app, ["candidate", "baseline", "--kernels", str(kernels), "--board", "apollo510_evb", "--out", str(out)])
    assert result.exit_code == 2


def test_object_check_rejects_after_build(tmp_path, kernels, monkeypatch) -> None:
    """The post-build scan can still reject."""
    out, _ = _baseline(tmp_path, kernels)
    report = {"ok": False, "findings": [{"rule": "scs_address", "path": "Source/Conv/a.c"}]}
    monkeypatch.setattr(candidate_eval, "object_check", lambda *a: report)
    verdict = _eval(kernels, out, FakeRun(tmp_path / "reports"))
    assert verdict["verdict"] == "rejected" and verdict["stage"] == "objects"


def test_snapshot_skips_fifos_and_links(tmp_path, kernels) -> None:
    """No hang, no followed links."""
    import os

    out, _ = _baseline(tmp_path, kernels)
    os.mkfifo(kernels / "Source/Conv/pipe.c")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "s.c").write_text("hidden\n")
    (kernels / "Source/Leak").symlink_to(secret)
    snap = candidate_eval.snapshot(kernels, out, candidate_eval.read_baseline(out)["base_commit"])
    assert (snap / "Source/Conv/a.c").is_file() and not (snap / "Source/Conv/pipe.c").exists()
    # Linked dirs stay links, unread.
    assert (snap / "Source/Leak").is_symlink() and not (snap / "Source/Leak").resolve().is_relative_to(snap)


def test_eval_cli_interrupt_exits_130(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)

    def _stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(candidate_eval, "evaluate", _stop)
    result, _ = _cli_eval(kernels, out)
    assert result.exit_code == 130


class OtherBuild(FakeRun):
    """Builds kernels other than the snapshot."""

    def __call__(self, args: list[str], log: Path):
        rc, summary = super().__call__(args, log)
        manifest_path = Path(summary["bundle"]) / "session_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["build"]["kernels"]["tree_hash"] = "other"
        manifest_path.write_text(json.dumps(manifest))
        return rc, summary


def test_build_of_other_kernels_is_not_comparable(tmp_path, kernels) -> None:
    """Score ties the bundle to the checked snapshot."""
    out, _ = _baseline(tmp_path, kernels)
    verdict = _eval(kernels, out, OtherBuild(tmp_path / "reports"))
    assert verdict["verdict"] == "not_comparable"
    assert any("tree hash" in f["reason"] for f in verdict["failures"])


@pytest.mark.parametrize("edit", [
    lambda meta: [],
    lambda meta: {**meta, "schema_version": 99},
    lambda meta: {k: v for k, v in meta.items() if k != "run"},
    lambda meta: {**meta, "sessions": []},
    lambda meta: {**meta, "run": {**meta["run"], "board": "nope_evb"}},
])
def test_bad_baseline_is_a_refused_verdict(tmp_path, kernels, edit) -> None:
    out, _ = _baseline(tmp_path, kernels)
    path = out / candidate_eval.BASELINE_FILE
    path.write_text(json.dumps(edit(json.loads(path.read_text()))))
    with pytest.raises(ValueError):
        candidate_eval.read_baseline(out)
    result, verdict = _cli_eval(kernels, out)
    assert result.exit_code == 3 and verdict["verdict"] == "refused" and verdict["stage"] == "baseline"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_min_score_must_be_finite(tmp_path, kernels, value) -> None:
    out, _ = _baseline(tmp_path, kernels)
    result, _ = _cli_eval(kernels, out, "--min-score", value)
    assert result.exit_code == 2


def test_missing_hidden_case_refuses_without_its_id(tmp_path, kernels, monkeypatch) -> None:
    monkeypatch.setattr(sb, "FIELDS", sb.FIELDS + ["hidden"])
    out, _ = _baseline(tmp_path, kernels, rows={"dw_a": {"hidden": "true"}})
    verdict = _eval(kernels, out, FakeRun(tmp_path / "reports", drop=("dw_a", "fc_a")))
    # Score refuses a changed hidden set.
    assert verdict["verdict"] == "not_comparable" and "dw_a" not in json.dumps(verdict)


def test_all_cases_missing_refuses(tmp_path, kernels) -> None:
    out, _ = _baseline(tmp_path, kernels)
    verdict = _eval(kernels, out, FakeRun(tmp_path / "reports", drop=tuple(sb.CASES)))
    assert verdict["verdict"] == "refused"


@pytest.mark.parametrize(("budget", "make"), [
    (candidate_eval.CopyBudget(file_bytes=1000), lambda k: (k / "Source/big.c").write_bytes(b"x" * 2000)),
    # Sparse: few blocks, huge size.
    (candidate_eval.CopyBudget(), lambda k: os.truncate(_touch(k / "Source/sparse.c"), 1 << 40)),
    (candidate_eval.CopyBudget(total_bytes=100), lambda k: [(k / f"Source/f{i}.c").write_bytes(b"y" * 60) for i in range(2)]),
    (candidate_eval.CopyBudget(files=5), lambda k: [_touch(k / f"Source/n{i}.c") for i in range(3)]),
])
def test_oversized_candidate_refuses(tmp_path, kernels, budget, make) -> None:
    out, _ = _baseline(tmp_path, kernels)
    make(kernels)
    run = FakeRun(tmp_path / "reports")
    verdict = candidate_eval.evaluate(kernels, out, candidate_eval.read_baseline(out), 0.005, run=run, budget=budget)
    assert verdict["verdict"] == "refused" and verdict["stage"] == "check" and not run.calls


def _touch(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def test_baseline_out_must_be_a_dir(tmp_path, kernels) -> None:
    out = tmp_path / "file"
    out.write_text("x")
    result = runner.invoke(app, ["candidate", "baseline", "--kernels", str(kernels), "--board", "apollo510_evb", "--out", str(out)])
    assert result.exit_code == 2
