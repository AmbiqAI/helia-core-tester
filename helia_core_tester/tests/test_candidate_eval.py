from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware import candidate_eval, code_graph
from helia_core_tester.hardware.candidate_check import CheckError
from helia_core_tester.hardware.toolchain import arm_tool
from helia_core_tester.tests import test_score_bundles as sb
from helia_core_tester.tests.test_harness_lock import _git, _repo

runner = CliRunner()
REAL_TESTER_DIRTY = candidate_eval.tester_dirty
needs_gcc = pytest.mark.skipif(shutil.which(arm_tool("arm-none-eabi-gcc")) is None, reason="needs arm-none-eabi-gcc")


@pytest.fixture(autouse=True)
def clean_tester(monkeypatch, tmp_path) -> None:
    """CLI tests assume a committed tester."""
    monkeypatch.setattr(candidate_eval, "tester_dirty", lambda: False)
    monkeypatch.setattr(candidate_eval, "scan_cache_dir", lambda: tmp_path / "scan_cache")
    # Fake runs build no objects.
    monkeypatch.setattr(candidate_eval, "object_check", lambda *args: None)
    monkeypatch.setattr(candidate_eval, "kernel_graph", lambda board: None)


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

    def __call__(self, args: list[str], log: Path, on_built=None):
        self.calls.append(args)
        if on_built is not None:
            on_built()
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


@needs_gcc
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
    verdict = _eval(kernels, out, lambda args, log, on_built=None: (rc, None))
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
    assert verdict["hidden"]["failed"] == [{"kind": "comparison_failed", "symbol": "arm_depthwise_conv_wrapper_s8",
                                            "via": None, "touched": None, "count": 1}]


def test_hidden_failures_name_kernels_only() -> None:
    def case(case_id, inner, touched):
        return {"case_id": case_id, "timed_symbol": "arm_depthwise_conv_wrapper_s8", "inner_symbol": inner, "touched": touched}

    cases = [case("h_1x7x7", "arm_convolve_s8", True), case("h_2x9x9", "arm_convolve_s8", True),
             case("h_3x5x5", None, False), case("pub", "arm_convolve_s8", True)]
    failures = [{"kind": "regression", "case_id": c, "reason": "+1.24% slower than band 1.00%"} for c in ("h_1x7x7", "h_2x9x9", "pub")]
    failures += [{"kind": "comparison_failed", "case_id": "h_3x5x5", "reason": "s: output mismatch"},
                 {"kind": "family_regression", "case_id": None, "reason": "conv all"}]
    failed = candidate_eval.hidden_failures({"cases": cases, "failures": failures}, {"h_1x7x7", "h_2x9x9", "h_3x5x5"})
    assert failed == [
        {"kind": "regression", "symbol": "arm_convolve_s8", "via": "arm_depthwise_conv_wrapper_s8", "touched": True, "count": 2},
        {"kind": "comparison_failed", "symbol": "arm_depthwise_conv_wrapper_s8", "via": None, "touched": False, "count": 1},
    ]
    text = json.dumps(failed)
    assert "h_" not in text and "%" not in text and "x7" not in text


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


def test_object_check_overlaps_run(tmp_path, kernels, monkeypatch) -> None:
    """The scan runs once, inside the run."""
    out, _ = _baseline(tmp_path, kernels)
    order = []
    monkeypatch.setattr(candidate_eval, "object_check", lambda *a: order.append("objects"))

    class Run(FakeRun):
        def __call__(self, args, log, on_built=None):
            result = super().__call__(args, log, on_built)
            order.append("run_end")
            return result

    _eval(kernels, out, Run(tmp_path / "reports"))
    assert order == ["objects", "run_end"]


def test_run_failure_beats_object_check(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)
    monkeypatch.setattr(candidate_eval, "object_check", lambda *a: {"ok": False, "findings": []})

    def run(args, log, on_built=None):
        on_built()
        return 5, None

    verdict = _eval(kernels, out, run)
    assert verdict["verdict"] == "error" and verdict["stage"] == "run"


def test_object_check_error_waits_for_run(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)
    ended = []

    def broken(*a):
        raise OSError("disk")

    monkeypatch.setattr(candidate_eval, "object_check", broken)

    def run(args, log, on_built=None):
        on_built()
        ended.append(True)
        return FakeRun(tmp_path / "reports")(args, log)

    with pytest.raises(OSError, match="disk"):
        _eval(kernels, out, run)
    assert ended == [True]


def _fake_cli(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake-python"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize(("body", "built"), [
    # dash cannot redirect to fds above 9.
    ('printf 1 > "/dev/fd/$HCT_BUILT_FD"; echo \'{"bundle": "b"}\'', True),
    ('echo \'{"bundle": "b"}\'', False),
])
def test_hardware_run_signals_build(tmp_path, monkeypatch, body, built) -> None:
    monkeypatch.setattr(candidate_eval.sys, "executable", _fake_cli(tmp_path, body))
    seen = []
    rc, summary = candidate_eval.hardware_run(["x"], tmp_path / "log", lambda: seen.append(True))
    assert rc == 0 and summary == {"bundle": "b"} and seen == ([True] if built else [])


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

    def __call__(self, args: list[str], log: Path, on_built=None):
        rc, summary = super().__call__(args, log, on_built)
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
    (candidate_eval.CopyBudget(entries=6), lambda k: [_touch(k / f"Source/n{i}.c") for i in range(3)]),
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


@pytest.mark.parametrize("shape", ["deep", "wide"])
def test_empty_dir_trees_refuse_before_mkdir(tmp_path, kernels, shape) -> None:
    out, _ = _baseline(tmp_path, kernels)
    if shape == "deep":
        (kernels / "Source" / Path(*["d"] * 10)).mkdir(parents=True)
        budget = candidate_eval.CopyBudget(depth=5)
    else:
        for i in range(20):
            (kernels / "Include" / f"w{i}").mkdir()
        budget = candidate_eval.CopyBudget(entries=10)
    verdict = candidate_eval.evaluate(kernels, out, candidate_eval.read_baseline(out), 0.005,
                                      run=FakeRun(tmp_path / "reports"), budget=budget)
    assert verdict["verdict"] == "refused"
    snap = out / "snapshot"
    assert not (snap / "Source" / Path(*["d"] * 6)).exists() and len(list((snap / "Include").glob("w*"))) <= 10


def test_file_growing_during_copy_hits_total(tmp_path, kernels, monkeypatch) -> None:
    """Bytes read count, not the stat."""
    out, _ = _baseline(tmp_path, kernels)
    real = candidate_eval._read_chunks
    # Each file "grows" by 1 KiB while read.
    monkeypatch.setattr(candidate_eval, "_read_chunks", lambda handle: [*real(handle), b"z" * 1024])
    budget = candidate_eval.CopyBudget(total_bytes=2048)
    verdict = candidate_eval.evaluate(kernels, out, candidate_eval.read_baseline(out), 0.005,
                                      run=FakeRun(tmp_path / "reports"), budget=budget)
    assert verdict["verdict"] == "refused" and "in total" in verdict["reason"]


def test_baseline_keeps_run_refusal(tmp_path, kernels, monkeypatch) -> None:
    """A refused run refuses the baseline (3)."""
    monkeypatch.setattr(candidate_eval, "hardware_run", lambda args, log, on_built=None: (3, None))
    out = tmp_path / "base"
    result = runner.invoke(app, ["candidate", "baseline", "--kernels", str(kernels), "--board", "apollo510_evb", "--out", str(out)])
    assert result.exit_code == 3
    monkeypatch.setattr(candidate_eval, "hardware_run", lambda args, log, on_built=None: (5, None))
    result = runner.invoke(app, ["candidate", "baseline", "--kernels", str(kernels), "--board", "apollo510_evb",
                                 "--out", str(tmp_path / "base2")])
    assert result.exit_code == 5


def test_default_budget_spans_trees(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)
    small = candidate_eval.CopyBudget
    monkeypatch.setattr(candidate_eval, "CopyBudget", lambda: small(total_bytes=3000))
    (kernels / "Source/s.c").write_bytes(b"s" * 2000)
    (kernels / "Include/i.h").write_bytes(b"i" * 2000)
    # Each tree fits; together they do not.
    with pytest.raises(candidate_eval.TooLarge, match="in total"):
        candidate_eval.snapshot(kernels, out, candidate_eval.read_baseline(out)["base_commit"])


def test_hidden_count_survives_not_comparable(tmp_path, kernels, monkeypatch) -> None:
    monkeypatch.setattr(sb, "FIELDS", sb.FIELDS + ["hidden"])
    out, _ = _baseline(tmp_path, kernels, rows={"dw_a": {"hidden": "true"}, "fc_a": {"hidden": "true"}})
    verdict = _eval(kernels, out, OtherBuild(tmp_path / "reports", rows={"dw_a": {"hidden": "true"}, "fc_a": {"hidden": "true"}}))
    assert verdict["verdict"] == "not_comparable" and verdict["hidden"]["cases"] == 2


def test_dirty_tester_refuses_before_check(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)
    (kernels / "nsx/CMakeLists.txt").write_text("y\n")
    monkeypatch.setattr(candidate_eval, "tester_dirty", lambda: True)
    result, verdict = _cli_eval(kernels, out)
    assert result.exit_code == 3 and verdict["verdict"] == "refused" and verdict["stage"] == "tester"
    result = runner.invoke(app, ["candidate", "baseline", "--kernels", str(kernels), "--board", "apollo510_evb",
                                 "--out", str(tmp_path / "b2")])
    assert result.exit_code == 3


@pytest.mark.parametrize("which", [0, -1])
def test_missing_bundle_refuses_before_run(tmp_path, kernels, which, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels, repeats=3)
    meta = candidate_eval.read_baseline(out)
    shutil.rmtree(out / "bundles" / meta["sessions"][which])
    run = FakeRun(tmp_path / "reports")
    monkeypatch.setattr(candidate_eval, "hardware_run", run)
    result, verdict = _cli_eval(kernels, out)
    assert result.exit_code == 3 and verdict["stage"] == "baseline" and not run.calls


@pytest.mark.parametrize(("dirty", "expected"), [(False, False), (True, True), (None, True)])
def test_unknown_tester_counts_as_dirty(monkeypatch, dirty, expected) -> None:
    from helia_core_tester.hardware import harness_lock

    monkeypatch.setattr(harness_lock, "tester_state", lambda root: {"dirty": dirty})
    assert REAL_TESTER_DIRTY() is expected


def test_readme_lists_every_stage() -> None:
    from helia_core_tester.hardware.boards import repo_root

    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    line = readme[readme.index("`stage`:"):readme.index("- `findings`")]
    assert all(f"`{stage}`" in line for stage in candidate_eval.STAGES)


def test_unreadable_tester_state_is_a_tester_verdict(tmp_path, kernels, monkeypatch) -> None:
    """A non-UTF-8 name must not crash."""
    tester = _repo(tmp_path / "tester", {"a.txt": "a\n"})
    (tester / b"bad\xff".decode("utf-8", "surrogateescape")).write_text("x")
    monkeypatch.setattr(candidate_eval, "repo_root", lambda: tester)
    monkeypatch.setattr(candidate_eval, "tester_dirty", REAL_TESTER_DIRTY)
    assert REAL_TESTER_DIRTY() is True
    out = tmp_path / "base"
    out.mkdir()
    result, verdict = _cli_eval(kernels, out)
    assert result.exit_code == 3 and verdict["stage"] == "tester"


def test_eval_never_exits_without_a_verdict(tmp_path, kernels, monkeypatch) -> None:
    out, _ = _baseline(tmp_path, kernels)

    def _boom():
        raise KeyError("bug")

    monkeypatch.setattr(candidate_eval, "read_baseline", lambda path: _boom())
    result, verdict = _cli_eval(kernels, out)
    assert result.exit_code == 5 and verdict["verdict"] == "error" and verdict["stage"] == "eval"


@pytest.mark.parametrize(("board", "placement"), [("apollo510_evb", "sram"), ("apollo3p_evb", "mram")])
def test_baseline_refuses_bad_placement_before_mkdir(tmp_path, kernels, board, placement) -> None:
    out = tmp_path / "new"
    result = runner.invoke(app, ["candidate", "baseline", "--kernels", str(kernels), "--board", board,
                                 "--placement", placement, "--out", str(out)])
    assert result.exit_code == 2 and not out.exists()


def test_hints_report_percent_of_peak(tmp_path, monkeypatch) -> None:
    """Hints carry percent, like the score's peak fields."""
    from helia_core_tester.hardware.pmu_explain import explain_case

    row = {"case_id": "c", "timed_symbol": "arm_convolve_wrapper_s8", "inner_symbol": "arm_convolve_s8",
           "median_cycles": 4000, "macs": 16000, "timing_status": "valid"}
    case = explain_case(row, cpu="cortex-m55")
    monkeypatch.setattr(candidate_eval, "explain_bundle", lambda bundle: {"cases": [case]})
    [hint] = candidate_eval._hints(tmp_path, hidden=set())
    assert hint["pct_of_peak"] == pytest.approx(50.0)


CONV4 = {f"conv_{i}": ("arm_convolve_wrapper_s8", 1000.0) for i in "abcd"}


def _graph(digest: str) -> dict:
    nodes = {"arm_convolve_wrapper_s8": {"digest": digest, "refs": ["arm_nn_mat_mult_s8"]},
             "arm_nn_mat_mult_s8": {"digest": "m", "refs": []}}
    return {"schema": code_graph.SCHEMA, "schema_version": code_graph.SCHEMA_VERSION, "nodes": nodes}


@pytest.mark.parametrize("base_graph, cand_graph, verdict, scope, reason", [
    (_graph("w"), _graph("w"), "no_gain", "touched", None),
    (_graph("w"), _graph("x"), "fail", "touched", None),
    (None, _graph("w"), "fail", "all", "baseline has no code graph"),
])
def test_case_gate_needs_changed_code(tmp_path, kernels, monkeypatch, base_graph, cand_graph, verdict, scope, reason) -> None:
    monkeypatch.setattr(candidate_eval, "kernel_graph", lambda board: base_graph)
    out, _ = _baseline(tmp_path, kernels, cases=CONV4)
    assert (out / candidate_eval.GRAPH_FILE).is_file() == (base_graph is not None)
    monkeypatch.setattr(candidate_eval, "kernel_graph", lambda board: cand_graph)
    # Past case band, inside family band.
    result = _eval(kernels, out, FakeRun(tmp_path / "reports", cases=CONV4, cycles={"conv_a": 1015.0}))
    assert result["verdict"] == verdict
    assert [f["kind"] for f in result["failures"]] == (["regression"] if verdict == "fail" else [])
    assert result["case_gate"] == {"scope": scope, "reason": reason}
    conv = next(c for c in result["cases"] if c["case_id"] == "conv_a")
    assert conv["touched"] == (None if scope == "all" else cand_graph != base_graph)


def test_unreadable_candidate_objects_gate_all(tmp_path, kernels, monkeypatch) -> None:
    monkeypatch.setattr(candidate_eval, "kernel_graph", lambda board: _graph("w"))
    out, _ = _baseline(tmp_path, kernels, cases=CONV4)

    def broken(board):
        raise ValueError("no kernel objects found")

    monkeypatch.setattr(candidate_eval, "kernel_graph", broken)
    result = _eval(kernels, out, FakeRun(tmp_path / "reports", cases=CONV4))
    assert result["case_gate"]["scope"] == "all" and "unreadable" in result["case_gate"]["reason"]


def test_placements_get_own_build_dirs(tmp_path, kernels) -> None:
    """No tcm/mram flip rebuilds."""
    dirs = {placement: candidate_eval.run_args(candidate_eval.RunSpec("apollo510_evb", kernels, placement), "s")
            for placement in ("tcm", "mram")}
    found = {p: Path(a[a.index("--build-dir") + 1]) for p, a in dirs.items()}
    assert found["tcm"] != found["mram"] and found["tcm"].name == "apollo510_evb-eval-tcm"
