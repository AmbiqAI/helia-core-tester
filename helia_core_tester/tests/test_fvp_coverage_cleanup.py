from __future__ import annotations

from types import SimpleNamespace
import subprocess
from pathlib import Path

from helia_core_tester.fvp.coverage import generate_coverage_reports


def test_generate_coverage_reports_clears_stale_cpu_dir(tmp_path: Path, monkeypatch) -> None:
    cpu = "cortex-m55"
    suite = "int"
    source_dir = tmp_path / "Tests" / "UnitTest"
    source_dir.mkdir(parents=True, exist_ok=True)
    build_dir = tmp_path / "artifacts" / f"build-{suite}-{cpu}-gcc"
    build_dir.mkdir(parents=True, exist_ok=True)

    report_dir = tmp_path / "artifacts" / "reports" / "coverage" / suite / cpu
    stale_file = report_dir / "stale.txt"
    report_dir.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("stale")

    monkeypatch.setattr("helia_core_tester.fvp.coverage.REPO_ROOT", tmp_path)
    monkeypatch.setattr("helia_core_tester.fvp.coverage.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(
        "helia_core_tester.fvp.coverage.shutil.which",
        lambda name, path=None: "/usr/bin/gcovr" if name == "gcovr" else None,
    )
    monkeypatch.setattr("helia_core_tester.fvp.coverage.resolve_gcov_executable", lambda env: None)

    def _fake_run(cmd, cwd, env, capture_output, text, check):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("helia_core_tester.fvp.coverage.subprocess.run", _fake_run)

    generate_coverage_reports(
        cpus=[cpu],
        suite=suite,
        args=SimpleNamespace(coverage=True),
        env={"PATH": ""},
        source_dir=source_dir,
        compiler_tag="gcc",
        verbosity=0,
    )

    assert not stale_file.exists()
    assert (report_dir / "summary.txt").exists()


def test_generate_coverage_reports_clears_stale_dir_even_without_build(tmp_path: Path, monkeypatch) -> None:
    cpu = "cortex-m0"
    suite = "int"
    source_dir = tmp_path / "Tests" / "UnitTest"
    source_dir.mkdir(parents=True, exist_ok=True)

    report_dir = tmp_path / "artifacts" / "reports" / "coverage" / suite / cpu
    stale_file = report_dir / "stale.txt"
    report_dir.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("stale")

    monkeypatch.setattr("helia_core_tester.fvp.coverage.REPO_ROOT", tmp_path)
    monkeypatch.setattr("helia_core_tester.fvp.coverage.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(
        "helia_core_tester.fvp.coverage.shutil.which",
        lambda name, path=None: "/usr/bin/gcovr" if name == "gcovr" else None,
    )
    monkeypatch.setattr("helia_core_tester.fvp.coverage.resolve_gcov_executable", lambda env: None)

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("gcovr should not run when build dir is missing")

    monkeypatch.setattr("helia_core_tester.fvp.coverage.subprocess.run", _should_not_run)

    generate_coverage_reports(
        cpus=[cpu],
        suite=suite,
        args=SimpleNamespace(coverage=True),
        env={"PATH": ""},
        source_dir=source_dir,
        compiler_tag="gcc",
        verbosity=0,
    )

    assert not stale_file.exists()
    assert report_dir.exists()
    assert not (report_dir / "summary.txt").exists()


def test_generate_coverage_reports_can_use_float_mve_report_lane(tmp_path: Path, monkeypatch) -> None:
    cpu = "cortex-m55"
    suite = "float"
    source_dir = tmp_path / "Tests" / "UnitTest"
    source_dir.mkdir(parents=True, exist_ok=True)
    build_dir = tmp_path / "artifacts" / f"build-{suite}-{cpu}-gcc"
    build_dir.mkdir(parents=True, exist_ok=True)

    report_dir = tmp_path / "artifacts" / "reports" / "coverage" / "float-mve" / cpu

    monkeypatch.setattr("helia_core_tester.fvp.coverage.REPO_ROOT", tmp_path)
    monkeypatch.setattr("helia_core_tester.fvp.coverage.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(
        "helia_core_tester.fvp.coverage.shutil.which",
        lambda name, path=None: "/usr/bin/gcovr" if name == "gcovr" else None,
    )
    monkeypatch.setattr("helia_core_tester.fvp.coverage.resolve_gcov_executable", lambda env: None)

    captured = {}

    def _fake_run(cmd, cwd, env, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("helia_core_tester.fvp.coverage.subprocess.run", _fake_run)

    generate_coverage_reports(
        cpus=[cpu],
        suite=suite,
        args=SimpleNamespace(coverage=True, coverage_report_suite="float-mve"),
        env={"PATH": ""},
        source_dir=source_dir,
        compiler_tag="gcc",
        verbosity=0,
    )

    assert (report_dir / "summary.txt").exists()
    assert str(build_dir) in captured["cmd"]
    assert str(report_dir / "coverage.info") in captured["cmd"]
