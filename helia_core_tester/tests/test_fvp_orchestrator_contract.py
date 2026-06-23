from __future__ import annotations

from pathlib import Path

from helia_core_tester.fvp import build_and_run_fvp as facade
from helia_core_tester.fvp.cli import build_arg_parser
from helia_core_tester.fvp.env import _resolve_downloaded_fvp_executable as env_downloaded_fvp_resolver
from helia_core_tester.fvp.runner import (
    ProcessRecord as RunnerProcessRecord,
    ProcessSupervisor as RunnerProcessSupervisor,
    _resolve_run_jobs as runner_resolve_run_jobs,
)


def _fake_detect_paths(tmp_path: Path, compiler_tag: str = "gcc") -> dict:
    return {
        "env": {"PATH": ""},
        "dl": tmp_path / "downloads",
        "ethos": tmp_path / "ethos-u-core-platform",
        "cmsis5": tmp_path / "CMSIS_5",
        "toolchain_file": tmp_path / "toolchain.cmake",
        "compiler_tag": compiler_tag,
        "fvp_exe": tmp_path / "FVP_Corstone_SSE-300_Ethos-U55",
    }


def test_facade_compatibility_exports_preserved() -> None:
    assert facade.ProcessRecord is RunnerProcessRecord
    assert facade.ProcessSupervisor is RunnerProcessSupervisor
    assert facade._resolve_run_jobs is runner_resolve_run_jobs
    assert facade._resolve_downloaded_fvp_executable is env_downloaded_fvp_resolver
    assert callable(facade.main)


def test_fvp_parser_option_surface_unchanged() -> None:
    parser = build_arg_parser(
        default_downloads_dir=Path("/tmp/downloads"),
        default_source_dir=Path("/tmp/source"),
    )
    actual = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option
    }
    expected = {
        "-h", "--help",
        "-c", "--cpu",
        "--suite",
        "-o", "--opt",
        "--verbosity",
        "-b", "--no-build",
        "-r", "--no-run",
        "-e", "--no-setup",
        "-a", "--use-arm-compiler",
        "-p", "--no-venv",
        "-f", "--no-fvp-from-download",
        "-g", "--no-gcc-from-download",
        "-u", "--ethos-path",
        "-C", "--cmsis5-path",
        "-D", "--cmake-def",
        "--coverage",
        "--downloads-dir",
        "--source-dir",
        "--generator",
        "-j", "--jobs",
        "--run-jobs",
        "--timeout-run",
        "--fail-fast", "--no-fail-fast",
        "--fvp-arg",
        "--no-report",
        "--report-formats",
        "--quiet",
    }
    assert actual == expected


def test_main_rejects_non_linux_platform(monkeypatch, capsys) -> None:
    monkeypatch.setattr("helia_core_tester.fvp.orchestrator.is_linux", lambda: False)

    rc = facade.main([])

    assert rc == 2
    assert "supports Linux only" in capsys.readouterr().err


def test_main_rejects_invalid_run_jobs(monkeypatch, tmp_path: Path, capsys) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("helia_core_tester.fvp.orchestrator.is_linux", lambda: True)
    monkeypatch.setattr(
        "helia_core_tester.fvp.orchestrator.detect_paths",
        lambda args: _fake_detect_paths(tmp_path, compiler_tag="gcc"),
    )

    rc = facade.main(["-e", "--source-dir", str(tmp_path), "--run-jobs", "-1"])

    assert rc == 2
    assert "--run-jobs must be >= 0" in capsys.readouterr().err


def test_main_rejects_coverage_with_arm_compiler(monkeypatch, tmp_path: Path, capsys) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("helia_core_tester.fvp.orchestrator.is_linux", lambda: True)
    monkeypatch.setattr(
        "helia_core_tester.fvp.orchestrator.detect_paths",
        lambda args: _fake_detect_paths(tmp_path, compiler_tag="arm-compiler"),
    )

    rc = facade.main(
        [
            "-e",
            "--source-dir",
            str(tmp_path),
            "--coverage",
            "--use-arm-compiler",
        ]
    )

    assert rc == 2
    assert "--coverage is only supported with GCC builds" in capsys.readouterr().err


def test_no_build_no_run_short_circuits_without_cmake(monkeypatch, tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("helia_core_tester.fvp.orchestrator.is_linux", lambda: True)
    monkeypatch.setattr(
        "helia_core_tester.fvp.orchestrator.detect_paths",
        lambda args: _fake_detect_paths(tmp_path, compiler_tag="gcc"),
    )

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("cmake/build/run path should not execute for --no-build --no-run")

    monkeypatch.setattr("helia_core_tester.fvp.orchestrator.cmake_configure", _should_not_be_called)
    monkeypatch.setattr("helia_core_tester.fvp.orchestrator.cmake_build", _should_not_be_called)
    monkeypatch.setattr("helia_core_tester.fvp.orchestrator.find_elves", _should_not_be_called)

    rc = facade.main(
        [
            "-e",
            "--source-dir",
            str(tmp_path),
            "--no-report",
            "--no-build",
            "--no-run",
        ]
    )

    assert rc == 0
