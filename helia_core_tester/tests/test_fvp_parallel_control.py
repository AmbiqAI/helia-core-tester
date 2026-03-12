import subprocess
import time
from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.core.steps.run import RunStep
from helia_core_tester.fvp.build_and_run_fvp import (
    ProcessRecord,
    ProcessSupervisor,
    _resolve_downloaded_fvp_executable,
    _resolve_run_jobs,
)


def _make_config_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_config_run_jobs_default_and_auto(tmp_path: Path, monkeypatch) -> None:
    root = _make_config_root(tmp_path)
    cfg_default = Config(project_root=root, run_jobs=1)
    assert cfg_default.run_jobs == 1

    monkeypatch.setattr("helia_core_tester.core.config.os.cpu_count", lambda: 12)
    cfg_auto = Config(project_root=root, run_jobs=0)
    assert cfg_auto.run_jobs == 12


def test_config_run_jobs_negative_rejected(tmp_path: Path) -> None:
    root = _make_config_root(tmp_path)
    with pytest.raises(ValueError, match="run_jobs must be >= 0"):
        Config(project_root=root, run_jobs=-1)


def test_run_step_passes_fail_fast_and_run_jobs(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, cwd, check, text, bufsize):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr("helia_core_tester.core.steps.run.subprocess.run", fake_run)

    root = _make_config_root(tmp_path)
    cfg = Config(project_root=root, run_jobs=4, fail_fast=True, enable_reporting=False)
    result = RunStep(cfg)._do_execute()

    assert result.success
    cmd = result.details["command"]
    assert "--fail-fast" in cmd
    assert "--run-jobs" in cmd
    assert "4" in cmd
    assert "--no-fail-fast" not in cmd


def test_resolve_run_jobs_caps_to_test_count() -> None:
    assert _resolve_run_jobs(0, 8) >= 1
    assert _resolve_run_jobs(16, 5) == 5
    assert _resolve_run_jobs(1, 5) == 1


def test_process_supervisor_terminates_registered_process() -> None:
    proc = subprocess.Popen(
        ["python3", "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    supervisor = ProcessSupervisor(grace_seconds=0.2, verbosity=0)
    supervisor.register(
        ProcessRecord(
            elf=Path("sleep.elf"),
            cpu="cortex-m55",
            descriptor_name="sleep",
            popen=proc,
            start_time=time.time(),
        )
    )
    assert supervisor.active_count() == 1

    supervisor.terminate_all("test")
    proc.wait(timeout=2.0)
    supervisor.unregister(proc.pid)

    assert proc.poll() is not None
    assert supervisor.active_count() == 0


def test_resolve_downloaded_fvp_executable_prefers_matching_arch(tmp_path: Path) -> None:
    x86 = tmp_path / "corstone300_download" / "models" / "Linux64_GCC-9.3" / "FVP_Corstone_SSE-300_Ethos-U55"
    arm = tmp_path / "corstone300_download" / "models" / "Linux64_armv8l_GCC-9.3" / "FVP_Corstone_SSE-300_Ethos-U55"
    x86.parent.mkdir(parents=True, exist_ok=True)
    arm.parent.mkdir(parents=True, exist_ok=True)
    x86.write_text("")
    arm.write_text("")

    exe, checked = _resolve_downloaded_fvp_executable(tmp_path, "x86_64")
    assert exe == x86
    assert checked[0] == x86


def test_resolve_downloaded_fvp_executable_falls_back_to_other_dir(tmp_path: Path) -> None:
    arm = tmp_path / "corstone300_download" / "models" / "Linux64_armv8l_GCC-9.3" / "FVP_Corstone_SSE-300_Ethos-U55"
    arm.parent.mkdir(parents=True, exist_ok=True)
    arm.write_text("")

    exe, checked = _resolve_downloaded_fvp_executable(tmp_path, "x86_64")
    assert exe == arm
    assert len(checked) == 2
