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
from helia_core_tester.fvp.runner import _sanitize_terminal_text, _signal_process_group


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
    cmd = result.details["commands"][0]
    assert "--fail-fast" in cmd
    assert "--run-jobs" in cmd
    assert "4" in cmd
    assert "--no-fail-fast" not in cmd


def test_run_step_routes_suite_both_into_cpu_groups(tmp_path: Path, monkeypatch) -> None:
    seen = []

    def fake_run(cmd, cwd, check, text, bufsize):
        seen.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr("helia_core_tester.core.steps.run.subprocess.run", fake_run)

    root = _make_config_root(tmp_path)
    cfg = Config(
        project_root=root,
        cpu="cortex-m0,cortex-m4,cortex-m55",
        suite="both",
        float_precision="f16",
        fail_fast=False,
        enable_reporting=False,
        _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
    )

    result = RunStep(cfg)._do_execute()

    assert result.success
    assert len(result.details["commands"]) == 3
    rendered = [" ".join(cmd) for cmd in seen]
    assert any("--suite int" in cmd and "--cpu cortex-m0,cortex-m4,cortex-m55" in cmd for cmd in rendered)
    assert any("--suite float" in cmd and "--cpu cortex-m4" in cmd for cmd in rendered)
    assert any("--suite float" in cmd and "--cpu cortex-m55" in cmd for cmd in rendered)


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


def test_sanitize_terminal_text_escapes_control_chars() -> None:
    raw = "ok\x03line\x1b"
    sanitized = _sanitize_terminal_text(raw)
    assert sanitized == "ok\\x03line\\x1b"


def test_signal_process_group_falls_back_to_direct_signal_when_same_group(monkeypatch) -> None:
    class DummyProc:
        pid = 4242

        def poll(self):
            return None

        def send_signal(self, sig):
            self.sent = sig

    proc = DummyProc()
    called = {"killpg": False}

    monkeypatch.setattr("helia_core_tester.fvp.runner.os.getpgid", lambda _pid: 100)
    monkeypatch.setattr("helia_core_tester.fvp.runner.os.getpgrp", lambda: 100)

    def _killpg(*_args, **_kwargs):
        called["killpg"] = True

    monkeypatch.setattr("helia_core_tester.fvp.runner.os.killpg", _killpg)

    _signal_process_group(proc, 15)

    assert getattr(proc, "sent", None) == 15
    assert called["killpg"] is False
