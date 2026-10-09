"""`hardware run` host-checks the generated int cases against the firmware's
kernels after generation and before anything is flashed."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.core.steps.base import StepResult, StepStatus
from helia_core_tester.hardware import firmware_build, hardware_pipeline
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.errors import RunRefused
from helia_core_tester.hardware.hardware_pipeline import StreamOptions, run_hardware_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARD = resolve_board("apollo510_evb")


def _wire(monkeypatch, tmp_path, order, host_check):
    monkeypatch.setattr(hardware_pipeline, "stage_kernels", lambda *a, **k: Path("/kernels"))
    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", lambda *a, **k: (order.append("generate"), 41)[1])
    monkeypatch.setattr(hardware_pipeline, "host_check_tests_for_board", host_check)
    monkeypatch.setattr(
        hardware_pipeline,
        "flash_firmware",
        lambda *a, **k: (order.append("flash"), firmware_build.FlashDecision(True, "abc", "test"))[1],
    )
    monkeypatch.setattr(
        hardware_pipeline,
        "stream_generated_tests",
        lambda *a, **k: (order.append("stream"), hardware_pipeline.HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[]))[1],
    )


def test_host_check_runs_between_generate_and_flash(tmp_path, monkeypatch) -> None:
    order, seen = [], {}

    def host_check(repo_root, board, suite, cmsis_nn_root, host_kernels, seed, jobs):
        order.append("host-check")
        seen.update(root=cmsis_nn_root, kernels=tuple(host_kernels), seed=seed, suite=suite)
        return "host check passed: 3 case run(s)"

    _wire(monkeypatch, tmp_path, order, host_check)
    run_hardware_pipeline(
        tmp_path, BOARD, 42, options=StreamOptions(), app_options=object(), echo=lambda _m: None, host_kernels=("m0", "dsp"),
    )
    assert order == ["generate", "host-check", "flash", "stream"]
    # The check runs against the kernels the firmware compiles, with the run's seed.
    assert seen == {"root": Path("/kernels"), "kernels": ("m0", "dsp"), "seed": 41, "suite": "int"}


def test_failing_host_check_refuses_before_flash(tmp_path, monkeypatch) -> None:
    order = []

    def host_check(*a, **k):
        order.append("host-check")
        raise RunRefused("Host check failed: 1 host-check failure(s)")

    _wire(monkeypatch, tmp_path, order, host_check)
    with pytest.raises(RunRefused, match="Host check failed"):
        run_hardware_pipeline(tmp_path, BOARD, 42, options=StreamOptions(), app_options=object(), echo=lambda _m: None)
    assert order == ["generate", "host-check"]


def test_skip_host_check_and_skip_generate_bypass_it(tmp_path, monkeypatch) -> None:
    order, messages = [], []

    def host_check(*a, **k):
        raise AssertionError("host check must not run")

    _wire(monkeypatch, tmp_path, order, host_check)
    run_hardware_pipeline(
        tmp_path, BOARD, 42, options=StreamOptions(), app_options=object(), echo=messages.append, skip_host_check=True,
    )
    assert order == ["generate", "flash", "stream"]
    assert any("--skip-host-check" in m for m in messages)
    order.clear()
    run_hardware_pipeline(
        tmp_path, BOARD, 42, options=StreamOptions(), app_options=object(), echo=lambda _m: None, skip_generate=True,
    )
    assert order == ["flash", "stream"]


def test_host_check_for_board_maps_step_results(monkeypatch, tmp_path) -> None:
    from helia_core_tester.core import steps

    seen = {}

    def execute(self):
        seen.update(
            cpus=self.config.cpus, kernels=self.config.host_kernels, root=self.config.cmsis_nn_root, seed=self.config.seed
        )
        return StepResult(name="host-check", status=status, message="detail")

    monkeypatch.setattr(steps.HostCheckStep, "execute", execute)
    status = StepStatus.SUCCESS
    message = hardware_pipeline.host_check_tests_for_board(
        PROJECT_ROOT, BOARD, "int", tmp_path, host_kernels=("dsp",), seed=5, jobs=3
    )
    assert message == "detail"
    assert seen == {"cpus": [BOARD.cpu], "kernels": ["dsp"], "root": tmp_path.resolve(), "seed": 5}
    status = StepStatus.FAILED
    with pytest.raises(RunRefused, match="Host check failed: detail"):
        hardware_pipeline.host_check_tests_for_board(PROJECT_ROOT, BOARD, "int", tmp_path)
    status = StepStatus.SKIPPED
    assert hardware_pipeline.host_check_tests_for_board(PROJECT_ROOT, BOARD, "float", tmp_path) == "detail"
