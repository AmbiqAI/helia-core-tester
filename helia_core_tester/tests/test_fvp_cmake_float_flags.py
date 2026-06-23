from __future__ import annotations

from pathlib import Path

from helia_core_tester.core.config import Config
from helia_core_tester.core.steps.build import BuildStep
from helia_core_tester.core.steps.generate import GenerateStep


def _init_repo_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_generate_step_propagates_suite_and_precision_flags(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(
        project_root=root,
        suite="float",
        float_precision="f16",
        _explicit_overrides={"project_root", "suite", "float_precision"},
    )

    cmd = GenerateStep(cfg)._build_cmd("cortex-m4", "float")

    assert "--suite" in cmd
    assert "float" in cmd
    assert "--float-precision" in cmd
    assert "f16" in cmd


def test_generate_step_routes_suite_both_by_cpu_capability(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(
        project_root=root,
        cpu="cortex-m0,cortex-m4,cortex-m55",
        suite="both",
        float_precision="f16",
        _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
    )

    commands = GenerateStep(cfg)._plan_details().commands
    rendered = [" ".join(cmd) for cmd in commands]

    assert len(commands) == 5
    assert any("--cpu cortex-m0" in cmd and "--suite int" in cmd for cmd in rendered)
    assert not any("--cpu cortex-m0" in cmd and "--suite float" in cmd for cmd in rendered)
    assert any(
        "--cpu cortex-m4" in cmd
        and "--suite float" in cmd
        and "--float-precision f32" in cmd
        for cmd in rendered
    )
    assert any(
        "--cpu cortex-m55" in cmd
        and "--suite float" in cmd
        and "--float-precision both" in cmd
        for cmd in rendered
    )


def test_build_step_emits_float_cmake_defines(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(
        project_root=root,
        suite="float",
        float_precision="both",
        _explicit_overrides={"project_root", "suite", "float_precision"},
    )

    cmd = BuildStep(cfg)._plan_details().commands[0]

    assert "--cmake-def" in cmd
    assert "ARM_NN_ENABLE_F32=ON" in cmd
    assert "ARM_NN_ENABLE_F16=ON" in cmd


def test_build_step_int_suite_disables_float_cmake_defines(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(project_root=root, suite="int", _explicit_overrides={"project_root", "suite"})

    cmd = BuildStep(cfg)._plan_details().commands[0]

    assert "ARM_NN_ENABLE_F32=OFF" in cmd
    assert "ARM_NN_ENABLE_F16=OFF" in cmd


def test_build_step_splits_float_commands_by_effective_precision(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(
        project_root=root,
        cpu="cortex-m0,cortex-m4,cortex-m55",
        suite="both",
        float_precision="f16",
        _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
    )

    commands = BuildStep(cfg)._plan_details().commands
    rendered = [" ".join(cmd) for cmd in commands]

    assert len(commands) == 3

    int_cmd = next(cmd for cmd in rendered if "--suite int" in cmd)
    assert "--cpu cortex-m0,cortex-m4,cortex-m55" in int_cmd
    assert "ARM_NN_ENABLE_F32=OFF" in int_cmd
    assert "ARM_NN_ENABLE_F16=OFF" in int_cmd

    float_cmds = [cmd for cmd in rendered if "--suite float" in cmd]
    assert len(float_cmds) == 2
    assert any(
        "--cpu cortex-m4" in cmd
        and "ARM_NN_ENABLE_F32=ON" in cmd
        and "ARM_NN_ENABLE_F16=OFF" in cmd
        for cmd in float_cmds
    )
    assert any(
        "--cpu cortex-m55" in cmd
        and "ARM_NN_ENABLE_F32=ON" in cmd
        and "ARM_NN_ENABLE_F16=ON" in cmd
        for cmd in float_cmds
    )


def test_build_step_execute_success_message_no_name_error(tmp_path: Path, monkeypatch) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(
        project_root=root,
        cpu="cortex-m0,cortex-m4,cortex-m55",
        suite="both",
        float_precision="f16",
        _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
    )

    def fake_run_command(cmd, cwd, verbosity):
        return None

    monkeypatch.setattr("helia_core_tester.core.steps.build.run_command", fake_run_command)

    result = BuildStep(cfg)._do_execute()

    assert result.success
    assert "cpus=cortex-m0,cortex-m4,cortex-m55" in result.message
    assert len(result.details["commands"]) == 3
