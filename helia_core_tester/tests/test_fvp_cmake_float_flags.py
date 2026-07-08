from __future__ import annotations

from pathlib import Path
import os

from helia_core_tester.core.config import Config
from helia_core_tester.core.runtime_env import RuntimeEnvContext
from helia_core_tester.core.steps.build import BuildStep
from helia_core_tester.core.steps.generate import GenerateStep


def _init_repo_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _runtime_env(root: Path) -> RuntimeEnvContext:
    return RuntimeEnvContext(
        downloads_dir=root / "artifacts" / "downloads",
        ethos_path=root / "artifacts" / "downloads" / "ethos-u-core-platform",
        cmsis5_path=root / "artifacts" / "downloads" / "CMSIS_5",
        toolchain_file=root / "artifacts" / "downloads" / "ethos-u-core-platform" / "cmake" / "toolchain" / "arm-none-eabi-gcc.cmake",
        compiler_tag="gcc",
        fvp_exe=root / "artifacts" / "downloads" / "corstone300_download" / "models" / "Linux64_GCC-9.3" / "FVP_Corstone_SSE-300_Ethos-U55",
        child_env={"PATH": os.environ.get("PATH", "")},
    )


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
    assert "--no-setup" in cmd
    assert "--no-gcc-from-download" in cmd
    assert "--no-fvp-from-download" in cmd
    assert "ARM_NN_ENABLE_F32=ON" in cmd
    assert "ARM_NN_ENABLE_F16=ON" in cmd
    assert "ENABLE_COVERAGE_MVE_FLOAT=ON" not in cmd


def test_build_step_emits_mve_float_coverage_define(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(
        project_root=root,
        cpu="cortex-m55",
        suite="float",
        float_precision="both",
        coverage=True,
        coverage_mve_float=True,
        _explicit_overrides={
            "project_root",
            "cpu",
            "suite",
            "float_precision",
            "coverage",
            "coverage_mve_float",
        },
    )

    cmd = BuildStep(cfg)._plan_details().commands[0]

    assert "ARM_NN_ENABLE_F32=ON" in cmd
    assert "ARM_NN_ENABLE_F16=ON" in cmd
    assert "ENABLE_COVERAGE_MVE_FLOAT=ON" in cmd


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

    def fake_run_command(cmd, cwd, verbosity, env=None):
        return None

    monkeypatch.setattr("helia_core_tester.core.steps.build.run_command", fake_run_command)

    result = BuildStep(cfg, runtime_env=_runtime_env(root))._do_execute()

    assert result.success
    assert "cpus=cortex-m0,cortex-m4,cortex-m55" in result.message
    assert len(result.details["commands"]) == 3
