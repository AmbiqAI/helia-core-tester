from __future__ import annotations

from pathlib import Path

from helia_core_tester.core.config import Config
from helia_core_tester.core.steps.build import BuildStep
from helia_core_tester.core.steps.generate import GenerateStep


def _init_repo_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_generate_step_propagates_include_float_flag(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(project_root=root, include_float=True, _explicit_overrides={"project_root", "include_float"})

    cmd = GenerateStep(cfg)._build_cmd("cortex-m4")

    assert "--include-float" in cmd


def test_build_step_emits_float_cmake_define(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(project_root=root, include_float=True, _explicit_overrides={"project_root", "include_float"})

    cmd = BuildStep(cfg)._plan_details().commands[0]

    assert "--cmake-def" in cmd
    assert "ARM_NN_ENABLE_F32=ON" in cmd
