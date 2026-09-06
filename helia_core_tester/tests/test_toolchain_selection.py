from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.core import runtime_env as runtime_env_module
from helia_core_tester.core import path_layout
from helia_core_tester.core.config import Config
from helia_core_tester.core.discovery import find_build_dir
from helia_core_tester.core.errors import ConfigurationError
from helia_core_tester.core.runtime_env import RuntimeEnvContext, bootstrap_runtime_env, build_locked_fvp_flags
from helia_core_tester.core.steps import BuildStep, RunStep


def _init_repo_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _config(root: Path, **overrides) -> Config:
    return Config(project_root=root, _explicit_overrides={"project_root", *overrides}, **overrides)


def _runtime_env(root: Path, compiler_tag: str = "gcc") -> RuntimeEnvContext:
    dl = root / "artifacts" / "downloads"
    return RuntimeEnvContext(
        downloads_dir=dl,
        ethos_path=dl / "ethos-u-core-platform",
        cmsis5_path=dl / "CMSIS_5",
        toolchain_file=dl / "ethos-u-core-platform" / "cmake" / "toolchain" / "x.cmake",
        compiler_tag=compiler_tag,
        fvp_exe=dl / "corstone300_download" / "models" / "Linux64_GCC-9.3" / "FVP_Corstone_SSE-300_Ethos-U55",
        child_env={"PATH": os.environ.get("PATH", "")},
    )


# --- mapping helper -----------------------------------------------------------


def test_compiler_tag_mapping() -> None:
    assert path_layout.compiler_tag_for_toolchain("gcc") == "gcc"
    assert path_layout.compiler_tag_for_toolchain("armclang") == "arm-compiler"
    assert path_layout.compiler_tag_for_toolchain(" ArmClang ") == "arm-compiler"
    with pytest.raises(ValueError, match="Unsupported toolchain"):
        path_layout.compiler_tag_for_toolchain("atfe")


def test_build_dir_helper_for_armclang(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert find_build_dir("cortex-m55", repo_root=root, suite="int", toolchain="armclang") == (
        root / "artifacts" / "build-int-cortex-m55-arm-compiler"
    )
    assert find_build_dir("cortex-m55", repo_root=root, suite="int") == root / "artifacts" / "build-int-cortex-m55-gcc"


# --- locked flags / bootstrap -------------------------------------------------


def test_locked_flags_emit_use_arm_compiler_only_for_armclang(tmp_path: Path) -> None:
    gcc_flags = build_locked_fvp_flags(None, tmp_path)
    default_flags = build_locked_fvp_flags(None, tmp_path, "gcc")
    armclang_flags = build_locked_fvp_flags(None, tmp_path, "armclang")

    assert gcc_flags == default_flags
    assert "--use-arm-compiler" not in gcc_flags
    assert "--no-gcc-from-download" in gcc_flags
    assert armclang_flags == [*gcc_flags, "--use-arm-compiler"]


def test_bootstrap_runtime_env_passes_use_arm_compiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list = []

    def fake_detect_paths(args):
        seen.append(args)
        return {
            "env": {"PATH": ""},
            "dl": tmp_path,
            "ethos": tmp_path / "ethos",
            "cmsis5": tmp_path / "cmsis5",
            "toolchain_file": tmp_path / "armclang.cmake",
            "compiler_tag": "arm-compiler" if args.use_arm_compiler else "gcc",
            "fvp_exe": tmp_path / "fvp",
        }

    def fail_setup(_dl):
        raise AssertionError("ensure_setup=False must not call setup_dependencies")

    monkeypatch.setattr(runtime_env_module, "detect_paths", fake_detect_paths)
    monkeypatch.setattr(runtime_env_module, "call_setup_dependencies", fail_setup)

    ctx = bootstrap_runtime_env(downloads_dir=tmp_path, ensure_setup=False, toolchain="armclang")
    assert seen[-1].use_arm_compiler is True
    assert ctx.compiler_tag == "arm-compiler"

    ctx = bootstrap_runtime_env(downloads_dir=tmp_path, ensure_setup=False)
    assert seen[-1].use_arm_compiler is False
    assert ctx.compiler_tag == "gcc"

    with pytest.raises(ValueError, match="Unsupported toolchain"):
        bootstrap_runtime_env(downloads_dir=tmp_path, ensure_setup=False, toolchain="atfe")


# --- Config -------------------------------------------------------------------


def test_config_defaults_to_gcc(tmp_path: Path) -> None:
    cfg = _config(_init_repo_root(tmp_path))
    assert cfg.toolchain == "gcc"
    assert cfg.compiler_tag == "gcc"
    assert cfg.to_dict()["toolchain"] == "gcc"
    assert cfg.build_dir_for("cortex-m55", suite="int").name == "build-int-cortex-m55-gcc"


def test_config_armclang_build_dir(tmp_path: Path) -> None:
    cfg = _config(_init_repo_root(tmp_path), toolchain="armclang")
    assert cfg.compiler_tag == "arm-compiler"
    assert cfg.build_dir_for("cortex-m55", suite="int").name == "build-int-cortex-m55-arm-compiler"


def test_config_rejects_unknown_toolchain(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Invalid toolchain"):
        _config(_init_repo_root(tmp_path), toolchain="atfe")


def test_config_rejects_coverage_with_armclang(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--coverage is only supported with the gcc toolchain"):
        _config(_init_repo_root(tmp_path), toolchain="armclang", coverage=True)


def test_config_toolchain_from_toml_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    (root / "helia_core_tester.toml").write_text('[helia_core_tester]\ntoolchain = "armclang"\n')
    monkeypatch.delenv("HELIA_CORE_TESTER_TOOLCHAIN", raising=False)
    assert _config(root).toolchain == "armclang"

    monkeypatch.setenv("HELIA_CORE_TESTER_TOOLCHAIN", "gcc")
    assert _config(root).toolchain == "gcc"

    assert _config(root, toolchain="armclang").toolchain == "armclang"


# --- steps --------------------------------------------------------------------


@pytest.mark.parametrize("step_cls", [BuildStep, RunStep])
def test_step_commands_carry_use_arm_compiler_only_for_armclang(tmp_path: Path, step_cls) -> None:
    root = _init_repo_root(tmp_path)

    gcc_cmds = step_cls(_config(root), runtime_env=_runtime_env(root))._plan_details().commands
    armclang_cmds = step_cls(
        _config(root, toolchain="armclang"), runtime_env=_runtime_env(root, "arm-compiler")
    )._plan_details().commands

    assert gcc_cmds and armclang_cmds
    assert all("--use-arm-compiler" not in cmd for cmd in gcc_cmds)
    assert all("--use-arm-compiler" in cmd for cmd in armclang_cmds)
    assert all("--no-gcc-from-download" in cmd for cmd in armclang_cmds)


def test_run_step_validate_looks_for_armclang_build_dir(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    (root / "helia_core_tester" / "fvp").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp" / "build_and_run_fvp.py").write_text("")
    (root / "artifacts" / "build-int-cortex-m55-gcc").mkdir(parents=True)

    assert RunStep(_config(root)).validate() is None
    error = RunStep(_config(root, toolchain="armclang")).validate()
    assert error is not None and "build-int-cortex-m55-arm-compiler" in error


def test_steps_bootstrap_with_config_toolchain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    seen: list[str] = []

    def fake_bootstrap(*, downloads_dir, ensure_setup, toolchain="gcc"):
        seen.append(toolchain)
        return _runtime_env(root, "arm-compiler")

    for module in ("helia_core_tester.core.steps.build", "helia_core_tester.core.steps.run", "helia_core_tester.core.pipeline"):
        monkeypatch.setattr(f"{module}.bootstrap_runtime_env", fake_bootstrap)

    cfg = _config(root, toolchain="armclang")
    BuildStep(cfg)._ensure_runtime_env()
    RunStep(cfg)._ensure_runtime_env()
    from helia_core_tester.core.pipeline import FullTestPipeline

    FullTestPipeline(cfg)._ensure_runtime_env()
    assert seen == ["armclang", "armclang", "armclang"]


# --- CLI ----------------------------------------------------------------------


def _cli_text(result) -> str:
    return "".join(getattr(result, attr, "") or "" for attr in ("output", "stdout", "stderr"))


def test_cli_toolchain_reaches_config_for_full(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from helia_core_tester import cli as cli_module

    root = _init_repo_root(tmp_path)
    captured: list[Config] = []

    class FakePipeline:
        def __init__(self, config):
            captured.append(config)

        def print_plan(self):
            pass

    monkeypatch.setattr(cli_module, "FullTestPipeline", FakePipeline)
    result = CliRunner().invoke(cli_module.app, ["full", "--plan", "--toolchain", "armclang", "--repo-root", str(root)])
    assert result.exit_code == 0, _cli_text(result)
    assert captured[0].toolchain == "armclang"
    assert "toolchain" in captured[0]._explicit_overrides

    captured.clear()
    result = CliRunner().invoke(cli_module.app, ["full", "--plan", "--repo-root", str(root)])
    assert result.exit_code == 0, _cli_text(result)
    assert captured[0].toolchain == "gcc"


# `full` prints its plan via the logger, not typer.echo, so it is covered by the
# FakePipeline test above instead.
@pytest.mark.parametrize("command", ["build", "run"])
def test_cli_plan_output_shows_use_arm_compiler(tmp_path: Path, command: str) -> None:
    from helia_core_tester.cli import app

    root = _init_repo_root(tmp_path)
    (root / "helia_core_tester" / "fvp").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp" / "build_and_run_fvp.py").write_text("")
    args = [command, "--plan", "--toolchain", "armclang", "--repo-root", str(root)]

    result = CliRunner().invoke(app, args)
    text = _cli_text(result)
    assert result.exit_code == 0, text
    assert "--use-arm-compiler" in text


@pytest.mark.parametrize("command", ["build", "run", "full"])
def test_cli_rejects_coverage_with_armclang(tmp_path: Path, command: str) -> None:
    from helia_core_tester.cli import app

    root = _init_repo_root(tmp_path)
    result = CliRunner().invoke(app, [command, "--plan", "--toolchain", "armclang", "--coverage", "--repo-root", str(root)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ConfigurationError) or "only supported with the gcc toolchain" in _cli_text(result)
