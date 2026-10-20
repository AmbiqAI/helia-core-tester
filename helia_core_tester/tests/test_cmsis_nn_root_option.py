"""--cmsis-nn-root on generate/build/full, its env spelling, and the fail-early check (#117)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester import cli as cli_module
from helia_core_tester.core.config import Config
from helia_core_tester.core.steps import BuildStep, GenerateStep
from helia_core_tester.core.steps.base import StepStatus

ENV_NAME = "HELIA_CORE_TESTER_CMSIS_NN_ROOT"


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "tester"
    (root / "helia_core_tester" / "generation").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp" / "build_and_run_fvp.py").write_text("")
    return root


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "ns-cmsis-nn"
    root.mkdir()
    (root / "CMakeLists.txt").write_text("project(cmsis-nn)\n")
    return root


def _config(root: Path, **overrides) -> Config:
    return Config(project_root=root, _explicit_overrides={"project_root", *overrides}, **overrides)


def _text(result) -> str:
    return "".join(getattr(result, attr, "") or "" for attr in ("output", "stdout", "stderr"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    monkeypatch.delenv("CMSIS_NN_ROOT", raising=False)


@pytest.fixture
def captured(monkeypatch) -> list[Config]:
    seen: list[Config] = []

    class _FakeStep:
        def __init__(self, config, *args, **kwargs):
            seen.append(config)

        def plan(self):
            return None

    class _FakePipeline:
        def __init__(self, config):
            seen.append(config)

        def print_plan(self):
            pass

    monkeypatch.setattr(cli_module, "GenerateStep", _FakeStep)
    monkeypatch.setattr(cli_module, "BuildStep", _FakeStep)
    monkeypatch.setattr(cli_module, "FullTestPipeline", _FakePipeline)
    monkeypatch.setattr(cli_module, "_print_plan_item", lambda item: None)
    return seen


@pytest.mark.parametrize("command", ["generate", "build", "full"])
def test_option_reaches_config(tmp_path, captured, command) -> None:
    repo, checkout = _repo(tmp_path), _checkout(tmp_path)
    result = CliRunner().invoke(
        cli_module.app, [command, "--plan", "--cmsis-nn-root", str(checkout), "--repo-root", str(repo)]
    )
    assert result.exit_code == 0, _text(result)
    assert captured[0].cmsis_nn_root == checkout.resolve()
    assert "cmsis_nn_root" in captured[0]._explicit_overrides


@pytest.mark.parametrize("command", ["generate", "build", "full"])
def test_unset_option_leaves_the_sibling_default(tmp_path, captured, command) -> None:
    result = CliRunner().invoke(cli_module.app, [command, "--plan", "--repo-root", str(_repo(tmp_path))])
    assert result.exit_code == 0, _text(result)
    assert captured[0].cmsis_nn_root is None


@pytest.mark.parametrize("command", ["generate", "build"])
def test_prefixed_env_var_reaches_config(tmp_path, captured, monkeypatch, command) -> None:
    repo, checkout = _repo(tmp_path), _checkout(tmp_path)
    monkeypatch.setenv(ENV_NAME, str(checkout))
    result = CliRunner().invoke(cli_module.app, [command, "--plan", "--repo-root", str(repo)])
    assert result.exit_code == 0, _text(result)
    assert captured[0].cmsis_nn_root == checkout.resolve()


def test_option_wins_over_env_var(tmp_path, captured, monkeypatch) -> None:
    repo, checkout = _repo(tmp_path), _checkout(tmp_path)
    monkeypatch.setenv(ENV_NAME, str(tmp_path / "elsewhere"))
    result = CliRunner().invoke(
        cli_module.app, ["build", "--plan", "--cmsis-nn-root", str(checkout), "--repo-root", str(repo)]
    )
    assert result.exit_code == 0, _text(result)
    assert captured[0].cmsis_nn_root == checkout.resolve()


def test_bare_cmsis_nn_root_env_is_not_read(tmp_path, captured, monkeypatch) -> None:
    monkeypatch.setenv("CMSIS_NN_ROOT", str(_checkout(tmp_path)))
    result = CliRunner().invoke(cli_module.app, ["build", "--plan", "--repo-root", str(_repo(tmp_path))])
    assert result.exit_code == 0, _text(result)
    assert captured[0].cmsis_nn_root is None


def test_root_error_none_when_unset_or_valid(tmp_path) -> None:
    repo = _repo(tmp_path)
    assert _config(repo).cmsis_nn_root_error() is None
    assert _config(repo, cmsis_nn_root=_checkout(tmp_path)).cmsis_nn_root_error() is None


@pytest.mark.parametrize("layout", ["missing", "empty_dir", "cmakelists_is_dir", "file_not_dir"])
def test_root_error_names_option_and_env(tmp_path, layout) -> None:
    repo = _repo(tmp_path)
    bad = tmp_path / "bad"
    if layout == "empty_dir":
        bad.mkdir()
    elif layout == "cmakelists_is_dir":
        (bad / "CMakeLists.txt").mkdir(parents=True)
    elif layout == "file_not_dir":
        bad.write_text("x")
    error = _config(repo, cmsis_nn_root=bad).cmsis_nn_root_error()
    assert error is not None
    assert str(bad.resolve()) in error
    assert "--cmsis-nn-root" in error and ENV_NAME in error


def test_build_fails_before_cmake_on_a_bad_root(tmp_path, monkeypatch) -> None:
    from helia_core_tester.core.steps import build as build_module

    calls: list = []
    monkeypatch.setattr(build_module, "run_command", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(build_module, "bootstrap_runtime_env", lambda **k: pytest.fail("must not bootstrap"))
    step = BuildStep(_config(_repo(tmp_path), cmsis_nn_root=tmp_path / "missing"))
    result = step.execute()
    assert result.status == StepStatus.FAILED
    assert ENV_NAME in result.message
    assert calls == []


def test_generate_fails_validation_on_a_bad_root(tmp_path) -> None:
    step = GenerateStep(_config(_repo(tmp_path), cmsis_nn_root=tmp_path / "missing"))
    error = step.validate()
    assert error is not None and "--cmsis-nn-root" in error


def test_build_passes_a_valid_root_to_cmake(tmp_path) -> None:
    checkout = _checkout(tmp_path)
    step = BuildStep(_config(_repo(tmp_path), cmsis_nn_root=checkout))
    assert step.validate() is None
    for cmd in step._build_commands():
        assert f"CMSIS_NN_ROOT={checkout.resolve()}" in cmd


def test_cmake_error_names_the_option_and_env_var() -> None:
    text = (Path(cli_module.__file__).resolve().parents[1] / "CMakeLists.txt").read_text()
    block = text[text.index('message(FATAL_ERROR\n    "cmsis-nn CMakeLists.txt not found'):]
    block = block[: block.index(")\n")]
    assert "--cmsis-nn-root" in block and ENV_NAME in block


def test_tilde_root_is_expanded(tmp_path, monkeypatch) -> None:
    checkout = _checkout(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _config(_repo(tmp_path), cmsis_nn_root=Path("~") / checkout.name)
    assert config.cmsis_nn_root == checkout.resolve()
    assert config.cmsis_nn_root_error() is None


def test_tilde_root_from_env_is_expanded(tmp_path, monkeypatch) -> None:
    checkout = _checkout(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV_NAME, f"~/{checkout.name}")
    config = Config(project_root=_repo(tmp_path), _explicit_overrides={"project_root"})
    assert config.cmsis_nn_root == checkout.resolve()


def test_tilde_root_from_toml_is_expanded(tmp_path, monkeypatch) -> None:
    checkout = _checkout(tmp_path)
    repo = _repo(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HELIA_CORE_TESTER_CONFIG", raising=False)
    (repo / "helia_core_tester.toml").write_text(f'[helia_core_tester]\ncmsis_nn_root = "~/{checkout.name}"\n')
    config = Config(project_root=repo, _explicit_overrides={"project_root"})
    assert config.cmsis_nn_root == checkout.resolve()
    assert config.cmsis_nn_root_error() is None


def test_error_names_the_flag_as_the_source(tmp_path) -> None:
    error = _config(_repo(tmp_path), cmsis_nn_root=tmp_path / "missing").cmsis_nn_root_error()
    assert "(from --cmsis-nn-root)" in error


def test_error_names_the_env_var_as_the_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_NAME, str(tmp_path / "missing"))
    error = Config(project_root=_repo(tmp_path), _explicit_overrides={"project_root"}).cmsis_nn_root_error()
    assert f"(from {ENV_NAME})" in error


def test_error_names_the_toml_file_as_the_source(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.delenv("HELIA_CORE_TESTER_CONFIG", raising=False)
    toml = repo / "helia_core_tester.toml"
    toml.write_text(f'[helia_core_tester]\ncmsis_nn_root = "{tmp_path / "missing"}"\n')
    error = Config(project_root=repo, _explicit_overrides={"project_root"}).cmsis_nn_root_error()
    assert f"(from cmsis_nn_root in {toml.resolve()})" in error


def test_flag_source_wins_over_env_in_the_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_NAME, str(tmp_path / "env-missing"))
    error = _config(_repo(tmp_path), cmsis_nn_root=tmp_path / "flag-missing").cmsis_nn_root_error()
    assert "flag-missing" in error and "(from --cmsis-nn-root)" in error
