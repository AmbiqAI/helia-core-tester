from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.core.errors import ConfigurationError


def _init_repo_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_config_precedence_defaults_then_toml_then_env_then_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    (root / "helia_core_tester.toml").write_text(
        """
[helia_core_tester]
cpu = "cortex-m4"
verbosity = 1
coverage = false
report_formats = ["json"]
""".strip()
    )

    monkeypatch.setenv("HELIA_CORE_TESTER_CPU", "cortex-m0")
    monkeypatch.setenv("HELIA_CORE_TESTER_VERBOSITY", "2")
    monkeypatch.setenv("HELIA_CORE_TESTER_COVERAGE", "true")
    monkeypatch.setenv("HELIA_CORE_TESTER_REPORT_FORMATS", "json,junit")

    cfg = Config(
        project_root=root,
        cpu="cortex-m55",
        verbosity=3,
        _explicit_overrides={"project_root", "cpu", "verbosity"},
    )

    assert cfg.cpu == "cortex-m55"  # CLI override beats env+toml
    assert cfg.verbosity == 3  # CLI override beats env+toml
    assert cfg.coverage is True  # env beats toml
    assert cfg.report_formats == ["json", "junit"]  # env beats toml


def test_config_is_immutable_after_init(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(project_root=root)

    with pytest.raises(AttributeError, match="immutable"):
        cfg.cpu = "cortex-m4"


def test_invalid_env_bool_raises_configuration_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    monkeypatch.setenv("HELIA_CORE_TESTER_COVERAGE", "maybe")

    with pytest.raises(ConfigurationError, match="Invalid env override HELIA_CORE_TESTER_COVERAGE"):
        Config(project_root=root)
