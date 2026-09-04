from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from helia_core_tester.core import config as config_module
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


def test_suite_both_routes_float_by_cpu_capability(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    cfg = Config(
        project_root=root,
        cpu="cortex-m0,cortex-m4,cortex-m55",
        suite="both",
        float_precision="f16",
        _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
    )

    assert cfg.effective_suites_for_cpu("cortex-m0") == ["int", "float"]
    assert cfg.effective_suites_for_cpu("cortex-m4") == ["int", "float"]
    assert cfg.effective_suites_for_cpu("cortex-m55") == ["int", "float"]

    # m0 and m4 narrow to f32 for opposite reasons -- m4 has a single-precision FPU,
    # m0 has none and runs f32 through the soft-float ABI -- but neither has an f16 path,
    # so the requested f16 precision must not reach them.
    assert cfg.effective_float_precision_for_cpu("cortex-m0", suite="float") == "f32"
    assert cfg.effective_float_precision_for_cpu("cortex-m4", suite="float") == "f32"
    assert cfg.effective_float_precision_for_cpu("cortex-m55", suite="float") == "both"


def test_float_f16_rejected_for_cpus_without_fp16_execution(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    with pytest.raises(ConfigurationError, match="requires FP16 execution support"):
        Config(
            project_root=root,
            cpu="cortex-m4,cortex-m55",
            suite="float",
            float_precision="both",
            _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
        )


def test_float_f32_suite_accepts_the_soft_float_cpu(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    cfg = Config(
        project_root=root,
        cpu="cortex-m0,cortex-m55",
        suite="float",
        float_precision="f32",
        _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
    )

    assert cfg.cpus == ["cortex-m0", "cortex-m55"]


def test_float_suite_rejected_for_cpu_without_fp32_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shipped profile now has fp32_execution, so the guard needs a stand-in.

    Keeping it covered matters because the guard is what turns an unsupported
    (cpu, precision) pair into a configuration error instead of a build that emits
    kernels the target cannot execute.
    """
    root = _init_repo_root(tmp_path)
    real_profile = config_module.get_cpu_profile

    def _stripped(cpu: str):
        profile = real_profile(cpu)
        if profile.cpu == "cortex-m0":
            return replace(profile, capabilities=frozenset())
        return profile

    monkeypatch.setattr(config_module, "get_cpu_profile", _stripped)

    with pytest.raises(ConfigurationError, match="requires FP32 execution support"):
        Config(
            project_root=root,
            cpu="cortex-m0,cortex-m55",
            suite="float",
            float_precision="f32",
            _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
        )


def test_float_f16_allowed_for_fp16_capable_cpu(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    cfg = Config(
        project_root=root,
        cpu="cortex-m55",
        suite="float",
        float_precision="f16",
        _explicit_overrides={"project_root", "cpu", "suite", "float_precision"},
    )

    assert cfg.cpus == ["cortex-m55"]
    assert cfg.suite == "float"
    assert cfg.float_precision == "f16"


def test_coverage_mve_float_requires_coverage(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    with pytest.raises(ConfigurationError, match="requires --coverage"):
        Config(
            project_root=root,
            cpu="cortex-m55",
            suite="float",
            coverage_mve_float=True,
            _explicit_overrides={"project_root", "cpu", "suite", "coverage_mve_float"},
        )


def test_coverage_mve_float_requires_float_suite(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    with pytest.raises(ConfigurationError, match="requires --suite float"):
        Config(
            project_root=root,
            cpu="cortex-m55",
            suite="int",
            coverage=True,
            coverage_mve_float=True,
            _explicit_overrides={"project_root", "cpu", "suite", "coverage", "coverage_mve_float"},
        )


def test_coverage_mve_float_requires_cortex_m55(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    with pytest.raises(ConfigurationError, match="only supported for cortex-m55"):
        Config(
            project_root=root,
            cpu="cortex-m4",
            suite="float",
            float_precision="f32",
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


def test_coverage_mve_float_allowed_for_m55_float_coverage(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)

    cfg = Config(
        project_root=root,
        cpu="cortex-m55",
        suite="float",
        coverage=True,
        coverage_mve_float=True,
        _explicit_overrides={"project_root", "cpu", "suite", "coverage", "coverage_mve_float"},
    )

    assert cfg.coverage_mve_float is True
