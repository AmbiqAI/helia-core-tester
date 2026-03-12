"""
Configuration management for Helia Core Tester.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from helia_core_tester.core.cpu_targets import parse_cpu_list
from helia_core_tester.core.discovery import find_repo_root
from helia_core_tester.core.errors import ConfigurationError, PathNotFoundError
from helia_core_tester.core.path_layout import (
    artifacts_root,
    build_dir,
    coverage_merged_dir,
    coverage_report_dir,
    generated_tests_dir,
    generated_tests_root,
    generation_report_dir,
    reports_root,
    tests_report_dir,
)

ENV_PREFIX = "HELIA_CORE_TESTER_"
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

PATH_KEYS = frozenset(
    {
        "project_root",
        "config_file",
        "downloads_dir",
        "generation_dir",
        "generated_tests_root",
        "reports_root",
    }
)


@dataclass
class Config:
    """Configuration class for helia-core-tester."""

    project_root: Optional[Path] = None
    config_file: Optional[Path] = None
    downloads_dir: Optional[Path] = None
    generation_dir: Optional[Path] = None
    generated_tests_root: Optional[Path] = None
    reports_root: Optional[Path] = None

    cpu: str = "cortex-m55"
    cpus: list[str] = field(default_factory=list)
    optimization: str = "-Ofast"
    jobs: Optional[int] = None
    coverage: bool = False

    timeout: float = 0.0
    fail_fast: bool = True
    run_jobs: int = 1
    verbosity: int = 0
    dry_run: bool = False
    plan: bool = False

    op_filter: Optional[str] = None
    dtype_filter: Optional[str] = None
    name_filter: Optional[str] = None
    limit: Optional[int] = None
    seed: Optional[int] = 500

    skip_generation: bool = False
    skip_build: bool = False
    skip_run: bool = False

    enable_reporting: bool = True
    report_formats: list[str] = field(default_factory=lambda: ["json"])

    _explicit_overrides: set[str] = field(default_factory=set, repr=False, compare=False)
    _frozen: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise AttributeError("Config is immutable after initialization")
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        self._resolve_project_root()
        self._load_config_file_defaults()
        self._apply_env_overrides()
        self._resolve_paths()
        self._validate_and_finalize()
        object.__setattr__(self, "_frozen", True)

    def _resolve_project_root(self) -> None:
        if self.project_root is None:
            try:
                self.project_root = find_repo_root()
            except Exception as e:  # pragma: no cover - surfaced with clear message
                raise ConfigurationError(
                    f"Could not discover repository root: {e}. "
                    "Set project_root explicitly or set CMSIS_NN_REPO_ROOT."
                ) from e
        else:
            self.project_root = Path(self.project_root).resolve()

    def _field_default(self, key: str) -> Any:
        for f in fields(self):
            if f.name != key:
                continue
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:  # type: ignore[attr-defined]
                return f.default_factory()  # type: ignore[misc]
            return MISSING
        return MISSING

    def _load_config_file_defaults(self) -> None:
        env_path = os.environ.get("HELIA_CORE_TESTER_CONFIG")
        if env_path:
            self.config_file = Path(env_path).resolve()
        elif self.config_file is None:
            self.config_file = self.project_root / "helia_core_tester.toml"

        if not self.config_file or not self.config_file.exists():
            return

        try:
            try:
                import tomllib  # Python 3.11+
            except ModuleNotFoundError:  # pragma: no cover - py3.8 fallback
                import tomli as tomllib

            data = tomllib.loads(self.config_file.read_text())
            table = data.get("helia_core_tester") or data.get("tool", {}).get("helia_core_tester", {})
            if not isinstance(table, dict):
                return
        except Exception as e:
            raise ConfigurationError(f"Failed to read config file: {self.config_file}: {e}") from e

        explicit = set(self._explicit_overrides or set())
        for key, value in table.items():
            if value is None or not hasattr(self, key):
                continue
            if key in explicit:
                continue
            default_value = self._field_default(key)
            current_value = getattr(self, key)
            if default_value is MISSING or current_value == default_value:
                setattr(self, key, value)

    def _resolve_paths(self) -> None:
        if self.config_file is not None:
            self.config_file = Path(self.config_file).resolve()

        if self.downloads_dir is None:
            self.downloads_dir = artifacts_root(self.project_root) / "downloads"
        else:
            self.downloads_dir = Path(self.downloads_dir).resolve()

        if self.generation_dir is None:
            self.generation_dir = self.project_root / "helia_core_tester" / "generation"
        else:
            self.generation_dir = Path(self.generation_dir).resolve()

        if self.generated_tests_root is None:
            self.generated_tests_root = generated_tests_root(self.project_root)
        else:
            self.generated_tests_root = Path(self.generated_tests_root).resolve()

        if self.reports_root is None:
            self.reports_root = reports_root(self.project_root)
        else:
            self.reports_root = Path(self.reports_root).resolve()

    def _parse_bool(self, key: str, value: str) -> bool:
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
        raise ConfigurationError(
            f"Invalid boolean for {ENV_PREFIX}{key.upper()}: {value!r} "
            "(expected one of: true/false, 1/0, yes/no, on/off)"
        )

    def _parse_env_value(self, key: str, value: str) -> Any:
        if key in PATH_KEYS:
            return Path(value)
        if key in {"jobs", "run_jobs", "limit", "seed", "verbosity"}:
            return int(value)
        if key == "timeout":
            return float(value)
        if key in {
            "coverage",
            "fail_fast",
            "dry_run",
            "plan",
            "skip_generation",
            "skip_build",
            "skip_run",
            "enable_reporting",
        }:
            return self._parse_bool(key, value)
        if key == "report_formats":
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def _apply_env_overrides(self) -> None:
        explicit = set(self._explicit_overrides or set())
        for key in self.__dataclass_fields__.keys():
            if key in {"_explicit_overrides", "_frozen", "cpus", "config_file"}:
                continue
            if key in explicit:
                continue
            env_key = f"{ENV_PREFIX}{key.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            try:
                parsed = self._parse_env_value(key, raw)
            except Exception as e:
                raise ConfigurationError(f"Invalid env override {env_key}={raw!r}: {e}") from e
            setattr(self, key, parsed)

    def _validate_and_finalize(self) -> None:
        if not self.project_root.exists():
            raise PathNotFoundError(f"Project root does not exist: {self.project_root}")

        if not self.generation_dir.exists():
            raise PathNotFoundError(f"Generation directory does not exist: {self.generation_dir}")

        try:
            self.cpus = parse_cpu_list(self.cpu)
            self.cpu = self.cpus[0]
        except ValueError as e:
            raise ConfigurationError(str(e)) from e

        if not 0 <= self.verbosity <= 3:
            raise ValueError(f"verbosity must be between 0 and 3, got {self.verbosity}")

        if self.jobs is None:
            self.jobs = os.cpu_count() or 4

        if self.run_jobs is None:
            self.run_jobs = 1
        if self.run_jobs < 0:
            raise ValueError(f"run_jobs must be >= 0, got {self.run_jobs}")
        if self.run_jobs == 0:
            self.run_jobs = os.cpu_count() or 4

        self.downloads_dir.parent.mkdir(parents=True, exist_ok=True)
        self.generated_tests_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

    def generated_tests_dir_for(self, cpu: str) -> Path:
        return generated_tests_dir(self.project_root, cpu)

    def generation_report_dir_for(self, cpu: str) -> Path:
        return generation_report_dir(self.project_root, cpu)

    def tests_report_dir_for(self, cpu: str) -> Path:
        return tests_report_dir(self.project_root, cpu)

    def coverage_report_dir_for(self, cpu: str) -> Path:
        return coverage_report_dir(self.project_root, cpu)

    def coverage_merged_report_dir(self) -> Path:
        return coverage_merged_dir(self.project_root)

    def build_dir_for(self, cpu: str) -> Path:
        return build_dir(self.project_root, cpu)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "config_file": str(self.config_file) if self.config_file else None,
            "downloads_dir": str(self.downloads_dir),
            "generation_dir": str(self.generation_dir),
            "generated_tests_root": str(self.generated_tests_root),
            "reports_root": str(self.reports_root),
            "cpu": self.cpu,
            "cpus": list(self.cpus),
            "optimization": self.optimization,
            "jobs": self.jobs,
            "coverage": self.coverage,
            "timeout": self.timeout,
            "fail_fast": self.fail_fast,
            "run_jobs": self.run_jobs,
            "verbosity": self.verbosity,
            "dry_run": self.dry_run,
            "plan": self.plan,
            "op_filter": self.op_filter,
            "dtype_filter": self.dtype_filter,
            "name_filter": self.name_filter,
            "limit": self.limit,
            "seed": self.seed,
            "skip_generation": self.skip_generation,
            "skip_build": self.skip_build,
            "skip_run": self.skip_run,
            "enable_reporting": self.enable_reporting,
            "report_formats": list(self.report_formats),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        payload = dict(data)
        for key in PATH_KEYS:
            if key in payload and isinstance(payload[key], str):
                payload[key] = Path(payload[key])
        return cls(**payload)
