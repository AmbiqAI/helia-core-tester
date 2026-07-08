from __future__ import annotations

import os
from pathlib import Path

from helia_core_tester.core.config import Config
from helia_core_tester.core.pipeline import FullTestPipeline
from helia_core_tester.core.runtime_env import RuntimeEnvContext


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


def test_full_pipeline_bootstraps_runtime_env_once_and_reuses(monkeypatch, tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(
        project_root=root,
        skip_generation=True,
        _explicit_overrides={"project_root", "skip_generation"},
    )

    expected_runtime_env = _runtime_env(root)
    bootstrap_calls = {"count": 0}
    step_runtime_env_ids = []

    def fake_bootstrap_runtime_env(*, downloads_dir, ensure_setup):
        bootstrap_calls["count"] += 1
        assert ensure_setup is True
        assert Path(downloads_dir) == cfg.downloads_dir
        return expected_runtime_env

    def fake_run_step(step, logger, verbosity, fail_fast):
        step_runtime_env_ids.append(id(step.runtime_env))
        return True, False

    monkeypatch.setattr(
        "helia_core_tester.core.pipeline.bootstrap_runtime_env",
        fake_bootstrap_runtime_env,
    )
    monkeypatch.setattr("helia_core_tester.core.pipeline._run_step", fake_run_step)

    ok = FullTestPipeline(cfg).run()

    assert ok is True
    assert bootstrap_calls["count"] == 1
    assert len(step_runtime_env_ids) == 2
    assert len(set(step_runtime_env_ids)) == 1
