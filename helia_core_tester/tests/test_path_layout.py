from __future__ import annotations

from pathlib import Path

from helia_core_tester.core.discovery import find_build_dir
from helia_core_tester.core.config import Config
from helia_core_tester.core import path_layout as layout


def _init_repo_root(tmp_path: Path) -> Path:
    (tmp_path / "helia_core_tester" / "generation").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_path_layout_matches_config_helpers(tmp_path: Path) -> None:
    root = _init_repo_root(tmp_path)
    cfg = Config(project_root=root)

    assert layout.artifacts_root(root) == root / "artifacts"
    assert layout.generated_tests_root(root) == root / "artifacts" / "generated_tests"
    assert layout.reports_root(root) == root / "artifacts" / "reports"

    cpu = "cortex-m55"
    assert cfg.generated_tests_dir_for(cpu) == layout.generated_tests_dir(root, cpu)
    assert layout.generated_tests_family_dir(root, cpu, "PoolingFunctions") == root / "artifacts" / "generated_tests" / cpu / "PoolingFunctions"
    assert layout.generated_test_case_dir(root, cpu, "PoolingFunctions", "avg_pool_case_s8") == root / "artifacts" / "generated_tests" / cpu / "PoolingFunctions" / "avg_pool_case_s8"
    assert cfg.generation_report_dir_for(cpu) == layout.generation_report_dir(root, cpu)
    assert cfg.tests_report_dir_for(cpu) == layout.tests_report_dir(root, cpu)
    assert cfg.coverage_report_dir_for(cpu) == layout.coverage_report_dir(root, cpu)
    assert cfg.coverage_merged_report_dir() == layout.coverage_merged_dir(root)
    assert layout.build_tests_family_dir(root, cpu, "PoolingFunctions") == root / "artifacts" / f"build-{cpu}-gcc" / "tests" / "PoolingFunctions"
    assert layout.build_test_elf_path(root, cpu, "PoolingFunctions", "avg_pool_case_s8") == root / "artifacts" / f"build-{cpu}-gcc" / "tests" / "PoolingFunctions" / "avg_pool_case_s8.elf"


def test_find_build_dir_uses_canonical_layout(tmp_path: Path) -> None:
    cpu = "cortex-m55"
    resolved = find_build_dir(cpu=cpu, repo_root=tmp_path)
    assert resolved == layout.build_dir(tmp_path, cpu, compiler_tag="gcc")
