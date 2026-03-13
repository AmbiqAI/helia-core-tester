"""
Canonical artifact/report path layout for helia-core-tester.
"""

from __future__ import annotations

from pathlib import Path


def _root(project_root: Path) -> Path:
    return Path(project_root).resolve()


def artifacts_root(project_root: Path) -> Path:
    return _root(project_root) / "artifacts"


def build_dir(project_root: Path, cpu: str, compiler_tag: str = "gcc") -> Path:
    return artifacts_root(project_root) / f"build-{cpu}-{compiler_tag}"


def generated_tests_root(project_root: Path) -> Path:
    return artifacts_root(project_root) / "generated_tests"


def generated_tests_dir(project_root: Path, cpu: str) -> Path:
    return generated_tests_root(project_root) / cpu


def generated_tests_family_dir(project_root: Path, cpu: str, family: str) -> Path:
    return generated_tests_dir(project_root, cpu) / family


def generated_test_case_dir(project_root: Path, cpu: str, family: str, test_name: str) -> Path:
    return generated_tests_family_dir(project_root, cpu, family) / test_name


def reports_root(project_root: Path) -> Path:
    return artifacts_root(project_root) / "reports"


def generation_report_dir(project_root: Path, cpu: str) -> Path:
    return reports_root(project_root) / "generation" / cpu


def tests_report_dir(project_root: Path, cpu: str) -> Path:
    return reports_root(project_root) / "tests" / cpu


def coverage_root(project_root: Path) -> Path:
    return reports_root(project_root) / "coverage"


def coverage_report_dir(project_root: Path, cpu: str) -> Path:
    return coverage_root(project_root) / cpu


def coverage_merged_dir(project_root: Path) -> Path:
    return coverage_root(project_root) / "merged"


def build_tests_dir(project_root: Path, cpu: str, compiler_tag: str = "gcc") -> Path:
    return build_dir(project_root, cpu, compiler_tag) / "tests"


def build_tests_family_dir(project_root: Path, cpu: str, family: str, compiler_tag: str = "gcc") -> Path:
    return build_tests_dir(project_root, cpu, compiler_tag) / family


def build_test_elf_path(
    project_root: Path,
    cpu: str,
    family: str,
    test_name: str,
    compiler_tag: str = "gcc",
) -> Path:
    return build_tests_family_dir(project_root, cpu, family, compiler_tag) / f"{test_name}.elf"
