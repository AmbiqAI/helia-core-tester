"""
Canonical artifact/report path layout for helia-core-tester.
"""

from __future__ import annotations

from pathlib import Path


VALID_SUITES = {"int", "float", "float-mve"}

# toolchain (CLI/config name) -> compiler tag used in build-<suite>-<cpu>-<tag>.
COMPILER_TAGS = {"gcc": "gcc", "armclang": "arm-compiler"}
VALID_TOOLCHAINS = frozenset(COMPILER_TAGS)


def normalize_suite(suite: str) -> str:
    normalized = str(suite).strip().lower()
    if normalized not in VALID_SUITES:
        raise ValueError(f"Unsupported suite: {suite}")
    return normalized


def normalize_toolchain(toolchain: str) -> str:
    normalized = str(toolchain).strip().lower()
    if normalized not in VALID_TOOLCHAINS:
        raise ValueError(f"Unsupported toolchain: {toolchain} (expected one of: {', '.join(sorted(VALID_TOOLCHAINS))})")
    return normalized


def compiler_tag_for_toolchain(toolchain: str) -> str:
    return COMPILER_TAGS[normalize_toolchain(toolchain)]


def _root(project_root: Path) -> Path:
    return Path(project_root).resolve()


def artifacts_root(project_root: Path) -> Path:
    return _root(project_root) / "artifacts"


def build_dir(project_root: Path, cpu: str, compiler_tag: str = "gcc", suite: str = "int") -> Path:
    suite_name = normalize_suite(suite)
    return artifacts_root(project_root) / f"build-{suite_name}-{cpu}-{compiler_tag}"


def generated_tests_root(project_root: Path) -> Path:
    return artifacts_root(project_root) / "generated_tests"


def generated_tests_dir(project_root: Path, cpu: str, suite: str = "int") -> Path:
    suite_name = normalize_suite(suite)
    return generated_tests_root(project_root) / suite_name / cpu


def generated_tests_family_dir(project_root: Path, cpu: str, family: str, suite: str = "int") -> Path:
    return generated_tests_dir(project_root, cpu, suite=suite) / family


def generated_test_case_dir(project_root: Path, cpu: str, family: str, test_name: str, suite: str = "int") -> Path:
    return generated_tests_family_dir(project_root, cpu, family, suite=suite) / test_name


def reports_root(project_root: Path) -> Path:
    return artifacts_root(project_root) / "reports"


def generation_report_dir(project_root: Path, cpu: str, suite: str = "int") -> Path:
    suite_name = normalize_suite(suite)
    return reports_root(project_root) / "generation" / suite_name / cpu


def tests_report_dir(project_root: Path, cpu: str, suite: str = "int") -> Path:
    suite_name = normalize_suite(suite)
    return reports_root(project_root) / "tests" / suite_name / cpu


def coverage_root(project_root: Path) -> Path:
    return reports_root(project_root) / "coverage"


def coverage_report_dir(project_root: Path, cpu: str, suite: str = "int") -> Path:
    suite_name = normalize_suite(suite)
    return coverage_root(project_root) / suite_name / cpu


def coverage_merged_dir(project_root: Path) -> Path:
    return coverage_root(project_root) / "merged"


def build_tests_dir(project_root: Path, cpu: str, compiler_tag: str = "gcc", suite: str = "int") -> Path:
    return build_dir(project_root, cpu, compiler_tag, suite=suite) / "tests"


def build_tests_family_dir(
    project_root: Path,
    cpu: str,
    family: str,
    compiler_tag: str = "gcc",
    suite: str = "int",
) -> Path:
    return build_tests_dir(project_root, cpu, compiler_tag, suite=suite) / family


def build_test_elf_path(
    project_root: Path,
    cpu: str,
    family: str,
    test_name: str,
    compiler_tag: str = "gcc",
    suite: str = "int",
) -> Path:
    return build_tests_family_dir(project_root, cpu, family, compiler_tag, suite=suite) / f"{test_name}.elf"
