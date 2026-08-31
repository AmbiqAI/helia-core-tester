"""Regression coverage for helia-core-tester issue #66 (reporting side).

``fvp/reporting.py::run_tests_with_reporting`` builds its descriptor-aware
report from two sources: the tests it actually ran (already correctly
filtered by ``find_elves(build_dir, active_test_list(...))``, see
``test_fvp_active_test_list.py``), and a fallback scan that adds any
descriptor whose ``.tflite``/header/``.elf`` still exists on disk -- so a
report could still count a descriptor outside the active test list purely
because a stale artifact was never pruned (e.g. ``--skip-build`` +
``--skip-generation`` reusing an older, wider build that was never
reconfigured this run, so ``cmake_configure``'s prune never ran). This test
pins that a descriptor outside the active list never re-enters the report
through that fallback scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import helia_core_tester.fvp.reporting as reporting_module
from helia_core_tester.reporting.models import TestResult, TestStatus


class _FakeTracker:
    """Stands in for DescriptorTracker: fixed descriptor set, no YAML I/O."""

    _DESCRIPTORS = {
        "add_s8_broadcast": {"name": "add_s8_broadcast", "_family": "BasicMathFunctions"},
        "conv_s8_1x1": {"name": "conv_s8_1x1", "_family": "ConvolutionFunctions"},
    }

    def __init__(self, descriptors_dir: Path) -> None:
        self.descriptors_dir = descriptors_dir

    def load_all_descriptors(self) -> Dict[str, Dict]:
        return dict(self._DESCRIPTORS)

    def map_test_to_descriptor(self, test_name: str, descriptors: Dict[str, Dict]) -> Optional[Dict]:
        return descriptors.get(test_name)

    def generated_test_dir_for(self, descriptor_name: str, generated_tests_dir: Path) -> Path:
        family = self._DESCRIPTORS[descriptor_name]["_family"]
        return generated_tests_dir / family / descriptor_name

    def elf_path_for(self, descriptor_name: str, build_dir: Path) -> Path:
        family = self._DESCRIPTORS[descriptor_name]["_family"]
        return build_dir / "tests" / family / f"{descriptor_name}.elf"

    def determine_descriptor_status(
        self, descriptor_name: str, test_result, build_dir: Path, generated_tests_dir: Path
    ):
        if test_result:
            return TestStatus.PASS, None, None
        return TestStatus.NOT_RUN, None, "Test not executed"

    def get_descriptor_path(self, descriptor_name: str) -> Path:
        return self.descriptors_dir / f"{descriptor_name}.yaml"


class _FakeReportGenerator:
    """Captures the built TestReport instead of writing report files."""

    captured: List = []

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(self, report, formats=None):
        _FakeReportGenerator.captured.append(report)
        return {}


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF")
    return path


def _fake_run_elf_jobs_with_reporting(elf_entries, cpu, **_kwargs) -> Tuple[List[TestResult], bool]:
    results = [
        TestResult(
            test_name=elf.stem,
            status=TestStatus.PASS,
            duration=0.01,
            cpu=cpu,
            elf_path=str(elf),
            descriptor_name=descriptor_name,
        )
        for elf, descriptor_name in elf_entries
    ]
    return results, False


def test_stale_build_artifact_outside_active_list_is_excluded_from_report(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(reporting_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(reporting_module, "find_descriptors_dir", lambda: tmp_path / "descriptors")
    monkeypatch.setattr(reporting_module, "DescriptorTracker", _FakeTracker)
    monkeypatch.setattr(reporting_module, "get_git_sha", lambda source_dir: "deadbeef")
    monkeypatch.setattr(reporting_module, "generate_coverage_reports", lambda *a, **k: None)
    monkeypatch.setattr(reporting_module, "run_elf_jobs_with_reporting", _fake_run_elf_jobs_with_reporting)
    monkeypatch.setattr(reporting_module, "ReportGenerator", _FakeReportGenerator)
    _FakeReportGenerator.captured = []

    source_dir = tmp_path / "src"
    build_dir = tmp_path / "artifacts" / "build-int-cortex-m55-gcc"
    generated_tests_dir = tmp_path / "src" / "artifacts" / "generated_tests" / "int" / "cortex-m55"

    # add_s8_broadcast is the only case the active filter admits this run.
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    (generated_tests_dir / "manifest.json").write_text(
        '{"tests": [{"name": "add_s8_broadcast"}]}'
    )

    # conv_s8_1x1's .elf is a leftover from an earlier, wider build that was
    # never reconfigured (hence never pruned) this run -- e.g. --skip-build.
    _touch(build_dir / "tests" / "BasicMathFunctions" / "add_s8_broadcast.elf")
    _touch(build_dir / "tests" / "ConvolutionFunctions" / "conv_s8_1x1.elf")

    class _Args:
        no_build = True
        no_run = False
        coverage = False
        opt = "-Ofast"
        cmake_def: List[str] = []
        generator = None
        downloads_dir = tmp_path / "downloads"
        timeout_run = 0.0
        fvp_arg: List[str] = []
        fail_fast = True
        run_jobs = 1
        quiet = True
        report_formats = ["json"]

    all_results, success = reporting_module.run_tests_with_reporting(
        cpus=["cortex-m55"],
        suite="int",
        source_dir=source_dir,
        toolchain_file=tmp_path / "toolchain.cmake",
        cmsis5=tmp_path / "CMSIS_5",
        fvp_exe=tmp_path / "fvp_exe",
        compiler_tag="gcc",
        args=_Args(),
        env={},
    )

    assert success
    assert {r.test_name for r in all_results} == {"add_s8_broadcast"}

    assert len(_FakeReportGenerator.captured) == 1
    report = _FakeReportGenerator.captured[0]
    assert set(report.descriptor_results.keys()) == {"add_s8_broadcast"}
    assert report.total_tests == 1
