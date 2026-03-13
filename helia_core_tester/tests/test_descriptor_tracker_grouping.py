from __future__ import annotations

from pathlib import Path

from helia_core_tester.reporting.descriptor_tracker import DescriptorTracker
from helia_core_tester.reporting.models import TestStatus


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_descriptor_tracker_uses_grouped_descriptor_paths() -> None:
    tracker = DescriptorTracker(_repo_root() / "assets" / "descriptors")
    descriptors = tracker.load_all_descriptors()

    descriptor = next(desc for desc in descriptors.values() if desc["operator"] == "AvgPool")
    descriptor_name = descriptor["name"]

    descriptor_path = tracker.get_descriptor_path(descriptor_name)
    assert descriptor_path == _repo_root() / "assets" / "descriptors" / descriptor["_source_relpath"]


def test_descriptor_status_uses_grouped_generated_and_elf_paths(tmp_path: Path) -> None:
    tracker = DescriptorTracker(_repo_root() / "assets" / "descriptors")
    descriptors = tracker.load_all_descriptors()
    descriptor = next(desc for desc in descriptors.values() if desc["operator"] == "AvgPool")
    descriptor_name = descriptor["name"]
    family = descriptor["_family"]

    generated_tests_dir = tmp_path / "generated_tests"
    build_dir = tmp_path / "build"
    test_dir = generated_tests_dir / family / descriptor_name
    includes_dir = test_dir / "includes"
    includes_dir.mkdir(parents=True, exist_ok=True)
    (includes_dir / f"{descriptor_name}_avg_pool.h").write_text("/* generated */")

    status, failure_stage, failure_reason = tracker.determine_descriptor_status(
        descriptor_name,
        test_result=None,
        build_dir=build_dir,
        generated_tests_dir=generated_tests_dir,
    )
    assert status == TestStatus.BUILD_FAILED
    assert failure_stage == "build"
    assert failure_reason == "ELF file not found in build directory"

    elf_path = build_dir / "tests" / family / f"{descriptor_name}.elf"
    elf_path.parent.mkdir(parents=True, exist_ok=True)
    elf_path.write_bytes(b"\x7fELF")

    status, failure_stage, failure_reason = tracker.determine_descriptor_status(
        descriptor_name,
        test_result=None,
        build_dir=build_dir,
        generated_tests_dir=generated_tests_dir,
    )
    assert status == TestStatus.NOT_RUN
    assert failure_stage is None
    assert failure_reason == "Test not executed"
