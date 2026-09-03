"""Descriptor-aware reporting orchestration for FVP runs."""

from __future__ import annotations

from datetime import datetime
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.core.cpu_targets import target_cpu_cmake_value
from helia_core_tester.core.path_layout import build_dir as canonical_build_dir
from helia_core_tester.core.path_layout import generated_tests_dir as canonical_generated_tests_dir
from helia_core_tester.core.path_layout import tests_report_dir as canonical_tests_report_dir
from helia_core_tester.reporting.models import DescriptorResult, TestReport, TestResult, TestStatus
from helia_core_tester.reporting.descriptor_tracker import DescriptorTracker
from helia_core_tester.reporting.generator import ReportGenerator
from helia_core_tester.generation.artifact_identity import generated_case_artifact_sha256

from .cmake import active_test_list, cmake_build, cmake_configure, find_elves
from .coverage import generate_coverage_reports, new_coverage_context
from .env import REPO_ROOT, get_git_sha
from .runner import ProcessSupervisor, run_elf_jobs_with_reporting


def resolve_generated_tests_dir(source_dir: Path, cpu: str, suite: str) -> Path:
    return canonical_generated_tests_dir(source_dir, cpu, suite=suite)


def _best_effort_artifact_sha256(generated_test_dir: Path) -> Optional[str]:
    """Compute the artifact digest, leaving it unset if outputs are missing.

    ``generated_case_artifact_sha256`` raises when ``generated_test_dir`` doesn't
    exist or has no deterministic files yet -- e.g. for failures that occur
    before generation/build outputs are produced. That's expected here, not a
    reporting bug, so treat the digest as best-effort rather than aborting
    report generation.
    """
    try:
        return generated_case_artifact_sha256(generated_test_dir)
    except (OSError, ValueError):
        return None


def _tests_report_dir(cpu: str, suite: str) -> Path:
    return canonical_tests_report_dir(REPO_ROOT, cpu, suite=suite)


def run_tests_with_reporting(
    cpus: List[str],
    suite: str,
    source_dir: Path,
    toolchain_file: Path,
    cmsis5: Path,
    fvp_exe: Path,
    compiler_tag: str,
    args,
    env: dict,
    supervisor: Optional[ProcessSupervisor] = None,
) -> Tuple[List[TestResult], bool]:
    all_results: List[TestResult] = []
    any_fail = False
    verbosity = getattr(args, "verbosity", 0)
    descriptors_dir = find_descriptors_dir()
    tracker = DescriptorTracker(descriptors_dir)
    all_descriptors_dict = tracker.load_all_descriptors()

    for cpu in cpus:
        cpu_start_time = datetime.now()
        if verbosity >= 1:
            print(f"\nTarget: {cpu} ({compiler_tag}, suite={suite})")
        build_dir = canonical_build_dir(REPO_ROOT, cpu, compiler_tag, suite=suite)
        cpu_generated_tests_dir = resolve_generated_tests_dir(source_dir, cpu, suite=suite)
        cpu_tests_report_dir = _tests_report_dir(cpu, suite=suite)
        coverage_ctx = new_coverage_context(build_dir, getattr(args, "_gcov_tool", None)) if args.coverage else None

        if cpu_tests_report_dir.exists():
            if verbosity >= 1:
                print(f"Removing previous tests report directory: {cpu_tests_report_dir}")
            shutil.rmtree(cpu_tests_report_dir, ignore_errors=True)
        cpu_tests_report_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_build:
            if build_dir.exists():
                if verbosity >= 1:
                    print(f"Removing previous build directory: {build_dir}")
                shutil.rmtree(build_dir, ignore_errors=True)
            cmake_configure(
                source_dir=source_dir,
                build_dir=build_dir,
                toolchain_file=toolchain_file,
                cpu=target_cpu_cmake_value(cpu),
                cmsis5=cmsis5,
                optimization=args.opt,
                extra_defs=args.cmake_def,
                generator=args.generator,
                generated_tests_dir=cpu_generated_tests_dir,
                enable_coverage=args.coverage,
                verbosity=verbosity,
                env=env,
            )
            cmake_build(build_dir=build_dir, verbosity=verbosity, env=env, jobs=args.jobs)

        if args.no_run:
            continue

        active_names = active_test_list(cpu_generated_tests_dir)
        elves = find_elves(build_dir, active_names)
        if not elves:
            if verbosity >= 1:
                print(f"(no .elf found under {build_dir}, nothing to run)")
            continue

        elf_entries: List[Tuple[Path, str]] = []
        for elf in elves:
            test_name = elf.stem
            descriptor = tracker.map_test_to_descriptor(test_name, all_descriptors_dict)
            descriptor_name = descriptor.get("name") if descriptor else test_name
            elf_entries.append((elf, descriptor_name))

        cpu_results, cpu_failed = run_elf_jobs_with_reporting(
            elf_entries=elf_entries,
            fvp_exe=fvp_exe,
            timeout=args.timeout_run,
            verbosity=verbosity,
            extra_args=args.fvp_arg,
            env=env,
            cpu=cpu,
            coverage_ctx=coverage_ctx,
            run_jobs=getattr(args, "run_jobs", 1),
            fail_fast=args.fail_fast,
            supervisor=supervisor,
        )
        all_results.extend(cpu_results)
        any_fail = any_fail or cpu_failed

        cpu_end_time = datetime.now()
        generator = ReportGenerator(output_dir=cpu_tests_report_dir)

        test_result_map: dict[str, TestResult] = {}
        for result in cpu_results:
            desc_name = result.descriptor_name or result.test_name
            if desc_name not in test_result_map or result.status == TestStatus.PASS:
                test_result_map[desc_name] = result

        active_descriptors: set[str] = set(test_result_map.keys())
        for desc_name in all_descriptors_dict.keys():
            # A descriptor outside the active test list must never re-enter
            # the report via a stale artifact still on disk (e.g. a build_dir
            # never reconfigured/pruned this run, such as --skip-build +
            # --skip-generation reusing an older, wider build) -- otherwise
            # the descriptor-aware count above can exceed the active filter
            # even though find_elves() correctly never ran it. See issue #66.
            if active_names is not None and desc_name not in active_names:
                continue
            generated_test_dir = tracker.generated_test_dir_for(desc_name, cpu_generated_tests_dir)
            tflite_file = generated_test_dir / f"{desc_name}.tflite"
            includes_dir = generated_test_dir / "includes"
            model_headers = list(includes_dir.glob(f"{desc_name}_*.h")) if includes_dir.exists() else []
            model_header_old = generated_test_dir / "includes" / f"{desc_name}_model.h"
            has_model_header = len(model_headers) > 0 or model_header_old.exists()
            elf_path = tracker.elf_path_for(desc_name, build_dir)
            if tflite_file.exists() or has_model_header or elf_path.exists():
                active_descriptors.add(desc_name)

        descriptor_results: dict[str, DescriptorResult] = {}
        for desc_name in sorted(active_descriptors):
            desc_content = all_descriptors_dict.get(desc_name)
            if not desc_content:
                continue
            test_result = test_result_map.get(desc_name)
            status, failure_stage, failure_reason = tracker.determine_descriptor_status(
                descriptor_name=desc_name,
                test_result=test_result,
                build_dir=build_dir,
                generated_tests_dir=cpu_generated_tests_dir,
            )
            desc_path = tracker.get_descriptor_path(desc_name)
            generated_test_dir = tracker.generated_test_dir_for(desc_name, cpu_generated_tests_dir)
            descriptor_results[desc_name] = DescriptorResult(
                descriptor_name=desc_name,
                descriptor_path=desc_path,
                descriptor_content=desc_content,
                status=status,
                test_result=test_result,
                failure_stage=failure_stage,
                failure_reason=failure_reason,
                artifact_sha256=_best_effort_artifact_sha256(generated_test_dir),
            )

        for result in cpu_results:
            if not result.descriptor_name or result.descriptor_name in descriptor_results:
                continue
            desc = tracker.map_test_to_descriptor(result.test_name, all_descriptors_dict)
            if not desc:
                continue
            desc_name = desc.get("name", result.descriptor_name)
            if desc_name in descriptor_results:
                continue
            status, failure_stage, failure_reason = tracker.determine_descriptor_status(
                descriptor_name=desc_name,
                test_result=result,
                build_dir=build_dir,
                generated_tests_dir=cpu_generated_tests_dir,
            )
            desc_path = tracker.get_descriptor_path(desc_name)
            descriptor_results[desc_name] = DescriptorResult(
                descriptor_name=desc_name,
                descriptor_path=desc_path,
                descriptor_content=desc,
                status=status,
                test_result=result,
                failure_stage=failure_stage,
                failure_reason=failure_reason,
                artifact_sha256=_best_effort_artifact_sha256(
                    tracker.generated_test_dir_for(desc_name, cpu_generated_tests_dir)
                ),
            )

        metadata = {
            "cpu": cpu,
            "suite": suite,
            "optimization": args.opt,
            "compiler": compiler_tag,
            "toolchain_file": str(toolchain_file),
            "cmsis5_path": str(cmsis5),
            "fvp_exe": str(fvp_exe),
            "downloads_dir": str(args.downloads_dir),
            "source_dir": str(source_dir),
            "generated_tests_dir": str(cpu_generated_tests_dir),
            "tests_report_dir": str(cpu_tests_report_dir),
            "git_sha": get_git_sha(source_dir),
        }
        report = TestReport(
            run_id=f"run_{cpu}_{cpu_start_time.strftime('%Y%m%d_%H%M%S')}",
            start_time=cpu_start_time,
            end_time=cpu_end_time,
            cpu=cpu,
            descriptor_results=descriptor_results,
            all_descriptors=list(all_descriptors_dict.values()),
            project_root=source_dir,
            metadata=metadata,
        )
        report_formats = getattr(args, "report_formats", None) or ["json"]
        generated_files = generator.generate_reports(report, report_formats)
        if not getattr(args, "quiet", False):
            print(
                f"Summary ({cpu}): total={report.total_tests} "
                f"passed={report.passed} failed={report.failed} skipped={report.skipped} "
                f"duration={report.duration:.2f}s"
            )
        if verbosity >= 1:
            for format_type, file_path in generated_files.items():
                print(f"{cpu} {format_type.upper()} report: {file_path}")

        if cpu_failed and args.fail_fast and verbosity >= 1:
            print("Stopping early due to failure (--fail-fast).")

        if any_fail and args.fail_fast:
            break

    generate_coverage_reports(cpus, suite, args, env, source_dir, compiler_tag, verbosity)
    all_results = sorted(all_results, key=lambda item: (item.cpu, item.test_name))
    return all_results, not any_fail
