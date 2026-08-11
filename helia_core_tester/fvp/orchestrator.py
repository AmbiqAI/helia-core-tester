"""Top-level orchestration for FVP build/run command."""

from __future__ import annotations

from pathlib import Path
from typing import List

from helia_core_tester.core.cpu_targets import parse_cpu_list, target_cpu_cmake_value
from helia_core_tester.core.path_layout import build_dir as canonical_build_dir

from .cli import build_arg_parser
from .cmake import cmake_build, cmake_configure, find_elves
from .coverage import generate_coverage_reports, new_coverage_context
from .env import DEFAULT_DL, DEFAULT_SOURCE, REPO_ROOT, detect_paths, is_linux, resolve_gcov_tool
from .errors import FvpScriptError
from .reporting import resolve_generated_tests_dir, run_tests_with_reporting
from .runner import ProcessSupervisor, run_elf_jobs_with_reporting


def parse_cpus(cpu_str: str) -> List[str]:
    return parse_cpu_list(cpu_str)


def run_main(argv: List[str]) -> int:
    parser = build_arg_parser(default_downloads_dir=DEFAULT_DL, default_source_dir=DEFAULT_SOURCE)
    args = parser.parse_args(argv)

    # The FVP binaries are Linux-only prebuilt executables (matching the
    # original bash script). --no-run skips the FVP entirely (e.g. a
    # cross-compile-only build, or a real-hardware flash/benchmark flow
    # driven separately), so only enforce the Linux requirement when the
    # FVP is actually going to be invoked.
    if not args.no_run and not is_linux():
        raise FvpScriptError("This script supports Linux only (matching the original bash script).")

    if not args.no_setup:
        from .env import call_setup_dependencies

        call_setup_dependencies(args.downloads_dir)

    ctx = detect_paths(args)
    env = ctx["env"]
    toolchain_file = ctx["toolchain_file"]
    compiler_tag = ctx["compiler_tag"]
    fvp_exe = ctx["fvp_exe"]
    cmsis5 = ctx["cmsis5"]
    source_dir = args.source_dir.resolve()

    if not source_dir.exists():
        raise FvpScriptError(f"CMake source dir not found: {source_dir}")

    try:
        cpus = parse_cpus(args.cpu)
    except ValueError as exc:
        raise FvpScriptError(str(exc)) from exc

    if args.run_jobs < 0:
        raise FvpScriptError(f"--run-jobs must be >= 0, got {args.run_jobs}")

    if args.coverage and args.use_arm_compiler:
        raise FvpScriptError("--coverage is only supported with GCC builds")
    if args.coverage:
        gcov_tool = resolve_gcov_tool(env)
        if not gcov_tool:
            raise FvpScriptError("Coverage requested but no gcov-tool found on PATH (expected arm-none-eabi-gcov-tool).")
        setattr(args, "_gcov_tool", gcov_tool)

    enable_reporting = not args.no_report
    verbosity = getattr(args, "verbosity", 0)
    supervisor = ProcessSupervisor(verbosity=verbosity)
    try:
        if enable_reporting:
            _, success = run_tests_with_reporting(
                cpus=cpus,
                suite=args.suite,
                source_dir=source_dir,
                toolchain_file=toolchain_file,
                cmsis5=cmsis5,
                fvp_exe=fvp_exe,
                compiler_tag=compiler_tag,
                args=args,
                env=env,
                supervisor=supervisor,
            )
            if success:
                if verbosity >= 1:
                    print("\nAll requested builds/runs completed successfully.")
                return 0
            return 1

        any_fail = False
        for cpu in cpus:
            if verbosity >= 1:
                print(f"\nTarget: {cpu} ({compiler_tag}, suite={args.suite})")
            build_dir = canonical_build_dir(REPO_ROOT, cpu, compiler_tag, suite=args.suite)
            cpu_generated_tests_dir = resolve_generated_tests_dir(source_dir, cpu, suite=args.suite)
            coverage_ctx = new_coverage_context(build_dir, getattr(args, "_gcov_tool", None)) if args.coverage else None

            if not args.no_build:
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

            elves = find_elves(build_dir)
            if not elves:
                if verbosity >= 1:
                    print(f"(no .elf found under {build_dir}, nothing to run)")
                continue

            elf_entries = [(elf, elf.stem) for elf in elves]
            _, cpu_failed = run_elf_jobs_with_reporting(
                elf_entries=elf_entries,
                fvp_exe=fvp_exe,
                timeout=args.timeout_run,
                verbosity=verbosity,
                extra_args=args.fvp_arg,
                env=env,
                cpu=cpu,
                coverage_ctx=coverage_ctx,
                run_jobs=args.run_jobs,
                fail_fast=args.fail_fast,
                supervisor=supervisor,
            )
            any_fail = any_fail or cpu_failed
            if cpu_failed and args.fail_fast:
                if verbosity >= 1:
                    print("Stopping early due to failure (--fail-fast).")
                break

        if any_fail:
            return 1

        generate_coverage_reports(cpus, args.suite, args, env, source_dir, compiler_tag, verbosity)

        if verbosity >= 1:
            print("\nAll requested builds/runs completed successfully.")
        return 0
    finally:
        if supervisor.active_count() > 0:
            supervisor.terminate_all("shutdown")
