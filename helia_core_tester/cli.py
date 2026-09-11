"""
Command-line interface for helia-core-tester.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import MISSING
from pathlib import Path
from typing import Optional

import typer

from helia_core_tester.core.config import DEFAULT_RUN_JOBS_CAP, DEFAULT_TIMEOUT_SECONDS, Config
from helia_core_tester.core.discovery import ensure_arm_toolchain_on_path
from helia_core_tester.core.logging import setup_logger
from helia_core_tester.core.path_layout import artifacts_root
from helia_core_tester.core.pipeline import FullTestPipeline
from helia_core_tester.core.steps import BuildStep, CleanStep, GenerateStep, RunStep
from helia_core_tester.reporting.coverage_merge import run_coverage_merge
from helia_core_tester.perf_stream.cli import app as perf_stream_app

# Once, for every subcommand (including perf-stream's) for the lifetime of this
# process -- see ensure_arm_toolchain_on_path()'s own docstring for why this can't
# just live at each subprocess call site.
ensure_arm_toolchain_on_path()

app = typer.Typer(
    name="helia_core_tester",
    help="CMSIS-NN testing toolkit - generate, build, and run tests for CMSIS-NN kernels",
    add_completion=False,
)

app.add_typer(perf_stream_app, name="perf-stream")


def _print_plan_item(plan_item) -> None:
    typer.echo(f"1. {plan_item.name}: {'will run' if plan_item.will_run else 'skipped'} ({plan_item.reason})")
    for cmd in plan_item.commands:
        typer.echo(f"   cmd: {' '.join(cmd)}")
    if plan_item.outputs:
        outputs = ", ".join(f"{k}={v}" for k, v in plan_item.outputs.items() if v)
        if outputs:
            typer.echo(f"   outputs: {outputs}")


def get_config(
    cpu: Optional[str] = None,
    verbosity: Optional[int] = None,
    dry_run: bool = False,
    project_root: Optional[Path] = None,
    **kwargs,
) -> Config:
    """Create Config with explicit CLI precedence over TOML defaults."""
    def _default_for(field_name: str):
        field_def = Config.__dataclass_fields__[field_name]
        if field_def.default is not MISSING:
            return field_def.default
        if field_def.default_factory is not MISSING:  # type: ignore[attr-defined]
            return field_def.default_factory()  # type: ignore[misc]
        return MISSING

    init_kwargs = {}
    explicit: set[str] = set()

    cpu_default = _default_for("cpu")
    if cpu is not None and cpu != cpu_default:
        init_kwargs["cpu"] = cpu
        explicit.add("cpu")

    dry_run_default = _default_for("dry_run")
    if dry_run != dry_run_default:
        init_kwargs["dry_run"] = dry_run
        explicit.add("dry_run")

    if verbosity is not None:
        if not 0 <= verbosity <= 3:
            raise ValueError(f"verbosity must be between 0 and 3, got {verbosity}")
        verbosity_default = _default_for("verbosity")
        if verbosity != verbosity_default:
            init_kwargs["verbosity"] = verbosity
            explicit.add("verbosity")

    if project_root is not None:
        init_kwargs["project_root"] = project_root
        explicit.add("project_root")

    for key, value in kwargs.items():
        if key not in Config.__dataclass_fields__ or value is None:
            continue
        default_value = _default_for(key)
        if value != default_value:
            init_kwargs[key] = value
            explicit.add(key)

    init_kwargs["_explicit_overrides"] = explicit
    return Config(**init_kwargs)


def run_step_exit(step, config: Config, success_msg: str, failure_prefix: Optional[str] = None) -> None:
    """Run a step, echo result, and exit with appropriate code."""
    setup_logger(verbosity=config.verbosity)
    result = step.execute()
    if result.success:
        typer.echo(success_msg if success_msg else result.message)
        sys.exit(0)
    if result.skipped:
        typer.echo(f"⊘ {result.message}")
        sys.exit(0)
    msg = f"{failure_prefix}: {result.message}" if failure_prefix else result.message
    typer.echo(f"✗ {msg}", err=True)
    sys.exit(1)


@app.command()
def generate(
    op: Optional[str] = typer.Option(None, help="Generate only specific operator"),
    dtype: Optional[str] = typer.Option(None, help="Generate only specific dtype"),
    name: Optional[str] = typer.Option(None, help="Generate only specific test by name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(None, help="Random seed for test generation"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    suite: str = typer.Option("int", "--suite", help="Test suite selection: int, float, or both"),
    float_precision: str = typer.Option("both", "--float-precision", help="Float precision filter: f16, f32, or both"),
    force_generate: bool = typer.Option(False, "--force-generate", help="Regenerate every case even when its reuse stamp still matches"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Generate TFLite models and template C/H files."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        plan=plan,
        project_root=project_root,
        op_filter=op,
        dtype_filter=dtype,
        name_filter=name,
        limit=limit,
        seed=seed,
        suite=suite,
        float_precision=float_precision,
        force_generate=force_generate,
    )
    if config.plan:
        _print_plan_item(GenerateStep(config).plan())
        sys.exit(0)
    run_step_exit(
        GenerateStep(config),
        config,
        "✓ Generation completed successfully",
        failure_prefix="Generation failed",
    )


@app.command()
def build(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    toolchain: str = typer.Option("gcc", "--toolchain", help="Toolchain: gcc (default) or armclang (Arm Compiler 6 on PATH; not combinable with --coverage)"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    coverage: bool = typer.Option(False, "--coverage", help="Enable ns-cmsis-nn code coverage instrumentation"),
    coverage_mve_float: bool = typer.Option(False, "--coverage-mve-float", help="Enable Cortex-M55 float MVE paths during coverage builds"),
    suite: str = typer.Option("int", "--suite", help="Test suite selection: int, float, or both"),
    float_precision: str = typer.Option("both", "--float-precision", help="Float precision selection for float suite: f16, f32, or both"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Build test executables using CMake."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        plan=plan,
        project_root=project_root,
        optimization=opt,
        toolchain=toolchain,
        jobs=jobs,
        coverage=coverage,
        coverage_mve_float=coverage_mve_float,
        suite=suite,
        float_precision=float_precision,
    )
    if config.plan:
        _print_plan_item(BuildStep(config).plan())
        sys.exit(0)
    run_step_exit(
        BuildStep(config),
        config,
        f"✓ Build completed successfully for {cpu}",
        failure_prefix="Build failed",
    )


@app.command()
def run(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    timeout: Optional[float] = typer.Option(None, help=f"Per-case FVP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}; 0 disables it and lets a hung kernel block the run)"),
    run_jobs: Optional[int] = typer.Option(None, "--run-jobs", help=f"Parallel FVP run jobs (default: min(host cores, {DEFAULT_RUN_JOBS_CAP}); 0 = every host core). FVP boot dominates per-case time so parallelism is the lever, but unbounded jobs on a shared or metered runner is a cost risk"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Do not stop on first failure"),
    coverage: bool = typer.Option(False, "--coverage", help="Collect and merge ns-cmsis-nn gcov streams"),
    coverage_mve_float: bool = typer.Option(False, "--coverage-mve-float", help="Write MVE float coverage to the float-mve report lane"),
    suite: str = typer.Option("int", "--suite", help="Test suite selection: int, float, or both"),
    toolchain: str = typer.Option("gcc", "--toolchain", help="Toolchain: gcc (default) or armclang (Arm Compiler 6 on PATH; not combinable with --coverage)"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md, junit)"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run tests on FVP simulator."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        plan=plan,
        project_root=project_root,
        timeout=timeout,
        fail_fast=not no_fail_fast,
        enable_reporting=not no_report,
        report_formats=report_formats,
        coverage=coverage,
        coverage_mve_float=coverage_mve_float,
        run_jobs=run_jobs,
        suite=suite,
        toolchain=toolchain,
    )
    if config.plan:
        _print_plan_item(RunStep(config).plan())
        sys.exit(0)
    run_step_exit(
        RunStep(config),
        config,
        "✓ All tests completed successfully",
        failure_prefix="Test execution failed",
    )


@app.command()
def full(
    op: Optional[str] = typer.Option(None, help="Generate only specific operator"),
    dtype: Optional[str] = typer.Option(None, help="Generate only specific dtype"),
    name: Optional[str] = typer.Option(None, help="Generate only specific test by name"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(None, help="Random seed for test generation"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    suite: str = typer.Option("int", "--suite", help="Test suite selection: int, float, or both"),
    float_precision: str = typer.Option("both", "--float-precision", help="Float precision selection for float suite: f16, f32, or both"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    toolchain: str = typer.Option("gcc", "--toolchain", help="Toolchain: gcc (default) or armclang (Arm Compiler 6 on PATH; not combinable with --coverage)"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    timeout: Optional[float] = typer.Option(None, help=f"Per-case FVP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}; 0 disables it and lets a hung kernel block the run)"),
    run_jobs: Optional[int] = typer.Option(None, "--run-jobs", help=f"Parallel FVP run jobs (default: min(host cores, {DEFAULT_RUN_JOBS_CAP}); 0 = every host core). FVP boot dominates per-case time so parallelism is the lever, but unbounded jobs on a shared or metered runner is a cost risk"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Do not stop on first failure"),
    coverage: bool = typer.Option(False, "--coverage", help="Enable ns-cmsis-nn coverage collection/reporting"),
    coverage_mve_float: bool = typer.Option(False, "--coverage-mve-float", help="Enable Cortex-M55 float MVE paths during coverage builds"),
    force_generate: bool = typer.Option(False, "--force-generate", help="Regenerate every case even when its reuse stamp still matches"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip TFLite generation"),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip FVP build"),
    skip_run: bool = typer.Option(False, "--skip-run", help="Skip FVP test execution"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md, junit)"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help="Override the ns-cmsis-nn checkout used for the CMake build (defaults to CMakeLists.txt's ../.. sibling checkout)"),
):
    """Run the complete pipeline (generate -> build -> run)."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        dry_run=dry_run,
        plan=plan,
        project_root=project_root,
        op_filter=op,
        dtype_filter=dtype,
        name_filter=name,
        limit=limit,
        seed=seed,
        suite=suite,
        float_precision=float_precision,
        optimization=opt,
        toolchain=toolchain,
        jobs=jobs,
        timeout=timeout,
        run_jobs=run_jobs,
        fail_fast=not no_fail_fast,
        coverage=coverage,
        coverage_mve_float=coverage_mve_float,
        force_generate=force_generate,
        skip_generation=skip_generation,
        skip_build=skip_build,
        skip_run=skip_run,
        enable_reporting=not no_report,
        report_formats=report_formats,
        cmsis_nn_root=cmsis_nn_root,
    )

    setup_logger(verbosity=config.verbosity)
    pipeline = FullTestPipeline(config)
    if config.plan:
        pipeline.print_plan()
        sys.exit(0)

    success = pipeline.run()
    if success:
        typer.echo("✓ Pipeline completed successfully")
        sys.exit(0)
    typer.echo("✗ Pipeline failed", err=True)
    sys.exit(1)


@app.command()
def clean(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    suite: str = typer.Option("both", "--suite", help="Suite(s) to clean: int, float, or both"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Remove generated tests + reports + build outputs for selected CPU(s)."""
    config = get_config(cpu=cpu, suite=suite, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root)
    if config.plan:
        _print_plan_item(CleanStep(config).plan())
        sys.exit(0)
    run_step_exit(CleanStep(config), config, "✓ Clean completed", failure_prefix="Clean failed")


@app.command(name="clean-all")
def clean_all(
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Remove all generated tests, reports, and build outputs."""
    config = get_config(verbosity=verbosity, dry_run=dry_run, project_root=project_root)
    art_root = artifacts_root(config.project_root)
    targets = [
        config.generated_tests_root,
        config.reports_root,
    ]
    # Collect build dirs for both suites (build-int-<cpu>-<compiler> and build-float-<cpu>-<compiler>)
    for pattern in ("build-int-*", "build-float-*"):
        targets.extend([p for p in art_root.glob(pattern) if p.is_dir()])

    existing = [p for p in targets if p.exists()]
    if not existing:
        typer.echo("No generated-tests/reports/build outputs to clean.")
        sys.exit(0)

    if dry_run:
        typer.echo("DRY RUN: Would remove:")
        for p in existing:
            typer.echo(f"  - {p}")
        sys.exit(0)

    for p in existing:
        shutil.rmtree(p, ignore_errors=True)

    if config.verbosity >= 1:
        typer.echo(f"Removed {len(existing)} path(s)")
    typer.echo("✓ Clean-all completed")


def _tool_version(exe: str) -> str:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10, check=False).stdout
        return out.strip().splitlines()[0] if out.strip() else "version unknown"
    except Exception:
        return "version unknown"


@app.command()
def doctor(
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run preflight checks (verify tools, paths, permissions)."""
    typer.echo("Running preflight checks...")

    try:
        from .core.discovery import find_repo_root

        repo_root = Path(project_root).resolve() if project_root else find_repo_root()
        typer.echo(f"✓ Repository root: {repo_root}")
    except Exception as e:
        typer.echo(f"✗ Repository root not found: {e}", err=True)
        sys.exit(1)

    tools = {
        "python3": "Python interpreter",
        "pytest": "pytest (for test generation)",
        "cmake": "CMake (for building)",
    }

    all_ok = True
    for tool, description in tools.items():
        if shutil.which(tool):
            typer.echo(f"✓ {tool} found ({description})")
        else:
            typer.echo(f"✗ {tool} not found ({description})", err=True)
            all_ok = False

    gcc = shutil.which("arm-none-eabi-gcc")
    if gcc:
        typer.echo(f"✓ arm-none-eabi-gcc found ({_tool_version(gcc)}) at {gcc}")
    else:
        typer.echo("⚠ arm-none-eabi-gcc not on PATH (run setup_dependencies.py; --toolchain gcc needs it)", err=True)
    armclang = shutil.which("armclang")
    if armclang:
        typer.echo(f"✓ armclang found ({_tool_version(armclang)}) at {armclang}")
    else:
        typer.echo("⚠ armclang not on PATH (only needed for --toolchain armclang)")

    key_dirs = {
        "assets/descriptors": "Test descriptors",
        "artifacts/generated_tests": "Generated tests (will be created)",
        "artifacts/reports": "Canonical reports root",
    }
    for dir_name, description in key_dirs.items():
        dir_path = repo_root / dir_name
        if dir_path.exists() or dir_name in ["artifacts/generated_tests", "artifacts/reports"]:
            typer.echo(f"✓ {dir_name}/ exists or will be created ({description})")
        else:
            typer.echo(f"⚠ {dir_name}/ not found ({description})", err=True)

    if all_ok:
        typer.echo("\n✓ All preflight checks passed")
        sys.exit(0)
    typer.echo("\n✗ Some preflight checks failed", err=True)
    sys.exit(1)


@app.command(name="coverage-merge")
def coverage_merge(
    cpu: str = typer.Option("cortex-m0,cortex-m4,cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    suite: str = typer.Option("both", "--suite", help="Coverage suite selection: int, float, or both"),
    include_mve_float: bool = typer.Option(
        False,
        "--include-mve-float",
        help="Also merge cortex-m55 MVE float coverage (reports/coverage/float-mve).",
    ),
    expected_zero_config: Optional[Path] = typer.Option(
        None,
        help="Path to expected-zero JSON config (default: assets/coverage_expected_zero.json)",
    ),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Merge per-CPU coverage.info files and classify zero-hit files."""
    config = get_config(cpu=cpu, suite=suite, project_root=project_root)
    expected_zero_path = Path(expected_zero_config).resolve() if expected_zero_config else None

    merge_suites = list(config.suites)
    if include_mve_float and "float-mve" not in merge_suites:
        merge_suites.append("float-mve")

    exit_code, report = run_coverage_merge(
        project_root=config.project_root,
        cpus=config.cpus,
        suites=merge_suites,
        report_dir=config.coverage_merged_report_dir(),
        expected_zero_config=expected_zero_path,
    )

    typer.echo(f"Merged LCOV: {report.merged_lcov_path}")
    typer.echo(f"Summary JSON: {report.summary_json_path}")
    typer.echo(f"Summary MD:   {report.summary_md_path}")
    typer.echo(f"Summary HTML: {report.summary_html_path}")
    typer.echo(f"HTML generator: {report.html_generator}")
    if report.html_generation_note:
        typer.echo(f"HTML note: {report.html_generation_note}")
    typer.echo(f"Overall line coverage: {report.total_lh}/{report.total_lf} ({report.overall_line_rate:.2f}%)")
    typer.echo(
        "Counts: "
        f"covered={len(report.covered_files)}, "
        f"zero_reachable={len(report.zero_reachable_files)}, "
        f"expected_zero={len(report.expected_zero_files)}, "
        f"expected_zero_but_covered={len(report.expected_zero_but_covered_files)}"
    )

    if exit_code != 0:
        typer.echo("✗ Coverage merge failed: missing required coverage.info inputs", err=True)
    else:
        typer.echo("✓ Coverage merge completed")
    sys.exit(exit_code)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
