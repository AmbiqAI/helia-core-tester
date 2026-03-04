"""
Command-line interface for CMSIS-NN Tools.

This module provides a subcommand-based CLI using Typer.
"""

import sys
import shutil
from pathlib import Path
from typing import Optional

import typer

from helia_core_tester.core.config import Config
from helia_core_tester.core.logging import setup_logger
from helia_core_tester.core.pipeline import FullTestPipeline
from helia_core_tester.core.steps import (
    GenerateStep,
    BuildStep,
    RunStep,
    CleanStep,
)
from helia_core_tester.reporting.gap_gate import run_gap_check

app = typer.Typer(
    name="helia_core_tester",
    help="CMSIS-NN testing toolkit - Generate, build, and run tests for CMSIS-NN kernels",
    add_completion=False,
)

def get_config(
    cpu: str = "cortex-m55",
    verbosity: Optional[int] = None,
    dry_run: bool = False,
    project_root: Optional[Path] = None,
    **kwargs
) -> Config:
    """Create and configure Config object."""
    init_kwargs = {
        "cpu": cpu,
        "dry_run": dry_run,
    }
    if project_root:
        init_kwargs["project_root"] = project_root
    for key, value in kwargs.items():
        if key in Config.__dataclass_fields__ and value is not None:
            init_kwargs[key] = value
    config = Config(**init_kwargs)
    if verbosity is not None:
        if not 0 <= verbosity <= 3:
            raise ValueError(f"verbosity must be between 0 and 3, got {verbosity}")
        config.verbosity = verbosity
    return config


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
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Generate TFLite models and template C/H files."""
    config = get_config(
        cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root,
        op_filter=op, dtype_filter=dtype, name_filter=name, limit=limit, seed=seed,
    )
    if config.plan:
        plan_item = GenerateStep(config).plan()
        typer.echo(f"1. {plan_item.name}: {'will run' if plan_item.will_run else 'skipped'} ({plan_item.reason})")
        for cmd in plan_item.commands:
            typer.echo(f"   cmd: {' '.join(cmd)}")
        if plan_item.outputs:
            outputs = ", ".join(f"{k}={v}" for k, v in plan_item.outputs.items() if v)
            if outputs:
                typer.echo(f"   outputs: {outputs}")
        sys.exit(0)
    run_step_exit(
        GenerateStep(config), config,
        "✓ Generation completed successfully",
        failure_prefix="Generation failed",
    )


@app.command()
def build(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    coverage: bool = typer.Option(False, "--coverage", help="Enable ns-cmsis-nn code coverage instrumentation"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Build test executables using CMake."""
    config = get_config(
        cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root,
        optimization=opt, jobs=jobs, coverage=coverage,
    )
    if config.plan:
        plan_item = BuildStep(config).plan()
        typer.echo(f"1. {plan_item.name}: {'will run' if plan_item.will_run else 'skipped'} ({plan_item.reason})")
        for cmd in plan_item.commands:
            typer.echo(f"   cmd: {' '.join(cmd)}")
        if plan_item.outputs:
            outputs = ", ".join(f"{k}={v}" for k, v in plan_item.outputs.items() if v)
            if outputs:
                typer.echo(f"   outputs: {outputs}")
        sys.exit(0)
    run_step_exit(
        BuildStep(config), config,
        f"✓ Build completed successfully for {cpu}",
        failure_prefix="Build failed",
    )


@app.command()
def run(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    timeout: float = typer.Option(0.0, help="Per-test timeout in seconds (0 = none)"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Don't stop on first failure"),
    coverage: bool = typer.Option(False, "--coverage", help="Collect and merge ns-cmsis-nn gcov streams"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md, junit)"),
    report_dir: Optional[Path] = typer.Option(None, help="Directory to save reports"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run tests on FVP simulator."""
    config = get_config(
        cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root,
        timeout=timeout, fail_fast=not no_fail_fast, enable_reporting=not no_report,
        report_formats=report_formats, report_dir=report_dir, coverage=coverage,
    )
    if config.plan:
        plan_item = RunStep(config).plan()
        typer.echo(f"1. {plan_item.name}: {'will run' if plan_item.will_run else 'skipped'} ({plan_item.reason})")
        for cmd in plan_item.commands:
            typer.echo(f"   cmd: {' '.join(cmd)}")
        if plan_item.outputs:
            outputs = ", ".join(f"{k}={v}" for k, v in plan_item.outputs.items() if v)
            if outputs:
                typer.echo(f"   outputs: {outputs}")
        sys.exit(0)
    run_step_exit(
        RunStep(config), config,
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
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    timeout: float = typer.Option(0.0, help="Per-test timeout in seconds (0 = none)"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Don't stop on first failure"),
    coverage: bool = typer.Option(False, "--coverage", help="Enable ns-cmsis-nn coverage collection/reporting"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip TFLite generation"),
    skip_conversion: bool = typer.Option(False, "--skip-conversion", help="Skip TFLite to C conversion"),
    skip_runners: bool = typer.Option(False, "--skip-runners", help="Skip test runner generation"),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip FVP build"),
    skip_run: bool = typer.Option(False, "--skip-run", help="Skip FVP test execution"),
    no_report: bool = typer.Option(False, "--no-report", help="Disable test reporting"),
    report_formats: list[str] = typer.Option(["json"], help="Report formats (json, html, md, junit)"),
    report_dir: Optional[Path] = typer.Option(None, help="Directory to save reports"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run the complete pipeline (generate → runners → build → run)."""
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
        optimization=opt,
        jobs=jobs,
        timeout=timeout,
        fail_fast=not no_fail_fast,
        coverage=coverage,
        skip_generation=skip_generation,
        skip_conversion=skip_conversion,
        skip_runners=skip_runners,
        skip_build=skip_build,
        skip_run=skip_run,
        enable_reporting=not no_report,
        report_formats=report_formats,
        report_dir=report_dir,
    )
    
    logger = setup_logger(verbosity=config.verbosity)
    
    pipeline = FullTestPipeline(config)
    if config.plan:
        pipeline.print_plan()
        sys.exit(0)
    success = pipeline.run()
    
    if success:
        typer.echo("✓ Pipeline completed successfully")
        sys.exit(0)
    else:
        typer.echo("✗ Pipeline failed", err=True)
        sys.exit(1)


@app.command()
def clean(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Clean build artifacts and reports."""
    config = get_config(cpu=cpu, verbosity=verbosity, dry_run=dry_run, plan=plan, project_root=project_root)
    if config.plan:
        plan_item = CleanStep(config).plan()
        typer.echo(f"1. {plan_item.name}: {'will run' if plan_item.will_run else 'skipped'} ({plan_item.reason})")
        for cmd in plan_item.commands:
            typer.echo(f"   cmd: {' '.join(cmd)}")
        if plan_item.outputs:
            outputs = ", ".join(f"{k}={v}" for k, v in plan_item.outputs.items() if v)
            if outputs:
                typer.echo(f"   outputs: {outputs}")
        sys.exit(0)
    run_step_exit(CleanStep(config), config, "", failure_prefix="Clean failed")


@app.command(name="clean-all")
def clean_all(
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Remove all artifacts (generated tests, downloads, reports, builds)."""
    config = get_config(verbosity=verbosity, dry_run=dry_run, project_root=project_root)
    artifacts_root = config.project_root / "artifacts"
    if not artifacts_root.exists():
        typer.echo("No artifacts directory to clean.")
        sys.exit(0)

    if dry_run:
        typer.echo(f"DRY RUN: Would remove {artifacts_root}")
        sys.exit(0)

    if config.verbosity >= 1:
        typer.echo(f"Removing artifacts directory: {artifacts_root}")
    shutil.rmtree(artifacts_root, ignore_errors=True)
    typer.echo("✓ Clean-all completed")


@app.command()
def doctor(
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Run preflight checks (verify tools, paths, permissions)."""
    typer.echo("Running preflight checks...")
    
    # Check repository root
    try:
        from .core.discovery import find_repo_root
        if project_root:
            repo_root = Path(project_root).resolve()
        else:
            repo_root = find_repo_root()
        typer.echo(f"✓ Repository root: {repo_root}")
    except Exception as e:
        typer.echo(f"✗ Repository root not found: {e}", err=True)
        sys.exit(1)
    
    # Check required tools
    import shutil
    
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
    
    # Check key directories
    key_dirs = {
        "assets/descriptors": "Test descriptors",
        "artifacts/generated_tests": "Generated tests (will be created)",
        "artifacts/downloads": "Downloads (will be created)",
    }
    
    for dir_name, description in key_dirs.items():
        dir_path = repo_root / dir_name
        if dir_path.exists() or dir_name in ["artifacts/generated_tests", "artifacts/downloads"]:
            typer.echo(f"✓ {dir_name}/ exists or will be created ({description})")
        else:
            typer.echo(f"⚠ {dir_name}/ not found ({description})", err=True)
    
    if all_ok:
        typer.echo("\n✓ All preflight checks passed")
        sys.exit(0)
    else:
        typer.echo("\n✗ Some preflight checks failed", err=True)
        sys.exit(1)


@app.command(name="gap-check")
def gap_check(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    report_dir: Optional[Path] = typer.Option(None, help="Directory to save gap reports"),
    generated_tests_dir: Optional[Path] = typer.Option(None, help="Override generated tests directory"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
):
    """Check for new gaps between descriptors and generated tests."""
    config = get_config(cpu=cpu, project_root=project_root)
    gen_dir = Path(generated_tests_dir).resolve() if generated_tests_dir else None
    out_dir = Path(report_dir).resolve() if report_dir else None

    exit_code, report, (json_path, md_path) = run_gap_check(
        project_root=config.project_root,
        cpu=config.cpu,
        report_dir=out_dir,
        generated_tests_dir=gen_dir,
    )

    typer.echo(f"Gap report JSON: {json_path}")
    typer.echo(f"Gap report MD:   {md_path}")
    if exit_code != 0:
        typer.echo("✗ Gap check failed: unexpected gaps detected", err=True)
    else:
        typer.echo("✓ Gap check passed (no unexpected gaps)")
    sys.exit(exit_code)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
