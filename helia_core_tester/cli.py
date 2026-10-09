"""
Command-line interface for helia-core-tester.
"""

from __future__ import annotations

import shutil
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
from helia_core_tester.core.steps import BuildStep, CleanStep, GenerateStep, HostCheckStep, RunStep
from helia_core_tester.reporting.coverage_merge import run_coverage_merge
from helia_core_tester.contract.cli import contract_app
from helia_core_tester.hardware.cli import boards as boards_command
from helia_core_tester.hardware.cli import explain as explain_command
from helia_core_tester.hardware.cli import hardware_app, probes_app
from helia_core_tester.hardware.candidate_check import candidate_app
# Registers candidate baseline and eval.
import helia_core_tester.hardware.candidate_eval  # noqa: F401
from helia_core_tester.hardware.score import score as score_command
from helia_core_tester.agent_loop.cli import agent_loop_app

# Once, for every subcommand (including the hardware group's) for the lifetime of
# this process -- see ensure_arm_toolchain_on_path()'s own docstring for why this
# can't just live at each subprocess call site.
ensure_arm_toolchain_on_path()

app = typer.Typer(
    name="helia_core_tester",
    help="CMSIS-NN testing toolkit - generate, build, and run tests for CMSIS-NN kernels",
    add_completion=False,
)

app.add_typer(hardware_app, name="hardware")
app.add_typer(probes_app, name="probes")
app.add_typer(contract_app, name="contract")
app.add_typer(candidate_app, name="candidate")
app.add_typer(agent_loop_app, name="agent-loop")
app.command(name="boards")(boards_command)
app.command(name="explain")(explain_command)
app.command(name="score")(score_command)


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
    op: Optional[str] = typer.Option(None, help="Only these operators, comma-separated"),
    dtype: Optional[str] = typer.Option(None, help="Only this case dtype: activations, or S4 weights"),
    name: Optional[str] = typer.Option(None, help="Only these exact test names, comma-separated"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(None, help="Random seed for test generation"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    suite: str = typer.Option("int", "--suite", help="Test suite selection: int, float, or both"),
    float_precision: str = typer.Option("both", "--float-precision", help="Float precision filter: f16, f32, or both"),
    force_generate: bool = typer.Option(False, "--force-generate", help="Regenerate every case even when its reuse stamp still matches"),
    random_shapes: Optional[int] = typer.Option(None, "--random-shapes", help="Draw N random shapes per matching op"),
    shape_seed: Optional[int] = typer.Option(None, "--shape-seed", help="Seed for --random-shapes, 0 to 2**32-1 (default 0)"),
    hidden_dir: Optional[Path] = typer.Option(None, "--hidden-dir", help="Write secret-seeded shapes here, outside the tree"),
    hidden_seed_file: Optional[Path] = typer.Option(None, "--hidden-seed-file", help="Secret seed file; else HCT_HIDDEN_SEED"),
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
        random_shapes=random_shapes,
        shape_seed=shape_seed,
        hidden_dir=hidden_dir,
        hidden_seed_file=hidden_seed_file,
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
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    coverage: bool = typer.Option(False, "--coverage", help="Enable ns-cmsis-nn code coverage instrumentation"),
    coverage_mve_float: bool = typer.Option(False, "--coverage-mve-float", help="Enable Cortex-M55 float MVE paths during coverage builds"),
    coverage_mve_int: bool = typer.Option(False, "--coverage-mve-int", help="Enable Cortex-M55 integer MVE paths (no ARM_MATH_AUTOVECTORIZE) during coverage builds"),
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
        jobs=jobs,
        coverage=coverage,
        coverage_mve_float=coverage_mve_float,
        coverage_mve_int=coverage_mve_int,
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


@app.command(name="host-check")
def host_check(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s) whose generated int trees to check, comma-separated"),
    host_kernels: str = typer.Option("m0", "--host-kernels", help="Host kernel builds, comma-separated: m0 (pure C, no ARM_MATH_*) and/or dsp (Armv7E-M via dsp_shim.h)"),
    jobs: Optional[int] = typer.Option(None, help="Parallel compile/run jobs"),
    seed: Optional[int] = typer.Option(None, help="Run seed for the reproduce hints (default: the one manifest.json records)"),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help="Verbosity level (0-3)"),
    plan: bool = typer.Option(False, "--plan", help="Print execution plan and exit"),
    project_root: Optional[Path] = typer.Option(None, "--repo-root", help="Repository root directory"),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help="ns-cmsis-nn checkout whose kernels to check against (default: CMSIS_NN_ROOT, else the enclosing checkout)"),
):
    """Compile and run every generated int case on the host against the ns-cmsis-nn kernels."""
    config = get_config(
        cpu=cpu,
        verbosity=verbosity,
        plan=plan,
        project_root=project_root,
        jobs=jobs,
        seed=seed,
        suite="int",
        host_kernels=[m.strip() for m in host_kernels.split(",") if m.strip()],
        cmsis_nn_root=cmsis_nn_root,
    )
    if config.plan:
        _print_plan_item(HostCheckStep(config).plan())
        sys.exit(0)
    run_step_exit(HostCheckStep(config), config, "", failure_prefix="Host check failed")


@app.command()
def run(
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    timeout: Optional[float] = typer.Option(None, help=f"Per-case FVP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}; 0 disables it and lets a hung kernel block the run)"),
    run_jobs: Optional[int] = typer.Option(None, "--run-jobs", help=f"Parallel FVP run jobs (default: min(host cores, {DEFAULT_RUN_JOBS_CAP}); 0 = every host core). FVP boot dominates per-case time so parallelism is the lever, but unbounded jobs on a shared or metered runner is a cost risk"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Do not stop on first failure"),
    coverage: bool = typer.Option(False, "--coverage", help="Collect and merge ns-cmsis-nn gcov streams"),
    coverage_mve_float: bool = typer.Option(False, "--coverage-mve-float", help="Write MVE float coverage to the float-mve report lane"),
    coverage_mve_int: bool = typer.Option(False, "--coverage-mve-int", help="Write MVE integer coverage to the int-mve report lane"),
    suite: str = typer.Option("int", "--suite", help="Test suite selection: int, float, or both"),
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
        coverage_mve_int=coverage_mve_int,
        run_jobs=run_jobs,
        suite=suite,
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
    op: Optional[str] = typer.Option(None, help="Only these operators, comma-separated"),
    dtype: Optional[str] = typer.Option(None, help="Only this case dtype: activations, or S4 weights"),
    name: Optional[str] = typer.Option(None, help="Only these exact test names, comma-separated"),
    limit: Optional[int] = typer.Option(None, help="Limit number of models to generate"),
    seed: Optional[int] = typer.Option(None, help="Random seed for test generation"),
    cpu: str = typer.Option("cortex-m55", help="Target CPU(s), comma-separated (e.g. m0,m4,m55)"),
    suite: str = typer.Option("int", "--suite", help="Test suite selection: int, float, or both"),
    float_precision: str = typer.Option("both", "--float-precision", help="Float precision selection for float suite: f16, f32, or both"),
    opt: str = typer.Option("-Ofast", help="Optimization level"),
    jobs: Optional[int] = typer.Option(None, help="Parallel build jobs"),
    timeout: Optional[float] = typer.Option(None, help=f"Per-case FVP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}; 0 disables it and lets a hung kernel block the run)"),
    run_jobs: Optional[int] = typer.Option(None, "--run-jobs", help=f"Parallel FVP run jobs (default: min(host cores, {DEFAULT_RUN_JOBS_CAP}); 0 = every host core). FVP boot dominates per-case time so parallelism is the lever, but unbounded jobs on a shared or metered runner is a cost risk"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast", help="Do not stop on first failure"),
    coverage: bool = typer.Option(False, "--coverage", help="Enable ns-cmsis-nn coverage collection/reporting"),
    coverage_mve_float: bool = typer.Option(False, "--coverage-mve-float", help="Enable Cortex-M55 float MVE paths during coverage builds"),
    coverage_mve_int: bool = typer.Option(False, "--coverage-mve-int", help="Enable Cortex-M55 integer MVE paths (no ARM_MATH_AUTOVECTORIZE) during coverage builds"),
    force_generate: bool = typer.Option(False, "--force-generate", help="Regenerate every case even when its reuse stamp still matches"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip TFLite generation"),
    skip_host_check: bool = typer.Option(False, "--skip-host-check", help="Skip the host check of the generated int cases (runs after generation, before the FVP build, and blocks it on failure)"),
    host_kernels: str = typer.Option("m0", "--host-kernels", help="Host check kernel builds, comma-separated: m0 and/or dsp"),
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
        jobs=jobs,
        timeout=timeout,
        run_jobs=run_jobs,
        fail_fast=not no_fail_fast,
        coverage=coverage,
        coverage_mve_float=coverage_mve_float,
        coverage_mve_int=coverage_mve_int,
        force_generate=force_generate,
        skip_generation=skip_generation,
        skip_host_check=skip_host_check,
        host_kernels=[m.strip() for m in host_kernels.split(",") if m.strip()],
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

    # Host toolchain: the reference library (goldens) and the host check need it,
    # so a missing C/C++ compiler fails doctor.
    from .generation.reference.host_build import describe_cache
    from .utils.host_compiler import describe_host_toolchain

    typer.echo("\nHost toolchain (reference goldens and host check):")
    toolchain = describe_host_toolchain()
    for key, label in (("cc", "C compiler"), ("cxx", "C++ compiler")):
        if toolchain.get(key):
            typer.echo(f"✓ {label}: {toolchain[key]}")
        else:
            typer.echo(f"✗ {label}: {toolchain.get(f'{key}_error')}", err=True)
            all_ok = False
    cache = describe_cache()
    if "error" in cache:
        typer.echo(f"✗ reference library: {cache['error']}", err=True)
        all_ok = False
    else:
        state = "built" if cache["built"] else "not built yet (builds on first use)"
        typer.echo(f"✓ reference library {cache['key']}: {state} ({cache['library']})")

    # Hardware (J-Link/RTT) checks are informational: the FVP path never needs
    # them, so a missing tool is reported as missing without failing doctor.
    from .hardware.doctor import hardware_checks

    typer.echo("\nHardware (helia_core_tester hardware ...):")
    for check in hardware_checks(repo_root):
        marker = "✓" if check.ok else "⚠"
        typer.echo(f"{marker} {check.label}: {check.detail}")

    # The kernel contract export is optional for a checkout (older ns-cmsis-nn have
    # none) but never allowed to be present and wrong.
    from .contract.ir import ContractError, load_contract_set
    from .generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

    typer.echo("\nKernel contract (ns-cmsis-nn Tests/KernelContracts):")
    cmsis_root = resolve_cmsis_nn_root()
    try:
        contracts = load_contract_set(cmsis_root)
    except ContractError as error:
        typer.echo(f"✗ kernel contract: {error}", err=True)
        all_ok = False
    else:
        if contracts.present:
            kinds = {}
            for decl in contracts.functions.values():
                kinds[decl.kind] = kinds.get(decl.kind, 0) + 1
            typer.echo(f"✓ {contracts.path}: {len(contracts.functions)} public functions "
                       f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})")
        else:
            typer.echo(f"⚠ kernel contract: absent ({cmsis_root or 'no ns-cmsis-nn checkout resolved'}; "
                       "contract-driven commands are unavailable)")

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
    include_mve_int: bool = typer.Option(
        False,
        "--include-mve-int",
        help="Also merge cortex-m55 MVE integer coverage (reports/coverage/int-mve).",
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
    if include_mve_int and "int-mve" not in merge_suites:
        merge_suites.append("int-mve")

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
        for source_key, path in sorted(report.missing_coverage_inputs.items()):
            typer.echo(f"  {source_key}: {path}", err=True)
    else:
        typer.echo("✓ Coverage merge completed")
    sys.exit(exit_code)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
