"""
CLI for mutation scoring (issue #76): ``python -m helia_core_tester.mutation``.

Kept as its own Typer app rather than a subcommand registered in
helia_core_tester/cli.py so it does not collide with in-flight CLI work;
wiring it into the main CLI is a one-line follow-up once that lands.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

from helia_core_tester.core.cpu_targets import get_cpu_profile
from helia_core_tester.mutation.catalog import MUTANTS_V1, get_mutants
from helia_core_tester.mutation.runner import run_mutation_scoring
from helia_core_tester.mutation.host_build import discover_cases

app = typer.Typer(
    name="mutation",
    help="Mutation scoring: measure whether generated cases can discriminate a correct kernel from a broken one.",
    add_completion=False,
)

DEFAULT_OPS = "Add,Sub,Mul,SquaredDifference,ChunkedEquivalence,Maximum,Minimum,Convolve"

# The widest capability set in the matrix, so the default run generates every
# case the catalog's killers need. A narrower CPU is a valid choice, but it
# silently removes killers from the corpus, which is why the capabilities of
# this CPU are passed to the scorer and mutants they cannot cover come back
# NOT_APPLICABLE rather than SURVIVED.
DEFAULT_CPU = "cortex-m55"


def _tester_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _generate_cases(tester_root: Path, workdir: Path, ops: List[str], cpu: str, seed: int) -> List[Path]:
    """Generate int-suite cases per op via the generation pytest entry point."""
    gen_root = workdir / "gen"
    roots: List[Path] = []
    for op in ops:
        out_dir = gen_root / op
        typer.echo(f"[mutation] generating cases for {op} (cpu={cpu}, seed={seed})")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "test_ops.py::test_generation",
            "-q",
            "--cpu",
            cpu,
            "--suite",
            "int",
            "--op",
            op,
            "--seed",
            str(seed),
            "--generated-tests-dir",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, cwd=tester_root / "helia_core_tester" / "generation")
        if proc.returncode != 0:
            typer.echo(f"✗ generation failed for op {op}", err=True)
            raise typer.Exit(1)
        roots.append(out_dir)
    return roots


@app.command("list")
def list_mutants():
    """List the catalogued mutants."""
    for m in MUTANTS_V1:
        refs = f" [{', '.join(m.refs)}]" if m.refs else ""
        needs = f" (requires {', '.join(m.requires_capabilities)})" if m.requires_capabilities else ""
        typer.echo(f"{m.mutant_id}: {m.description}{needs}{refs}")


@app.command()
def run(
    cmsis_nn_root: Path = typer.Option(..., "--cmsis-nn-root", help="Path to an ns-cmsis-nn checkout (read-only; mutants are applied to a copy)"),
    ops: str = typer.Option(DEFAULT_OPS, "--ops", help="Comma-separated operator filter (generation op names)"),
    cases_root: Optional[Path] = typer.Option(None, "--cases-root", help="Reuse already-generated cases under this directory instead of generating"),
    mutants: Optional[str] = typer.Option(None, "--mutants", help="Comma-separated mutant ids (default: full v1 catalog)"),
    workdir: Path = typer.Option(Path("artifacts/mutation"), "--workdir", help="Scratch + report directory"),
    cpu: str = typer.Option(DEFAULT_CPU, "--cpu", help="CPU whose capabilities the generated corpus has; also the generation target, and with --cases-root it must name the CPU those cases were generated for"),
    seed: int = typer.Option(500, "--seed", help="Generation seed (fixed for determinism)"),
    jobs: int = typer.Option(8, "--jobs", help="Parallel compile/run jobs"),
    cc: str = typer.Option("gcc", "--cc", help="Host C compiler"),
    fail_on_survivor: bool = typer.Option(False, "--fail-on-survivor", help="Exit nonzero if any mutant survives"),
):
    """Score the generated cases against the mutant catalog on the host."""
    tester_root = _tester_root()
    if shutil.which(cc) is None:
        typer.echo(f"✗ host compiler '{cc}' not found", err=True)
        raise typer.Exit(1)

    mutant_list = get_mutants([m.strip() for m in mutants.split(",")] if mutants else None)

    if cases_root is not None:
        case_roots = [cases_root]
    else:
        op_list = [o.strip() for o in ops.split(",") if o.strip()]
        case_roots = _generate_cases(tester_root, workdir, op_list, cpu, seed)

    case_dirs = discover_cases(case_roots)
    if not case_dirs:
        typer.echo("✗ no generated cases found", err=True)
        raise typer.Exit(1)

    report = run_mutation_scoring(
        cmsis_nn_root=cmsis_nn_root,
        case_dirs=case_dirs,
        mutants=mutant_list,
        tester_root=tester_root,
        workdir=workdir,
        cc=cc,
        jobs=jobs,
        capabilities=get_cpu_profile(cpu).capabilities,
        log=typer.echo,
    )
    typer.echo("")
    typer.echo(report.render_text())
    typer.echo(f"\nJSON report: {workdir / 'mutation_report.json'}")

    if report.apply_failures:
        typer.echo("✗ one or more mutants failed to apply or build; the catalog has drifted", err=True)
        raise typer.Exit(2)
    # Only a genuine survivor fails the run: a NOT_APPLICABLE mutant was never
    # sampled, and apply/build failures already exited above.
    if fail_on_survivor and report.survivors:
        raise typer.Exit(3)


if __name__ == "__main__":
    app()
