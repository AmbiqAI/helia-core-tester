"""
Mutation-scoring orchestration (issue #76).

Given an ns-cmsis-nn checkout and a set of generated cases, the runner:

1. copies Source/ + Include/ into a scratch tree (the user's checkout is
   never modified),
2. builds the pristine host kernel library and runs every case as a
   baseline -- cases that fail unmutated are excluded from scoring and
   reported,
3. for each catalogued mutant: applies the patch to the scratch tree,
   rebuilds the library, re-runs every scored case, and restores the tree,
4. reports the kill matrix: which cases killed each mutant, the headline
   mutants-killed / total, and the cases that killed *nothing* across all
   mutants (vacuous-case candidates).

A mutant whose patch fails to apply is reported as APPLY_FAILED and makes
the run exit nonzero; source drift must be loud.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from helia_core_tester.mutation.catalog import Mutant
from helia_core_tester.mutation.host_build import (
    CaseResult,
    build_kernel_lib,
    build_runtime_obj,
    run_all_cases,
)
from helia_core_tester.mutation.patching import AppliedMutant, MutantApplyError

STATUS_KILLED = "KILLED"
STATUS_SURVIVED = "SURVIVED"
STATUS_APPLY_FAILED = "APPLY_FAILED"
STATUS_BUILD_FAILED = "BUILD_FAILED"


@dataclass
class MutantOutcome:
    mutant: Mutant
    status: str
    killed_by: List[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class MutationReport:
    baseline_total: int
    baseline_failed: List[CaseResult]
    scored_cases: List[str]
    outcomes: List[MutantOutcome]
    wall_time_s: float

    @property
    def killed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == STATUS_KILLED)

    @property
    def apply_failures(self) -> List[MutantOutcome]:
        return [o for o in self.outcomes if o.status in (STATUS_APPLY_FAILED, STATUS_BUILD_FAILED)]

    def vacuous_case_candidates(self) -> List[str]:
        """Cases that killed no mutant at all, across every applied mutant."""
        killers = set()
        for outcome in self.outcomes:
            killers.update(outcome.killed_by)
        return [c for c in self.scored_cases if c not in killers]

    def to_dict(self) -> Dict:
        return {
            "schema": "helia-core-tester/mutation-report/v1",
            "baseline": {
                "total_cases": self.baseline_total,
                "failed_cases": [
                    {"name": r.name, "family": r.family, "detail": r.detail}
                    for r in self.baseline_failed
                ],
                "scored_cases": self.scored_cases,
            },
            "mutants": [
                {
                    "id": o.mutant.mutant_id,
                    "description": o.mutant.description,
                    "family": o.mutant.family,
                    "refs": list(o.mutant.refs),
                    "expected_detected_by": o.mutant.expected_detected_by,
                    "status": o.status,
                    "killed_by": o.killed_by,
                    "detail": o.detail,
                }
                for o in self.outcomes
            ],
            "headline": {
                "mutants_killed": self.killed_count,
                "mutants_total": len(self.outcomes),
                "vacuous_case_candidates": self.vacuous_case_candidates(),
            },
            "wall_time_s": round(self.wall_time_s, 1),
        }

    def render_text(self, max_killers: int = 6) -> str:
        lines: List[str] = []
        lines.append(
            f"Mutation score: {self.killed_count}/{len(self.outcomes)} mutants killed "
            f"({self.baseline_total} cases, {len(self.scored_cases)} scored, "
            f"{self.wall_time_s:.0f}s wall)"
        )
        if self.baseline_failed:
            lines.append(f"Baseline failures (excluded from scoring): {len(self.baseline_failed)}")
            for r in self.baseline_failed:
                lines.append(f"  ! {r.name}: {r.detail}")
        lines.append("")
        for o in self.outcomes:
            if o.status == STATUS_KILLED:
                shown = ", ".join(o.killed_by[:max_killers])
                more = len(o.killed_by) - max_killers
                suffix = f" (+{more} more)" if more > 0 else ""
                lines.append(f"  KILLED    {o.mutant.mutant_id}  by {len(o.killed_by)} case(s): {shown}{suffix}")
            elif o.status == STATUS_SURVIVED:
                lines.append(f"  SURVIVED  {o.mutant.mutant_id}  <-- no case detects this bug class")
            else:
                lines.append(f"  {o.status}  {o.mutant.mutant_id}: {o.detail}")
        vacuous = self.vacuous_case_candidates()
        lines.append("")
        lines.append(f"Vacuous-case candidates (killed nothing across all mutants): {len(vacuous)}/{len(self.scored_cases)}")
        return "\n".join(lines)


def prepare_tree(cmsis_nn_root: Path, workdir: Path) -> Path:
    """Copy Source/ + Include/ of the checkout into the scratch tree."""
    cmsis_nn_root = Path(cmsis_nn_root)
    for required in ("Source", "Include"):
        if not (cmsis_nn_root / required).is_dir():
            raise FileNotFoundError(
                f"{cmsis_nn_root} does not look like an ns-cmsis-nn checkout (missing {required}/)"
            )
    tree = workdir / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    shutil.copytree(cmsis_nn_root / "Source", tree / "Source")
    shutil.copytree(cmsis_nn_root / "Include", tree / "Include")
    return tree


def run_mutation_scoring(
    cmsis_nn_root: Path,
    case_dirs: Sequence[Path],
    mutants: Sequence[Mutant],
    tester_root: Path,
    workdir: Path,
    cc: str = "gcc",
    jobs: int = 8,
    log=print,
) -> MutationReport:
    start = time.monotonic()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tree = prepare_tree(cmsis_nn_root, workdir)
    build_dir = workdir / "build"
    build_dir.mkdir(exist_ok=True)
    runtime_obj = build_runtime_obj(tester_root, build_dir, cc=cc)

    log(f"[mutation] baseline: building pristine kernel library + {len(case_dirs)} case(s)")
    lib = build_kernel_lib(tree, build_dir, cc=cc, jobs=jobs)
    baseline = run_all_cases(case_dirs, tree, lib, runtime_obj, tester_root, build_dir / "bin", cc=cc, jobs=jobs)
    baseline_failed = [r for r in baseline if not r.passed]
    scored = [d for d, r in zip(case_dirs, baseline) if r.passed]
    scored_names = [d.name for d in scored]
    for r in baseline_failed:
        log(f"[mutation] baseline FAIL (excluded): {r.name}: {r.detail}")

    outcomes: List[MutantOutcome] = []
    for mutant in mutants:
        log(f"[mutation] mutant {mutant.mutant_id}: applying + rebuilding")
        try:
            with AppliedMutant(tree, mutant):
                try:
                    lib = build_kernel_lib(tree, build_dir, cc=cc, jobs=jobs)
                except Exception as exc:  # build of a mutated tree failed
                    outcomes.append(
                        MutantOutcome(mutant, STATUS_BUILD_FAILED, detail=str(exc)[:2000])
                    )
                    continue
                results = run_all_cases(
                    scored, tree, lib, runtime_obj, tester_root, build_dir / "bin", cc=cc, jobs=jobs
                )
        except MutantApplyError as exc:
            outcomes.append(MutantOutcome(mutant, STATUS_APPLY_FAILED, detail=str(exc)))
            log(f"[mutation] APPLY FAILED: {exc}")
            continue
        killed_by = [r.name for r in results if not r.passed]
        status = STATUS_KILLED if killed_by else STATUS_SURVIVED
        outcomes.append(MutantOutcome(mutant, status, killed_by=killed_by))
        log(f"[mutation] mutant {mutant.mutant_id}: {status} ({len(killed_by)} case(s))")

    # The pristine library is rebuilt from the restored tree so a later
    # consumer of the workdir never sees mutated objects.
    build_kernel_lib(tree, build_dir, cc=cc, jobs=jobs)

    report = MutationReport(
        baseline_total=len(case_dirs),
        baseline_failed=baseline_failed,
        scored_cases=scored_names,
        outcomes=outcomes,
        wall_time_s=time.monotonic() - start,
    )
    (workdir / "mutation_report.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return report
