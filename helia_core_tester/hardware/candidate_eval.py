"""Evaluate a kernel candidate on hardware: one command, one verdict.

`candidate baseline` runs a clean ns-cmsis-nn tree N times with pinned
options and keeps the bundles plus baseline.json in --out. `candidate
eval` checks a candidate tree against that base commit, runs it once
with the same options and the first baseline run's outputs as goldens,
scores it against every baseline repeat and prints one JSON verdict.

Both shell out to `hardware run --json`, so the board session, build
and bundle work exactly as for a human run. Hidden cases (bundle
column `hidden`) count in the verdict, but their ids never print.

Exit codes: 0 pass, 1 fail, 2 usage, 3 refused or rejected, 4 no_gain,
5 error (build, board, transport).
"""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from .boards import UnknownBoardError, repo_root, resolve_board
from .candidate_check import CheckError, check_candidate
from .pmu_explain import AGENT_PMU_SELECTION, explain_bundle
from .score import DEFAULT_MIN_SCORE, EXITS, load_bundle, load_scoring, score_bundles

SCHEMA = "hct.candidate_eval"
BASELINE_SCHEMA = "hct.candidate_baseline"
SCHEMA_VERSION = 1
BASELINE_FILE = "baseline.json"
EXIT_REFUSED, EXIT_ERROR = 3, 5
# hardware run exits 3 when it refuses.
RUN_REFUSED = 3
VERDICT_EXITS = {**EXITS, "rejected": EXIT_REFUSED, "refused": EXIT_REFUSED, "error": EXIT_ERROR}


@dataclass(frozen=True)
class RunSpec:
    """Everything pinned between baseline and candidate."""

    board: str
    kernels: Path
    placement: str = "tcm"
    inline_asm: bool = True
    ops: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    hidden_set: Optional[Path] = None
    pmu: tuple[str, ...] = field(default_factory=tuple)

    def selection(self) -> dict[str, Any]:
        return {"ops": list(self.ops), "dtypes": list(self.dtypes), "case_ids": list(self.case_ids)}


def agent_pmu(board: str) -> tuple[str, ...]:
    """Counters explain reads; none on DWT boards."""
    if resolve_board(board).pmu_tier == "dwt":
        return ()
    return tuple(f"{group}:{','.join(names)}" for group, names in AGENT_PMU_SELECTION.items())


def run_args(
    spec: RunSpec, session_id: str, *, golden_from: Optional[Path] = None, skip_generate: bool = False, skip_flash: bool = False,
) -> list[str]:
    """`hardware run` flags with every option pinned."""
    args = [
        "hardware", "run", "--board", spec.board, "--cmsis-nn-root", str(spec.kernels),
        "--placement", spec.placement, "--inline-asm" if spec.inline_asm else "--no-inline-asm",
        "--fvp-gate", "off", "--session-id", session_id, "--json",
    ]
    for flag, values in (("--pmu-counters", spec.pmu), ("--op", spec.ops), ("--dtype", spec.dtypes), ("--case-id", spec.case_ids)):
        for value in values:
            args += [flag, value]
    if spec.hidden_set is not None:
        # TODO(hidden-run): flag lands with the hidden-set PR.
        args += ["--hidden-set", str(spec.hidden_set)]
    if golden_from is not None:
        args += ["--golden-from", str(golden_from)]
    if skip_generate:
        args.append("--skip-generate")
    if skip_flash:
        args.append("--skip-flash")
    return args


def hardware_run(args: list[str]) -> tuple[int, Optional[dict]]:
    """Run the CLI; stdout holds its JSON summary."""
    proc = subprocess.run(
        [sys.executable, "-m", "helia_core_tester", *args], cwd=repo_root(), stdout=subprocess.PIPE, text=True, check=False,
    )
    try:
        summary = json.loads(proc.stdout)
    except ValueError:
        summary = None
    return proc.returncode, summary if isinstance(summary, dict) and summary.get("bundle") else None


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _head(tree: Path) -> str:
    out = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise CheckError(f"{tree} is not a git checkout")
    return out.stdout.strip()


def _emit(verdict: dict) -> None:
    verdict["exit_code"] = VERDICT_EXITS[verdict["verdict"]]
    typer.echo(json.dumps(verdict, indent=2))
    raise typer.Exit(verdict["exit_code"])


# --- baseline -----------------------------------------------------------------------


def write_baseline(spec: RunSpec, out: Path, repeats: int, run=None) -> dict:
    """Run the clean tree `repeats` times; save bundles."""
    run = run or hardware_run
    base = _head(spec.kernels)
    report = check_candidate(spec.kernels, base)
    if report["files"]:
        raise CheckError("Baseline tree has changes; commit or stash them.")
    stamp, sessions = _stamp(), []
    for index in range(repeats):
        session = f"baseline-{stamp}-{index + 1}"
        # Repeats reuse the first build.
        rc, summary = run(run_args(spec, session, skip_generate=index > 0, skip_flash=index > 0))
        if summary is None:
            raise RuntimeError(f"Baseline run {session} exited {rc} without a bundle.")
        if summary["totals"]["failed"]:
            raise RuntimeError(f"Baseline run {session} failed {summary['totals']['failed']} case(s).")
        shutil.copytree(summary["bundle"], out / "bundles" / session)
        sessions.append(session)
    meta = {
        "schema": BASELINE_SCHEMA, "schema_version": SCHEMA_VERSION, "created_at": _stamp(),
        "board": spec.board, "base_commit": base, "base_tree": str(spec.kernels),
        "placement": spec.placement, "inline_asm": spec.inline_asm, "pmu": list(spec.pmu), "selection": spec.selection(),
        "hidden_set": str(spec.hidden_set) if spec.hidden_set else None, "sessions": sessions,
    }
    (out / BASELINE_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def read_baseline(path: Path) -> dict:
    meta = json.loads((path / BASELINE_FILE).read_text(encoding="utf-8"))
    if meta.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"{path}: not a candidate baseline")
    return meta


def spec_from(meta: dict, kernels: Path) -> RunSpec:
    selection = meta["selection"]
    return RunSpec(
        board=meta["board"], kernels=kernels, placement=meta["placement"], inline_asm=meta["inline_asm"],
        ops=tuple(selection["ops"]), dtypes=tuple(selection["dtypes"]), case_ids=tuple(selection["case_ids"]),
        hidden_set=Path(meta["hidden_set"]) if meta.get("hidden_set") else None, pmu=tuple(meta["pmu"]),
    )


# --- eval ---------------------------------------------------------------------------


def _hidden_ids(bundles: list) -> set[str]:
    return {case_id for b in bundles for case_id, row in b.rows.items() if row.get("hidden") == "true"}


def _score(baselines: list, candidate, scoring: dict, check: dict) -> dict:
    # TODO(score-trust): always pass check once merged.
    if "check" in inspect.signature(score_bundles).parameters:
        return score_bundles(baselines, [candidate], scoring, check=check)
    return score_bundles(baselines, [candidate], scoring)


def _case_view(case: dict) -> dict:
    keys = ("case_id", "family", "timed_symbol", "excluded_by", "timing_lost", "baseline_cycles", "candidate_cycles",
            "speedup", "delta_pct", "band_pct", "regression", "cycles_per_mac_candidate")
    return {key: case.get(key) for key in keys}


def _hints(bundle: Path, hidden: set[str]) -> list[dict]:
    """PMU diagnosis per public MAC case."""
    try:
        result = explain_bundle(bundle)
    except (OSError, ValueError, KeyError):
        return []
    return [
        {"case_id": e.case_id, "pct_of_peak": e.pct_of_peak, "diagnosis": e.diagnosis, "hints": e.hints}
        for e in result["cases"] if e.case_id not in hidden
    ]


def verdict_from(report: dict, hidden: set[str], meta: dict, candidate: Path) -> dict:
    """The agent-facing verdict; hidden ids redacted."""
    public = [c for c in report["cases"] if c["case_id"] not in hidden and not c.get("hidden")]
    secret = len(report["cases"]) - len(public)
    failures, hidden_kinds = [], Counter()
    for failure in report["failures"]:
        if failure.get("case_id") in hidden:
            hidden_kinds[failure["kind"]] += 1
        else:
            failures.append(failure)
    return {
        "verdict": report["verdict"], "stage": "score", "score": report["score"], "board": report["board"],
        "base_commit": meta["base_commit"], "candidate_session": report["candidate"][0], "baseline_sessions": report["baseline"],
        "families": report["families"], "failures": failures, "cases": [_case_view(c) for c in public],
        "hidden": {"cases": secret, "failures": dict(hidden_kinds), "subscores": (report.get("subscores") or {}).get("hidden")}
        if hidden else None,
        "hints": _hints(candidate, hidden),
    }


def evaluate(kernels: Path, baseline: Path, min_score: float, run=None) -> dict:
    """Check, run, score: one verdict dict."""
    run = run or hardware_run
    meta = read_baseline(baseline)
    head = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "board": meta["board"], "base_commit": meta["base_commit"]}
    try:
        check = check_candidate(kernels, meta["base_commit"])
    except CheckError as exc:
        return {**head, "verdict": "refused", "stage": "check", "reason": str(exc)}
    if not check["ok"]:
        return {**head, "verdict": "rejected", "stage": "check", "findings": check["findings"]}
    spec = spec_from(meta, kernels.resolve())
    bundles = [baseline / "bundles" / session for session in meta["sessions"]]
    # Baseline generated the cases; goldens check inputs.
    rc, summary = run(run_args(spec, f"eval-{_stamp()}", golden_from=bundles[0], skip_generate=True))
    if summary is None:
        verdict = "refused" if rc == RUN_REFUSED else "error"
        return {**head, "verdict": verdict, "stage": "run", "reason": f"hardware run exited {rc}"}
    baselines, candidate = [load_bundle(path) for path in bundles], load_bundle(Path(summary["bundle"]))
    scoring = load_scoring(meta["board"]) | {"min_score": min_score}
    report = _score(baselines, candidate, scoring, check)
    return {**head, **verdict_from(report, _hidden_ids(baselines + [candidate]), meta, candidate.path)}


# --- commands -----------------------------------------------------------------------


def baseline_command(
    kernels: Path = typer.Option(..., "--kernels", exists=True, file_okay=False, resolve_path=True, help="Clean ns-cmsis-nn checkout at the base."),
    board: str = typer.Option(..., "--board", help="Board id from assets/hardware_boards.yaml."),
    out: Path = typer.Option(..., "--out", resolve_path=True, help="New directory for bundles and baseline.json."),
    repeats: int = typer.Option(3, "--repeats", min=1, help="Runs to pool for the noise band."),
    placement: str = typer.Option("tcm", "--placement", help="tcm or mram."),
    inline_asm: bool = typer.Option(True, "--inline-asm/--no-inline-asm", help="Requantize inline asm."),
    op: Optional[list[str]] = typer.Option(None, "--op", help="Only these operators (repeatable)."),
    dtype: Optional[list[str]] = typer.Option(None, "--dtype", help="Only these dtypes (repeatable)."),
    case_id: Optional[list[str]] = typer.Option(None, "--case-id", help="Only these case ids (repeatable)."),
    hidden_set: Optional[Path] = typer.Option(
        None, "--hidden-set", exists=True, file_okay=False, resolve_path=True, help="Hidden case dir from `generate --hidden-dir`.",
    ),
) -> None:
    """Run a clean base tree N times for `candidate eval`."""
    if out.exists() and any(out.iterdir()):
        raise typer.BadParameter(f"{out} is not empty", param_hint="--out")
    try:
        spec = RunSpec(board, kernels, placement, inline_asm, tuple(op or ()), tuple(dtype or ()), tuple(case_id or ()),
                       hidden_set, agent_pmu(board))
    except UnknownBoardError as exc:
        raise typer.BadParameter(str(exc), param_hint="--board") from exc
    out.mkdir(parents=True, exist_ok=True)
    try:
        meta = write_baseline(spec, out, repeats)
    except CheckError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(EXIT_REFUSED)
    except RuntimeError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(EXIT_ERROR)
    typer.echo(json.dumps(meta, indent=2))


def eval_command(
    kernels: Path = typer.Option(..., "--kernels", exists=True, file_okay=False, resolve_path=True, help="Candidate ns-cmsis-nn worktree."),
    baseline: Path = typer.Option(..., "--baseline", exists=True, file_okay=False, resolve_path=True, help="Dir from `candidate baseline`."),
    board: Optional[str] = typer.Option(None, "--board", help="Must match the baseline's board."),
    min_score: float = typer.Option(DEFAULT_MIN_SCORE, "--min-score", help="Score to beat; 0.005 is about 0.5 % gain."),
) -> None:
    """Check, run and score a candidate; print one JSON verdict.

    Exit 0 pass, 1 fail, 2 usage, 3 refused or rejected, 4 no gain, 5 error.
    """
    try:
        meta = read_baseline(baseline)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--baseline") from exc
    if board is not None and board != meta["board"]:
        raise typer.BadParameter(f"baseline ran on {meta['board']}", param_hint="--board")
    _emit(evaluate(kernels, baseline, min_score))
