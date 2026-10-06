"""Evaluate a kernel candidate on hardware: one command, one verdict.

`candidate baseline` runs a clean ns-cmsis-nn tree N times with pinned
options and keeps, in --out, the bundles, baseline.json and a shallow
copy of the base commit. `candidate eval` copies the candidate's
Source/, Include/, cmake/ and nsx/ into a fresh checkout of that base,
checks the copy, builds and runs the copy with the baseline's options
and the first baseline run's outputs as goldens, scores it against
every baseline repeat and prints one JSON verdict.

The agent's own repo is only read as files: no git command runs in
it, and later edits do not reach the build. Both commands shell out
to `hardware run --json`; its stderr, which names every case, goes to
--out/logs. Hidden cases (bundle column `hidden`) count in the verdict
and in family totals, but their ids never print.

Exit codes: 0 pass, 1 fail, 2 usage, 3 refused, rejected or not
comparable, 4 no_gain, 5 error (build, board, transport).
"""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from .boards import UnknownBoardError, repo_root, resolve_board
from .candidate_check import CheckError, _git, candidate_app, check_candidate
from .nsx_app import KERNEL_TREES
from .pmu_explain import AGENT_PMU_SELECTION, explain_bundle
from .score import DEFAULT_MIN_SCORE, EXIT_REFUSED, EXITS, load_bundle, load_scoring, score_bundles

SCHEMA = "hct.candidate_eval"
BASELINE_SCHEMA = "hct.candidate_baseline"
SCHEMA_VERSION = 1
BASELINE_FILE = "baseline.json"
EXIT_ERROR = 5
# Needs hardware run's refusal code.
RUN_REFUSED = 3
VERDICT_EXITS = {**EXITS, "rejected": EXIT_REFUSED, "refused": EXIT_REFUSED, "error": EXIT_ERROR}
# The build copies these trees.
SNAPSHOT_TREES = (*KERNEL_TREES, "nsx")


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
    pmu: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict[str, Any], kernels: Path) -> "RunSpec":
        hidden = data.get("hidden_set")
        tuples = {key: tuple(data[key]) for key in ("ops", "dtypes", "case_ids", "pmu")}
        return cls(**{**data, **tuples, "kernels": kernels, "hidden_set": Path(hidden) if hidden else None})


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


def hardware_run(args: list[str], log: Path) -> tuple[int, Optional[dict]]:
    """Run the CLI; stdout is JSON, stderr goes to log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            [sys.executable, "-m", "helia_core_tester", *args], cwd=repo_root(), stdout=subprocess.PIPE, stderr=err,
            text=True, check=False,
        )
    try:
        summary = json.loads(proc.stdout)
    except ValueError:
        summary = None
    return proc.returncode, summary if isinstance(summary, dict) and summary.get("bundle") else None


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _emit(verdict: dict) -> None:
    verdict["exit_code"] = VERDICT_EXITS[verdict["verdict"]]
    typer.echo(json.dumps(verdict, indent=2))
    raise typer.Exit(verdict["exit_code"])


# --- baseline -----------------------------------------------------------------------


def write_baseline(spec: RunSpec, out: Path, repeats: int, run=None) -> dict:
    """Run the clean tree `repeats` times; save bundles."""
    run = run or hardware_run
    base = _git(spec.kernels, "rev-parse", "HEAD").decode().strip()
    report = check_candidate(spec.kernels, base)
    if report["files"] or not report["ok"]:
        raise CheckError("Baseline tree has changes; commit or stash them.")
    # Trusted base objects for every eval.
    _git(out, "init", "-q", "--bare", "kernels.git")
    _git(out / "kernels.git", "fetch", "-q", "--depth", "1", f"file://{spec.kernels}", f"{base}:refs/heads/base")
    stamp, sessions = _stamp(), []
    for index in range(repeats):
        session = f"baseline-{stamp}-{index + 1}"
        # Repeats reuse the first build.
        rc, summary = run(run_args(spec, session, skip_generate=index > 0, skip_flash=index > 0), out / "logs" / f"{session}.log")
        if summary is None:
            raise RuntimeError(f"Baseline run {session} exited {rc} without a bundle.")
        if summary["totals"]["failed"]:
            raise RuntimeError(f"Baseline run {session} failed {summary['totals']['failed']} case(s).")
        shutil.copytree(summary["bundle"], out / "bundles" / session)
        sessions.append(session)
    meta = {
        "schema": BASELINE_SCHEMA, "schema_version": SCHEMA_VERSION, "created_at": _stamp(),
        "base_commit": base, "run": spec.to_json(), "sessions": sessions,
    }
    (out / BASELINE_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def read_baseline(path: Path) -> dict:
    meta = json.loads((path / BASELINE_FILE).read_text(encoding="utf-8"))
    if meta.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"{path}: not a candidate baseline")
    return meta


# --- eval ---------------------------------------------------------------------------


def snapshot(candidate: Path, baseline: Path, base: str) -> Path:
    """Base checkout with the candidate's trees copied in."""
    snap = baseline / "snapshot"
    shutil.rmtree(snap, ignore_errors=True)
    _git(baseline, "clone", "-q", "--branch", "base", str(baseline / "kernels.git"), "snapshot")
    if _git(snap, "rev-parse", "HEAD").decode().strip() != base:
        raise CheckError("Baseline base copy does not match.")
    for name in SNAPSHOT_TREES:
        src, dst = candidate / name, snap / name
        shutil.rmtree(dst, ignore_errors=True)
        if src.is_symlink():
            # The check flags symlinks.
            dst.symlink_to(src.readlink())
        elif src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
    return snap


def object_check(snap: Path, base: str, board: str) -> Optional[dict]:
    """Recheck with the built objects."""
    from .firmware_build import resolve_build_dir

    # TODO(object-scan): always run once merged.
    if "build_dir" not in inspect.signature(check_candidate).parameters:
        return None
    build_dir = resolve_build_dir(repo_root(), resolve_board(board), None)
    report = check_candidate(snap, base, build_dir=build_dir)
    if (report.get("objects") or {}).get("kernels_hash") != report.get("tree_hash"):
        report["ok"] = False
        report["findings"].append({"rule": "objects_mismatch", "message": "built objects differ from the snapshot"})
    return report


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
    except Exception:  # noqa: BLE001 -- hints are optional
        return []
    return [
        {"case_id": e.case_id, "pct_of_peak": e.pct_of_peak, "diagnosis": e.diagnosis, "hints": e.hints}
        for e in result["cases"] if e.case_id not in hidden
    ]


def verdict_from(report: dict, hidden: set[str], candidate: Path) -> dict:
    """The agent-facing verdict; hidden ids redacted."""
    public = [c for c in report["cases"] if c["case_id"] not in hidden]
    failures, hidden_kinds = [], Counter()
    for failure in report["failures"]:
        if failure.get("case_id") in hidden:
            hidden_kinds[failure["kind"]] += 1
        else:
            failures.append(failure)
    verdict = report["verdict"]
    # Lost cases mean the tests moved.
    if failures and all(f["kind"] == "missing_case" for f in failures) and not hidden_kinds:
        verdict = "refused"
    return {
        "verdict": verdict, "stage": "score", "score": report["score"], "board": report["board"],
        "candidate_session": report["candidate"][0], "baseline_sessions": report["baseline"],
        "families": report["families"], "failures": failures, "cases": [_case_view(c) for c in public],
        # TODO(score-hidden): subscores land with score-hidden.
        "hidden": {"cases": len(report["cases"]) - len(public), "failures": dict(hidden_kinds),
                   "subscores": (report.get("subscores") or {}).get("hidden")} if hidden else None,
        "hints": _hints(candidate, hidden),
    }


def evaluate(kernels: Path, baseline: Path, meta: dict, min_score: float, run=None) -> dict:
    """Snapshot, check, run, score: one verdict dict."""
    run = run or hardware_run
    head = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "board": meta["run"]["board"], "base_commit": meta["base_commit"]}
    session = f"eval-{_stamp()}"
    try:
        snap = snapshot(kernels, baseline, meta["base_commit"])
        check = check_candidate(snap, meta["base_commit"])
    except (CheckError, OSError) as exc:
        return {**head, "verdict": "refused", "stage": "check", "reason": str(exc)}
    if not check["ok"]:
        return {**head, "verdict": "rejected", "stage": "check", "findings": check["findings"]}
    spec = RunSpec.from_json(meta["run"], snap)
    bundles = [baseline / "bundles" / name for name in meta["sessions"]]
    # Baseline generated the cases; goldens check inputs.
    rc, summary = run(run_args(spec, session, golden_from=bundles[0], skip_generate=True), baseline / "logs" / f"{session}.log")
    if summary is None:
        verdict = "refused" if rc == RUN_REFUSED else "error"
        return {**head, "verdict": verdict, "stage": "run", "reason": f"hardware run exited {rc}"}
    built = object_check(snap, meta["base_commit"], head["board"])
    if built is not None and not built["ok"]:
        return {**head, "verdict": "rejected", "stage": "objects", "findings": built["findings"]}
    baselines, candidate = [load_bundle(path) for path in bundles], load_bundle(Path(summary["bundle"]))
    scoring = load_scoring(head["board"]) | {"min_score": min_score}
    report = _score(baselines, candidate, scoring, check)
    return {**head, **verdict_from(report, _hidden_ids(baselines + [candidate]), candidate.path)}


# --- commands -----------------------------------------------------------------------


@candidate_app.command("baseline")
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
        typer.echo(f"✗ {exc} Logs: {out / 'logs'}", err=True)
        raise typer.Exit(EXIT_ERROR)
    typer.echo(json.dumps(meta, indent=2))


@candidate_app.command("eval")
def eval_command(
    kernels: Path = typer.Option(..., "--kernels", exists=True, file_okay=False, resolve_path=True, help="Candidate ns-cmsis-nn worktree."),
    baseline: Path = typer.Option(..., "--baseline", exists=True, file_okay=False, resolve_path=True, help="Dir from `candidate baseline`."),
    board: Optional[str] = typer.Option(None, "--board", help="Must match the baseline's board."),
    min_score: float = typer.Option(DEFAULT_MIN_SCORE, "--min-score", help="Score to beat; 0.005 is about 0.5 % gain."),
) -> None:
    """Check, run and score a candidate; print one JSON verdict.

    Exit 0 pass, 1 fail, 2 usage, 3 refused, rejected or not comparable,
    4 no gain, 5 error.
    """
    try:
        meta = read_baseline(baseline)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--baseline") from exc
    if board is not None and board != meta["run"]["board"]:
        raise typer.BadParameter(f"baseline ran on {meta['run']['board']}", param_hint="--board")
    try:
        verdict = evaluate(kernels, baseline, meta, min_score)
    except Exception as exc:  # noqa: BLE001 -- one verdict, always
        log = baseline / "logs" / f"eval-error-{_stamp()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        verdict = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "verdict": "error", "stage": "eval",
                   "reason": type(exc).__name__}
    _emit(verdict)
