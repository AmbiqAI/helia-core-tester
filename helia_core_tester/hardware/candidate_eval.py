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
comparable, 4 no_gain, 5 error (build, board, transport), 130
interrupted (no verdict).
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import shutil
import stat
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import click
import typer

from .boards import UnknownBoardError, repo_root, resolve_board
from .candidate_check import CheckError, _git, candidate_app, check_candidate
from .cli import _check_placement
from .errors import RunRefused
from . import nsx_cli
from .nsx_app import KERNEL_TREES, AppRenderError, write_kernels
from .pmu_explain import AGENT_PMU_SELECTION, explain_bundle
from .score import DEFAULT_MIN_SCORE, EXIT_REFUSED, EXITS, load_bundle, load_scoring, score_bundles

SCHEMA = "hct.candidate_eval"
BASELINE_SCHEMA = "hct.candidate_baseline"
# Verdict 2: hints pct_of_peak in percent.
SCHEMA_VERSION = 2
BASELINE_VERSION = 1
BASELINE_FILE = "baseline.json"
# Every verdict stage, in order.
STAGES = ("tester", "baseline", "check", "run", "objects", "score", "eval")
# Files eval and score read.
BUNDLE_FILES = ("case_summary.csv", "session_manifest.json", "cases.json")
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
            # Keep hardware run's refusal.
            error = RunRefused if rc == RUN_REFUSED else RuntimeError
            raise error(f"Baseline run {session} exited {rc} without a bundle.")
        if summary["totals"]["failed"]:
            raise RuntimeError(f"Baseline run {session} failed {summary['totals']['failed']} case(s).")
        shutil.copytree(summary["bundle"], out / "bundles" / session)
        sessions.append(session)
    meta = {
        "schema": BASELINE_SCHEMA, "schema_version": BASELINE_VERSION, "created_at": _stamp(),
        "base_commit": base, "run": spec.to_json(), "sessions": sessions,
    }
    (out / BASELINE_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def read_baseline(path: Path) -> dict:
    """baseline.json, validated; ValueError if unusable."""
    try:
        meta = json.loads((path / BASELINE_FILE).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path}: no readable {BASELINE_FILE}") from exc
    if not isinstance(meta, dict) or meta.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"{path}: not a candidate baseline")
    if meta.get("schema_version") != BASELINE_VERSION:
        raise ValueError(f"{path}: unsupported baseline version {meta.get('schema_version')!r}")
    base, sessions, run = meta.get("base_commit"), meta.get("sessions"), meta.get("run")
    if not (isinstance(base, str) and re.fullmatch(r"[0-9a-f]{40}", base)):
        raise ValueError(f"{path}: base_commit is not a full SHA")
    if not (isinstance(sessions, list) and sessions and all(isinstance(s, str) and s for s in sessions)):
        raise ValueError(f"{path}: no baseline sessions")
    try:
        RunSpec.from_json(run, path)
        resolve_board(run["board"])
    except (TypeError, KeyError, AttributeError, UnknownBoardError) as exc:
        raise ValueError(f"{path}: bad run options ({exc})") from exc
    for session in sessions:
        bundle = path / "bundles" / session
        missing = [name for name in BUNDLE_FILES if not (bundle / name).is_file()]
        if missing:
            raise ValueError(f"{path}: bundle {session} lacks {missing[0]}")
    return meta


def tester_dirty() -> bool:
    """Dirty or unknown tester state."""
    from .harness_lock import tester_state

    try:
        return tester_state(repo_root())["dirty"] is not False
    except Exception:  # noqa: BLE001 -- unreadable state is unknown
        return True


# --- eval ---------------------------------------------------------------------------


class TooLarge(CheckError):
    """The candidate trees exceed the copy limits."""


@dataclass
class CopyBudget:
    """Size, entry and depth limits for one snapshot."""

    file_bytes: int = 4 << 20
    total_bytes: int = 64 << 20
    entries: int = 5000
    depth: int = 64
    used_bytes: int = 0
    used_entries: int = 0

    def entry(self, path: str) -> None:
        """Count a file, dir or link."""
        self.used_entries += 1
        if self.used_entries > self.entries:
            raise TooLarge(f"Over {self.entries} files and dirs")

    def fits(self, size: int, path: str) -> None:
        """Refuse a file before reading it."""
        if size > self.file_bytes:
            raise TooLarge(f"{path}: over {self.file_bytes} bytes")
        if self.used_bytes + size > self.total_bytes:
            raise TooLarge(f"Over {self.total_bytes} bytes in total")

    def spend(self, copied: int, chunk: int, path: str) -> None:
        """Charge bytes actually copied."""
        self.used_bytes += chunk
        if copied > self.file_bytes:
            raise TooLarge(f"{path}: over {self.file_bytes} bytes")
        if self.used_bytes > self.total_bytes:
            raise TooLarge(f"Over {self.total_bytes} bytes in total")


def copy_tree(src: Path, dst: Path, budget: CopyBudget) -> None:
    """Copy by dir fd; never follow links.

    A racing agent cannot swap a dir for a symlink mid-copy. Symlinks
    copy as links; FIFOs, sockets and devices are skipped. Every entry
    and every copied byte is charged; dirs count before they are made.
    """
    for root, dirs, files, root_fd in os.fwalk(src, follow_symlinks=False):
        rel = Path(root).relative_to(src)
        out = dst / rel
        # Charged when the parent listed it.
        out.mkdir(parents=True, exist_ok=True)
        for name in dirs:
            budget.entry(name)
            if len(rel.parts) + 1 > budget.depth:
                raise TooLarge(f"Dirs nest deeper than {budget.depth}")
        for name in files:
            budget.entry(name)
        # Dir links sit in dirs, unwalked.
        for name in dirs + files:
            mode = os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                (out / name).symlink_to(os.readlink(name, dir_fd=root_fd))
            elif stat.S_ISREG(mode):
                _copy_file(name, root_fd, out / name, budget, f"{rel}/{name}")


def _read_chunks(handle) -> Iterator[bytes]:
    while chunk := handle.read(1 << 16):
        yield chunk


def _copy_file(name: str, dir_fd: int, dst: Path, budget: CopyBudget, path: str) -> None:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(fd)
        # Swapped for a FIFO since stat.
        if not stat.S_ISREG(info.st_mode):
            return
        # Sparse files: apparent size counts.
        budget.fits(max(info.st_size, info.st_blocks * 512), path)
        copied = 0
        with dst.open("wb") as out:
            for chunk in _read_chunks(handle):
                copied += len(chunk)
                budget.spend(copied, len(chunk), path)
                out.write(chunk)


def snapshot(candidate: Path, baseline: Path, base: str, budget: Optional[CopyBudget] = None) -> Path:
    """Base checkout with the candidate's trees copied in."""
    snap = baseline / "snapshot"
    shutil.rmtree(snap, ignore_errors=True)
    _git(baseline, "clone", "-q", "--branch", "base", str(baseline / "kernels.git"), "snapshot")
    if _git(snap, "rev-parse", "HEAD").decode().strip() != base:
        raise CheckError("Baseline base copy does not match.")
    # One budget spans every tree.
    budget = budget or CopyBudget()
    for name in SNAPSHOT_TREES:
        src, dst = candidate / name, snap / name
        shutil.rmtree(dst, ignore_errors=True)
        if src.is_symlink():
            # The check flags symlinks.
            dst.symlink_to(src.readlink())
        elif src.is_dir():
            copy_tree(src, dst, budget)
    return snap


def object_check(snap: Path, base: str, board: str) -> Optional[dict]:
    """Recheck with the built objects."""
    from .firmware_build import resolve_build_dir

    build_dir = resolve_build_dir(repo_root(), resolve_board(board), None)
    report = check_candidate(snap, base, build_dir=build_dir)
    if (report.get("objects") or {}).get("kernels_hash") != report.get("tree_hash"):
        report["ok"] = False
        report["findings"].append({"rule": "objects_mismatch", "message": "built objects differ from the snapshot"})
    return report


def _hidden_ids(bundles: list) -> set[str]:
    return {case_id for b in bundles for case_id, row in b.rows.items() if row.get("hidden") == "true"}


def snapshot_hash(snap: Path) -> str:
    """The tree hash a build of snap records."""
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "module"
        write_kernels(snap, module)
        return nsx_cli.tree_hash(module)


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
    public = [c for c in report["cases"] if c["case_id"] not in hidden and not c.get("hidden")]
    failures, hidden_kinds = [], Counter()
    for failure in report["failures"]:
        if failure.get("case_id") in hidden:
            hidden_kinds[failure["kind"]] += 1
        else:
            failures.append(failure)
    verdict = report["verdict"]
    # Lost cases, hidden too: tests moved.
    kinds = [f["kind"] for f in report["failures"] if f["kind"] != "no_eligible_cases"]
    if kinds and all(kind == "missing_case" for kind in kinds):
        verdict = "refused"
    return {
        "verdict": verdict, "stage": "score", "score": report["score"], "board": report["board"],
        "candidate_session": report["candidate"][0], "baseline_sessions": report["baseline"],
        "families": report["families"], "failures": failures, "cases": [_case_view(c) for c in public],
        "hidden": {"cases": len(hidden), "failures": dict(hidden_kinds),
                   "subscores": (report.get("subscores") or {}).get("hidden")} if hidden else None,
        "hints": _hints(candidate, hidden),
    }


def evaluate(kernels: Path, baseline: Path, meta: dict, min_score: float, run=None, budget: Optional[CopyBudget] = None) -> dict:
    """Snapshot, check, run, score: one verdict dict."""
    run = run or hardware_run
    head = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "board": meta["run"]["board"], "base_commit": meta["base_commit"]}
    session = f"eval-{_stamp()}"
    try:
        snap = snapshot(kernels, baseline, meta["base_commit"], budget)
        check = check_candidate(snap, meta["base_commit"])
        # TODO(lock-scope): the check reports tree_hash.
        if check["ok"] and not check.get("tree_hash"):
            check = {**check, "tree_hash": snapshot_hash(snap)}
    except (CheckError, AppRenderError, OSError) as exc:
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
    report = score_bundles(baselines, [candidate], scoring, check=check)
    return {**head, **verdict_from(report, _hidden_ids(baselines + [candidate]), candidate.path)}


# --- commands -----------------------------------------------------------------------


@candidate_app.command("baseline")
def baseline_command(
    kernels: Path = typer.Option(..., "--kernels", exists=True, file_okay=False, resolve_path=True, help="Clean ns-cmsis-nn checkout at the base."),
    board: str = typer.Option(..., "--board", help="Board id from assets/hardware_boards.yaml."),
    out: Path = typer.Option(..., "--out", file_okay=False, resolve_path=True, help="New directory for bundles and baseline.json."),
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
    if tester_dirty():
        typer.echo("✗ Tester worktree is dirty or unknown; commit it.", err=True)
        raise typer.Exit(EXIT_REFUSED)
    if out.exists() and any(out.iterdir()):
        raise typer.BadParameter(f"{out} is not empty", param_hint="--out")
    try:
        spec = RunSpec(board, kernels, placement, inline_asm, tuple(op or ()), tuple(dtype or ()), tuple(case_id or ()),
                       hidden_set, agent_pmu(board))
    except UnknownBoardError as exc:
        raise typer.BadParameter(str(exc), param_hint="--board") from exc
    # Same rules as hardware run.
    _check_placement(placement, resolve_board(board))
    out.mkdir(parents=True, exist_ok=True)
    try:
        meta = write_baseline(spec, out, repeats)
    except (CheckError, RunRefused) as exc:
        typer.echo(f"✗ {exc} Logs: {out / 'logs'}", err=True)
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
    max_file_bytes: int = typer.Option(CopyBudget.file_bytes, "--max-file-bytes", min=1, help="Largest candidate file to copy."),
    max_total_bytes: int = typer.Option(CopyBudget.total_bytes, "--max-total-bytes", min=1, help="Most candidate bytes to copy."),
    max_files: int = typer.Option(CopyBudget.entries, "--max-files", min=1, help="Most candidate files and dirs to copy."),
) -> None:
    """Check, run and score a candidate; print one JSON verdict.

    Exit 0 pass, 1 fail, 2 usage, 3 refused, rejected or not comparable,
    4 no gain, 5 error.
    """
    if not math.isfinite(min_score):
        raise typer.BadParameter("--min-score must be finite", param_hint="--min-score")
    budget = CopyBudget(max_file_bytes, max_total_bytes, max_files)
    try:
        verdict = _eval_verdict(kernels, baseline, board, min_score, budget)
    except KeyboardInterrupt:
        # Shells report Ctrl-C as 130.
        raise typer.Exit(130)
    except click.ClickException:
        raise
    except Exception as exc:  # noqa: BLE001 -- one verdict, always
        _log_crash(baseline)
        verdict = _refusal("eval", type(exc).__name__, "error")
    _emit(verdict)


def _refusal(stage: str, reason: str, verdict: str = "refused") -> dict:
    return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "verdict": verdict, "stage": stage, "reason": reason}


def _log_crash(baseline: Path) -> None:
    """Traceback to logs, if writable."""
    try:
        log = baseline / "logs" / f"eval-error-{_stamp()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        pass


def _eval_verdict(kernels: Path, baseline: Path, board: Optional[str], min_score: float, budget: CopyBudget) -> dict:
    """Tester, baseline, then evaluate."""
    # No opt-out: verdicts need a committed tester.
    if tester_dirty():
        return _refusal("tester", "Tester worktree is dirty or unknown")
    try:
        meta = read_baseline(baseline)
    except ValueError as exc:
        return _refusal("baseline", str(exc))
    if board is not None and board != meta["run"]["board"]:
        raise typer.BadParameter(f"baseline ran on {meta['run']['board']}", param_hint="--board")
    return evaluate(kernels, baseline, meta, min_score, budget=budget)
