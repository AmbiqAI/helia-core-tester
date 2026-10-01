"""Run one selection across boards and summarize.

Usage:
  uv run python -m helia_core_tester.scripts.board_matrix run \\
      --board apollo510_evb:SERIAL --board apollo330mP_evb:SERIAL \\
      --suite int --limit 2 --pmu-counters mve:default [-- HARDWARE_RUN_ARGS]
  uv run python -m helia_core_tester.scripts.board_matrix summarize BUNDLE_DIR... --out DIR

Writes board_matrix.json (schema hct.hardware.board_matrix v1) and
board_matrix.md; see README "Board matrix". Exit 1 unless every board passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from helia_core_tester.hardware.boards import BoardSpec, default_session_id, repo_root, resolve_board
from helia_core_tester.hardware.hardware_pipeline import StreamOptions, fit_to_board, parse_pmu_counters
from helia_core_tester.hardware.probes import SERIAL_ENV_VAR
from helia_core_tester.hardware.session_runner import canonical_suite
from helia_core_tester.scripts.ab_bundles import load_bundle

SCHEMA = "hct.hardware.board_matrix"
MVE_RETIRED = "ARM_PMU_MVE_INST_RETIRED"
SUMMARY_KEYS = ("case_count", "passed_cases", "failed_cases", "rejected_cases")
# Per-leg options; build dirs stay per board.
OWNED_OPTIONS = (
    "--board", "--serial-no", "--session-id", "--suite", "--limit", "--family",
    "--precision", "--pmu-counters", "--pmu-groups", "--build-dir",
)


@dataclass(frozen=True)
class Leg:
    board: BoardSpec
    serial_no: int | None
    session_id: str
    pmu_args: tuple[str, ...]
    note: str | None = None


def plan_legs(board_args: list[str], pmu_counters: list[str], suite: str, now: datetime) -> list[Leg]:
    """Resolve boards and fit counters per board."""
    suite = canonical_suite(suite)
    selection = parse_pmu_counters(pmu_counters) if pmu_counters else None
    requested = tuple(arg for value in pmu_counters for arg in ("--pmu-counters", value))
    legs: list[Leg] = []
    for arg in board_args:
        board_id, _, serial = arg.partition(":")
        board = resolve_board(board_id)
        if any(leg.board.id == board.id for leg in legs):
            raise ValueError(f"Board {board.id} given twice.")
        pmu_args, note = requested, None
        if selection:
            try:
                fit_to_board(board, StreamOptions(suite=suite, pmu_counters=dict(selection)), explicit_pmu=True)
            except ValueError as exc:
                # Unsupported counters: use board default.
                pmu_args, note = (), f"{exc} Ran its default counters."
        legs.append(Leg(board, int(serial) if serial else None, default_session_id(board, now), pmu_args, note))
    return legs


def runs_parallel(legs: list[Leg]) -> bool:
    """Parallel only with distinct explicit probes."""
    serials = [leg.serial_no for leg in legs]
    return None not in serials and len(set(serials)) == len(serials)


def run_leg(leg: Leg, shared_args: list[str], log_path: Path, root: Path, env: dict) -> int:
    command = [
        sys.executable, "-m", "helia_core_tester", "hardware", "run", "--board", leg.board.id,
        "--session-id", leg.session_id, *shared_args, *leg.pmu_args,
    ]
    if leg.serial_no is not None:
        command += ["--serial-no", str(leg.serial_no)]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        try:
            return subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        except OSError as exc:
            # Launch failure: leg becomes error row.
            log.write(f"board_matrix: launch failed: {exc}\n")
            return 1


class BadBundle(ValueError):
    """Bundle breaks the result-bundle contract."""


def _summary_problem(summary: dict, row_count: int) -> str | None:
    """Why the summary counts disagree, if they do."""
    counts = [summary[key] for key in SUMMARY_KEYS[:3]]
    rejected = summary["rejected_cases"]
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts):
        return "case counts must be non-negative integers"
    if not isinstance(rejected, list) or not all(isinstance(case, str) for case in rejected):
        return "rejected_cases must list case ids"
    total, passed, failed = counts
    if passed + failed != total or len(rejected) > failed:
        return f"counts disagree: {passed} + {failed} != {total} or {len(rejected)} rejected"
    if row_count != total:
        return f"{row_count} CSV rows, {total} cases"
    return None


def empty_entry(board: str) -> dict:
    """Row with every key, values unknown."""
    return {
        "board": board, "cpu": None, "pmu_tier": None, "status": "error", "session_id": None, "bundle": None,
        "golden": None, "boot": None, "build_id": None, "kernels": None,
        "serial_no": None, "exit_code": None, "note": None, "log": None,
    }


def board_entry(bundle: Path) -> dict:
    """One board's row; read errors are BadBundle."""
    try:
        manifest = json.loads((bundle / "session_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((bundle / "session_summary.json").read_text(encoding="utf-8"))
        table = load_bundle(bundle)
        missing = [key for key in SUMMARY_KEYS if key not in summary]
        target = manifest.get("target") or {}
        if not isinstance(target.get("board"), str):
            missing.append("target.board")
        if missing:
            raise BadBundle(f"{bundle}: missing {', '.join(missing)}")
        problem = _summary_problem(summary, len(table.rows))
        if problem is not None:
            raise BadBundle(f"{bundle}: {problem}")
        # Parse every value the summary reads.
        values = [table.value(case_id, counter) for case_id in table.rows for counter in ("median_cycles", MVE_RETIRED)]
        boot = manifest.get("boot") or {}
        kernels = (manifest.get("build") or {}).get("kernels") or {}
        rejected = len(summary["rejected_cases"])
        failed = summary["failed_cases"] - rejected
        entry = empty_entry(target["board"])
        entry.update({
            "cpu": target.get("cpu"),
            "pmu_tier": target.get("pmu_tier"),
            "status": "passed" if failed == 0 and rejected == 0 else "failed",
            "session_id": manifest.get("session_id", bundle.name),
            "bundle": str(bundle),
            "golden": {"total": summary["case_count"], "passed": summary["passed_cases"], "failed": failed, "rejected": rejected},
            "boot": {"status": boot.get("status"), "core_clock_hz": boot.get("core_clock_hz")},
            "build_id": manifest.get("firmware_build_id"),
            "kernels": {key: kernels.get(key) for key in ("ref", "commit", "root")},
        })
        # Refuse NaN or Infinity anywhere.
        json.dumps([entry, values], allow_nan=False)
    except BadBundle:
        raise
    except (OSError, ValueError, KeyError, AttributeError, TypeError, csv.Error) as exc:
        raise BadBundle(f"{bundle}: {exc}") from exc
    return entry


def case_table(bundles: dict[str, Path | None]) -> list[dict]:
    """Shared cases; boards without bundles get null."""
    loaded = {board: load_bundle(path) for board, path in bundles.items() if path is not None}
    if not loaded:
        return []
    first, *rest = loaded.values()
    shared = [case_id for case_id in first.rows if all(case_id in other.rows for other in rest)]

    def column(case_id: str, counter: str) -> dict:
        return {board: loaded[board].value(case_id, counter) if board in loaded else None for board in bundles}

    return [{"case_id": c, "median_cycles": column(c, "median_cycles"), MVE_RETIRED: column(c, MVE_RETIRED)} for c in shared]


def build_summary(matrix_id: str, selection: dict, boards: list[dict]) -> dict:
    bundles = {entry["board"]: Path(entry["bundle"]) if entry.get("bundle") else None for entry in boards}
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "matrix_id": matrix_id,
        "selection": selection,
        "boards": boards,
        "cases": case_table(bundles),
    }


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return " ".join(value)
    return format(value, ".10g") if isinstance(value, float) else str(value)


def _kernels(kernels: dict | None) -> str:
    kernels = kernels or {}
    if kernels.get("root"):
        return str(kernels["root"])
    commit = str(kernels.get("commit") or "")[:12]
    return "@".join(str(part) for part in (kernels.get("ref"), commit) if part) or "-"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def render_markdown(summary: dict) -> str:
    selection = summary["selection"]
    lines = [f"# Board matrix {summary['matrix_id']}", ""]
    if any(selection.values()):
        lines += [" · ".join(f"{key} `{_cell(value)}`" for key, value in selection.items() if value), ""]
    rows = []
    for entry in summary["boards"]:
        golden, boot = entry.get("golden") or {}, entry.get("boot") or {}
        clock = boot.get("core_clock_hz")
        rows.append([
            entry["board"], entry["status"], _cell(golden.get("passed")), _cell(golden.get("failed")),
            _cell(golden.get("rejected")), _cell(boot.get("status")), _cell(clock / 1e6 if isinstance(clock, (int, float)) and clock else clock),
            _cell(entry.get("build_id")), _kernels(entry.get("kernels")), entry.get("note") or "",
        ])
    header = ["board", "status", "passed", "failed", "rejected", "boot", "clock MHz", "build id", "kernels", "note"]
    lines += _table(header, rows)
    boards = [entry["board"] for entry in summary["boards"]]
    for counter in ("median_cycles", MVE_RETIRED):
        rows = [[case["case_id"], *(_cell(case[counter][b]) for b in boards)] for case in summary["cases"]]
        if counter != "median_cycles" and all(cell == "-" for row in rows for cell in row[1:]):
            continue
        lines += ["", f"## {counter} ({len(rows)} shared cases)", ""] + _table(["case_id", *boards], rows)
    return "\n".join(lines) + "\n"


def write_summary(summary: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "board_matrix.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    (out_dir / "board_matrix.md").write_text(markdown, encoding="utf-8")
    return markdown


def run_matrix(args: argparse.Namespace) -> int:
    root = repo_root()
    now = datetime.now(timezone.utc)
    matrix_id = f"matrix-{now.strftime('%Y%m%dT%H%M%SZ')}"
    try:
        owned = [arg for arg in args.extra if arg.split("=", 1)[0] in OWNED_OPTIONS]
        if owned:
            raise ValueError(f"The matrix owns {', '.join(owned)}; drop it after --.")
        legs = plan_legs(args.boards, args.pmu_counters or [], args.suite, now)
    except ValueError as exc:
        print(f"board_matrix: {exc}", file=sys.stderr)
        return 2
    out_dir = args.out or root / "artifacts" / "reports" / "hardware" / matrix_id
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    shared = ["--suite", args.suite]
    if args.limit is not None:
        shared += ["--limit", str(args.limit)]
    if args.family:
        shared += ["--family", args.family]
    shared += args.extra
    parallel = not args.sequential and runs_parallel(legs)
    print(f"board_matrix: {len(legs)} boards, {'parallel' if parallel else 'sequential'}; logs in {out_dir / 'logs'}", file=sys.stderr)
    # Env serial would hit one probe.
    env = {k: v for k, v in os.environ.items() if len(legs) == 1 or k != SERIAL_ENV_VAR}
    with ThreadPoolExecutor(max_workers=len(legs) if parallel else 1) as pool:
        codes = list(pool.map(lambda leg: run_leg(leg, shared, out_dir / "logs" / f"{leg.board.id}.log", root, env), legs))

    boards = []
    for leg, code in zip(legs, codes):
        bundle = root / "artifacts" / "reports" / "hardware" / leg.session_id
        problem = None
        try:
            entry = board_entry(bundle)
        except BadBundle as exc:
            entry = empty_entry(leg.board.id) | {"session_id": leg.session_id}
            problem = str(exc) if bundle.exists() else None
        if code != 0 and entry["status"] == "passed":
            entry["status"] = "error"
        note = "; ".join(part for part in (leg.note, problem) if part) or None
        entry.update(serial_no=leg.serial_no, exit_code=code, note=note, log=str(out_dir / "logs" / f"{leg.board.id}.log"))
        boards.append(entry)
    summary = build_summary(matrix_id, selection_of(args), boards)
    print(write_summary(summary, out_dir))
    return 0 if all(entry["status"] == "passed" for entry in boards) else 1


def selection_of(args: argparse.Namespace) -> dict:
    """Run selection; None when summarizing."""
    keys = {"suite": "suite", "limit": "limit", "family": "family", "pmu_counters": "pmu_counters", "extra_args": "extra"}
    return {key: getattr(args, attr, None) or None for key, attr in keys.items()}


def summarize(args: argparse.Namespace) -> int:
    try:
        boards = [board_entry(path) for path in args.bundles]
    except BadBundle as exc:
        print(f"board_matrix: bad bundle {exc}", file=sys.stderr)
        return 2
    ids = [entry["board"] for entry in boards]
    if len(set(ids)) != len(ids):
        print("board_matrix: one bundle per board, please.", file=sys.stderr)
        return 2
    summary = build_summary(args.matrix_id or args.out.name, selection_of(args), boards)
    print(write_summary(summary, args.out))
    return 0 if all(entry["status"] == "passed" for entry in boards) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one selection across boards.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run every board, then summarize.")
    run.add_argument("--board", dest="boards", action="append", required=True, metavar="ID[:SERIAL]", help="Board id, optional probe serial (repeatable).")
    run.add_argument("--suite", default="int", help="int, float or both (default int).")
    run.add_argument("--limit", type=int, default=None, help="Cases per family.")
    run.add_argument("--family", default=None, help="One operator family.")
    run.add_argument("--pmu-counters", action="append", metavar="GROUP:SELECTION", help="Counters, hpx syntax (repeatable).")
    run.add_argument("--sequential", action="store_true", help="Run one board at a time.")
    run.add_argument("--out", type=Path, default=None, help="Summary directory.")
    run.add_argument("extra", nargs="*", help="Args after -- for hardware run.")
    run.set_defaults(handler=run_matrix)
    summary = commands.add_parser("summarize", help="Summarize existing bundles.")
    summary.add_argument("bundles", type=Path, nargs="+", help="Bundle directories.")
    summary.add_argument("--out", type=Path, required=True, help="Summary directory.")
    summary.add_argument("--matrix-id", default=None, help="Summary title (default: --out name).")
    summary.set_defaults(handler=summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
