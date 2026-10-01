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
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from helia_core_tester.hardware.boards import BoardSpec, default_session_id, repo_root, resolve_board
from helia_core_tester.hardware.hardware_pipeline import StreamOptions, fit_to_board, parse_pmu_counters
from helia_core_tester.hardware.pmu_catalog import default_selection
from helia_core_tester.hardware.session_runner import canonical_suite
from helia_core_tester.scripts.ab_bundles import load_bundle

SCHEMA = "hct.hardware.board_matrix"
MVE_RETIRED = "ARM_PMU_MVE_INST_RETIRED"


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
    selection = parse_pmu_counters(pmu_counters) if pmu_counters else default_selection()
    requested = tuple(arg for value in pmu_counters for arg in ("--pmu-counters", value))
    legs: list[Leg] = []
    for arg in board_args:
        board_id, _, serial = arg.partition(":")
        board = resolve_board(board_id)
        if any(leg.board.id == board.id for leg in legs):
            raise ValueError(f"Board {board.id} given twice.")
        pmu_args, note = requested, None
        try:
            fit_to_board(board, StreamOptions(suite=suite, pmu_counters=dict(selection)), explicit_pmu=bool(pmu_counters))
        except ValueError as exc:
            # Unsupported counters: use board default.
            fit_to_board(board, StreamOptions(suite=suite), explicit_pmu=False)
            pmu_args, note = (), f"{exc} Ran its default counters."
        legs.append(Leg(board, int(serial) if serial else None, default_session_id(board, now), pmu_args, note))
    return legs


def runs_parallel(legs: list[Leg]) -> bool:
    """Parallel only with distinct explicit probes."""
    serials = [leg.serial_no for leg in legs]
    return None not in serials and len(set(serials)) == len(serials)


def run_leg(leg: Leg, shared_args: list[str], log_path: Path, root: Path) -> int:
    command = [
        sys.executable, "-m", "helia_core_tester", "hardware", "run", "--board", leg.board.id,
        "--session-id", leg.session_id, *shared_args, *leg.pmu_args,
    ]
    if leg.serial_no is not None:
        command += ["--serial-no", str(leg.serial_no)]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        return subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def board_entry(bundle: Path) -> dict:
    """One board's row from its bundle."""
    manifest = _read_json(bundle / "session_manifest.json")
    summary = _read_json(bundle / "session_summary.json")
    target = manifest.get("target") or {}
    boot = manifest.get("boot") or {}
    kernels = (manifest.get("build") or {}).get("kernels") or {}
    rejected = len(summary.get("rejected_cases", []))
    failed = int(summary.get("failed_cases", 0)) - rejected
    return {
        "board": target.get("board", bundle.name),
        "cpu": target.get("cpu"),
        "pmu_tier": target.get("pmu_tier"),
        "status": "passed" if failed == 0 and rejected == 0 else "failed",
        "session_id": manifest.get("session_id", bundle.name),
        "bundle": str(bundle),
        "golden": {"total": summary.get("case_count"), "passed": summary.get("passed_cases"), "failed": failed, "rejected": rejected},
        "boot": {"status": boot.get("status"), "core_clock_hz": boot.get("core_clock_hz")},
        "build_id": manifest.get("firmware_build_id"),
        "kernels": {key: kernels.get(key) for key in ("ref", "commit", "root")},
        "note": None,
    }


def case_table(bundles: dict[str, Path]) -> list[dict]:
    """Shared cases: median cycles and MVE retired."""
    loaded = {board: load_bundle(path) for board, path in bundles.items()}
    if not loaded:
        return []
    first, *rest = loaded.values()
    shared = [case_id for case_id in first.rows if all(case_id in other.rows for other in rest)]

    def column(case_id: str, counter: str) -> dict:
        return {board: bundle.value(case_id, counter) for board, bundle in loaded.items()}

    return [{"case_id": c, "median_cycles": column(c, "median_cycles"), MVE_RETIRED: column(c, MVE_RETIRED)} for c in shared]


def build_summary(matrix_id: str, selection: dict, boards: list[dict]) -> dict:
    bundles = {entry["board"]: Path(entry["bundle"]) for entry in boards if entry.get("bundle")}
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
        return kernels["root"]
    commit = (kernels.get("commit") or "")[:12]
    return "@".join(part for part in (kernels.get("ref"), commit) if part) or "-"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def render_markdown(summary: dict) -> str:
    selection = summary["selection"]
    lines = [
        f"# Board matrix {summary['matrix_id']}",
        "",
        " · ".join(f"{key} `{_cell(value)}`" for key, value in selection.items()),
        "",
    ]
    rows = []
    for entry in summary["boards"]:
        golden, boot = entry.get("golden") or {}, entry.get("boot") or {}
        clock = boot.get("core_clock_hz")
        rows.append([
            entry["board"], entry["status"], _cell(golden.get("passed")), _cell(golden.get("failed")),
            _cell(golden.get("rejected")), _cell(boot.get("status")), _cell(clock / 1e6 if clock else None),
            _cell(entry.get("build_id")), _kernels(entry.get("kernels")), entry.get("note") or "",
        ])
    header = ["board", "status", "passed", "failed", "rejected", "boot", "clock MHz", "build id", "kernels", "note"]
    lines += _table(header, rows)
    boards = [entry["board"] for entry in summary["boards"] if entry.get("bundle")]
    for counter in ("median_cycles", MVE_RETIRED):
        rows = [[case["case_id"], *(_cell(case[counter][b]) for b in boards)] for case in summary["cases"]]
        if counter != "median_cycles" and all(cell == "-" for row in rows for cell in row[1:]):
            continue
        lines += ["", f"## {counter} ({len(rows)} shared cases)", ""] + _table(["case_id", *boards], rows)
    return "\n".join(lines) + "\n"


def write_summary(summary: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "board_matrix.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    (out_dir / "board_matrix.md").write_text(markdown, encoding="utf-8")
    return markdown


def run_matrix(args: argparse.Namespace) -> int:
    root = repo_root()
    now = datetime.now(timezone.utc)
    matrix_id = f"matrix-{now.strftime('%Y%m%dT%H%M%SZ')}"
    try:
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
    with ThreadPoolExecutor(max_workers=len(legs) if parallel else 1) as pool:
        codes = list(pool.map(lambda leg: run_leg(leg, shared, out_dir / "logs" / f"{leg.board.id}.log", root), legs))

    boards = []
    for leg, code in zip(legs, codes):
        bundle = root / "artifacts" / "reports" / "hardware" / leg.session_id
        if (bundle / "session_summary.json").is_file():
            entry = board_entry(bundle)
        else:
            entry = {"board": leg.board.id, "status": "error", "session_id": leg.session_id, "bundle": None}
        if code != 0 and entry["status"] == "passed":
            entry["status"] = "error"
        entry.update(serial_no=leg.serial_no, exit_code=code, note=leg.note, log=str(out_dir / "logs" / f"{leg.board.id}.log"))
        boards.append(entry)
    selection = {
        "suite": args.suite, "limit": args.limit, "family": args.family,
        "pmu_counters": args.pmu_counters or None, "extra_args": args.extra or None,
    }
    summary = build_summary(matrix_id, selection, boards)
    print(write_summary(summary, out_dir))
    return 0 if all(entry["status"] == "passed" for entry in boards) else 1


def summarize(args: argparse.Namespace) -> int:
    boards = [board_entry(path) for path in args.bundles]
    ids = [entry["board"] for entry in boards]
    if len(set(ids)) != len(ids):
        print("board_matrix: one bundle per board, please.", file=sys.stderr)
        return 2
    summary = build_summary(args.matrix_id or args.out.name, {"bundles": len(boards)}, boards)
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
