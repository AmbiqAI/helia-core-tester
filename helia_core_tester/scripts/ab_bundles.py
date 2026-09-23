"""Compare two hardware result bundles per case and counter.

Usage: uv run python -m helia_core_tester.scripts.ab_bundles A_DIR B_DIR

Retired-instruction counters (names ending in _RETIRED) must not move
across a harness-only change, so they gate at --max-delta-pct (default 0).
Every other counter, median_cycles included, gates at
--max-cycle-delta-pct, which is off by default. Cases flagged with
overflow, valid_for_regression=false or a correctness mismatch are
reported but excluded from gating. A gated counter that is empty on
one side only, or non-finite on either side, counts as a violation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from helia_core_tester.hardware.result_bundle import CASE_SUMMARY_BASE_FIELDS, CASE_SUMMARY_FLAG_FIELDS

# median_cycles is the one base field that is a measurement.
_NON_COUNTER_COLUMNS = (set(CASE_SUMMARY_BASE_FIELDS) | set(CASE_SUMMARY_FLAG_FIELDS)) - {"median_cycles"}
_FLAG_COLUMNS = (("overflow_detected", "true", "overflow"), ("valid_for_regression", "false", "invalid"), ("comparison_passed", "false", "mismatch"))


@dataclass(frozen=True)
class Bundle:
    session_id: str
    counters: list[str]
    rows: dict[str, dict[str, str]]

    def value(self, case_id: str, counter: str) -> float | None:
        cell = self.rows[case_id].get(counter, "")
        return None if cell in ("", None) else float(cell)

    def flags(self, case_id: str) -> list[str]:
        row = self.rows[case_id]
        return [flag for column, bad, flag in _FLAG_COLUMNS if row.get(column) == bad]


def load_bundle(root: Path) -> Bundle:
    with (root / "case_summary.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {row["case_id"]: row for row in reader}
        counters = [name for name in reader.fieldnames or [] if name not in _NON_COUNTER_COLUMNS]
    session_id = root.name
    manifest = root / "session_manifest.json"
    if manifest.exists():
        session_id = json.loads(manifest.read_text(encoding="utf-8")).get("session_id", session_id)
    return Bundle(session_id=session_id, counters=counters, rows=rows)


def delta_pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0 if b == 0 else math.inf
    return (b - a) / a * 100.0


def is_retired_counter(counter: str) -> bool:
    return counter.endswith("_RETIRED")


def _format_value(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "-"
    return "inf" if math.isinf(delta) else f"{delta:+.3f}%"


def _section(title: str, entries: list[str]) -> list[str]:
    return ["", f"== {title} ({len(entries)})", *(f"  {entry}" for entry in entries)] if entries else []


def compare(a: Bundle, b: Bundle, counters: list[str], *, retired_limit: float, cycle_limit: float | None) -> tuple[list[str], int]:
    """Return report lines and exit status."""
    lines = [f"A: {a.session_id}", f"B: {b.session_id}", ""]
    shared = [case_id for case_id in a.rows if case_id in b.rows]
    deltas: dict[str, list[float]] = {counter: [] for counter in counters}
    violations: list[str] = []
    flagged: list[str] = []
    width = max(len("counter"), *(len(counter) for counter in counters))
    for case_id in shared:
        case_flags = sorted(set(a.flags(case_id)) | set(b.flags(case_id)))
        if case_flags:
            flagged.append(f"{case_id} ({', '.join(case_flags)})")
        lines.append(f"== {case_id}" + (f" [{', '.join(case_flags)}]" if case_flags else ""))
        lines.append(f"{'counter':<{width}} {'A':>14} {'B':>14} {'delta':>10}")
        for counter in counters:
            value_a, value_b = a.value(case_id, counter), b.value(case_id, counter)
            missing = value_a is None or value_b is None
            finite = not missing and math.isfinite(value_a) and math.isfinite(value_b)
            delta = delta_pct(value_a, value_b) if finite else None
            lines.append(f"{counter:<{width}} {_format_value(value_a):>14} {_format_value(value_b):>14} {_format_delta(delta):>10}")
            if case_flags or (value_a is None and value_b is None):
                continue
            limit = retired_limit if is_retired_counter(counter) else cycle_limit
            if delta is None:
                if limit is not None:
                    reason = "missing on one side" if missing else "non-finite value"
                    violations.append(f"{case_id} {counter} {reason}")
                continue
            deltas[counter].append(delta)
            if limit is not None and abs(delta) > limit:
                violations.append(f"{case_id} {counter} {_format_value(value_a)} -> {_format_value(value_b)} {_format_delta(delta)} > {limit:g}%")
        lines.append("")

    lines.append("== median delta per counter")
    for counter in counters:
        median = statistics.median(deltas[counter]) if deltas[counter] else None
        lines.append(f"{counter:<{width}} {_format_delta(median):>10}")

    lines += _section("counters only in A", [c for c in a.counters if c not in b.counters])
    lines += _section("counters only in B", [c for c in b.counters if c not in a.counters])
    lines += _section("cases only in A", [c for c in a.rows if c not in b.rows])
    lines += _section("cases only in B", [c for c in b.rows if c not in a.rows])
    lines += _section("flagged, not gated", flagged)

    lines.append("")
    if not shared:
        lines.append("== FAIL: no shared cases")
        return lines, 1
    if violations:
        lines.append(f"== FAIL: {len(violations)} counter deltas over limit")
        lines.extend(f"  {entry}" for entry in violations)
        return lines, 1
    lines.append(f"== PASS: {len(shared)} shared cases within limits")
    return lines, 0


def select_counters(a: Bundle, b: Bundle, requested: list[str] | None) -> list[str]:
    common = [counter for counter in a.counters if counter in b.counters]
    if not requested:
        return common
    requested = list(dict.fromkeys(requested))
    unknown = [counter for counter in requested if counter not in common]
    if unknown:
        raise SystemExit(f"Counter not in both bundles: {', '.join(unknown)}")
    return requested


def parse_limit(text: str) -> float:
    """Finite, non-negative percentage."""
    try:
        limit = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not math.isfinite(limit) or limit < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be finite and >= 0")
    return limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two hardware result bundles.")
    parser.add_argument("bundle_a", type=Path, help="Baseline bundle directory")
    parser.add_argument("bundle_b", type=Path, help="Candidate bundle directory")
    parser.add_argument("--counter", action="append", dest="counters", metavar="NAME", help="Counter to compare (repeatable); default: all shared.")
    parser.add_argument("--max-delta-pct", type=parse_limit, default=0.0, help="Limit for *_RETIRED counters (default 0).")
    parser.add_argument("--max-cycle-delta-pct", type=parse_limit, default=None, help="Limit for every other counter (default none).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_a, bundle_b = load_bundle(args.bundle_a), load_bundle(args.bundle_b)
    counters = select_counters(bundle_a, bundle_b, args.counters)
    lines, status = compare(bundle_a, bundle_b, counters, retired_limit=args.max_delta_pct, cycle_limit=args.max_cycle_delta_pct)
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(main())
