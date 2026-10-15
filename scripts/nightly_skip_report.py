#!/usr/bin/env python3
"""Hardware-nightly skip and untimed-entry report, with growth against a baseline run.

Reads the TCM leg's `hct-hardware-<board>-<date>-<run>` artifacts: the bundle's
`session_summary.json` (`skipped_cases`, each with its family) and `coverage.json`
(`deployed.missing`). Writes a Markdown section per board and prints one GitHub
`::warning::` per family whose skip count grew and per newly untimed entry point.
The baseline for each board is the newest earlier run (one `<run id>/` directory
each under --previous) with a readable result and the same selection, so a night
that left no result does not hide growth. A board with no current result, or with
no baseline because earlier runs left no result, also gets a warning: growth went
unchecked.

Stdlib only: the summary job runs it with the runner's python3, without uv.

`select-baselines` picks those earlier runs: the newest --count completed, non-cancelled
scheduled runs on main older than --before, paging the Actions API (via `gh api`) until
it has them or the history ends, at most --max-pages pages. It appends `runs=` (and
`note=` when the history holds none) to --output; an API error exits 1.

Usage: nightly_skip_report.py --current DIR [--previous DIR] --boards JSON --out FILE
       nightly_skip_report.py select-baselines --repo OWNER/NAME --before RUN_ID --output FILE
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ARTIFACT_PREFIX = "hct-hardware"
RUN_JSON = "hardware-nightly-run.json"
UNKNOWN_FAMILY = "(unknown)"
WORKFLOW_FILE = "hardware-nightly.yml"
PER_PAGE = 100
GH_TIMEOUT_S = 120


class ReportError(Exception):
    """A leg's result exists but cannot be read."""


@dataclass
class LegSkips:
    """Skip counts and untimed entry points from one leg's artifact."""

    artifact: str
    selection: Any
    skipped_by_family: dict[str, int]
    untimed: Optional[list[str]]
    skipped_total: int = field(init=False)

    def __post_init__(self) -> None:
        self.skipped_total = sum(self.skipped_by_family.values())


def _artifact_dir(root: Path, board: str) -> Optional[Path]:
    """The board's TCM artifact under root; the newest run when several match."""
    pattern = re.compile(rf"^{re.escape(ARTIFACT_PREFIX)}-{re.escape(board)}-\d{{8}}-(\d+)$")
    matches = []
    if root.is_dir():
        for child in root.iterdir():
            match = pattern.match(child.name)
            if match and child.is_dir():
                matches.append((int(match.group(1)), child))
    if not matches:
        return None
    return max(matches)[1]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ReportError(f"{path.name}: {exc}") from exc


def load_leg(root: Path, board: str) -> Optional[LegSkips]:
    """The board's leg under root, None when it left no result. ReportError when unreadable."""
    directory = _artifact_dir(root, board)
    if directory is None:
        return None
    run_json = directory / RUN_JSON
    if not run_json.is_file():
        return None
    run = _read_json(run_json)
    session = run.get("session_id") if isinstance(run, dict) else None
    if not isinstance(session, str) or not session or "/" in session or session in {".", ".."}:
        raise ReportError(f"{RUN_JSON}: no usable session_id")
    bundle = directory / "hardware" / session
    summary = _read_json(bundle / "session_summary.json")
    cases = summary.get("skipped_cases") if isinstance(summary, dict) else None
    if not isinstance(cases, list):
        raise ReportError("session_summary.json: no skipped_cases list")
    by_family: dict[str, int] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ReportError("session_summary.json: skipped_cases holds a non-object")
        family = case.get("family")
        key = family if isinstance(family, str) and family else UNKNOWN_FAMILY
        by_family[key] = by_family.get(key, 0) + 1
    untimed: Optional[list[str]] = None
    coverage_path = bundle / "coverage.json"
    if coverage_path.is_file():
        coverage = _read_json(coverage_path)
        if not isinstance(coverage, dict):
            raise ReportError("coverage.json: not an object")
        deployed = coverage.get("deployed")
        if not isinstance(deployed, dict):
            raise ReportError("coverage.json: deployed is not an object")
        missing = deployed.get("missing")
        if not isinstance(missing, list) or not all(isinstance(name, str) for name in missing):
            raise ReportError("coverage.json: deployed.missing is not a list of names")
        untimed = sorted(set(missing))
    return LegSkips(directory.name, run.get("selection"), dict(sorted(by_family.items())), untimed)


def _delta(now: int, before: Optional[int]) -> str:
    if before is None:
        return "new" if now else ""
    diff = now - before
    return f"{diff:+d}" if diff else "0"


def board_section(board: str, current: Optional[LegSkips], previous: Optional[LegSkips],
                  baseline_note: Optional[str]) -> tuple[list[str], list[str]]:
    """Markdown lines and workflow warnings for one board; previous shares current's selection."""
    lines = [f"#### {board}", ""]
    warnings: list[str] = []
    if current is None:
        return lines + ["No TCM-leg result.", ""], warnings
    comparable = previous is not None
    if previous is None:
        lines.append(f"No baseline: {baseline_note or 'no earlier run has a result for this board'}.")
    else:
        lines.append(f"Baseline: `{previous.artifact}`.")
    lines.append("")
    base = previous.skipped_by_family if comparable else {}
    families = sorted(set(current.skipped_by_family) | set(base))
    lines += ["| Family | Skipped | Δ |", "|---|---|---|"]
    for family in families:
        now = current.skipped_by_family.get(family, 0)
        before = base.get(family)
        grew = comparable and now > (before or 0)
        delta = _delta(now, before) if comparable else ""
        lines.append(f"| {family} | {now} | {delta}{' ⚠' if grew else ''} |")
        if grew:
            warnings.append(f"{board}: {family} skipped cases grew from {before or 0} to {now}")
    total_before = previous.skipped_total if comparable else None
    lines.append(f"| **Total** | **{current.skipped_total}** | {_delta(current.skipped_total, total_before) if comparable else ''} |")
    lines.append("")
    if current.untimed is None:
        lines += ["Untimed deployed entry points: no coverage.json.", ""]
        return lines, warnings
    new_untimed = (
        sorted(set(current.untimed) - set(previous.untimed))
        if comparable and previous.untimed is not None else []
    )
    new_note = f", {len(new_untimed)} new" if new_untimed else ""
    lines += [f"<details><summary>Untimed deployed entry points ({len(current.untimed)}{new_note})</summary>", ""]
    for name in current.untimed:
        lines.append(f"- `{name}`{' (new)' if name in new_untimed else ''}")
    for name in new_untimed:
        warnings.append(f"{board}: {name} is newly untimed")
    lines += ["", "</details>", ""]
    return lines, warnings


def _safe_load(root: Path, board: str, label: str, errors: list[str]) -> tuple[Optional[LegSkips], bool]:
    """The leg and whether it was unreadable."""
    try:
        return load_leg(root, board), False
    except ReportError as exc:
        errors.append(f"{board}: {label} result unreadable: {exc}")
        return None, True


def baseline_runs(previous_root: Optional[Path]) -> list[Path]:
    """Earlier runs' directories, newest run id first."""
    if previous_root is None or not previous_root.is_dir():
        return []
    runs = [child for child in previous_root.iterdir() if child.is_dir() and child.name.isdigit()]
    return sorted(runs, key=lambda child: int(child.name), reverse=True)


def find_baseline(runs: list[Path], board: str, current: LegSkips,
                  errors: list[str]) -> tuple[Optional[LegSkips], str, int]:
    """The newest readable leg with current's selection, else None and why; plus how many runs had no result."""
    unreadable = other_selection = no_result = 0
    for run in runs:
        leg, bad = _safe_load(run, board, f"baseline run {run.name}", errors)
        unreadable += bad
        if leg is None:
            no_result += not bad
            continue
        if leg.selection != current.selection:
            other_selection += 1
            continue
        return leg, "", no_result
    if not runs:
        return None, "", 0
    reasons = []
    if no_result:
        reasons.append(f"{no_result} had no result")
    if other_selection:
        reasons.append(f"{other_selection} used a different selection")
    if unreadable:
        reasons.append(f"{unreadable} unreadable")
    detail = f" ({', '.join(reasons)})" if reasons else ""
    return (None, f"none of the {len(runs)} earlier run(s) searched has a comparable result for this board{detail}",
            no_result)


def render(current_root: Path, previous_root: Optional[Path], boards: list[str],
           baseline_note: Optional[str] = None) -> tuple[str, list[str]]:
    """The whole Markdown section and every workflow warning.

    previous_root given without baseline_note and holding no run means every baseline
    download failed, which is warned like a baseline search that found only missing results.
    """
    lines = ["### Skipped cases and untimed entry points (TCM leg)", ""]
    warnings: list[str] = []
    runs = baseline_runs(previous_root)
    nothing_downloaded = previous_root is not None and not runs and not baseline_note
    if nothing_downloaded:
        baseline_note = "no earlier run's artifacts were downloaded"
    for board in boards:
        errors: list[str] = []
        current, current_bad = _safe_load(current_root, board, "current", errors)
        if current_bad:
            section, board_warnings = [f"#### {board}", "", "TCM-leg result unreadable; see warnings.", ""], []
        elif current is None:
            section, board_warnings = board_section(board, None, None, None)
            errors.append(f"{board}: no current TCM-leg result; skip growth not checked")
        else:
            previous, why, no_result = find_baseline(runs, board, current, errors)
            section, board_warnings = board_section(board, current, previous, why or baseline_note)
            if previous is None and (no_result or nothing_downloaded):
                errors.append(f"{board}: no baseline because earlier runs left no result; "
                              f"skip growth not checked ({why or baseline_note})")
        lines += section
        warnings += errors + board_warnings
    return "\n".join(lines) + "\n", warnings


def select_baseline_runs(fetch_page: Callable[[int], Any], before: int, count: int,
                         max_pages: int, per_page: int = PER_PAGE) -> tuple[list[int], bool]:
    """The newest count non-cancelled run ids below before, and whether the history was exhausted.

    fetch_page(n) returns page n (from 1) of completed runs, newest first. A short page ends
    the history; max_pages bounds the search. ReportError on a malformed page.
    """
    if count < 1 or max_pages < 1 or per_page < 1:
        raise ReportError("count, max_pages and per_page must be positive")
    chosen: list[int] = []
    for page in range(1, max_pages + 1):
        runs = fetch_page(page)
        if not isinstance(runs, list):
            raise ReportError(f"page {page}: not a list of runs")
        for run in runs:
            run_id = run.get("id") if isinstance(run, dict) else None
            if isinstance(run_id, bool) or not isinstance(run_id, int):
                raise ReportError(f"page {page}: run without an integer id")
            if run_id < before and run.get("conclusion") != "cancelled" and run_id not in chosen:
                chosen.append(run_id)
                if len(chosen) == count:
                    return chosen, False
        if len(runs) < per_page:
            return chosen, True
    return chosen, False


def _gh_runs_page(repo: str, page: int, per_page: int = PER_PAGE) -> Any:
    """One page of completed scheduled runs of the nightly on main, newest first."""
    command = ["gh", "api", "-X", "GET", f"repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs",
               "-f", "event=schedule", "-f", "branch=main", "-f", "status=completed",
               "-f", f"per_page={per_page}", "-f", f"page={page}"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=GH_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReportError(f"gh api page {page}: {exc}") from exc
    if done.returncode != 0:
        raise ReportError(f"gh api page {page} failed: {done.stderr.strip() or done.returncode}")
    try:
        body = json.loads(done.stdout)
    except ValueError as exc:
        raise ReportError(f"gh api page {page}: {exc}") from exc
    return body.get("workflow_runs") if isinstance(body, dict) else None


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return value


def select_main(argv: list[str], fetch_page: Optional[Callable[[str, int], Any]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nightly_skip_report.py select-baselines",
                                     description="Pick the earlier scheduled runs to compare with")
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--before", type=_positive_int, required=True, help="This run's id")
    parser.add_argument("--count", type=_positive_int, default=3)
    parser.add_argument("--max-pages", type=_positive_int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="GITHUB_OUTPUT file to append to")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error(f"--repo is not OWNER/NAME: {args.repo!r}")
    fetch = fetch_page or _gh_runs_page
    try:
        runs, exhausted = select_baseline_runs(lambda page: fetch(args.repo, page), args.before,
                                               args.count, args.max_pages)
    except ReportError as exc:
        print(f"::error::baseline run selection failed: {_escape_command(str(exc))}")
        return 1
    lines = [f"runs={' '.join(str(run) for run in runs)}"]
    if not runs and exhausted:
        lines.append("note=no earlier scheduled run")
    elif len(runs) < args.count and not exhausted:
        print(f"::warning::baseline run selection stopped after {args.max_pages} page(s) "
              f"with {len(runs)} of {args.count} run(s)")
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0


def _escape_command(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["select-baselines"]:
        return select_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, required=True, help="Downloaded artifacts of this run")
    parser.add_argument("--previous", type=Path, help="Earlier runs' artifacts, one <run id>/ directory each")
    parser.add_argument("--baseline-note", help="Why there is no baseline, for the summary")
    parser.add_argument("--boards", required=True, help="JSON list of board ids")
    parser.add_argument("--out", type=Path, required=True, help="File to append the Markdown to")
    args = parser.parse_args(argv)
    try:
        boards = json.loads(args.boards)
    except ValueError as exc:
        parser.error(f"--boards is not JSON: {exc}")
    if not isinstance(boards, list) or not boards or not all(isinstance(board, str) and board for board in boards):
        parser.error("--boards must be a non-empty JSON list of non-empty strings")
    markdown, warnings = render(args.current, args.previous, boards, args.baseline_note)
    with args.out.open("a", encoding="utf-8") as handle:
        handle.write(markdown)
    for warning in warnings:
        print(f"::warning::{_escape_command(warning)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
