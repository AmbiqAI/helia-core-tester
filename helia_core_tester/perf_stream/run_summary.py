"""Human and JSON summaries of a hardware streaming run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

import typer


def _format_case_line(case, *, id_width: int = 0) -> str:
    passed = case.comparison.passed
    label = "PASS" if passed else "FAIL"
    # Pad the plain label to a fixed width *before* applying ANSI color styling --
    # styling first would make python's string padding count the (invisible)
    # escape codes as characters and silently break column alignment.
    status = typer.style(f"{label:<4}", fg=typer.colors.GREEN if passed else typer.colors.RED, bold=True)
    line = (
        f"  {case.case_bundle.case_id:<{id_width}}  {status}  "
        f"median_cycles={case.statistics.median_cycles:>10.1f}"
    )
    if not passed:
        line += f"  mismatches={case.comparison.mismatch_count}"
    return line


def make_live_progress_printer(total_hint: Optional[int] = None, *, id_width: int = 0, err: bool = False) -> Callable:
    """Return an on_case_complete callback that prints one line per case as soon
    as it finishes running on hardware, so a long multi-batch suite shows visible
    progress instead of going silent until the very end (or until it errors out).

    id_width/total_hint (when known ahead of time) keep the case_id, PASS/FAIL,
    and progress-counter columns aligned across every printed line, regardless
    of how long individual case_ids are or how many cases/digits the total has.
    """
    count = 0
    total_width = len(str(total_hint)) if total_hint else 0

    def _on_case_complete(case) -> None:
        nonlocal count
        count += 1
        if total_hint:
            progress = f"[{count:>{total_width}}/{total_hint}]"
        else:
            progress = f"[{count}]"
        typer.echo(f"{progress} {_format_case_line(case, id_width=id_width)}", err=err)

    return _on_case_complete


def print_case_results(cases, *, err: bool = False) -> tuple[int, list[str]]:
    """Print a readable, colorized per-case listing grouped by operator family.

    Returns (passed_count, failed_case_ids) for use in a trailing summary.
    """
    passed_count = 0
    failed_case_ids: list[str] = []
    current_family: Optional[str] = None
    id_width = max((len(c.case_bundle.case_id) for c in cases), default=0)

    for case in cases:
        family = str(case.case_bundle.manifest.get("family", "?"))
        if family != current_family:
            typer.echo(typer.style(f"\n[{family}]", bold=True), err=err)
            current_family = family

        if case.comparison.passed:
            passed_count += 1
        else:
            failed_case_ids.append(case.case_bundle.case_id)

        typer.echo(_format_case_line(case, id_width=id_width), err=err)

    return passed_count, failed_case_ids


def print_result_summary(total: int, passed_count: int, failed_case_ids: list[str], *, err: bool = False) -> None:
    """Print a compact, always-visible pass/fail summary at the end of a run."""
    typer.echo("\n" + "-" * 60, err=err)
    if failed_case_ids:
        typer.echo(
            typer.style(f"Summary: {passed_count}/{total} passed, {len(failed_case_ids)} failed", fg=typer.colors.RED, bold=True),
            err=err,
        )
        typer.echo("Failed cases:", err=err)
        for case_id in failed_case_ids:
            typer.echo(f"  - {case_id}", err=err)
    else:
        typer.echo(typer.style(f"Summary: {passed_count}/{total} passed", fg=typer.colors.GREEN, bold=True), err=err)
    typer.echo("-" * 60, err=err)


_BRIDGED_TODAY_RE = re.compile(r"\s*\(bridged today: \[.*?\]\)\.?$")
_NAMES_PER_LINE = 6


def _clean_skip_reason(test_name: str, reason: str) -> str:
    """Strip the redundant leading '{test_name}: ' (the CLI already prints the
    name once) and collapse the ever-growing 'bridged today: [...]' family/operator
    list -- otherwise identical for every not-yet-bridged case -- down to a
    pointer at the authoritative source, instead of repeating it per case.
    """
    prefix = f"{test_name}: "
    if reason.startswith(prefix):
        reason = reason[len(prefix):]
    return _BRIDGED_TODAY_RE.sub(
        " (see generated_test_bridge.bridged_families() for the current list).", reason
    )


def print_skipped_summary(skipped: list[tuple], *, err: bool = False) -> None:
    """Print skipped generated-tests grouped by (deduplicated) reason instead of
    repeating an identical, verbose reason string once per case -- keeps the
    signal (why + how many + which cases) readable even with hundreds of
    not-yet-bridged cases.
    """
    groups: dict[str, list[str]] = {}
    for test, reason in skipped:
        cleaned = _clean_skip_reason(test.name, reason)
        groups.setdefault(cleaned, []).append(test.name)

    typer.echo(
        typer.style(
            f"\n  Skipped {len(skipped)} generated test(s) with no real firmware dispatch support yet, grouped by reason:",
            fg=typer.colors.YELLOW,
        ),
        err=err,
    )
    for reason, names in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        typer.echo(typer.style(f"\n  [{len(names)}x] {reason}", fg=typer.colors.YELLOW), err=err)
        names = sorted(names)
        for i in range(0, len(names), _NAMES_PER_LINE):
            typer.echo("      " + ", ".join(names[i : i + _NAMES_PER_LINE]), err=err)


def print_run_report(result, skipped: list[tuple], bundle: Path, *, err: bool = False) -> list[str]:
    """The human report for `hardware run`/`hardware stream`. Returns the failed case ids."""
    typer.echo("\nFinal per-case results:", err=err)
    passed_count, failed_case_ids = print_case_results(result.cases, err=err)
    if skipped:
        print_skipped_summary(skipped, err=err)
    print_result_summary(len(result.cases), passed_count, failed_case_ids, err=err)
    typer.echo(f"\n✓ Result bundle: {bundle}", err=err)
    return failed_case_ids


def build_json_summary(result, skipped: list[tuple], *, session_id: str, board_id: str, bundle: Path) -> dict[str, Any]:
    """The single JSON document `--json` prints on stdout."""
    cases: list[dict[str, Any]] = []
    passed = 0
    for case in result.cases:
        ok = bool(case.comparison.passed)
        passed += int(ok)
        cases.append(
            {
                "case_id": case.case_bundle.case_id,
                "passed": ok,
                "median_cycles": float(case.statistics.median_cycles),
                "skipped_reason": None,
            }
        )
    for test, reason in skipped:
        cases.append(
            {
                "case_id": test.name,
                "passed": None,
                "median_cycles": None,
                "skipped_reason": _clean_skip_reason(test.name, reason),
            }
        )
    ran = len(result.cases)
    return {
        "session_id": session_id,
        "board": board_id,
        "bundle": str(bundle),
        "totals": {"ran": ran, "passed": passed, "failed": ran - passed, "skipped": len(skipped)},
        "cases": cases,
    }
