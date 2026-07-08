"""
Publish multi-CPU coverage and test summary table for CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from helia_core_tester.core.cpu_targets import parse_cpu_list
from helia_core_tester.core.path_layout import coverage_report_dir, tests_report_dir


def _to_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _normalize_source_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    for marker in ("/Source/", "/Include/"):
        idx = normalized.rfind(marker)
        if idx != -1:
            return normalized[idx + 1 :].lstrip("./")
    return normalized.lstrip("./")


def _parse_lcov(path: Path) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int], Dict[Tuple[str, str], int], Dict[Tuple[str, int, str, str], int]]:
    if not path.exists():
        raise FileNotFoundError(f"coverage.info not found: {path}")

    totals = {"lh": 0, "lf": 0, "fnh": 0, "fnf": 0, "brh": 0, "brf": 0}
    line_hits: Dict[Tuple[str, int], int] = {}
    function_hits: Dict[Tuple[str, str], int] = {}
    branch_hits: Dict[Tuple[str, int, str, str], int] = {}
    current_file: str | None = None

    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("SF:"):
            current_file = _normalize_source_path(line[3:].strip())
            continue

        if not current_file:
            continue

        if line.startswith("DA:"):
            payload = line[3:].split(",")
            if len(payload) >= 2:
                line_no = _to_int(payload[0])
                hit_count = _to_int(payload[1])
                key = (current_file, line_no)
                line_hits[key] = line_hits.get(key, 0) + hit_count
            continue

        if line.startswith("FNDA:"):
            payload = line[5:].split(",", 1)
            if len(payload) == 2:
                hit_count = _to_int(payload[0])
                fn_name = payload[1].strip()
                key = (current_file, fn_name)
                function_hits[key] = function_hits.get(key, 0) + hit_count
            continue

        if line.startswith("FN:"):
            payload = line[3:].split(",", 1)
            if len(payload) == 2:
                fn_name = payload[1].strip()
                key = (current_file, fn_name)
                function_hits.setdefault(key, 0)
            continue

        if line.startswith("BRDA:"):
            payload = line[5:].split(",")
            if len(payload) == 4:
                line_no = _to_int(payload[0])
                block_id = payload[1].strip()
                branch_id = payload[2].strip()
                taken_raw = payload[3].strip()
                taken = 0 if taken_raw == "-" else _to_int(taken_raw)
                key = (current_file, line_no, block_id, branch_id)
                branch_hits[key] = branch_hits.get(key, 0) + taken
            continue

        if line.startswith("LH:"):
            totals["lh"] += _to_int(line[3:])
            continue

        if line.startswith("LF:"):
            totals["lf"] += _to_int(line[3:])
            continue

        if line.startswith("FNH:"):
            totals["fnh"] += _to_int(line[4:])
            continue

        if line.startswith("FNF:"):
            totals["fnf"] += _to_int(line[4:])
            continue

        if line.startswith("BRH:"):
            totals["brh"] += _to_int(line[4:])
            continue

        if line.startswith("BRF:"):
            totals["brf"] += _to_int(line[4:])
            continue

    return totals, line_hits, function_hits, branch_hits


def _parse_test_report(tests_report_dir: Path, cpu: str) -> Dict[str, int]:
    pattern = f"test_report_{cpu}_*.json"
    candidates = sorted(tests_report_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(tests_report_dir.glob("test_report_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No test report JSON found for {cpu} in {tests_report_dir}")

    data = json.loads(candidates[-1].read_text())
    failed = (
        _to_int(data.get("failed"))
        + _to_int(data.get("errors"))
        + _to_int(data.get("timed_out"))
        + _to_int(data.get("build_failed"))
        + _to_int(data.get("generation_failed"))
        + _to_int(data.get("conversion_failed"))
    )
    skipped = _to_int(data.get("skipped")) + _to_int(data.get("not_run"))
    return {
        "number_of_tests": _to_int(data.get("total_tests")),
        "passed": _to_int(data.get("passed")),
        "failed": failed,
        "skipped": skipped,
    }


def _format_ratio(covered: int, total: int) -> str:
    pct = (covered / total * 100.0) if total else 0.0
    return f"{covered}/{total} ({pct:.1f}%)"


def _normalize_suites(suite: str) -> List[str]:
    normalized = str(suite).strip().lower()
    if normalized == "both":
        return ["int", "float"]
    if normalized in {"int", "float"}:
        return [normalized]
    raise ValueError(f"Invalid suite: {suite!r} (expected int, float, or both)")


def build_rows(
    artifacts_root: Path,
    cpus: Iterable[str],
    suites: Iterable[str] = ("int", "float"),
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    cpu_list = list(cpus)
    suite_list = [str(item).strip().lower() for item in suites if str(item).strip()]
    if not suite_list:
        suite_list = ["int", "float"]
    rows: List[Dict[str, object]] = []
    project_root = artifacts_root.parent
    union_lines: Dict[Tuple[str, int], bool] = {}
    union_functions: Dict[Tuple[str, str], bool] = {}
    union_branches: Dict[Tuple[str, int, str, str], bool] = {}

    total_tests = {"number_of_tests": 0, "passed": 0, "failed": 0, "skipped": 0}

    for cpu in cpu_list:
        cpu_lines: Dict[Tuple[str, int], bool] = {}
        cpu_functions: Dict[Tuple[str, str], bool] = {}
        cpu_branches: Dict[Tuple[str, int, str, str], bool] = {}
        cpu_tests = {"number_of_tests": 0, "passed": 0, "failed": 0, "skipped": 0}
        found_suite_coverage = False

        for suite in suite_list:
            coverage_path = coverage_report_dir(project_root, cpu, suite=suite) / "coverage.info"
            if not coverage_path.exists():
                continue

            found_suite_coverage = True
            _, line_hits, function_hits, branch_hits = _parse_lcov(coverage_path)

            for key, hits in line_hits.items():
                covered = hits > 0
                cpu_lines[key] = cpu_lines.get(key, False) or covered
                union_lines[key] = union_lines.get(key, False) or covered

            for key, hits in function_hits.items():
                covered = hits > 0
                cpu_functions[key] = cpu_functions.get(key, False) or covered
                union_functions[key] = union_functions.get(key, False) or covered

            for key, hits in branch_hits.items():
                covered = hits > 0
                cpu_branches[key] = cpu_branches.get(key, False) or covered
                union_branches[key] = union_branches.get(key, False) or covered

            # float-mve contributes additional MVE coverage only; its tests duplicate
            # the float suite, so skip test accounting to avoid double counting.
            if suite != "float-mve":
                test_totals = _parse_test_report(tests_report_dir(project_root, cpu, suite=suite), cpu)
                cpu_tests["number_of_tests"] += int(test_totals["number_of_tests"])
                cpu_tests["passed"] += int(test_totals["passed"])
                cpu_tests["failed"] += int(test_totals["failed"])
                cpu_tests["skipped"] += int(test_totals["skipped"])

        if not found_suite_coverage:
            checked = ", ".join(
                str(coverage_report_dir(project_root, cpu, suite=suite) / "coverage.info") for suite in suite_list
            )
            raise FileNotFoundError(f"coverage.info not found for {cpu} in suites [{', '.join(suite_list)}]: {checked}")

        total_tests["number_of_tests"] += int(cpu_tests["number_of_tests"])
        total_tests["passed"] += int(cpu_tests["passed"])
        total_tests["failed"] += int(cpu_tests["failed"])
        total_tests["skipped"] += int(cpu_tests["skipped"])

        rows.append(
            {
                "cpu": cpu,
                "coverage": {
                    "lf": len(cpu_lines),
                    "lh": sum(1 for covered in cpu_lines.values() if covered),
                    "fnf": len(cpu_functions),
                    "fnh": sum(1 for covered in cpu_functions.values() if covered),
                    "brf": len(cpu_branches),
                    "brh": sum(1 for covered in cpu_branches.values() if covered),
                },
                "tests": cpu_tests,
            }
        )

    total_row = {
        "cpu": "total",
        "coverage": {
            "lf": len(union_lines),
            "lh": sum(1 for covered in union_lines.values() if covered),
            "fnf": len(union_functions),
            "fnh": sum(1 for covered in union_functions.values() if covered),
            "brf": len(union_branches),
            "brh": sum(1 for covered in union_branches.values() if covered),
        },
        "tests": total_tests,
    }

    return rows, total_row


def render_markdown_table(rows: List[Dict[str, object]], total_row: Dict[str, object]) -> str:
    label_map = {
        "cortex-m0": "m0",
        "cortex-m4": "m4",
        "cortex-m55": "m55",
    }
    lines = [
        "| cpu | lines | functions | branches | number of tests | passed | failed | skipped (not generated) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        cpu = str(row["cpu"])
        coverage = dict(row["coverage"])
        tests = dict(row["tests"])
        lines.append(
            f"| {label_map.get(cpu, cpu)} | "
            f"{_format_ratio(int(coverage['lh']), int(coverage['lf']))} | "
            f"{_format_ratio(int(coverage['fnh']), int(coverage['fnf']))} | "
            f"{_format_ratio(int(coverage['brh']), int(coverage['brf']))} | "
            f"{int(tests['number_of_tests'])} | {int(tests['passed'])} | {int(tests['failed'])} | {int(tests['skipped'])} |"
        )

    cov_total = dict(total_row["coverage"])
    tests_total = dict(total_row["tests"])
    lines.append(
        f"| total | "
        f"{_format_ratio(int(cov_total['lh']), int(cov_total['lf']))} | "
        f"{_format_ratio(int(cov_total['fnh']), int(cov_total['fnf']))} | "
        f"{_format_ratio(int(cov_total['brh']), int(cov_total['brf']))} | "
        f"{int(tests_total['number_of_tests'])} | {int(tests_total['passed'])} | {int(tests_total['failed'])} | {int(tests_total['skipped'])} |"
    )
    return "\n".join(lines)


def publish_table(table: str, summary_file: Path | None, heading: str) -> None:
    print(table)
    if summary_file:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with summary_file.open("a", encoding="utf-8") as handle:
            handle.write(f"## {heading}\n\n")
            handle.write(table)
            handle.write("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish CPU coverage and test summary markdown table.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"), help="Artifacts root path")
    parser.add_argument(
        "--cpus",
        default="cortex-m0,cortex-m4,cortex-m55",
        help="Comma-separated CPU list (e.g. cortex-m0,cortex-m4,cortex-m55)",
    )
    parser.add_argument(
        "--suite",
        default="both",
        choices=["int", "float", "both"],
        help="Coverage/test suite selection for summary (default: both)",
    )
    parser.add_argument(
        "--include-mve-float",
        action="store_true",
        help="Also include cortex-m55 MVE float coverage (reports/coverage/float-mve) in the summary.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None,
        help="Markdown summary output path (defaults to $GITHUB_STEP_SUMMARY if set)",
    )
    parser.add_argument("--heading", default="Multi-CPU Coverage/Test Summary", help="Heading above markdown table")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cpus = parse_cpu_list(args.cpus)
        suites = _normalize_suites(args.suite)
        if args.include_mve_float and "float-mve" not in suites:
            suites.append("float-mve")
        rows, total = build_rows(args.artifacts_root, cpus, suites=suites)
        table = render_markdown_table(rows, total)
        publish_table(table, args.summary_file, args.heading)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
