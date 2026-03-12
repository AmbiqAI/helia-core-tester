from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from helia_core_tester.scripts.publish_cpu_coverage_summary import (
    _parse_lcov,
    _parse_test_report,
    _resolve_report_dir,
    build_rows,
    main,
    render_markdown_table,
)


def _write_lcov(path: Path, sf: str, da: list[tuple[int, int]], fns: list[tuple[int, str, int]], branches: list[tuple[int, str, str, str]]) -> None:
    lines: list[str] = []
    lines.append("TN:")
    lines.append(f"SF:{sf}")
    for line_no, count in da:
        lines.append(f"DA:{line_no},{count}")
    for line_no, fn_name, hits in fns:
        lines.append(f"FN:{line_no},{fn_name}")
        lines.append(f"FNDA:{hits},{fn_name}")
    for line_no, block_id, branch_id, taken in branches:
        lines.append(f"BRDA:{line_no},{block_id},{branch_id},{taken}")
    lines.append(f"LH:{sum(1 for _, count in da if count > 0)}")
    lines.append(f"LF:{len(da)}")
    lines.append(f"FNH:{sum(1 for _, _, hits in fns if hits > 0)}")
    lines.append(f"FNF:{len(fns)}")
    lines.append(f"BRH:{sum(1 for _, _, _, taken in branches if taken != '-' and int(taken) > 0)}")
    lines.append(f"BRF:{len(branches)}")
    lines.append("end_of_record")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_test_report(path: Path, *, cpu: str, total: int, passed: int, failed: int, skipped: int, not_run: int = 0, errors: int = 0, timed_out: int = 0, build_failed: int = 0, generation_failed: int = 0, conversion_failed: int = 0) -> None:
    payload = {
        "cpu": cpu,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "not_run": not_run,
        "errors": errors,
        "timed_out": timed_out,
        "build_failed": build_failed,
        "generation_failed": generation_failed,
        "conversion_failed": conversion_failed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _setup_cpu_artifacts(root: Path, cpu: str, sf_suffix: str, counts: dict) -> None:
    reports_dir = root / "artifacts" / f"build-{cpu}-gcc" / "reports"
    coverage_info = reports_dir / "coverage" / cpu / "coverage.info"
    sf = str(root / "Source" / sf_suffix)
    _write_lcov(
        coverage_info,
        sf=sf,
        da=counts["da"],
        fns=counts["fns"],
        branches=counts["branches"],
    )
    _write_test_report(
        reports_dir / f"test_report_{cpu}_20260311_120000.json",
        cpu=cpu,
        total=counts["total_tests"],
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        not_run=counts.get("not_run", 0),
        errors=counts.get("errors", 0),
        timed_out=counts.get("timed_out", 0),
        build_failed=counts.get("build_failed", 0),
        generation_failed=counts.get("generation_failed", 0),
        conversion_failed=counts.get("conversion_failed", 0),
    )


def test_parse_lcov_includes_fn_and_fnda_and_branches(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.info"
    _write_lcov(
        coverage,
        sf=str(tmp_path / "Source" / "A.c"),
        da=[(10, 1), (11, 0)],
        fns=[(7, "foo", 0), (20, "bar", 3)],
        branches=[(10, "0", "0", "1"), (10, "0", "1", "-")],
    )

    totals, lines, functions, branches = _parse_lcov(coverage)
    assert totals["lf"] == 2
    assert totals["lh"] == 1
    assert totals["fnf"] == 2
    assert totals["fnh"] == 1
    assert totals["brf"] == 2
    assert totals["brh"] == 1
    assert ("Source/A.c", 10) in lines
    assert ("Source/A.c", "foo") in functions
    assert ("Source/A.c", "bar") in functions
    assert ("Source/A.c", 10, "0", "0") in branches


def test_build_rows_and_table_with_union_totals_and_status_formulas(tmp_path: Path) -> None:
    _setup_cpu_artifacts(
        tmp_path,
        "cortex-m0",
        "ActivationFunctions/a0.c",
        {
            "da": [(10, 1), (11, 0)],
            "fns": [(5, "fn_shared", 1), (7, "fn_m0", 0)],
            "branches": [(10, "0", "0", "1"), (10, "0", "1", "0")],
            "total_tests": 5,
            "passed": 3,
            "failed": 1,
            "skipped": 1,
            "not_run": 2,
            "errors": 1,
            "timed_out": 0,
            "build_failed": 0,
            "generation_failed": 0,
            "conversion_failed": 0,
        },
    )
    _setup_cpu_artifacts(
        tmp_path,
        "cortex-m4",
        "ActivationFunctions/a4.c",
        {
            "da": [(10, 0), (12, 2)],
            "fns": [(5, "fn_shared", 0), (9, "fn_m4", 1)],
            "branches": [(12, "1", "0", "2"), (12, "1", "1", "-")],
            "total_tests": 6,
            "passed": 6,
            "failed": 0,
            "skipped": 0,
        },
    )
    _setup_cpu_artifacts(
        tmp_path,
        "cortex-m55",
        "ActivationFunctions/a55.c",
        {
            "da": [(20, 3)],
            "fns": [(11, "fn_m55", 2)],
            "branches": [(20, "2", "0", "1")],
            "total_tests": 7,
            "passed": 5,
            "failed": 0,
            "skipped": 1,
            "timed_out": 1,
        },
    )

    rows, total = build_rows(tmp_path / "artifacts", ["cortex-m0", "cortex-m4", "cortex-m55"])
    table = render_markdown_table(rows, total)

    assert "| cpu | lines | functions | branches | number of tests | passed | failed | skipped (not generated) |" in table
    assert "| m0 |" in table
    assert "| m4 |" in table
    assert "| m55 |" in table
    assert "| total |" in table

    # m0 failed = failed + errors + timed_out + build_failed + generation_failed + conversion_failed
    m0 = next(r for r in rows if r["cpu"] == "cortex-m0")
    assert m0["tests"]["failed"] == 2
    # m0 skipped = skipped + not_run
    assert m0["tests"]["skipped"] == 3

    # Total tests are sums across rows.
    assert total["tests"]["number_of_tests"] == 18
    assert total["tests"]["passed"] == 14
    assert total["tests"]["failed"] == 3
    assert total["tests"]["skipped"] == 4

    # Union coverage totals should be non-zero and reflect merged keys.
    assert total["coverage"]["lf"] >= 5
    assert total["coverage"]["fnf"] >= 5
    assert total["coverage"]["brf"] >= 5


def test_main_writes_junit_with_expected_rows_and_metrics(tmp_path: Path) -> None:
    _setup_cpu_artifacts(
        tmp_path,
        "cortex-m0",
        "ActivationFunctions/a0.c",
        {
            "da": [(1, 1), (2, 0)],
            "fns": [(1, "f0", 1)],
            "branches": [(1, "0", "0", "1")],
            "total_tests": 3,
            "passed": 2,
            "failed": 0,
            "skipped": 0,
            "not_run": 1,
        },
    )
    _setup_cpu_artifacts(
        tmp_path,
        "cortex-m4",
        "ActivationFunctions/a4.c",
        {
            "da": [(1, 1)],
            "fns": [(1, "f4", 1)],
            "branches": [(1, "0", "0", "1")],
            "total_tests": 4,
            "passed": 4,
            "failed": 0,
            "skipped": 0,
        },
    )
    _setup_cpu_artifacts(
        tmp_path,
        "cortex-m55",
        "ActivationFunctions/a55.c",
        {
            "da": [(1, 0), (2, 0)],
            "fns": [(1, "f55", 0)],
            "branches": [(1, "0", "0", "0")],
            "total_tests": 5,
            "passed": 3,
            "failed": 0,
            "skipped": 1,
            "errors": 1,
        },
    )

    summary_file = tmp_path / "summary.md"
    junit_file = tmp_path / "cpu_coverage_summary.junit.xml"
    rc = main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--cpus",
            "cortex-m0,cortex-m4,cortex-m55",
            "--summary-file",
            str(summary_file),
            "--junit-file",
            str(junit_file),
        ]
    )
    assert rc == 0
    assert junit_file.exists()

    root = ET.parse(junit_file).getroot()
    assert root.tag == "testsuite"
    testcases = root.findall("testcase")
    assert [t.attrib.get("name") for t in testcases] == ["m0", "m4", "m55", "total"]

    for testcase in testcases:
        out_text = testcase.findtext("system-out") or ""
        assert "lines=" in out_text
        assert "functions=" in out_text
        assert "branches=" in out_text
        assert "number_of_tests=" in out_text
        assert "passed=" in out_text
        assert "failed=" in out_text
        assert "skipped_not_generated=" in out_text

    assert "number_of_tests=3" in (testcases[0].findtext("system-out") or "")
    assert "skipped_not_generated=1" in (testcases[0].findtext("system-out") or "")
    assert "failed=1" in (testcases[2].findtext("system-out") or "")
    assert "number_of_tests=12" in (testcases[3].findtext("system-out") or "")


def test_nested_reports_fallback_and_generic_test_report_name(tmp_path: Path) -> None:
    cpu = "cortex-m0"
    base = tmp_path / "artifacts" / f"build-{cpu}-gcc" / "reports"
    nested = base / "reports"

    _write_lcov(
        nested / "coverage" / cpu / "coverage.info",
        sf=str(tmp_path / "Source" / "Nested" / "n.c"),
        da=[(1, 1)],
        fns=[(1, "nested_fn", 1)],
        branches=[(1, "0", "0", "1")],
    )
    _write_test_report(
        nested / "test_report_20260311_120000.json",
        cpu=cpu,
        total=1,
        passed=1,
        failed=0,
        skipped=0,
    )

    resolved = _resolve_report_dir(tmp_path / "artifacts", cpu)
    assert resolved == nested

    totals = _parse_test_report(resolved, cpu)
    assert totals["number_of_tests"] == 1
    assert totals["passed"] == 1


def test_main_fails_fast_when_missing_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--cpus",
            "cortex-m0,cortex-m4,cortex-m55",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "coverage.info not found" in err
