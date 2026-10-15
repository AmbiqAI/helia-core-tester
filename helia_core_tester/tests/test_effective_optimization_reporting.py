"""Run reports record the optimization ns-cmsis-nn was compiled at, not only the requested one (#87)."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from helia_core_tester.fvp import reporting
from helia_core_tester.fvp.cli import build_arg_parser
from helia_core_tester.fvp.cmake import cmake_truthy, read_cmake_cache_value
from helia_core_tester.fvp.reporting import (
    COVERAGE_CMSIS_NN_OPTIMIZATION,
    build_optimization_metadata,
    optimization_metadata,
)
from helia_core_tester.reporting.models import TestResult as _Result, TestStatus as _Status

REPO_ROOT = Path(reporting.__file__).resolve().parents[2]


@pytest.mark.parametrize("requested", ["-Ofast", "-O2", "-Os", "-O0"])
def test_without_coverage_effective_equals_requested(requested) -> None:
    assert optimization_metadata(requested, False) == {
        "optimization": requested,
        "cmsis_nn_optimization": requested,
        "coverage": False,
    }


@pytest.mark.parametrize("requested", ["-Ofast", "-O2", "-Os", "-O0", None])
def test_coverage_forces_o0_but_keeps_the_requested_level(requested) -> None:
    assert optimization_metadata(requested, True) == {
        "optimization": requested,
        "cmsis_nn_optimization": "-O0",
        "coverage": True,
    }


def test_unset_requested_without_coverage_stays_unset() -> None:
    assert optimization_metadata(None, False)["cmsis_nn_optimization"] is None


def test_coverage_constant_matches_the_cmake_flags() -> None:
    text = (REPO_ROOT / "CMakeLists.txt").read_text()
    lines = re.findall(r"target_compile_options\(cmsis-nn PRIVATE --coverage[^)]*\)", text)
    assert len(lines) == 1, lines
    opt_flags = re.findall(r"-O\S*", lines[0])
    assert opt_flags and opt_flags[-1] == COVERAGE_CMSIS_NN_OPTIMIZATION


def _cache(build_dir: Path, **entries: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# This is the CMakeCache file.", "ENABLE_COVERAGE_MVE_FLOAT:BOOL=ON"]
    lines += [f"{key}:STRING={value}" for key, value in entries.items()]
    (build_dir / "CMakeCache.txt").write_text("\n".join(lines) + "\n")
    return build_dir


@pytest.mark.parametrize(
    "value,expected",
    [("ON", True), ("on", True), ("1", True), ("TRUE", True), ("YES", True), ("Y", True), ("2", True),
     ("OFF", False), ("0", False), ("", False), ("NO", False), ("FALSE", False), ("NOTFOUND", False),
     ("X-NOTFOUND", False), ("IGNORE", False), ("garbage", False), ("0.0", False)],
)
def test_cmake_truthy(value, expected) -> None:
    assert cmake_truthy(value) is expected


def test_cache_value_reads_any_type_and_does_not_prefix_match(tmp_path) -> None:
    build = _cache(tmp_path / "b", CMSIS_OPTIMIZATION_LEVEL="-O2")
    assert read_cmake_cache_value(build, "CMSIS_OPTIMIZATION_LEVEL") == "-O2"
    assert read_cmake_cache_value(build, "ENABLE_COVERAGE") is None
    assert read_cmake_cache_value(build, "ENABLE_COVERAGE_MVE_FLOAT") == "ON"


def test_cache_value_missing_or_unreadable_is_none(tmp_path) -> None:
    assert read_cmake_cache_value(tmp_path / "nope", "X") is None
    (tmp_path / "dir" / "CMakeCache.txt").mkdir(parents=True)
    assert read_cmake_cache_value(tmp_path / "dir", "X") is None


def test_build_cache_wins_over_args(tmp_path) -> None:
    build = _cache(tmp_path / "b", CMSIS_OPTIMIZATION_LEVEL="-O2", ENABLE_COVERAGE="ON")
    meta = build_optimization_metadata(build, SimpleNamespace(opt="-Ofast", coverage=False))
    assert meta == {"optimization": "-O2", "cmsis_nn_optimization": "-O0", "coverage": True}


def test_cached_coverage_off_overrides_a_coverage_run_flag(tmp_path) -> None:
    build = _cache(tmp_path / "b", CMSIS_OPTIMIZATION_LEVEL="-Os", ENABLE_COVERAGE="OFF")
    meta = build_optimization_metadata(build, SimpleNamespace(opt="-Ofast", coverage=True))
    assert meta == {"optimization": "-Os", "cmsis_nn_optimization": "-Os", "coverage": False}


def test_no_cache_falls_back_to_args(tmp_path) -> None:
    meta = build_optimization_metadata(tmp_path / "none", SimpleNamespace(opt="-O1", coverage=True))
    assert meta == {"optimization": "-O1", "cmsis_nn_optimization": "-O0", "coverage": True}


def test_empty_cached_level_falls_back_to_args(tmp_path) -> None:
    build = _cache(tmp_path / "b", CMSIS_OPTIMIZATION_LEVEL="")
    meta = build_optimization_metadata(build, SimpleNamespace(opt="-O3", coverage=False))
    assert meta["optimization"] == meta["cmsis_nn_optimization"] == "-O3"


def _run(tmp_path: Path, monkeypatch, argv: list[str], **cache: str) -> Path:
    report_dir = tmp_path / "reports"
    build_dir = tmp_path / "build"
    if cache:
        _cache(build_dir, **cache)
    elf = build_dir / "case.elf"
    monkeypatch.setattr(reporting, "canonical_build_dir", lambda *a, **k: build_dir)
    monkeypatch.setattr(reporting, "cmake_configure", lambda **k: None)
    monkeypatch.setattr(reporting, "cmake_build", lambda **k: None)
    monkeypatch.setattr(reporting, "active_test_list", lambda d: None)
    monkeypatch.setattr(reporting, "find_elves", lambda b, names: [elf])
    monkeypatch.setattr(reporting, "find_descriptors_dir", lambda: tmp_path / "descriptors")
    monkeypatch.setattr(reporting, "new_coverage_context", lambda *a: object())
    monkeypatch.setattr(reporting, "generate_coverage_reports", lambda *a, **k: None)
    monkeypatch.setattr(reporting, "_tests_report_dir", lambda cpu, suite: report_dir)
    monkeypatch.setattr(reporting, "get_git_sha", lambda d: "deadbeef")
    monkeypatch.setattr(
        reporting,
        "run_elf_jobs_with_reporting",
        lambda **k: ([_Result(test_name="case", status=_Status.PASS, duration=0.1, cpu=k["cpu"], elf_path=elf)], False),
    )
    args = build_arg_parser(tmp_path, tmp_path).parse_args(["--report-formats", "json", "junit", *argv])
    args.quiet = True
    _, ok = reporting.run_tests_with_reporting(
        cpus=["cortex-m55"], suite="int", source_dir=tmp_path, toolchain_file=tmp_path / "tc.cmake",
        cmsis5=tmp_path / "CMSIS_5", fvp_exe=tmp_path / "fvp", compiler_tag="gcc", args=args, env={},
    )
    assert ok
    return report_dir


def _junit_props(report_dir: Path) -> dict[str, str]:
    (junit,) = report_dir.rglob("*.xml")
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    return {p.get("name"): p.get("value") for p in suite.iter("property")}


def _json_metadata(report_dir: Path) -> dict:
    (path,) = report_dir.rglob("*.json")
    return json.loads(path.read_text())["metadata"]


def _triple(d: dict) -> tuple:
    return d["optimization"], d["cmsis_nn_optimization"], d["coverage"]


def test_coverage_leg_reports_requested_and_effective(tmp_path, monkeypatch) -> None:
    """The #87 case: an -Ofast coverage leg, run step passes only --no-build --coverage."""
    report_dir = _run(
        tmp_path, monkeypatch, ["--no-build", "--coverage"], CMSIS_OPTIMIZATION_LEVEL="-Ofast", ENABLE_COVERAGE="ON"
    )
    assert _triple(_junit_props(report_dir)) == ("-Ofast", "-O0", "true")
    assert _triple(_json_metadata(report_dir)) == ("-Ofast", "-O0", True)


def test_run_step_reports_the_built_level_not_the_argparse_default(tmp_path, monkeypatch) -> None:
    report_dir = _run(tmp_path, monkeypatch, ["--no-build"], CMSIS_OPTIMIZATION_LEVEL="-O2", ENABLE_COVERAGE="OFF")
    assert _triple(_junit_props(report_dir)) == ("-O2", "-O2", "false")
    assert _triple(_json_metadata(report_dir)) == ("-O2", "-O2", False)


def test_without_a_cache_the_opt_flag_is_reported(tmp_path, monkeypatch) -> None:
    report_dir = _run(tmp_path, monkeypatch, ["--opt=-Os", "--coverage"])
    assert _triple(_json_metadata(report_dir)) == ("-Os", "-O0", True)


def test_default_opt_is_reported_when_nothing_else_is_known(tmp_path, monkeypatch) -> None:
    assert _triple(_junit_props(_run(tmp_path, monkeypatch, []))) == ("-Ofast", "-Ofast", "false")


def test_junit_writes_booleans_like_json(tmp_path) -> None:
    from datetime import datetime

    from helia_core_tester.reporting.generator import ReportGenerator
    from helia_core_tester.reporting.models import TestReport

    now = datetime.now()
    report = TestReport(
        run_id="r", start_time=now, end_time=now, cpu="cortex-m55", descriptor_results={},
        all_descriptors=[], project_root=tmp_path,
        metadata={"coverage": True, "other": False, "count": 1, "skipped": None, "label": "True"},
    )
    path = ReportGenerator(output_dir=tmp_path).generate_reports(report, ["junit"])["junit"]
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    props = {p.get("name"): p.get("value") for p in suite.iter("property")}
    assert props == {"coverage": "true", "other": "false", "count": "1", "label": "True"}
