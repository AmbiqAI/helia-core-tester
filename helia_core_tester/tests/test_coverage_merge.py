from pathlib import Path
import json

import pytest

from helia_core_tester.reporting import coverage_merge
from helia_core_tester.reporting.coverage_merge import run_coverage_merge


@pytest.fixture(autouse=True)
def builtin_html(monkeypatch):
    monkeypatch.setattr(
        coverage_merge,
        "_try_write_gcovr_html",
        lambda *args, **kwargs: (False, "test builtin renderer"),
    )


def _write_lcov(path: Path, records: list[tuple[str, list[tuple[int, int]]]]) -> None:
    lines: list[str] = []
    for sf, da in records:
        lines.append(f"SF:{sf}")
        for line_no, hits in da:
            lines.append(f"DA:{line_no},{hits}")
        lines.append(f"LF:{len(da)}")
        lines.append(f"LH:{sum(1 for _, hits in da if hits > 0)}")
        lines.append("end_of_record")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_coverage_merge_merges_and_classifies(tmp_path: Path) -> None:
    project_root = tmp_path

    file_a = project_root / "Source" / "ConvolutionFunctions" / "a.c"
    file_b = project_root / "Source" / "ConvolutionFunctions" / "b.c"
    file_c = project_root / "Source" / "NNSupportFunctions" / "c.c"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_b.parent.mkdir(parents=True, exist_ok=True)
    file_c.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("// a\n")
    file_b.write_text("// b\n")
    file_c.write_text("// c\n")

    _write_lcov(
        project_root / "artifacts" / "reports" / "coverage" / "int" / "cortex-m0" / "coverage.info",
        [
            (str(file_a), [(10, 1), (11, 0)]),
            (str(file_c), [(5, 0)]),
        ],
    )
    _write_lcov(
        project_root / "artifacts" / "reports" / "coverage" / "int" / "cortex-m4" / "coverage.info",
        [
            (str(file_b), [(20, 3)]),
        ],
    )
    _write_lcov(
        project_root / "artifacts" / "reports" / "coverage" / "float" / "cortex-m55" / "coverage.info",
        [
            (str(file_a), [(10, 0), (11, 0)]),
        ],
    )

    for suite, cpu in (
        ("int", "cortex-m55"),
        ("float", "cortex-m0"),
        ("float", "cortex-m4"),
    ):
        _write_lcov(
            project_root / "artifacts/reports/coverage" / suite / cpu / "coverage.info",
            [(str(file_a), [(10, 1), (11, 0)])],
        )

    expected_zero_config = project_root / "assets" / "coverage_expected_zero.json"
    expected_zero_config.parent.mkdir(parents=True, exist_ok=True)
    expected_zero_config.write_text(
        json.dumps(
            {
                "expected_zero_files": [
                    "Source/NNSupportFunctions/c.c",
                    "Source/NNSupportFunctions/missing.c",
                ]
            },
            indent=2,
        )
    )

    exit_code, report = run_coverage_merge(
        project_root=project_root,
        cpus="cortex-m0,cortex-m4,cortex-m55",
        suites=["int", "float"],
        report_dir=project_root / "artifacts" / "reports" / "coverage" / "merged",
        expected_zero_config=expected_zero_config,
    )

    assert exit_code == 0
    assert report.missing_coverage_inputs == {}
    assert "Source/ConvolutionFunctions/a.c" in report.covered_files
    assert "Source/ConvolutionFunctions/b.c" in report.covered_files
    assert "Source/NNSupportFunctions/c.c" in report.expected_zero_files
    assert "Source/NNSupportFunctions/missing.c" in report.expected_zero_missing_files
    assert report.overall_line_rate == 50.0
    assert report.html_generator in ("gcovr", "builtin")
    assert report.summary_json_path.exists()
    assert report.summary_md_path.exists()
    assert report.summary_html_path.exists()
    assert report.merged_lcov_path.exists()


def test_coverage_merge_fails_when_no_inputs(tmp_path: Path) -> None:
    project_root = tmp_path
    exit_code, report = run_coverage_merge(
        project_root=project_root,
        cpus="cortex-m55",
        suites=["int"],
        report_dir=project_root / "artifacts" / "reports" / "coverage" / "merged",
        expected_zero_config=project_root / "assets" / "coverage_expected_zero.json",
    )

    assert exit_code == 1
    assert report.coverage_inputs == {}


def test_coverage_merge_fails_when_any_requested_input_missing(tmp_path: Path) -> None:
    project_root = tmp_path
    source = project_root / "Source" / "NNSupportFunctions" / "a.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("// a\n")
    _write_lcov(
        project_root / "artifacts" / "reports" / "coverage" / "int" / "cortex-m55" / "coverage.info",
        [(str(source), [(1, 1)])],
    )

    exit_code, report = run_coverage_merge(
        project_root=project_root,
        cpus="cortex-m0,cortex-m55",
        suites=["int"],
        report_dir=project_root / "artifacts" / "reports" / "coverage" / "merged",
        expected_zero_config=project_root / "assets" / "coverage_expected_zero.json",
    )

    assert exit_code == 1
    assert "int:cortex-m55" in report.coverage_inputs
    assert "int:cortex-m0" in report.missing_coverage_inputs


def test_coverage_merge_includes_optional_float_mve_for_m55(tmp_path: Path) -> None:
    project_root = tmp_path
    file_a = project_root / "Source" / "ConvolutionFunctions" / "a.c"
    file_mve = project_root / "Source" / "ConvolutionFunctions" / "mve.c"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("// a\n")
    file_mve.write_text("// mve\n")

    for cpu in ("cortex-m0", "cortex-m4", "cortex-m55"):
        _write_lcov(
            project_root / "artifacts" / "reports" / "coverage" / "int" / cpu / "coverage.info",
            [(str(file_a), [(10, 1)])],
        )
    for cpu in ("cortex-m0", "cortex-m4", "cortex-m55"):
        _write_lcov(
            project_root / "artifacts/reports/coverage/float" / cpu / "coverage.info",
            [(str(file_a), [(10, 0)])],
        )
    _write_lcov(
        project_root / "artifacts" / "reports" / "coverage" / "float-mve" / "cortex-m55" / "coverage.info",
        [(str(file_mve), [(7, 2)])],
    )

    exit_code, report = run_coverage_merge(
        project_root=project_root,
        cpus="cortex-m0,cortex-m4,cortex-m55",
        suites=["int", "float", "float-mve"],
        report_dir=project_root / "artifacts" / "reports" / "coverage" / "merged",
        expected_zero_config=project_root / "assets" / "coverage_expected_zero.json",
    )

    assert exit_code == 0
    # float-mve coverage is merged in for cortex-m55.
    assert "float-mve:cortex-m55" in report.coverage_inputs
    assert "Source/ConvolutionFunctions/mve.c" in report.covered_files
    # float-mve is only probed for cortex-m55; other CPUs are never treated as missing.
    assert "float-mve:cortex-m0" not in report.missing_coverage_inputs
    assert "float-mve:cortex-m4" not in report.missing_coverage_inputs
    assert report.missing_coverage_inputs == {}


def test_coverage_merge_optional_float_mve_absent_does_not_fail(tmp_path: Path) -> None:
    project_root = tmp_path
    file_a = project_root / "Source" / "ConvolutionFunctions" / "a.c"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("// a\n")
    _write_lcov(
        project_root / "artifacts" / "reports" / "coverage" / "int" / "cortex-m55" / "coverage.info",
        [(str(file_a), [(10, 1)])],
    )

    # float-mve coverage is requested but not produced; it must be treated as optional.
    exit_code, report = run_coverage_merge(
        project_root=project_root,
        cpus="cortex-m55",
        suites=["int", "float-mve"],
        report_dir=project_root / "artifacts" / "reports" / "coverage" / "merged",
        expected_zero_config=project_root / "assets" / "coverage_expected_zero.json",
    )

    assert exit_code == 0
    assert "float-mve:cortex-m55" not in report.missing_coverage_inputs


@pytest.mark.parametrize(
    "missing",
    [
        (),
        ("float:cortex-m4", "float:cortex-m55"),
        ("float:cortex-m55",),
        ("int:cortex-m55",),
    ],
)
def test_required_pairs_cannot_be_replaced_by_other_coverage(tmp_path, missing):
    source = tmp_path / "Source/a.c"
    source.parent.mkdir()
    source.write_text("// a\n")
    inputs = {
        f"{suite}:{cpu}"
        for suite in ("int", "float")
        for cpu in ("cortex-m4", "cortex-m55")
    }
    inputs.add("float-mve:cortex-m55")
    paths = {
        key: tmp_path
        / "artifacts/reports/coverage"
        / key.split(":")[0]
        / key.split(":")[1]
        / "coverage.info"
        for key in inputs
    }
    for key in inputs - set(missing):
        _write_lcov(paths[key], [(str(source), [(1, 1), (2, 0)])])
    code, report = run_coverage_merge(
        tmp_path, "cortex-m4,cortex-m55", ["int", "float", "float-mve"]
    )
    expected_missing = {key: str(paths[key]) for key in missing}
    assert report.missing_coverage_inputs == expected_missing
    assert (
        json.loads(report.summary_json_path.read_text())["missing_coverage_inputs"]
        == expected_missing
    )
    assert set(report.coverage_inputs) == inputs - set(missing)
    assert report.overall_line_rate == 50.0
    assert f"DA:1,{len(inputs) - len(missing)}\n" in report.merged_lcov_path.read_text()
    assert report.summary_html_path.exists()
    for key, path in expected_missing.items():
        assert key in report.summary_md_path.read_text()
        assert path in report.summary_md_path.read_text()
    assert code == (1 if missing else 0)


@pytest.mark.parametrize(
    "suite,present", [("int", True), ("float-mve", True), ("float-mve", False)]
)
def test_single_suite_acceptance(tmp_path, suite, present):
    if present:
        source = tmp_path / "a.c"
        source.write_text("// a\n")
        _write_lcov(
            tmp_path
            / "artifacts/reports/coverage"
            / suite
            / "cortex-m55/coverage.info",
            [(str(source), [(1, 1)])],
        )
    code, report = run_coverage_merge(tmp_path, "cortex-m55", [suite])
    assert report.missing_coverage_inputs == {}
    assert code == (0 if present else 1)


def test_cli_names_missing_required_pairs(tmp_path):
    from typer.testing import CliRunner
    from helia_core_tester.cli import app

    for directory in (
        "helia_core_tester/generation",
        "assets/templates",
        "assets/descriptors",
    ):
        (tmp_path / directory).mkdir(parents=True)
    result = CliRunner().invoke(
        app,
        [
            "coverage-merge",
            "--repo-root",
            str(tmp_path),
            "--cpu",
            "cortex-m4,cortex-m55",
            "--suite",
            "both",
        ],
    )
    assert result.exit_code == 1
    for suite in ("int", "float"):
        for cpu in ("cortex-m4", "cortex-m55"):
            assert f"{suite}:{cpu}" in result.stderr
