from pathlib import Path
import json

from helia_core_tester.reporting.coverage_merge import run_coverage_merge


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
        project_root / "artifacts" / "build-cortex-m0-gcc" / "reports" / "coverage" / "cortex-m0" / "coverage.info",
        [
            (str(file_a), [(10, 1), (11, 0)]),
            (str(file_c), [(5, 0)]),
        ],
    )
    _write_lcov(
        project_root / "artifacts" / "build-cortex-m4-gcc" / "reports" / "coverage" / "cortex-m4" / "coverage.info",
        [
            (str(file_b), [(20, 3)]),
        ],
    )
    _write_lcov(
        project_root / "artifacts" / "build-cortex-m55-gcc" / "reports" / "coverage" / "cortex-m55" / "coverage.info",
        [
            (str(file_a), [(10, 0), (11, 0)]),
        ],
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
        report_dir=project_root / "artifacts" / "reports" / "coverage-merged",
        expected_zero_config=expected_zero_config,
    )

    assert exit_code == 0
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
        report_dir=project_root / "artifacts" / "reports" / "coverage-merged",
        expected_zero_config=project_root / "assets" / "coverage_expected_zero.json",
    )

    assert exit_code == 1
    assert report.coverage_inputs == {}
