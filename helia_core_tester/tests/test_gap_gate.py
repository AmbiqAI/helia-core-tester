from pathlib import Path
import json

import pytest

from helia_core_tester.reporting.gap_gate import (
    ALLOWLIST_DESCRIPTOR_NAMES,
    ALLOWLIST_OPERATORS,
    compute_gaps,
    run_gap_check,
)


def _write_manifest(base: Path, tests: list[dict]) -> None:
    manifest = {
        "generated_count": len(tests),
        "filters": {"cpu": "cortex-m55"},
        "tests": tests,
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _mk_desc_dir(base: Path, name: str, with_tflite: bool = True) -> None:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "descriptor.yaml").write_text("name: {}\n".format(name))
    if with_tflite:
        (d / f"{name}.tflite").write_bytes(b"tflite")


def _mk_header(base: Path, name: str) -> None:
    includes_dir = base / name / "includes"
    includes_dir.mkdir(parents=True, exist_ok=True)
    (includes_dir / f"{name}.h").write_text("// header")


def _write_descriptor_yaml(descriptors_dir: Path, name: str, operator: str) -> None:
    content = "\n".join([
        f"operator: {operator}",
        f"name: {name}",
        "activation_dtype: S8",
        "weight_dtype: S8",
        "activation: NONE",
        "hint:",
        "  call_style: per_tensor",
        "input_shape: [1, 1, 1, 1]",
        "",
    ])
    (descriptors_dir / f"{name}.yaml").write_text(content)


def test_gap_check_passes_with_no_gaps(tmp_path: Path) -> None:
    project_root = tmp_path
    gen_dir = project_root / "artifacts" / "generated_tests"
    gen_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)

    _mk_desc_dir(gen_dir, "ok_one", with_tflite=True)
    _write_manifest(gen_dir, [{"name": "ok_one", "operator": "Relu"}])
    _write_descriptor_yaml(desc_dir, "ok_one", "Relu")

    report = compute_gaps(project_root, "cortex-m55", generated_tests_dir=gen_dir)
    assert report.descriptor_without_tflite == []
    assert report.operators_zero_in_manifest == []


def test_gap_check_allows_allowlisted_descriptor_gaps(tmp_path: Path) -> None:
    project_root = tmp_path
    gen_dir = project_root / "artifacts" / "generated_tests"
    gen_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)

    name = next(iter(ALLOWLIST_DESCRIPTOR_NAMES))
    _mk_desc_dir(gen_dir, name, with_tflite=False)
    _write_descriptor_yaml(desc_dir, name, "ArgMax")
    _write_manifest(gen_dir, [])

    report = compute_gaps(project_root, "cortex-m55", generated_tests_dir=gen_dir)
    assert name in report.allowlisted_descriptor_gaps
    assert name not in report.unexpected_descriptor_gaps


def test_gap_check_fails_on_unexpected_descriptor_gap(tmp_path: Path) -> None:
    project_root = tmp_path
    gen_dir = project_root / "artifacts" / "generated_tests"
    gen_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)

    _mk_desc_dir(gen_dir, "unexpected_gap", with_tflite=False)
    _write_descriptor_yaml(desc_dir, "unexpected_gap", "Relu")
    _write_manifest(gen_dir, [])

    exit_code, report, _ = run_gap_check(project_root, "cortex-m55", generated_tests_dir=gen_dir)
    assert exit_code == 1
    assert "unexpected_gap" in report.unexpected_descriptor_gaps


def test_gap_check_allows_missing_tflite_with_headers(tmp_path: Path) -> None:
    project_root = tmp_path
    gen_dir = project_root / "artifacts" / "generated_tests"
    gen_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)

    _mk_desc_dir(gen_dir, "header_only", with_tflite=False)
    _mk_header(gen_dir, "header_only")
    _write_descriptor_yaml(desc_dir, "header_only", "Quantize")
    _write_manifest(gen_dir, [{"name": "header_only", "operator": "Quantize"}])

    report = compute_gaps(project_root, "cortex-m55", generated_tests_dir=gen_dir)
    assert "header_only" not in report.descriptor_without_tflite


def test_gap_check_fails_on_unexpected_zero_manifest_operator(tmp_path: Path) -> None:
    project_root = tmp_path
    gen_dir = project_root / "artifacts" / "generated_tests"
    gen_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)

    _mk_desc_dir(gen_dir, "present", with_tflite=True)
    _write_manifest(gen_dir, [{"name": "present", "operator": "Relu"}])
    _write_descriptor_yaml(desc_dir, "present", "Relu")
    _write_descriptor_yaml(desc_dir, "missing_op", "SVDF")

    exit_code, report, _ = run_gap_check(project_root, "cortex-m55", generated_tests_dir=gen_dir)
    assert exit_code == 1
    assert "SVDF" in report.unexpected_zero_manifest_ops


def test_gap_check_fails_on_manifest_missing_descriptor(tmp_path: Path) -> None:
    project_root = tmp_path
    gen_dir = project_root / "artifacts" / "generated_tests"
    gen_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)

    _write_manifest(gen_dir, [{"name": "missing_dir", "operator": "Relu"}])
    _write_descriptor_yaml(desc_dir, "missing_dir", "Relu")

    exit_code, report, _ = run_gap_check(project_root, "cortex-m55", generated_tests_dir=gen_dir)
    assert exit_code == 1
    assert "missing_dir" in report.manifest_missing_descriptor


def test_reports_are_written(tmp_path: Path) -> None:
    project_root = tmp_path
    gen_dir = project_root / "artifacts" / "generated_tests"
    gen_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)
    report_dir = project_root / "artifacts" / "reports"

    _mk_desc_dir(gen_dir, "ok_one", with_tflite=True)
    _write_manifest(gen_dir, [{"name": "ok_one", "operator": "Relu"}])
    _write_descriptor_yaml(desc_dir, "ok_one", "Relu")

    exit_code, report, (json_path, md_path) = run_gap_check(
        project_root, "cortex-m55", report_dir=report_dir, generated_tests_dir=gen_dir
    )
    assert exit_code == 0
    assert json_path.exists()
    assert md_path.exists()


def test_cpu_specific_dir_preferred(tmp_path: Path) -> None:
    project_root = tmp_path
    cpu_dir = project_root / "artifacts" / "generated_tests" / "cortex-m55"
    cpu_dir.mkdir(parents=True)
    desc_dir = project_root / "assets" / "descriptors"
    desc_dir.mkdir(parents=True)

    _mk_desc_dir(cpu_dir, "ok_one", with_tflite=True)
    _write_manifest(cpu_dir, [{"name": "ok_one", "operator": "Relu"}])
    _write_descriptor_yaml(desc_dir, "ok_one", "Relu")

    report = compute_gaps(project_root, "cortex-m55")
    assert report.generated_tests_dir == cpu_dir
