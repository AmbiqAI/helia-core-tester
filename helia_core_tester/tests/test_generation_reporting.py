from __future__ import annotations

import json
from pathlib import Path

import pytest

import helia_core_tester.generation.test_ops as generation_module


def _filters(generated_tests_dir: Path, cpu: str = "cortex-m55") -> dict[str, object]:
    return {
        "op": None,
        "dtype": None,
        "wtype": None,
        "name": None,
        "limit": None,
        "seed": 123,
        "cpu": cpu,
        "suite": "int",
        "float_precision": "both",
        "generated_tests_dir": str(generated_tests_dir),
    }


def test_generation_emits_canonical_report_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "int" / "cortex-m4"

    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors")
    monkeypatch.setattr(
        generation_module,
        "load_all_descriptors",
        lambda _path: [
            {
                "name": "fc_smoke",
                "operator": "FullyConnected",
                "activation_dtype": "S8",
                "weight_dtype": "S8",
                "_family": "FullyConnectedFunctions",
                "_parity_kind": "cmsis",
                "_source_family": "FullyConnectedFunctions",
                "_source_stem": "fully_connected",
                "_source_relpath": "FullyConnectedFunctions/fully_connected.yaml",
            }
        ],
    )

    def _fake_generate_test(desc, out_dir, seed=None, cpu="cortex-m55", conversion_failures=None, generation_failures=None):
        test_dir = Path(out_dir) / desc["_family"] / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")
        # tests.cmake only lists directories with runnable c_sources (see
        # test_ops.py's runnable_entries filter), so the fake must emit a .c
        # file too for this test to exercise the real tests.cmake contract.
        (test_dir / f"{desc['name']}_{desc['operator'].lower()}.c").write_text("// fake generated harness\n")

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)

    generation_module.test_generation(_filters(generated_tests_dir, cpu="cortex-m4"))

    report_dir = repo_root / "artifacts" / "reports" / "generation" / "int" / "cortex-m4"
    summary = json.loads((report_dir / "generation_summary.json").read_text())
    manifest_pointer = json.loads((report_dir / "manifest_pointer.json").read_text())
    conversion_failures = json.loads((report_dir / "conversion_failures.json").read_text())
    generation_failures = json.loads((report_dir / "generation_failures.json").read_text())
    capability_skips = json.loads((report_dir / "capability_skips.json").read_text())

    assert summary["status"] == "success"
    assert summary["families"] == ["FullyConnectedFunctions"]
    assert summary["parity_kind_counts"] == {"cmsis": 1}
    assert summary["counts"]["generated"] == 1
    assert summary["counts"]["skipped_capability"] == 0
    assert manifest_pointer["manifest_path"] == str(generated_tests_dir / "manifest.json")
    manifest = json.loads((generated_tests_dir / "manifest.json").read_text())
    assert manifest["tests"][0]["relative_test_dir"] == "FullyConnectedFunctions/fc_smoke"
    assert manifest["tests"][0]["resolved_tensor_dtypes"] == {"input": "S8", "output": "S8", "weights": "S8"}
    assert '"artifacts/generated_tests/int/cortex-m4/FullyConnectedFunctions/fc_smoke"' in (generated_tests_dir / "tests.cmake").read_text()
    assert conversion_failures == []
    assert generation_failures == []
    assert capability_skips == []


def test_generation_writes_reports_even_when_no_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "int" / "cortex-m4"

    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors")
    monkeypatch.setattr(
        generation_module,
        "load_all_descriptors",
        lambda _path: [
            {
                "name": "failing_case",
                "operator": "FullyConnected",
                "activation_dtype": "S8",
                "weight_dtype": "S8",
                "_family": "FullyConnectedFunctions",
                "_parity_kind": "cmsis",
                "_source_family": "FullyConnectedFunctions",
                "_source_stem": "fully_connected",
                "_source_relpath": "FullyConnectedFunctions/fully_connected.yaml",
            }
        ],
    )

    def _always_fail(desc, out_dir, seed=None, cpu="cortex-m55", conversion_failures=None, generation_failures=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(generation_module, "generate_test", _always_fail)

    with pytest.raises(AssertionError, match="No TFLite models were generated"):
        generation_module.test_generation(_filters(generated_tests_dir))

    report_dir = repo_root / "artifacts" / "reports" / "generation" / "int" / "cortex-m55"
    summary = json.loads((report_dir / "generation_summary.json").read_text())
    manifest_pointer = json.loads((report_dir / "manifest_pointer.json").read_text())

    assert summary["status"] == "failed_no_generated_tests"
    assert summary["counts"]["generated"] == 0
    assert manifest_pointer["manifest_path"] is None


def test_generation_records_capability_skips_in_manifest_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "int" / "cortex-m55"

    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors")
    monkeypatch.setattr(
        generation_module,
        "load_all_descriptors",
        lambda _path: [
            {
                "name": "future_fp16_case",
                "operator": "Abs",
                "tensor_dtypes": {"input": "FP16", "output": "FP16"},
                "resolved_tensor_dtypes": {"input": "FP16", "output": "FP16"},
                "_family": "BasicMathFunctions",
                "_parity_kind": "cmsis",
                "_source_family": "BasicMathFunctions",
                "_source_stem": "abs",
                "_source_relpath": "BasicMathFunctions/abs.yaml",
                "input_shape": [1, 4],
                "required_capabilities": ["fp16_execution"],
            }
        ],
    )

    generation_module.test_generation(_filters(generated_tests_dir, cpu="cortex-m4"))

    report_dir = repo_root / "artifacts" / "reports" / "generation" / "int" / "cortex-m4"
    summary = json.loads((report_dir / "generation_summary.json").read_text())
    manifest = json.loads((generated_tests_dir / "manifest.json").read_text())
    capability_skips = json.loads((report_dir / "capability_skips.json").read_text())

    assert summary["status"] == "skipped_only"
    assert summary["counts"]["generated"] == 0
    assert summary["counts"]["skipped_capability"] == 1
    assert manifest["generated_count"] == 0
    assert manifest["skipped_count"] == 1
    assert manifest["skipped"][0]["missing_capabilities"] == ["fp16_execution"]
    assert capability_skips[0]["name"] == "future_fp16_case"


def test_generation_excludes_float_suite_unless_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "int" / "cortex-m4"

    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors")
    monkeypatch.setattr(
        generation_module,
        "load_all_descriptors",
        lambda _path: [
            {
                "name": "softmax_float_default_f32",
                "operator": "Softmax",
                "tensor_dtypes": {"input": "FP32", "output": "FP32"},
                "resolved_tensor_dtypes": {"input": "FP32", "output": "FP32"},
                "_family": "SoftmaxFunctions",
                "_parity_kind": "cmsis",
                "_source_family": "SoftmaxFunctions",
                "_source_stem": "softmax_float",
                "_source_relpath": "SoftmaxFunctions/softmax_float.yaml",
                "_descriptor_suite": "float",
                "input_shape": [1, 4],
            }
        ],
    )

    with pytest.raises(AssertionError, match="Filter matched no descriptors"):
        generation_module.test_generation(_filters(generated_tests_dir, cpu="cortex-m4"))

    summary = json.loads((repo_root / "artifacts" / "reports" / "generation" / "int" / "cortex-m4" / "generation_summary.json").read_text())
    assert not (generated_tests_dir / "manifest.json").exists()
    assert summary["counts"]["descriptors_after_filters"] == 0
