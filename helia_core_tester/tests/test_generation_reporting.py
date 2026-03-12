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
        "generated_tests_dir": str(generated_tests_dir),
    }


def test_generation_emits_canonical_report_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "cortex-m55"

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
            }
        ],
    )

    def _fake_generate_test(desc, out_dir, seed=None, cpu="cortex-m55", conversion_failures=None, generation_failures=None):
        test_dir = Path(out_dir) / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)

    generation_module.test_generation(_filters(generated_tests_dir))

    report_dir = repo_root / "artifacts" / "reports" / "generation" / "cortex-m55"
    summary = json.loads((report_dir / "generation_summary.json").read_text())
    manifest_pointer = json.loads((report_dir / "manifest_pointer.json").read_text())
    conversion_failures = json.loads((report_dir / "conversion_failures.json").read_text())
    generation_failures = json.loads((report_dir / "generation_failures.json").read_text())

    assert summary["status"] == "success"
    assert summary["counts"]["generated"] == 1
    assert manifest_pointer["manifest_path"] == str(generated_tests_dir / "manifest.json")
    assert conversion_failures == []
    assert generation_failures == []


def test_generation_writes_reports_even_when_no_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "cortex-m55"

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
            }
        ],
    )

    def _always_fail(desc, out_dir, seed=None, cpu="cortex-m55", conversion_failures=None, generation_failures=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(generation_module, "generate_test", _always_fail)

    with pytest.raises(AssertionError, match="No TFLite models were generated"):
        generation_module.test_generation(_filters(generated_tests_dir))

    report_dir = repo_root / "artifacts" / "reports" / "generation" / "cortex-m55"
    summary = json.loads((report_dir / "generation_summary.json").read_text())
    manifest_pointer = json.loads((report_dir / "manifest_pointer.json").read_text())

    assert summary["status"] == "failed_no_generated_tests"
    assert summary["counts"]["generated"] == 0
    assert manifest_pointer["manifest_path"] is None
