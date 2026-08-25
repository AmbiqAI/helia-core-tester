from __future__ import annotations

from pathlib import Path

from helia_core_tester.perf_stream.generated_test_bridge import (
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, test_name: str) -> dict:
    cases = discover_generated_tests(PROJECT_ROOT, family="NNSupportFunctions", name_filter=test_name)
    assert cases, f"expected discoverable generated test {test_name}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    return bundle.manifest


def test_requantize_s8_case_extracts_scalars(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "requantize_default_s8")
    assert manifest["tensor_dtypes"] == {"input": "S8", "output": "S8"}
    assert manifest["serialized_scalar_parameters"] == {
        "out_mult": 1073741824,
        "out_shift": 0,
        "input_offset": 3,
        "output_offset": -2,
    }


def test_requantize_s16_case_uses_s16_kernel(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "requantize_default_s16")
    assert manifest["tensor_dtypes"] == {"input": "S16", "output": "S16"}
    assert manifest["required_target_capabilities"] == ["arm_requantize_s16_s16"]
