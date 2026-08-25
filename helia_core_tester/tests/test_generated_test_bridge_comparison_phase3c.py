from __future__ import annotations

from pathlib import Path

from helia_core_tester.perf_stream.generated_test_bridge import (
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, test_name: str) -> dict:
    cases = discover_generated_tests(PROJECT_ROOT, family="ComparisonFunctions", name_filter=test_name)
    assert cases, f"expected discoverable generated test {test_name}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    return bundle.manifest


def test_comparison_equal_batch_case_extracts_bool_output_and_dims(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "comparison_equal_batch_broadcast_s8")
    assert manifest["operator"] == "Equal"
    assert manifest["tensor_dtypes"] == {"input": "S8", "output": "BOOL"}
    assert manifest["serialized_scalar_parameters"]["output_n"] == 2
    assert manifest["correctness_comparison"] == {"mode": "bool"}


def test_comparison_less_equal_s16_uses_expected_kernel(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "comparison_less_equal_nhwc_s16")
    assert manifest["operator"] == "LessEqual"
    assert manifest["required_target_capabilities"] == ["arm_less_equal_s16"]
