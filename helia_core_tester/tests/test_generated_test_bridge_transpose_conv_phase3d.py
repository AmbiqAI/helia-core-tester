from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.perf_stream.generated_test_bridge import (
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "artifacts" / "generated_tests").is_dir(),
    reason="no generated-test artifacts under artifacts/generated_tests/ "
    "(artifacts/ is gitignored -- run `helia_core_tester generate` first)",
)


def _bridge(tmp_path: Path, test_name: str) -> dict:
    cases = discover_generated_tests(PROJECT_ROOT, family="ConvolutionFunctions", name_filter=test_name)
    assert cases, f"expected discoverable generated test {test_name}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    return bundle.manifest


def test_transpose_conv_no_bias_case_omits_bias_tensor_dtype(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8")
    assert manifest["tensor_dtypes"] == {"input": "S8", "weights": "S8", "output": "S8"}
    assert manifest["required_target_capabilities"] == ["arm_transpose_conv_wrapper_s8"]
    assert manifest["scratch_buffer"]["bytes"] > 0


def test_transpose_conv_bias_case_extracts_padding_offsets(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "transpose_conv_same_kernel6x6_stride2x2_bias_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["pad_offset_h"] == 0
    assert scalars["pad_offset_w"] == 0
    assert manifest["tensor_dtypes"]["bias"] == "S32"
    assert manifest["correctness_comparison"] == {"mode": "tolerant_int", "tolerance": 1}
