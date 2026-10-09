from __future__ import annotations

from pathlib import Path

from helia_core_tester.hardware.generated_test_bridge import (
    build_case_bundle_from_generated_test,
)
from helia_core_tester.tests.generated_inputs import discover_or_skip

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, test_name: str) -> dict:
    cases = discover_or_skip(PROJECT_ROOT, family="ConvolutionFunctions", name_filter=test_name)
    assert cases, f"expected discoverable generated test {test_name}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    return bundle.manifest


def test_transpose_conv_no_bias_case_omits_bias_tensor_dtype(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8")
    assert manifest["tensor_dtypes"] == {"input": "S8", "weights": "S8", "output": "S8"}
    assert manifest["required_target_capabilities"] == ["arm_transpose_conv_wrapper_s8"]
    assert manifest["scratch_buffer"]["bytes"] > 0


def test_transpose_conv_bias_case_extracts_padding_offsets(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "transpose_conv_reverse_same_kernel3x3_stride1x1_bias_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["pad_offset_h"] == 0
    assert scalars["pad_offset_w"] == 0
    assert manifest["tensor_dtypes"]["bias"] == "S32"
    # Bit-exact against the reference golden (see dtypes.py).
    assert manifest["correctness_comparison"] == {"mode": "exact_int"}


def test_batched_transpose_conv_case_is_refused_not_truncated(tmp_path: Path) -> None:
    # The descriptor declares batch 2; the golden now keeps it (the converter used to
    # collapse it to 1), and the bridge refuses batches rather than truncating them.
    import pytest

    from helia_core_tester.hardware.generated_test_bridge import UnsupportedGeneratedTestError

    with pytest.raises(UnsupportedGeneratedTestError, match="batch size > 1"):
        _bridge(tmp_path, "transpose_conv_same_kernel6x6_stride2x2_bias_s8")
