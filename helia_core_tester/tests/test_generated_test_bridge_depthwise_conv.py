"""Regression tests for the ConvolutionFunctions DepthwiseConv perf-stream hardware bridge.

Unlike Convolve's builder (which reorders `filter_dims` into (H, W, C, N) blob-storage
order), DepthwiseConv's generated header already defines `filter_dims` in native
(N=1, H, W, C_OUT) order matching `cmsis_nn_dw_conv_params`'s expectations, so
`_build_depthwise_conv_case()` must NOT reorder them -- see the builder's docstring. These
tests pin the extracted `cmsis_nn_dw_conv_params` scalars (including the DepthwiseConv-only
`ch_mult` field) against known-good values read directly from the generated headers, across
a default case, a large-`ch_mult` case, and a dilated case.

This test does not touch real hardware; it bridges real generated-test artifacts already
checked into `artifacts/generated_tests/` and asserts on the resulting CaseBundle manifest,
including the larger output-buffer allowance and the S4-weight DepthwiseConv bridge's
packed-weight handling.
"""

from __future__ import annotations

from pathlib import Path

from helia_core_tester.perf_stream.generated_test_bridge import (
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from helia_core_tester.perf_stream.case_bundle import load_case_bundle
from helia_core_tester.perf_stream.kernel_registry import lookup_kernel_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, name_filter: str) -> dict[str, object]:
    cases = discover_generated_tests(PROJECT_ROOT, family="ConvolutionFunctions", name_filter=name_filter)
    assert cases, f"expected a discoverable ConvolutionFunctions test matching {name_filter!r}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    loaded = load_case_bundle(bundle.manifest_path)
    return loaded.manifest


def test_depthwise_conv_kernel_support_case_extracts_true_scalars_and_kernel_id(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "depthwise_conv_kernel_support_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["stride_h"] == 1
    assert scalars["stride_w"] == 1
    assert scalars["pad_h"] == 1
    assert scalars["pad_w"] == 1
    assert scalars["dilation_h"] == 1
    assert scalars["dilation_w"] == 1
    assert scalars["output_h"] == 4
    assert scalars["output_w"] == 4
    assert scalars["output_c"] == 2
    assert scalars["input_offset"] == 0
    assert scalars["output_offset"] == -56
    assert scalars["activation_min"] == -128
    assert scalars["activation_max"] == 127
    assert scalars["ch_mult"] == 1
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT, family="ConvolutionFunctions", operator="DepthwiseConv", dtype="S8"
    )
    assert manifest["operator"] == "DepthwiseConv"
    role_names = {blob["role"] for blob in manifest["blob_roles"]}
    assert role_names == {"input_0", "weights", "bias", "multiplier", "shift", "expected_output"}


def test_depthwise_conv_large_channel_multiplier_case_bridges(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "depthwise_conv_no_bias_case_02_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["ch_mult"] == 8
    assert scalars["output_c"] == 16


def test_depthwise_conv_dilated_case_bridges(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "depthwise_conv_dilation_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["dilation_h"] == 3
    assert scalars["dilation_w"] == 2
    assert scalars["ch_mult"] == 3


def test_depthwise_conv_large_output_case_bridges_after_output_buffer_bump(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "depthwise_conv_eq_in_out_ch_s8")
    assert manifest["expected_output"]["byte_length"] == 8750


def test_depthwise_conv_s4_generic_case_bridges_with_packed_weights(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "depthwise_conv_generic_s4")
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT,
        family="ConvolutionFunctions",
        operator="DepthwiseConv",
        dtype="S8",
        weight_dtype="S4",
    )
    assert manifest["tensor_dtypes"]["weights"] == "S4"
    assert manifest["scratch_buffer"]["bytes"] == 0
    weights_blob = next(blob for blob in manifest["blob_roles"] if blob["role"] == "weights")
    assert weights_blob["byte_length"] == 36
    assert weights_blob["dimensions"] == [1, 3, 3, 8]


def test_depthwise_conv_s4_opt_case_bridges_with_wrapper_scratch(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "depthwise_conv_opt_s4")
    assert manifest["required_target_capabilities"] == ["depthwise_conv_s4"]
    assert manifest["scratch_buffer"]["bytes"] == 4464
