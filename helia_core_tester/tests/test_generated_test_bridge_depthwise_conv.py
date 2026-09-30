"""Regression tests for the ConvolutionFunctions DepthwiseConv hardware bridge.

Unlike Convolve's builder (which reorders `filter_dims` into (H, W, C, N) blob-storage
order), DepthwiseConv's generated header already defines `filter_dims` in native
(N=1, H, W, C_OUT) order matching `cmsis_nn_dw_conv_params`'s expectations, so
`_build_depthwise_conv_case()` must NOT reorder them -- see the builder's docstring. These
tests pin the extracted `cmsis_nn_dw_conv_params` scalars (including the DepthwiseConv-only
`ch_mult` field) across a default case, a large-`ch_mult` case, and a dilated case. Shape-
derived scalars are pinned; the quantization offsets depend on the generated data, so they are
checked against the zero points of the case's own TFLite model.

This test does not touch real hardware. Each test generates its case into `tmp_path` (seed
500, the CLI default), so it needs no pre-generated corpus, bridges it, and asserts on the
resulting CaseBundle manifest, including the larger output-buffer allowance and the S4-weight
DepthwiseConv bridge's packed-weight handling.
"""

from __future__ import annotations

from pathlib import Path

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.hardware.generated_test_bridge import (
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from helia_core_tester.hardware.case_bundle import load_case_bundle
from helia_core_tester.hardware.kernel_registry import lookup_kernel_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Mirrors the corpus layout, so discovery runs exactly as it does on a real corpus.
GENERATED_CPU_DIR = Path("artifacts", "generated_tests", "int", "cortex-m55")


def _bridge(tmp_path: Path, name: str) -> dict[str, object]:
    desc = next(d for d in load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(tmp_path / GENERATED_CPU_DIR), seed=500)
    cases = discover_generated_tests(tmp_path, family="ConvolutionFunctions", name_filter=name)
    assert [case.name for case in cases] == [name]
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, cases[0], output_root=tmp_path / "bundle", require_fvp_pass=False
    )
    return load_case_bundle(bundle.manifest_path).manifest


def _model_zero_points(case_dir: Path) -> tuple[int, int]:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(case_dir / f"{case_dir.name}.tflite"))
    (model_input,) = interpreter.get_input_details()
    (model_output,) = interpreter.get_output_details()
    return int(model_input["quantization"][1]), int(model_output["quantization"][1])


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
    input_zero_point, output_zero_point = _model_zero_points(
        tmp_path / GENERATED_CPU_DIR / "ConvolutionFunctions" / "depthwise_conv_kernel_support_s8"
    )
    assert scalars["input_offset"] == -input_zero_point
    assert scalars["output_offset"] == output_zero_point
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
