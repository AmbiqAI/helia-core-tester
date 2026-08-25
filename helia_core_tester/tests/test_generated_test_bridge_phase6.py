from __future__ import annotations

from pathlib import Path

from helia_core_tester.perf_stream.case_bundle import load_case_bundle
from helia_core_tester.perf_stream.generated_test_bridge import build_case_bundle_from_generated_test, discover_generated_tests
from helia_core_tester.perf_stream.kernel_registry import lookup_kernel_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, family: str, name_filter: str) -> dict[str, object]:
    cases = discover_generated_tests(PROJECT_ROOT, family=family, name_filter=name_filter)
    assert cases, f"expected a discoverable {family} test matching {name_filter!r}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    return load_case_bundle(bundle.manifest_path).manifest


def test_convolve_s4_case_bridges_to_distinct_kernel_id(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ConvolutionFunctions", "convolve_even_mve_s4")
    assert manifest["tensor_dtypes"]["weights"] == "S4"
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT,
        family="ConvolutionFunctions",
        operator="Convolve",
        dtype="S8",
        weight_dtype="S4",
    )


def test_fully_connected_s4_case_bridges_without_scratch(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "FullyConnectedFunctions", "fully_connected_bias_s4")
    assert manifest["tensor_dtypes"]["weights"] == "S4"
    assert manifest["scratch_buffer"]["bytes"] == 0
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT,
        family="FullyConnectedFunctions",
        operator="FullyConnected",
        dtype="S8",
        weight_dtype="S4",
    )


def test_prelu_batch_case_bridges_and_serializes_output_n(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ActivationFunctions", "prelu_broadcast_batch_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["output_n"] == 2
    assert scalars["output_h"] == 2
    assert scalars["output_w"] == 2
    assert scalars["output_c"] == 3


def test_prelu_scalar_multi_pixel_case_bridges(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ActivationFunctions", "prelu_pixel_scalar_input_broadcast_c_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["block_size"] == 3
    assert scalars["output_h"] == 2
    assert scalars["output_c"] == 3


def test_prelu_arg_error_cases_bridge_as_status_assertions(tmp_path: Path) -> None:
    for case_name in ("prelu_arg_error_output_mismatch_s8", "prelu_arg_error_output_mismatch_s16"):
        manifest = _bridge(tmp_path, "ActivationFunctions", case_name)
        assert manifest["correctness_comparison"] == {
            "mode": "exact_status",
            "expected_status": -1,
            "expected_status_name": "ARM_CMSIS_NN_ARG_ERROR",
        }
        assert manifest["expected_output"]["byte_length"] == 0


def test_convolve_batch_padded_case_truncates_to_header_dims(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ConvolutionFunctions", "convolve_kernel1x1_stride_xy_case_01_s8")
    blobs = {blob["role"]: blob for blob in manifest["blob_roles"]}
    assert blobs["input_0"]["dimensions"] == [1, 4, 4, 5]
    assert blobs["input_0"]["byte_length"] == 80
    assert manifest["expected_output"]["byte_length"] == 20


def test_depthwise_batch_padded_case_truncates_to_header_dims(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ConvolutionFunctions", "depthwise_conv_mult_batches_s8")
    blobs = {blob["role"]: blob for blob in manifest["blob_roles"]}
    assert blobs["input_0"]["dimensions"] == [1, 5, 3, 3]
    assert blobs["input_0"]["byte_length"] == 45
    assert manifest["expected_output"]["byte_length"] == 18


def test_pool_batch_padded_case_truncates_to_header_dims(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "PoolingFunctions", "avg_pool_valid_pool1x1_stride1x2_s16")
    blobs = {blob["role"]: blob for blob in manifest["blob_roles"]}
    assert blobs["input_0"]["dimensions"] == [1, 1, 9, 2]
    assert blobs["input_0"]["byte_length"] == 36
    assert manifest["expected_output"]["byte_length"] == 20


def test_fp16_pooling_expected_output_manifest_uses_fp16(tmp_path: Path) -> None:
    cases = discover_generated_tests(
        PROJECT_ROOT,
        suite="float",
        family="PoolingFunctions",
        name_filter="avg_pool_float_nhwc_alias_f16",
    )
    assert cases
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False
    )
    expected = bundle.manifest["expected_output"]
    blob = next(entry for entry in bundle.manifest["blob_roles"] if entry["blob_id"] == expected["blob_id"])
    assert expected["dtype"] == blob["dtype"] == "FP16"
    assert expected["byte_length"] == blob["byte_length"]


def test_grouped_convolve_case_01_now_bridges_with_unified_tolerance(tmp_path: Path) -> None:
    """Regression test: convolve_grouped_conv_case_01_s8 now bridges under
    tolerant_int/tolerance=1 (was previously unbridgeable under exact_int)."""
    cases = discover_generated_tests(PROJECT_ROOT, family="ConvolutionFunctions", name_filter="convolve_grouped_conv_case_01_s8")
    assert cases
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False
    )
    assert bundle.manifest["correctness_comparison"] == {"mode": "tolerant_int", "tolerance": 1}
