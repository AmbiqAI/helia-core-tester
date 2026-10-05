from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.hardware.case_bundle import load_case_bundle
from helia_core_tester.hardware.generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    _calculate_convolve_s4_scratch_bytes,
    _extract_define_int,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from helia_core_tester.hardware.kernel_registry import lookup_entry_id, lookup_kernel_id
from helia_core_tester.tests.generated_inputs import discover_or_skip

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, family: str, name_filter: str) -> dict[str, object]:
    cases = discover_or_skip(PROJECT_ROOT, family=family, name_filter=name_filter)
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


def test_convolve_batch_case_rejected_without_truncation(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedGeneratedTestError, match="batch size 2 > 1"):
        _bridge(tmp_path, "ConvolutionFunctions", "convolve_kernel1x1_stride_xy_case_01_s8")


def test_depthwise_batch_case_rejected_without_truncation(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedGeneratedTestError, match="batch size 2 > 1"):
        _bridge(tmp_path, "ConvolutionFunctions", "depthwise_conv_mult_batches_s8")


def test_pool_batch_padded_case_truncates_to_header_dims(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "PoolingFunctions", "avg_pool_valid_pool1x1_stride1x2_s16")
    blobs = {blob["role"]: blob for blob in manifest["blob_roles"]}
    assert blobs["input_0"]["dimensions"] == [1, 1, 9, 2]
    assert blobs["input_0"]["byte_length"] == 36
    assert manifest["expected_output"]["byte_length"] == 20


def test_fp16_pooling_expected_output_manifest_uses_fp16(tmp_path: Path) -> None:
    cases = discover_or_skip(
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
    cases = discover_or_skip(PROJECT_ROOT, family="ConvolutionFunctions", name_filter="convolve_grouped_conv_case_01_s8")
    assert cases
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False
    )
    assert bundle.manifest["correctness_comparison"] == {"mode": "tolerant_int", "tolerance": 1}


_PACK_BLOCK = {"FP16": 8, "FP32": 4}
_NUMPY_DTYPE = {"FP16": np.float16, "FP32": np.float32}
_PACKED_HINTS = {"NT_N_PACKED", "ARM_NN_WEIGHT_FORMAT_NT_N_PACKED"}


def _partially_filled_packed_descriptors() -> list[dict]:
    # NT_N_PACKED stores float convolve weights as [ceil(out_c / block)][K][block], block 8 for f16
    # and 4 for f32. Only an out_c that leaves the last block partly filled makes the array longer
    # than the shape. The hint spellings and the block follow the generator. Direct-entry cases
    # are left out: the bridge refuses them.
    return [
        desc
        for desc in load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors"))
        if desc["operator"] == "Convolve"
        and not desc.get("entry")
        and desc.get("activation_dtype") in _PACK_BLOCK
        and str((desc.get("hint") or {}).get("weight_format", "")).upper() in _PACKED_HINTS
        and desc["filter_shape"][3] % _PACK_BLOCK[desc["activation_dtype"]]
    ]


@pytest.mark.parametrize("desc", _partially_filled_packed_descriptors(), ids=lambda desc: desc["name"])
def test_packed_float_convolve_keeps_padded_weights(tmp_path: Path, desc: dict) -> None:
    generate_test(desc, str(tmp_path / "artifacts" / "generated_tests" / "float" / "cortex-m55"), seed=Config.seed)
    (case,) = discover_generated_tests(tmp_path, suite="float", family="ConvolutionFunctions", name_filter=desc["name"])
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, case, output_root=tmp_path / "bundle", require_fvp_pass=False
    )

    kernel_h, kernel_w, in_c, out_c = desc["filter_shape"]
    block = _PACK_BLOCK[desc["activation_dtype"]]
    packed_elements = -(-out_c // block) * block * kernel_h * kernel_w * in_c
    (header_path,) = sorted((case.directory / "includes").glob("*.h"))
    header = header_path.read_text()
    body = re.search(rf"{re.escape(desc['name'])}_weights\[\]\s*=\s*\{{([^}}]*)\}}", header).group(1)
    generated = np.array(
        [float(value.strip().rstrip("f")) for value in re.sub(r"\(float(?:16_t)?\)", "", body).split(",") if value.strip()],
        dtype=_NUMPY_DTYPE[desc["activation_dtype"]],
    )
    assert generated.size == packed_elements

    weights = next(blob for blob in bundle.blobs if blob.role == "weights")
    assert list(weights.dimensions) == desc["filter_shape"]
    sent, expected = weights.path.read_bytes(), generated.tobytes()
    assert len(sent) == len(expected)
    assert sent == expected
    assert bundle.manifest["serialized_scalar_parameters"]["weight_format_is_packed"] == 1


def test_packed_selection_covers_the_board_failures() -> None:
    # The Apollo510 run reported three cases (ns-cmsis-nn#633). The third,
    # convolve_float_entry_acc16_1d_k5_c4_oc13_packed_f16, is a direct-entry case the bridge now refuses.
    names = {desc["name"] for desc in _partially_filled_packed_descriptors()}
    assert {
        "convolve_float_1d_k5_fold_c12_oc13_packed_f16",
        "convolve_float_direct_fold_c20_k5_oc5_packed_f16",
    } <= names


def _entry_descriptors() -> list[dict]:
    return [d for d in load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors")) if d.get("entry")]


def test_unadapted_entry_cases_are_refused(tmp_path: Path) -> None:
    # Otherwise the wrapper's cycles carry the entry's name.
    desc = next(d for d in _entry_descriptors() if lookup_entry_id(PROJECT_ROOT, d["entry"]) is None)
    case = GeneratedTestCase(
        name=desc["name"], cpu="cortex-m55", family="ConvolutionFunctions", directory=tmp_path, descriptor=desc
    )
    with pytest.raises(UnsupportedGeneratedTestError, match=rf"direct-entry case \({desc['entry']}\) has no firmware adapter"):
        build_case_bundle_from_generated_test(PROJECT_ROOT, case, require_fvp_pass=False)


@pytest.mark.parametrize(
    "desc",
    [d for d in _entry_descriptors() if lookup_entry_id(PROJECT_ROOT, d["entry"]) and d.get("expected_status")],
    ids=lambda d: d["name"],
)
def test_declining_entry_cases_are_refused(tmp_path: Path, desc: dict) -> None:
    case = GeneratedTestCase(
        name=desc["name"], cpu="cortex-m55", family="ConvolutionFunctions", directory=tmp_path, descriptor=desc
    )
    with pytest.raises(UnsupportedGeneratedTestError, match="nothing to time"):
        build_case_bundle_from_generated_test(PROJECT_ROOT, case, require_fvp_pass=False)


def _bridge_entry(tmp_path: Path, family: str, name: str) -> tuple[GeneratedTestCase, dict]:
    desc = next(d for d in _entry_descriptors() if d["name"] == name)
    generate_test(desc, str(tmp_path / "artifacts" / "generated_tests" / "int" / "cortex-m55"), seed=Config.seed)
    (case,) = discover_generated_tests(tmp_path, family=family, name_filter=name)
    bundle = build_case_bundle_from_generated_test(
        PROJECT_ROOT, case, output_root=tmp_path / "bundle", require_fvp_pass=False
    )
    return case, bundle.manifest


@pytest.mark.parametrize(
    ("family", "name", "entry"),
    [
        ("ConvolutionFunctions", "convolve_entry_small_cin3_8x8_k3x3_co16_s8", "arm_convolve_s8_small_cin"),
        ("ConvolutionFunctions", "convolve_entry_3x3_c16_16x16_k3x3_co16_s8", "arm_convolve_s8_3x3_c16_s1"),
        ("ConvolutionFunctions", "convolve_entry_1x1_short_k_c8_2x4_co16_s8", "arm_convolve_1x1_s8_short_k"),
        ("ConvolutionFunctions", "depthwise_conv_entry_3x3_25x5_c64_s8", "arm_depthwise_conv_s8_opt_3x3"),
        ("ConvolutionFunctions", "depthwise_conv_entry_3x3_c64_s1_25x5_s8", "arm_depthwise_conv_s8_opt_3x3_c64_s1"),
        ("ConvolutionFunctions", "depthwise_conv_entry_planar_48x48_c8_s8", "arm_depthwise_conv_s8_opt_planar"),
        ("ConvolutionFunctions", "depthwise_conv_entry_channelwise_25x5_c64_s8", "arm_depthwise_conv_s8_opt_channelwise"),
        ("FullyConnectedFunctions", "fully_connected_entry_packed_k29_c7_b2_bias_s8",
         "arm_fully_connected_per_channel_packed_s8"),
    ],
)
def test_entry_case_times_its_entry(tmp_path: Path, family: str, name: str, entry: str) -> None:
    _case, manifest = _bridge_entry(tmp_path, family, name)
    catalog = json.loads((PROJECT_ROOT / "cmake" / "hardware" / "kernel_catalog.json").read_text())
    names = {item["kernel_id"]: item["canonical_name"] for item in catalog}
    assert names[manifest["kernel_id"]] == entry


def test_packed_fc_scratch_holds_sums_and_stream(tmp_path: Path) -> None:
    name = "fully_connected_entry_packed_k29_c7_b2_bias_s8"
    case, manifest = _bridge_entry(tmp_path, "FullyConnectedFunctions", name)
    (source,) = case.directory.glob("*.c")
    stream = _extract_define_int(source.read_text(), f"{name.upper()}_BUFFER_SIZE_MAX")
    # Seven int32 sums pad to 32 bytes.
    assert manifest["scratch_buffer"]["bytes"] == 32 + stream


@pytest.mark.parametrize(
    ("name", "wrapper"),
    [
        ("convolve_1x1_short_k8_12x12_co16_s8", "arm_convolve_wrapper_s8"),
        ("depthwise_conv_kernel_3x3_s8", "arm_depthwise_conv_wrapper_s8"),
    ],
)
def test_plain_s8_conv_times_wrapper(tmp_path: Path, name: str, wrapper: str) -> None:
    manifest = _bridge(tmp_path, "ConvolutionFunctions", name)
    catalog = json.loads((PROJECT_ROOT / "cmake" / "hardware" / "kernel_catalog.json").read_text())
    names = {entry["kernel_id"]: entry["canonical_name"] for entry in catalog}
    assert names[manifest["kernel_id"]] == wrapper


def test_s4_one_by_n_reserves_im2col() -> None:
    # VALID stride 2 leaves input unused.
    scratch = _calculate_convolve_s4_scratch_bytes(
        {"n": 1, "h": 1, "w": 12, "c": 4},
        {"n": 4, "h": 1, "w": 3, "c": 4},
        {"n": 1, "h": 1, "w": 5, "c": 4},
        pad_h=0, pad_w=0, dilation_h=1, dilation_w=1,
    )
    # arm_convolve_s4_get_buffer_size_mve: 4 * 16 * ceil(12 / 16).
    assert scratch >= 64
    # arm_convolve_s4_get_buffer_size: 2 * rhs_cols * 2.
    assert scratch >= 2 * (3 * 4) * 2
