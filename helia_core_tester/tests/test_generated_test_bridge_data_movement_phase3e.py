from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.perf_stream.generated_test_bridge import (
    UnsupportedGeneratedTestError,
    _find_header_file,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATUS_CASES = [
    ("BroadcastFunctions", "broadcast_to_null_input_s16"),
    ("BroadcastFunctions", "broadcast_to_null_input_s8"),
    ("BroadcastFunctions", "broadcast_to_null_output_s16"),
    ("BroadcastFunctions", "broadcast_to_null_output_s8"),
    ("BroadcastFunctions", "broadcast_to_null_params_s16"),
    ("BroadcastFunctions", "broadcast_to_null_params_s8"),
    ("BroadcastFunctions", "broadcast_to_rank0_s16"),
    ("BroadcastFunctions", "broadcast_to_rank0_s8"),
    ("BroadcastFunctions", "broadcast_to_rank9_s16"),
    ("BroadcastFunctions", "broadcast_to_rank9_s8"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_operand_s16"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_operand_s8"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_output_s16"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_output_s8"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_params_s16"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_params_s8"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_start_indices_s16"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_start_indices_s8"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_update_s16"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_null_update_s8"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_rank0_s16"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_rank0_s8"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_rank9_s16"),
    ("DynamicUpdateSliceFunctions", "dynamic_update_slice_rank9_s8"),
    ("GatherFunctions", "gather_nd_invalid_batch_dims_s8"),
    ("GatherFunctions", "gather_nd_invalid_params_rank_s16"),
    ("GatherFunctions", "gather_nd_negative_index_s16"),
    ("GatherFunctions", "gather_nd_oob_index_s8"),
    ("TransposeFunctions", "transpose_time_batch_invalid_perm_s16"),
    ("TransposeFunctions", "transpose_time_batch_invalid_perm_s8"),
]


def _bridge(tmp_path: Path, family: str, test_name: str) -> dict:
    cases = discover_generated_tests(PROJECT_ROOT, family=family, name_filter=test_name)
    assert cases, f"expected discoverable generated test {test_name}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path)
    return bundle.manifest


def test_select_v2_manifest_preserves_bool_condition_and_meta_blob(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "SelectFunctions", "select_v2_basic_s8")
    assert manifest["operator"] == "SelectV2"
    assert manifest["tensor_dtypes"] == {
        "condition": "BOOL",
        "x": "S8",
        "y": "S8",
        "meta": "S32",
        "output": "S8",
    }
    assert [blob["role"] for blob in manifest["blob_roles"]] == ["input_0", "input_1", "input_2", "meta_0", "expected_output"]
    assert manifest["required_target_capabilities"] == ["arm_select_v2_s8"]


def test_where_manifest_keeps_s64_coordinate_output(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "SelectFunctions", "where_2d_s8")
    assert manifest["operator"] == "Where"
    assert manifest["tensor_dtypes"] == {"condition": "S8", "meta": "S32", "output": "S64"}
    assert manifest["correctness_comparison"]["mode"] == "exact_int"


def test_gather_and_strided_slice_s32_extract_source_backed_metadata(tmp_path: Path) -> None:
    gather_manifest = _bridge(tmp_path, "GatherFunctions", "gather_axis1_s8")
    assert gather_manifest["serialized_scalar_parameters"] == {"output_n": 2, "output_h": 3, "output_w": 1, "output_c": 1}
    assert gather_manifest["required_target_capabilities"] == ["arm_gather_s8"]

    slice_manifest = _bridge(tmp_path, "StridedSliceFunctions", "strided_slice_crop_width_s32")
    assert slice_manifest["tensor_dtypes"] == {"input": "S32", "meta": "S32", "output": "S32"}
    assert slice_manifest["required_target_capabilities"] == ["arm_strided_slice_s32"]


def test_invalid_status_cases_bridge_as_exact_status_assertions(tmp_path: Path) -> None:
    for family, case_name in _STATUS_CASES:
        manifest = _bridge(tmp_path, family, case_name)
        assert manifest["correctness_comparison"] == {
            "mode": "exact_status",
            "expected_status": -1,
            "expected_status_name": "ARM_CMSIS_NN_ARG_ERROR",
        }

    broadcast_manifest = _bridge(tmp_path, "BroadcastFunctions", "broadcast_to_null_input_s8")
    assert broadcast_manifest["serialized_scalar_parameters"]["null_arg_mask"] == 1
    assert broadcast_manifest["expected_output"]["byte_length"] == 0

    dynamic_manifest = _bridge(tmp_path, "DynamicUpdateSliceFunctions", "dynamic_update_slice_null_output_s16")
    assert dynamic_manifest["serialized_scalar_parameters"]["null_arg_mask"] == 16
    assert dynamic_manifest["expected_output"]["byte_length"] == 0

    gather_manifest = _bridge(tmp_path, "GatherFunctions", "gather_nd_invalid_params_rank_s16")
    assert gather_manifest["serialized_scalar_parameters"]["output_n"] == 2
    assert gather_manifest["required_target_capabilities"] == ["arm_gather_nd_s16"]

    transpose_manifest = _bridge(tmp_path, "TransposeFunctions", "transpose_time_batch_invalid_perm_s8")
    assert transpose_manifest["required_target_capabilities"] == ["arm_transpose_s8"]


def test_strided_slice_batch_collapse_bug_is_fixed(tmp_path: Path) -> None:
    # strided_slice_case1_whole_slab_s8 used to trip the "non-empty output_dims
    # but empty expected_output" internal-consistency guard below: its
    # input_shape=[2,3,4,2]/begin=[1,0,0,0] descriptor explicitly targets the
    # "2nd of 2" batch row, but the TFLiteConverter's full-integer-quantization
    # path silently concretizes any batch dimension > 1 down to 1, so golden
    # generation used to run against a collapsed 1-row array where that row no
    # longer existed. helia_core_tester/generation/ops/StridedSliceFunctions/
    # strided_slice.py now detects that batch-collapse and computes golden data
    # directly via numpy against the descriptor's true full-size input instead,
    # so this case is bridgeable again -- see docs/perf-stream-expansion-progress.md
    # Phase 7b.
    cases = discover_generated_tests(PROJECT_ROOT, family="StridedSliceFunctions", name_filter="strided_slice_case1_whole_slab_s8")
    assert cases
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path)
    assert bundle is not None


def test_inconsistent_artifacts_stay_skipped(tmp_path: Path) -> None:
    # The guard itself (raising UnsupportedGeneratedTestError for a generated
    # header that declares a non-empty output_dims but an empty expected_output
    # array) must still fire for artifacts that are genuinely inconsistent --
    # exercise it directly against a synthetic malformed header rather than
    # relying on strided_slice_case1_whole_slab_s8, which is no longer such a
    # case (see test_strided_slice_batch_collapse_bug_is_fixed above).
    cases = discover_generated_tests(PROJECT_ROOT, family="StridedSliceFunctions", name_filter="strided_slice_case1_whole_slab_s8")
    assert cases
    case = cases[0]
    header_path = _find_header_file(case.directory)
    original_text = header_path.read_text()
    tampered_text = re.sub(
        r"(_expected_output\[\]\s*=\s*\{)[^}]*(\})",
        r"\1\2",
        original_text,
        count=1,
    )
    assert tampered_text != original_text
    header_path.write_text(tampered_text)
    try:
        with pytest.raises(UnsupportedGeneratedTestError, match="internally inconsistent"):
            build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path)
    finally:
        header_path.write_text(original_text)
