from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.perf_stream.generated_test_bridge import (
    UnsupportedGeneratedTestError,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    assert manifest["correctness_comparison"] == {"mode": "exact_int"}


def test_gather_and_strided_slice_s32_extract_source_backed_metadata(tmp_path: Path) -> None:
    gather_manifest = _bridge(tmp_path, "GatherFunctions", "gather_axis1_s8")
    assert gather_manifest["serialized_scalar_parameters"] == {"output_n": 2, "output_h": 3, "output_w": 1, "output_c": 1}
    assert gather_manifest["required_target_capabilities"] == ["arm_gather_s8"]

    slice_manifest = _bridge(tmp_path, "StridedSliceFunctions", "strided_slice_crop_width_s32")
    assert slice_manifest["tensor_dtypes"] == {"input": "S32", "meta": "S32", "output": "S32"}
    assert slice_manifest["required_target_capabilities"] == ["arm_strided_slice_s32"]


def test_invalid_status_and_inconsistent_artifacts_are_skipped(tmp_path: Path) -> None:
    invalid_cases = discover_generated_tests(PROJECT_ROOT, family="BroadcastFunctions", name_filter="broadcast_to_null_input_s8")
    assert invalid_cases
    with pytest.raises(UnsupportedGeneratedTestError, match="expects ARM_CMSIS_NN_ARG_ERROR"):
        build_case_bundle_from_generated_test(PROJECT_ROOT, invalid_cases[0], output_root=tmp_path)

    inconsistent_cases = discover_generated_tests(PROJECT_ROOT, family="StridedSliceFunctions", name_filter="strided_slice_case1_whole_slab_s8")
    assert inconsistent_cases
    with pytest.raises(UnsupportedGeneratedTestError, match="internally inconsistent"):
        build_case_bundle_from_generated_test(PROJECT_ROOT, inconsistent_cases[0], output_root=tmp_path)
