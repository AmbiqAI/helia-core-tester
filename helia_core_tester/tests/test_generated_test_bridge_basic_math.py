from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.perf_stream.case_bundle import load_case_bundle
from helia_core_tester.perf_stream.generated_test_bridge import (
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from helia_core_tester.perf_stream.kernel_registry import lookup_kernel_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "artifacts" / "generated_tests").is_dir(),
    reason="no generated-test artifacts under artifacts/generated_tests/ "
    "(artifacts/ is gitignored -- run `helia_core_tester generate` first)",
)


def _bridge(tmp_path: Path, name_filter: str) -> dict[str, object]:
    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter=name_filter)
    assert cases, f"expected a discoverable BasicMathFunctions test matching {name_filter!r}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    return load_case_bundle(bundle.manifest_path).manifest


def test_abs_s16_case_extracts_rescale_scalars(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "abs_rescale_s16")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["needs_rescale"] == 1
    assert manifest["kernel_id"] == lookup_kernel_id(PROJECT_ROOT, family="BasicMathFunctions", operator="Abs", dtype="S16")


def test_argmax_s16_case_uses_s32_output(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "argmax_axis0_s16")
    scalars = manifest["serialized_scalar_parameters"]
    assert manifest["expected_output"]["dtype"] == "S32"
    assert scalars["axis"] == 0
    assert (scalars["output_n"], scalars["output_h"], scalars["output_w"], scalars["output_c"]) == (1, 2, 3, 2)


def test_mean_s8_case_extracts_axis_and_quant_scalars(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "mean_default_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert (scalars["axis_n"], scalars["axis_h"], scalars["axis_w"], scalars["axis_c"]) == (0, 1, 1, 0)
    assert scalars["input_offset"] == 0
    assert scalars["output_offset"] == -2
    assert scalars["out_shift"] == -3
    assert manifest["correctness_comparison"] == {"mode": "tolerant_int", "tolerance": 1}


def test_sqrt_multi_batch_case_streams_lut_blob(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "sqrt_multi_batch_s8")
    blob_roles = {entry["role"]: entry for entry in manifest["blob_roles"]}
    assert blob_roles["weights"]["dtype"] == "S8"
    assert manifest["kernel_id"] == lookup_kernel_id(PROJECT_ROOT, family="BasicMathFunctions", operator="Sqrt", dtype="S8")


def test_rsqrt_universal_case_uses_s32_lut_and_universal_kernel_id(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "rsqrt_small_tensor_universal_s16")
    blob_roles = {entry["role"]: entry for entry in manifest["blob_roles"]}
    assert blob_roles["weights"]["dtype"] == "S32"
    assert manifest["kernel_id"] == lookup_kernel_id(PROJECT_ROOT, family="BasicMathFunctions", operator="RsqrtUniversal", dtype="S16")


def test_squared_difference_batch_case_keeps_output_n(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "squared_difference_batch_broadcast_input2_s16")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["output_n"] == 2
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT, family="BasicMathFunctions", operator="SquaredDifference", dtype="S16"
    )
