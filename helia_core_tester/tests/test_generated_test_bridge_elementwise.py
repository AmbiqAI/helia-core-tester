"""Regression tests for the BasicMathFunctions Add/Sub perf-stream hardware bridge.

Unlike ConvolutionFunctions' generated header (which has a named `cmsis_nn_conv_params`
scalar-params struct), the Add/Sub generated `.c` file inlines all quant scalars as
positional arguments to `arm_add_s8`/`arm_sub_s8` (with trailing `// name` comments), so
`_build_elementwise_binary_case()` extracts them via `_extract_call_args()` -- a naive
top-level comma split over the call-site text. These tests pin the extracted values against
known-good literals read directly from the generated `.c` files, and additionally exercise
a non-batch broadcast case (`add_broadcast_height_s8`, input1 h=1 broadcasting against
input2), since `arm_add_s8`/`arm_sub_s8` support broadcasting and the extraction path does
not special-case it (only batch-dimension broadcast, n>1, is rejected -- see
`_build_elementwise_binary_case`'s explicit `UnsupportedGeneratedTestError` guard).

This test does not touch real hardware; it bridges real generated-test artifacts already
checked into `artifacts/generated_tests/` and asserts on the resulting CaseBundle manifest.
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
    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter=name_filter)
    assert cases, f"expected a discoverable BasicMathFunctions test matching {name_filter!r}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    loaded = load_case_bundle(bundle.manifest_path)
    return loaded.manifest


def test_add_default_case_extracts_true_scalars_and_kernel_id(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "add_default_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["input1_offset"] == 0
    assert scalars["input1_mult"] == 1073741824
    assert scalars["input1_shift"] == 0
    assert scalars["input2_offset"] == 0
    assert scalars["input2_mult"] == 1073741824
    assert scalars["input2_shift"] == 0
    assert scalars["left_shift"] == 20
    assert scalars["output_offset"] == 0
    assert scalars["out_mult"] == 1073741824
    assert scalars["out_shift"] == -18
    assert scalars["activation_min"] == -128
    assert scalars["activation_max"] == 127
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT, family="BasicMathFunctions", operator="Add", dtype="S8"
    )


def test_sub_dual_inputs_case_extracts_true_scalars_and_kernel_id(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "sub_dual_inputs_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["left_shift"] == 20
    assert scalars["out_shift"] == -18
    assert scalars["activation_min"] == -128
    assert scalars["activation_max"] == 127
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT, family="BasicMathFunctions", operator="Sub", dtype="S8"
    )


def test_broadcast_height_case_bridges_with_mismatched_input_dims(tmp_path: Path) -> None:
    # Non-batch broadcast (input1 height differs from input2/output height): CMSIS-NN's
    # dims-based arm_add_s8 handles this internally, so the bridge should not reject it.
    manifest = _bridge(tmp_path, "add_broadcast_height_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["left_shift"] == 20
    assert (scalars["output_h"], scalars["output_w"], scalars["output_c"]) == (2, 3, 2)


def test_batch_broadcast_case_bridges_and_serializes_output_n(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "add_broadcast_batch_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["output_n"] == 2
    assert (scalars["output_h"], scalars["output_w"], scalars["output_c"]) == (2, 2, 3)


def test_mul_default_case_extracts_true_scalars_and_kernel_id(tmp_path: Path) -> None:
    # arm_mul_s8 has a shorter signature than Add/Sub -- no per-input mult/shift, no
    # left_shift -- so it is bridged via its own _build_mul_case() positional mapping.
    manifest = _bridge(tmp_path, "mul_default_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["input1_offset"] == 0
    assert scalars["input2_offset"] == 0
    assert scalars["output_offset"] == 0
    assert scalars["out_mult"] == 1073741824
    assert scalars["out_shift"] == -2
    assert scalars["activation_min"] == -128
    assert scalars["activation_max"] == 127
    assert "left_shift" not in scalars
    assert "input1_mult" not in scalars
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT, family="BasicMathFunctions", operator="Mul", dtype="S8"
    )


def test_mul_channel_broadcast_case_bridges(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "mul_channel_broadcast_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert (scalars["output_h"], scalars["output_w"], scalars["output_c"]) == (2, 2, 3)


def test_maximum_default_case_has_no_scalar_params(tmp_path: Path) -> None:
    # arm_maximum_s8/arm_minimum_s8 take no quantization scalars at all -- only dims/ctx.
    manifest = _bridge(tmp_path, "maximum_default_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert set(scalars) == {"output_h", "output_w", "output_c"}
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT, family="BasicMathFunctions", operator="Maximum", dtype="S8"
    )


def test_minimum_dual_case_bridges(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "minimum_dual_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert set(scalars) == {"output_h", "output_w", "output_c"}
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT, family="BasicMathFunctions", operator="Minimum", dtype="S8"
    )


def test_maximum_batch_broadcast_case_bridges(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "maximum_batch_broadcast_s8")
    assert manifest["serialized_scalar_parameters"]["output_n"] == 2
