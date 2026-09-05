"""Regression tests for the perf-stream firmware adapter codegen (single source of truth).

These tests exist to prevent exactly the kind of duplication/drift the codegen was built to
eliminate: `helia_core_tester/perf_stream/adapter_specs.py` is now the only place a bridged
kernel's real firmware C dispatch body is authored, and
`cmake/perf_stream/benchmark_server_session.c`'s generated block is produced from it by
`scripts/generate_perf_stream_adapters.py`. See that module's docstring for the full
rationale (and why the firmware still can't literally reuse the FVP-generated `.c.j2`
per-descriptor test files -- that would reintroduce the "one ELF per case" scalability
problem the streaming architecture exists to avoid).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from helia_core_tester.perf_stream.adapter_specs import (
    FIRMWARE_ADAPTERS,
    GENERATED_BLOCK_BEGIN,
    GENERATED_BLOCK_END,
    generated_test_bridge_scalar_fields,
    render_generated_adapters_block,
)
from helia_core_tester.perf_stream.generated_test_bridge import (
    _build_activation_case,
    _build_basic_math_lut_case,
    _build_basic_math_reduction_case,
    _build_batch_matmul_case,
    _build_batch_norm_case,
    _build_comparison_case,
    _build_convolve_case,
    _build_dequantize_case,
    _build_depthwise_conv_case,
    _build_fully_connected_case,
    _build_nn_activation_float_case,
    _build_pooling_case,
    _build_prelu_case,
    _build_prelu_scalar_case,
    _build_quantize_case,
    _build_reduce_sum_case,
    _build_requantize_case,
    _build_softmax_case,
    _build_squared_difference_case,
    _build_transpose_conv_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The builder cross-checks read a pre-generated artifacts/generated_tests/ tree,
# which is gitignored; the drift guards render straight from adapter_specs.py and
# the bias-bridge regression generates its own case, so those must not inherit
# this skip or they would never run in CI.
requires_generated_artifacts = pytest.mark.skipif(
    not (PROJECT_ROOT / "artifacts" / "generated_tests").is_dir(),
    reason="no generated-test artifacts under artifacts/generated_tests/ "
    "(artifacts/ is gitignored -- run `helia_core_tester generate` first)",
)
SESSION_C_PATH = PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_session.c"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_perf_stream_adapters.py"


def test_generated_block_is_present_exactly_once_in_session_c() -> None:
    text = SESSION_C_PATH.read_text(encoding="utf-8")
    assert text.count(GENERATED_BLOCK_BEGIN) == 1
    assert text.count(GENERATED_BLOCK_END) == 1
    assert text.index(GENERATED_BLOCK_BEGIN) < text.index(GENERATED_BLOCK_END)


def test_committed_session_c_matches_freshly_rendered_adapter_block() -> None:
    """Drift check: if adapter_specs.py is edited without rerunning the generator, the
    committed benchmark_server_session.c's generated block will no longer match a fresh
    render, and this test catches it (mirrors `--check` mode of the generator script).
    """
    text = SESSION_C_PATH.read_text(encoding="utf-8")
    begin = text.index(GENERATED_BLOCK_BEGIN)
    end = text.index(GENERATED_BLOCK_END) + len(GENERATED_BLOCK_END)
    committed_block = text[begin:end]
    fresh_block = render_generated_adapters_block().rstrip("\n")
    assert committed_block == fresh_block, (
        "benchmark_server_session.c's generated adapter block is out of date -- run "
        "`python scripts/generate_perf_stream_adapters.py` after editing adapter_specs.py."
    )


def test_generator_script_check_mode_passes_on_committed_file() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_data_movement_adapters_validate_all_meta_and_output_shapes() -> None:
    block = render_generated_adapters_block()
    assert "meta == NULL" not in block
    assert "meta_blob->dtype != HCT_DTYPE_S32" in block
    assert "meta_blob->alignment < sizeof(int32_t)" in block
    assert "hct_shape_product" not in block
    assert "hct_checked_shape_bytes" in block
    assert "hct_checked_dims_bytes" in block
    assert "const size_t required_meta_ints = is_batch_to_space ? 6u : 7u;" in block


def test_every_registered_adapter_has_a_unique_function_name() -> None:
    names = [adapter.function_name for adapter in FIRMWARE_ADAPTERS]
    assert len(names) == len(set(names)), f"duplicate function_name entries: {names}"


@requires_generated_artifacts
def test_convolve_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    """Cross-check: the Python bridge builder for Convolve must only send scalar keys the
    firmware's run_convolve_once() body (as documented in adapter_specs.py) actually reads --
    catches a renamed/typo'd scalar key at review/test time instead of only at hardware-run
    time (see the arena-capacity bug this same kind of drift caused previously).
    """
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="ConvolutionFunctions", name_filter="convolve_case_03_s8")
    assert cases, "expected at least one discoverable Convolve generated test"
    bundle = _build_convolve_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_convolve_once"))
    # "padding" (the SAME/VALID enum) is a pre-existing vestigial field: the Convolve
    # builder sends it and firmware's parse_scalar() stores it on the session struct, but
    # run_convolve_once() itself never reads session->padding -- it derives everything it
    # needs from the explicit pad_h/pad_w/output_h/w/c fields instead. It's harmless
    # (unused, not a missing/renamed-field bug) so it's carved out here rather than fixed,
    # to avoid touching already hardware-verified builder/firmware behavior in this change.
    manifest_keys.discard("padding")
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_depthwise_conv_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(
        PROJECT_ROOT, family="ConvolutionFunctions", name_filter="depthwise_conv_kernel_support_s8"
    )
    assert cases, "expected a discoverable DepthwiseConv generated test"
    bundle = _build_depthwise_conv_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_depthwise_conv_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_basic_math_reduction_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter="mean_default_s8")
    assert cases, "expected a discoverable Mean generated test"
    bundle = _build_basic_math_reduction_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_basic_math_reduction_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_basic_math_lut_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter="rsqrt_small_tensor_universal_s16")
    assert cases, "expected a discoverable Rsqrt generated test"
    bundle = _build_basic_math_lut_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_basic_math_lut_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_squared_difference_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(
        PROJECT_ROOT, family="BasicMathFunctions", name_filter="squared_difference_batch_broadcast_input2_s16"
    )
    assert cases, "expected a discoverable SquaredDifference generated test"
    bundle = _build_squared_difference_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_elementwise_binary_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_requantize_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="NNSupportFunctions", name_filter="requantize_default_s8")
    assert cases, "expected a discoverable Requantize generated test"
    bundle = _build_requantize_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_requantize_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_comparison_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="ComparisonFunctions", name_filter="comparison_equal_batch_broadcast_s8")
    assert cases, "expected a discoverable Comparison generated test"
    bundle = _build_comparison_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_comparison_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_transpose_conv_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(
        PROJECT_ROOT, family="ConvolutionFunctions", name_filter="transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8"
    )
    assert cases, "expected a discoverable TransposeConv generated test"
    bundle = _build_transpose_conv_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_transpose_conv_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


# The cross-checks below extend coverage past the 8 builders above -- previously the only
# ones with an automated guarantee that a builder's `serialized_scalar_parameters` keys are
# a subset of what the firmware body for that same kernel actually reads (see F007-style
# review finding: an unrecognized scalar name used to silently no-op in parse_scalar()
# rather than error, so a renamed/typo'd/new field added to any of these builders without a
# matching firmware entry would previously go undetected until a hardware run).


@requires_generated_artifacts
def test_fully_connected_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="FullyConnectedFunctions", name_filter="fully_connected_default_s8")
    assert cases, "expected a discoverable FullyConnected generated test"
    bundle = _build_fully_connected_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_fully_connected_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


def test_s8_mve_fully_connected_bridges_the_generated_bias(tmp_path: Path) -> None:
    """The bias an MVE s8 FC case folds into its weight sum still has to reach the adapter.

    The firmware body rebuilds the kernel sum with arm_vector_sum_s8() from the bias blob,
    so a missing or zero bias blob silently reproduces the golden minus the bias term -- an
    error of many LSBs against a tolerance of 1.  See AmbiqAI/helia-core-tester#77.

    Generates its own case so the regression runs without a pre-generated artifacts tree.
    """
    import numpy as np
    import yaml

    from helia_core_tester.generation.io.descriptors import load_all_descriptors
    from helia_core_tester.generation.io.dtypes import resolve_comparison
    from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import OpFullyConnected
    from helia_core_tester.perf_stream.generated_test_bridge import (
        GeneratedTestCase,
        _build_fully_connected_case,
        _extract_array,
        _find_header_file,
    )

    case_name = "fully_connected_mve_case_01_s8"
    descriptor = next(
        desc
        for desc in load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors"))
        if desc.get("name") == case_name
    )

    case_dir = tmp_path / case_name
    case_dir.mkdir()
    op = OpFullyConnected(descriptor, seed=1, target_cpu="cortex-m55")
    model = op.build_keras_model() if op.needs_keras_model() else None
    op.convert_to_tflite(model, str(case_dir / f"{case_name}.tflite"), 1)
    op.generate_c_files(case_dir)
    # The bridge reads (and hashes) the descriptor the generation pipeline drops next to
    # the sources, so mirror that layout rather than the in-memory descriptor alone.
    (case_dir / "descriptor.yaml").write_text(yaml.dump(descriptor, sort_keys=False), encoding="utf-8")

    case = GeneratedTestCase(
        name=case_name,
        cpu="cortex-m55",
        family="FullyConnectedFunctions",
        directory=case_dir,
        descriptor={
            **descriptor,
            "resolved_comparison": resolve_comparison(descriptor, descriptor.get("resolved_tensor_dtypes")),
        },
    )

    header_text = _find_header_file(case.directory).read_text(encoding="utf-8")
    generated_bias = np.array(_extract_array(header_text, f"{case.name}_biases"), dtype=np.int32)

    bundle = _build_fully_connected_case(PROJECT_ROOT, case, output_root=tmp_path / "bundle")
    bias_blobs = [blob for blob in bundle.blobs if blob.role == "bias"]
    assert bias_blobs, "s8 MVE FC case bridged without a bias blob"

    bridged_bias = np.frombuffer(bias_blobs[0].path.read_bytes(), dtype=np.int32)
    assert np.any(bridged_bias != 0), "bridged bias blob is all zero"
    assert np.array_equal(bridged_bias, generated_bias)


def test_s8_fully_connected_firmware_body_gates_the_kernel_sum_on_mve() -> None:
    """Non-MVE builds must get the bias pointer, not a NULL one.

    arm_nn_vec_mat_mult_t{,_per_ch}_s8 ignores the bias under ARM_MATH_MVEI and ignores
    the precomputed kernel sum otherwise, so folding the bias into the kernel sum
    unconditionally would drop the bias-add on every DSP/scalar build.
    """
    body = next(spec for spec in FIRMWARE_ADAPTERS if spec.function_name == "run_fully_connected_once").c_body

    mve_branch = body.split("#if defined(ARM_MATH_MVEI)")
    assert len(mve_branch) == 3, "expected the s8 FC body to gate both the bias pointer and the vector sum"
    guarded, fallback = mve_branch[1].split("#else", 1)
    assert "kernel_bias = NULL" in guarded
    assert "kernel_bias = bias_i32" in fallback.split("#endif", 1)[0]
    assert "arm_vector_sum_s8" in mve_branch[2].split("#endif", 1)[0]
    assert "arm_vector_sum_s8" not in mve_branch[0]
    assert "kernel_bias," in body and body.count("arm_vector_sum_s8") == 1


@requires_generated_artifacts
def test_batch_matmul_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="FullyConnectedFunctions", name_filter="batch_matmul_default_s8")
    assert cases, "expected a discoverable BatchMatMul generated test"
    bundle = _build_batch_matmul_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_batch_matmul_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_quantize_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    # Every shorter "quantize_..." suffix here is also a substring of some
    # "dequantize_..." directory name (discover_generated_tests() filters by substring),
    # which sorts first and would silently pick the wrong (unbridgeable) case -- this one
    # has no dequantize_* counterpart.
    cases = discover_generated_tests(PROJECT_ROOT, family="QuantizationFunctions", name_filter="quantize_relu6_tail_vec31_s8")
    assert cases, "expected a discoverable Quantize generated test"
    bundle = _build_quantize_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_quantize_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_dequantize_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="QuantizationFunctions", name_filter="dequantize_relu_s8")
    assert cases, "expected a discoverable Dequantize generated test"
    bundle = _build_dequantize_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_dequantize_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_softmax_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="SoftmaxFunctions", name_filter="softmax_default_s8")
    assert cases, "expected a discoverable Softmax generated test"
    bundle = _build_softmax_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_softmax_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_pooling_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="PoolingFunctions", name_filter="avg_pool_same_pool1x3_stride2x1_s8")
    assert cases, "expected a discoverable AvgPool generated test"
    bundle = _build_pooling_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_pooling_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_activation_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="ActivationFunctions", name_filter="clamp_default_s8")
    assert cases, "expected a discoverable Clamp generated test"
    bundle = _build_activation_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_activation_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_prelu_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="ActivationFunctions", name_filter="prelu_default_alpha_s8")
    assert cases, "expected a discoverable PReLU generated test"
    bundle = _build_prelu_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_prelu_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_prelu_scalar_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    """PReLUScalar's C body lives in the same run_prelu_once() firmware function as
    general PReLU (see the c_body comment in adapter_specs.py), so it cross-checks against
    the same scalar_fields list."""
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(
        PROJECT_ROOT, family="ActivationFunctions", name_filter="prelu_scalar_input_true_negative_s8"
    )
    assert cases, "expected a discoverable PReLUScalar generated test"
    bundle = _build_prelu_scalar_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_prelu_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_reduce_sum_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(
        PROJECT_ROOT, family="BasicMathFunctions", name_filter="reduce_sum_float_axis_c_f32", suite="float"
    )
    assert cases, "expected a discoverable ReduceSum generated test"
    bundle = _build_reduce_sum_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_reduce_sum_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_batch_norm_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(
        PROJECT_ROOT, family="NNSupportFunctions", name_filter="batch_norm_default_f32", suite="float"
    )
    assert cases, "expected a discoverable BatchNorm generated test"
    bundle = _build_batch_norm_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_batch_norm_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


@requires_generated_artifacts
def test_nn_activation_float_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(
        PROJECT_ROOT, family="ActivationFunctions", name_filter="nn_activation_float_hardswish_f32", suite="float"
    )
    assert cases, "expected a discoverable NNActivationFloat generated test"
    bundle = _build_nn_activation_float_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_nn_activation_float_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields
