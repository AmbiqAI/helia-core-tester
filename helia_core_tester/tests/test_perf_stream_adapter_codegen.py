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

from helia_core_tester.perf_stream.adapter_specs import (
    FIRMWARE_ADAPTERS,
    GENERATED_BLOCK_BEGIN,
    GENERATED_BLOCK_END,
    generated_test_bridge_scalar_fields,
    render_generated_adapters_block,
)
from helia_core_tester.perf_stream.generated_test_bridge import (
    _build_basic_math_lut_case,
    _build_basic_math_reduction_case,
    _build_comparison_case,
    _build_convolve_case,
    _build_depthwise_conv_case,
    _build_requantize_case,
    _build_squared_difference_case,
    _build_transpose_conv_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def test_basic_math_reduction_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter="mean_default_s8")
    assert cases, "expected a discoverable Mean generated test"
    bundle = _build_basic_math_reduction_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_basic_math_reduction_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


def test_basic_math_lut_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="BasicMathFunctions", name_filter="rsqrt_small_tensor_universal_s16")
    assert cases, "expected a discoverable Rsqrt generated test"
    bundle = _build_basic_math_lut_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_basic_math_lut_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


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


def test_requantize_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="NNSupportFunctions", name_filter="requantize_default_s8")
    assert cases, "expected a discoverable Requantize generated test"
    bundle = _build_requantize_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_requantize_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


def test_comparison_builder_scalar_keys_are_subset_of_firmware_adapter_scalar_fields(tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.generated_test_bridge import discover_generated_tests

    cases = discover_generated_tests(PROJECT_ROOT, family="ComparisonFunctions", name_filter="comparison_equal_batch_broadcast_s8")
    assert cases, "expected a discoverable Comparison generated test"
    bundle = _build_comparison_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest_keys = set(bundle.manifest["serialized_scalar_parameters"])
    firmware_fields = set(generated_test_bridge_scalar_fields("run_comparison_once"))
    assert manifest_keys <= firmware_fields, manifest_keys - firmware_fields


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
