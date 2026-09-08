"""Regression tests for issue #69: scratch sizers were only bounded above by a
Python re-derivation of the same C formula (`calculate_buffer_size_max` and
friends), so an under-reporting sizer, a sizer returning the `-1` overflow
sentinel, or a sizer returning a wrong-but-smaller value all passed silently.

These tests exercise the fix at the generation level (template rendering +
descriptor parsing), independent of the FVP-run suite:

  1. A descriptor/context carrying `expected_buffer_size` (scalar or a
     per-ISA `{generic, dsp, mve}` mapping) must render a
     `HELIA_VALIDATE_SCALAR_EQ_INT(..., <expected>, required_buffer_size)`
     call in addition to the existing `> BUFFER_SIZE_MAX` guard -- for every
     one of the 7 templates the issue named.
  2. The mapping form must gate each leg's expected value behind the matching
     `#if defined(ARM_MATH_MVEI) / #elif defined(ARM_MATH_DSP) / #else`
     preprocessor branch, including a leg carrying the `-1` sentinel (the
     mechanism must not special-case away a negative expected value).
  3. A context/descriptor WITHOUT `expected_buffer_size` must render
     unaffected: no `HELIA_VALIDATE_SCALAR_EQ_INT` call appears, only the
     pre-existing `> BUFFER_SIZE_MAX` guard -- older/unrelated descriptors
     must not be affected by this change.
  4. The checked-in descriptor cases actually modified for this issue parse
     `expected_buffer_size` (and, for transpose_conv,
     `expected_reverse_buffer_size`) to the documented values, so a future
     edit to those YAML files can't silently drop the field.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _templates_root() -> Path:
    return _repo_root() / "assets" / "templates"


def _descriptors_root() -> Path:
    return _repo_root() / "assets" / "descriptors"


def _render(template_name: str, context: dict[str, object]) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_templates_root())),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    render_context = TemplateContextBuilder.build_validation_context(template_name, context)
    return env.get_template(template_name).render(**render_context)


def _load_case(desc_path: Path, name: str) -> dict:
    for desc in load_descriptor(str(desc_path)):
        if desc.get("name") == name:
            return desc
    raise AssertionError(f"descriptor {name!r} not found in {desc_path}")


_MVEI_BRANCH = "#if defined(ARM_MATH_MVEI)"
_DSP_BRANCH = "#elif defined(ARM_MATH_DSP)"


def _assert_scalar_assertion(rendered: str, label: str, expected: int) -> None:
    assert f'HELIA_VALIDATE_SCALAR_EQ_INT("{label}", "buffer size", {expected}, required_buffer_size)' in rendered


def _assert_no_assertion(rendered: str) -> None:
    assert "HELIA_VALIDATE_SCALAR_EQ_INT" not in rendered
    assert "required_buffer_size > " in rendered  # the pre-existing guard survives


def _assert_isa_mapping(rendered: str, label: str, generic: int, dsp: int, mve: int) -> None:
    assert _MVEI_BRANCH in rendered
    assert _DSP_BRANCH in rendered
    assert f'HELIA_VALIDATE_SCALAR_EQ_INT("{label}", "buffer size", {mve}, required_buffer_size)' in rendered
    assert f'HELIA_VALIDATE_SCALAR_EQ_INT("{label}", "buffer size", {dsp}, required_buffer_size)' in rendered
    assert f'HELIA_VALIDATE_SCALAR_EQ_INT("{label}", "buffer size", {generic}, required_buffer_size)' in rendered


# ---------------------------------------------------------------------------
# convolve
# ---------------------------------------------------------------------------

def _convolve_base() -> dict:
    return {
        "name": "conv_probe",
        "buffer_size_max": 4096,
        "kernel_fn": "arm_convolve_wrapper_s16",
        "kernel_get_buffer_size_fn": "arm_convolve_wrapper_s16_get_buffer_size",
        "float_kernel": False,
        "input_dtype": "int16_t",
        "output_dtype": "int16_t",
        "bias_dtype": "int64_t",
        "has_biases": False,
        "filter_dims": {"n": 4, "h": 3, "w": 3, "c": 2},
        "output_dims": {"n": 1, "h": 4, "w": 4, "c": 4},
    }


def test_convolve_scalar_expected_buffer_size_renders_exact_assertion() -> None:
    rendered = _render(
        "ConvolutionFunctions/convolve/convolve.c.j2",
        {**_convolve_base(), "expected_buffer_size": 32},
    )
    _assert_scalar_assertion(rendered, "Convolve", 32)


def test_convolve_isa_mapping_gates_each_leg_including_negative_sentinel() -> None:
    rendered = _render(
        "ConvolutionFunctions/convolve/convolve.c.j2",
        {**_convolve_base(), "expected_buffer_size": {"generic": 32, "dsp": 32, "mve": -1}},
    )
    _assert_isa_mapping(rendered, "Convolve", generic=32, dsp=32, mve=-1)


def test_convolve_without_expected_buffer_size_is_unaffected() -> None:
    rendered = _render("ConvolutionFunctions/convolve/convolve.c.j2", {**_convolve_base(), "expected_buffer_size": None})
    _assert_no_assertion(rendered)


# ---------------------------------------------------------------------------
# depthwise_conv
# ---------------------------------------------------------------------------

def _depthwise_conv_base() -> dict:
    return {
        "name": "dw_probe",
        "buffer_size_max": 4096,
        "kernel_fn": "arm_depthwise_conv_wrapper_s4",
        "kernel_get_buffer_size_fn": "arm_depthwise_conv_wrapper_s4_get_buffer_size",
        "has_weight_sum": False,
        "float_kernel": False,
        "force_no_scratch": False,
        "input_dtype": "int8_t",
        "output_dtype": "int8_t",
        "has_biases": False,
        "output_dims": {"n": 1, "h": 3, "w": 3, "c": 9},
    }


def test_depthwise_conv_scalar_expected_buffer_size_renders_exact_assertion() -> None:
    rendered = _render(
        "ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2",
        {**_depthwise_conv_base(), "expected_buffer_size": 0},
    )
    _assert_scalar_assertion(rendered, "Depthwise Conv", 0)


def test_depthwise_conv_isa_mapping_gates_each_leg() -> None:
    rendered = _render(
        "ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2",
        {**_depthwise_conv_base(), "expected_buffer_size": {"generic": 0, "dsp": 0, "mve": 4464}},
    )
    _assert_isa_mapping(rendered, "Depthwise Conv", generic=0, dsp=0, mve=4464)


def test_depthwise_conv_without_expected_buffer_size_is_unaffected() -> None:
    rendered = _render(
        "ConvolutionFunctions/depthwise_conv/depthwise_conv.c.j2",
        {**_depthwise_conv_base(), "expected_buffer_size": None},
    )
    _assert_no_assertion(rendered)


# ---------------------------------------------------------------------------
# transpose_conv
# ---------------------------------------------------------------------------

def _transpose_conv_base() -> dict:
    return {
        "name": "tconv_probe",
        "buffer_size_max": 4096,
        "reverse_conv_ctx_size": 4096,
        "kernel_fn": "arm_transpose_conv_wrapper_s8",
        "kernel_get_buffer_size_fn": "arm_transpose_conv_s8_get_buffer_size",
        "kernel_get_reverse_buffer_size_fn": "arm_transpose_conv_s8_get_reverse_conv_buffer_size",
        "has_weight_sum": False,
        "float_kernel": False,
        "input_dtype": "int8_t",
        "output_dtype": "int8_t",
        "bias_dtype": "int32_t",
        "has_biases": False,
        "output_dims": {"n": 1, "h": 5, "w": 5, "c": 4},
    }


def test_transpose_conv_scalar_expected_buffer_sizes_render_exact_assertions() -> None:
    rendered = _render(
        "ConvolutionFunctions/transpose_conv/transpose_conv.c.j2",
        {**_transpose_conv_base(), "expected_buffer_size": 1024, "expected_reverse_buffer_size": 0},
    )
    _assert_scalar_assertion(rendered, "TransposeConv", 1024)
    assert (
        'HELIA_VALIDATE_SCALAR_EQ_INT("TransposeConv", "reverse buffer size", 0, '
        "reverse_required_buffer_size)" in rendered
    )


def test_transpose_conv_isa_mapping_gates_the_ctx_leg() -> None:
    rendered = _render(
        "ConvolutionFunctions/transpose_conv/transpose_conv.c.j2",
        {
            **_transpose_conv_base(),
            "expected_buffer_size": {"generic": 1024, "dsp": 1024, "mve": -1},
            "expected_reverse_buffer_size": 4096,
        },
    )
    _assert_isa_mapping(rendered, "TransposeConv", generic=1024, dsp=1024, mve=-1)


def test_transpose_conv_without_expected_sizes_is_unaffected() -> None:
    rendered = _render(
        "ConvolutionFunctions/transpose_conv/transpose_conv.c.j2",
        {**_transpose_conv_base(), "expected_buffer_size": None, "expected_reverse_buffer_size": None},
    )
    assert "HELIA_VALIDATE_SCALAR_EQ_INT" not in rendered
    assert "required_buffer_size > " in rendered
    assert "reverse_required_buffer_size > " in rendered


# ---------------------------------------------------------------------------
# fully_connected
# ---------------------------------------------------------------------------

def _fully_connected_base() -> dict:
    return {
        "name": "fc_probe",
        "buffer_size_max": 4096,
        "kernel_fn": "arm_fully_connected_wrapper_s8",
        "kernel_get_buffer_size_fn": "arm_fully_connected_s8_get_buffer_size",
        "float_kernel": False,
        "has_weight_sum": False,
        "quant_params": {"per_channel": False},
        "input_dtype": "int8_t",
        "output_dtype": "int8_t",
        "has_biases": False,
        "filter_dims": {"n": 20, "h": 1, "w": 1, "c": 6},
        "output_dims": {"n": 1, "h": 1, "w": 1, "c": 6},
    }


def test_fully_connected_scalar_expected_buffer_size_renders_exact_assertion() -> None:
    rendered = _render(
        "FullyConnectedFunctions/fully_connected/fully_connected.c.j2",
        {**_fully_connected_base(), "expected_buffer_size": 24},
    )
    _assert_scalar_assertion(rendered, "Fully Connected", 24)


def test_fully_connected_isa_mapping_gates_each_leg() -> None:
    rendered = _render(
        "FullyConnectedFunctions/fully_connected/fully_connected.c.j2",
        {**_fully_connected_base(), "expected_buffer_size": {"generic": 0, "dsp": 0, "mve": 24}},
    )
    _assert_isa_mapping(rendered, "Fully Connected", generic=0, dsp=0, mve=24)


def test_fully_connected_without_expected_buffer_size_is_unaffected() -> None:
    rendered = _render(
        "FullyConnectedFunctions/fully_connected/fully_connected.c.j2",
        {**_fully_connected_base(), "expected_buffer_size": None},
    )
    _assert_no_assertion(rendered)


def test_fully_connected_s4_hardcoded_zero_path_never_needs_the_assertion() -> None:
    """arm_fully_connected_s4 never calls a sizer at all (required_buffer_size is
    hardcoded 0 in the template), so no checked-in s4 descriptor carries
    expected_buffer_size -- confirm the template still renders cleanly for that
    kernel_fn even if a context accidentally supplied one (defensive only)."""
    base = {**_fully_connected_base(), "kernel_fn": "arm_fully_connected_s4"}
    rendered = _render("FullyConnectedFunctions/fully_connected/fully_connected.c.j2", base)
    assert "int32_t required_buffer_size = 0;" in rendered


# ---------------------------------------------------------------------------
# batch_matmul
# ---------------------------------------------------------------------------

def _batch_matmul_base() -> dict:
    return {
        "name": "bmm_probe",
        "buffer_size_max": 4096,
        "kernel_fn": "arm_batch_matmul_s8",
        "kernel_get_buffer_size_fn": "arm_fully_connected_s8_get_buffer_size",
        "float_kernel": False,
        "input_dtype": "int8_t",
        "input_rhs_dtype": "int8_t",
        "output_dtype": "int8_t",
        "input_rhs_dims": {"n": 1, "h": 1, "w": 2, "c": 3},
        "output_dims": {"n": 1, "h": 1, "w": 1, "c": 2},
    }


def test_batch_matmul_scalar_expected_buffer_size_renders_exact_assertion() -> None:
    rendered = _render(
        "FullyConnectedFunctions/batch_matmul/batch_matmul.c.j2",
        {**_batch_matmul_base(), "expected_buffer_size": 8},
    )
    _assert_scalar_assertion(rendered, "Batch Matmul", 8)


def test_batch_matmul_isa_mapping_gates_each_leg() -> None:
    rendered = _render(
        "FullyConnectedFunctions/batch_matmul/batch_matmul.c.j2",
        {**_batch_matmul_base(), "expected_buffer_size": {"generic": 0, "dsp": 0, "mve": 8}},
    )
    _assert_isa_mapping(rendered, "Batch Matmul", generic=0, dsp=0, mve=8)


def test_batch_matmul_without_expected_buffer_size_is_unaffected() -> None:
    rendered = _render(
        "FullyConnectedFunctions/batch_matmul/batch_matmul.c.j2",
        {**_batch_matmul_base(), "expected_buffer_size": None},
    )
    _assert_no_assertion(rendered)


# ---------------------------------------------------------------------------
# avg_pool / max_pool
# ---------------------------------------------------------------------------

def _pool_base(kernel_get_buffer_size_fn: str | None) -> dict:
    return {
        "name": "pool_probe",
        "buffer_size_max": 128,
        "kernel_fn": "arm_avgpool_s8",
        "kernel_get_buffer_size_fn": kernel_get_buffer_size_fn,
        "requires_kernel_prototype": False,
        "input_dtype": "int8_t",
        "output_dtype": "int8_t",
        "input_dims": {"n": 1, "h": 5, "w": 5, "c": 20},
        "output_dims": {"n": 1, "h": 3, "w": 3, "c": 20},
    }


def test_avg_pool_scalar_expected_buffer_size_renders_exact_assertion() -> None:
    rendered = _render(
        "PoolingFunctions/avg_pool/avg_pool.c.j2",
        {**_pool_base("arm_avgpool_s8_get_buffer_size"), "expected_buffer_size": 80},
    )
    _assert_scalar_assertion(rendered, "Avg Pool", 80)


def test_avg_pool_isa_mapping_gates_each_leg_matching_the_documented_zero_regime() -> None:
    """MVE and generic are both the documented degenerate-0 case for avgpool; only
    DSP needs ch_src * 4 -- see arm_avgpool_s8_get_buffer_size_{mve,dsp}()."""
    rendered = _render(
        "PoolingFunctions/avg_pool/avg_pool.c.j2",
        {**_pool_base("arm_avgpool_s8_get_buffer_size"), "expected_buffer_size": {"generic": 0, "dsp": 80, "mve": 0}},
    )
    _assert_isa_mapping(rendered, "Avg Pool", generic=0, dsp=80, mve=0)


def test_avg_pool_without_expected_buffer_size_is_unaffected() -> None:
    rendered = _render(
        "PoolingFunctions/avg_pool/avg_pool.c.j2",
        {**_pool_base("arm_avgpool_s8_get_buffer_size"), "expected_buffer_size": None},
    )
    _assert_no_assertion(rendered)


def test_max_pool_never_calls_a_sizer_so_the_assertion_branch_is_unreachable() -> None:
    """No MaxPool kernel_info entry ever sets kernel_get_buffer_size_fn (CMSIS-NN has
    no arm_max_pool*_get_buffer_size function) -- confirm the real (falsy)
    kernel_get_buffer_size_fn path renders the bufferless branch with no
    assertion, even if expected_buffer_size were somehow supplied."""
    rendered = _render(
        "PoolingFunctions/max_pool/max_pool.c.j2",
        {**_pool_base(None), "kernel_fn": "arm_max_pool_s8", "expected_buffer_size": 0},
    )
    assert "HELIA_VALIDATE_SCALAR_EQ_INT" not in rendered
    assert "Max pooling doesn't need a buffer" in rendered


def test_max_pool_assertion_mechanism_still_renders_if_ever_wired_up() -> None:
    """Parity check: if kernel_get_buffer_size_fn were ever set for MaxPool, the
    same assertion machinery as avg_pool.c.j2 would fire correctly (dead code
    today, kept only for parity -- see the ConvolutionFunctions/../*.c.j2 comment)."""
    rendered = _render(
        "PoolingFunctions/max_pool/max_pool.c.j2",
        {**_pool_base("arm_avgpool_s8_get_buffer_size"), "kernel_fn": "arm_max_pool_s8", "expected_buffer_size": 80},
    )
    _assert_scalar_assertion(rendered, "Max Pool", 80)


# ---------------------------------------------------------------------------
# Checked-in descriptor cases: the YAML actually parses to the documented
# values (protects against a future edit silently dropping/mistyping the
# field).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaml_file,case_name,expected",
    [
        ("ConvolutionFunctions/convolve.yaml", "convolve_default_s8", {"generic": 32, "dsp": 32, "mve": 64}),
        ("ConvolutionFunctions/convolve.yaml", "convolve_kernel1x1_s8", 0),
        ("ConvolutionFunctions/convolve_float.yaml", "convolve_float_default_f32", 864),
        (
            "ConvolutionFunctions/depthwise_conv.yaml",
            "depthwise_conv_case_02_s8",
            0,
        ),
        (
            "ConvolutionFunctions/depthwise_conv.yaml",
            "depthwise_conv_kernel_3x3_s8",
            {"generic": 0, "dsp": 0, "mve": 4464},
        ),
        ("ConvolutionFunctions/depthwise_conv_float.yaml", "depthwise_conv_float_default_f32", 432),
        (
            "FullyConnectedFunctions/fully_connected.yaml",
            "fully_connected_default_s8",
            {"generic": 0, "dsp": 0, "mve": 24},
        ),
        ("FullyConnectedFunctions/fully_connected.yaml", "fully_connected_default_s16", 44),
        ("FullyConnectedFunctions/fully_connected.yaml", "fully_connected_per_tensor_s16", 0),
        ("FullyConnectedFunctions/fully_connected_float.yaml", "fully_connected_float_default_f32", 0),
        (
            "FullyConnectedFunctions/batch_matmul.yaml",
            "batch_matmul_default_s8",
            {"generic": 0, "dsp": 0, "mve": 8},
        ),
        ("FullyConnectedFunctions/batch_matmul.yaml", "batch_matmul_batched_s16", 0),
        ("FullyConnectedFunctions/batch_matmul_float.yaml", "batch_matmul_float_default_f32", 0),
        (
            "PoolingFunctions/avg_pool.yaml",
            "avg_pool_same_pool5x6_stride5x9_s8",
            {"generic": 0, "dsp": 80, "mve": 0},
        ),
        (
            "PoolingFunctions/avg_pool.yaml",
            "avg_pool_same_pool3x2_stride1x2_s16",
            {"generic": 0, "dsp": 68, "mve": 0},
        ),
    ],
)
def test_checked_in_descriptor_expected_buffer_size(yaml_file: str, case_name: str, expected: object) -> None:
    desc = _load_case(_descriptors_root() / yaml_file, case_name)
    assert desc.get("expected_buffer_size") == expected


@pytest.mark.parametrize(
    "yaml_file,case_name,expected_ctx,expected_reverse",
    [
        (
            "ConvolutionFunctions/transpose_conv.yaml",
            "transpose_conv_same_kernel6x6_stride2x2_bias_s8",
            4752,
            0,
        ),
        (
            "ConvolutionFunctions/transpose_conv.yaml",
            "transpose_conv_reverse_same_kernel2x2_stride1x1_bias_s8",
            1024,
            4096,
        ),
        (
            "ConvolutionFunctions/transpose_conv_float.yaml",
            "transpose_conv_float_default_f32",
            0,
            0,
        ),
    ],
)
def test_checked_in_transpose_conv_descriptor_expected_buffer_sizes(
    yaml_file: str, case_name: str, expected_ctx: int, expected_reverse: int
) -> None:
    desc = _load_case(_descriptors_root() / yaml_file, case_name)
    assert desc.get("expected_buffer_size") == expected_ctx
    assert desc.get("expected_reverse_buffer_size") == expected_reverse


def test_descriptor_cases_without_expected_buffer_size_are_unaffected() -> None:
    """Backward compatibility: an ordinary, untouched descriptor case has no
    expected_buffer_size key at all, so TemplateContextBuilder callers see
    None and the templates skip the new assertion entirely (see the
    `*_without_expected_buffer_size_is_unaffected` render tests above)."""
    desc = _load_case(
        _descriptors_root() / "ConvolutionFunctions" / "convolve.yaml",
        "convolve_stride2pad1_s8",
    )
    assert desc.get("expected_buffer_size") is None
