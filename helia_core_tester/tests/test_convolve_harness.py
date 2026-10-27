"""Convolve through the generic harness: calls and scratch queries bound from its pool."""

from __future__ import annotations

import re

import pytest

from helia_core_tester.contract.bind import ContractBindError
from helia_core_tester.contract.ir import STATUS_PRESENT, ContractError, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.contract.render import load_current_contracts
from helia_core_tester.tests.harness_render import convolve_context, render_convolve


@pytest.fixture
def contracts() -> ContractSet:
    loaded = load_current_contracts()
    assert loaded.present, "the conftest fallback always provides the bound operators' contract"
    return loaded


def _strip(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.S)


def _calls(source: str, fn: str) -> list[list[str]]:
    return [[a.strip() for a in c.split(",")] for c in re.findall(rf"return {re.escape(fn)}\((.*?)\);", _strip(source), re.S)]


def _sizer(source: str) -> tuple[str, list[str]] | None:
    m = re.search(r"required_buffer_size = (arm_\w+)\((.*?)\);", _strip(source), re.S)
    return (m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]) if m else None


WRAP4 = ["&conv_case_conv_params", "&conv_case_input_dims", "&conv_case_filter_dims", "&conv_case_output_dims"]


@pytest.mark.parametrize("kernel_fn, flags, struct, expected", [
    ("arm_convolve_wrapper_s8", {}, False,
     ["&conv_case_ctx", "&conv_case_weight_sum_ctx", "&conv_case_conv_params", "&conv_case_quant_params",
      "&conv_case_input_dims", "input", "&conv_case_filter_dims", "conv_case_weights", "&conv_case_bias_dims",
      "conv_case_biases", "&conv_case_output_dims", "output"]),
    ("arm_convolve_wrapper_s4", {}, False,
     ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_quant_params", "&conv_case_input_dims", "input",
      "&conv_case_filter_dims", "conv_case_weights", "&conv_case_bias_dims", "conv_case_biases",
      "&conv_case_output_dims", "output"]),
    ("arm_convolve_wrapper_s16", {"input_dtype": "int16_t", "output_dtype": "int16_t", "bias_dtype": "int64_t"}, True,
     ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_quant_params", "&conv_case_input_dims", "input",
      "&conv_case_filter_dims", "conv_case_weights", "&conv_case_bias_dims", "&conv_case_bias_data",
      "&conv_case_output_dims", "output"]),
    ("arm_convolve_wrapper_f32", {"float_kernel": True}, False,
     ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_input_dims", "input", "&conv_case_filter_dims",
      "conv_case_weights", "&conv_case_bias_dims", "conv_case_biases", "&conv_case_output_dims", "output"]),
    ("arm_convolve_1x1_f16", {"float_kernel": True, "input_dtype": "float16_t", "output_dtype": "float16_t"}, False,
     ["&conv_case_ctx", "&conv_case_conv_params", "&conv_case_input_dims", "input", "&conv_case_filter_dims",
      "conv_case_weights", "&conv_case_bias_dims", "conv_case_biases", "&conv_case_output_dims", "output",
      "ARM_NN_LAYOUT_NHWC"]),
])
def test_table_kernels_bind_from_the_pool(contracts, kernel_fn, flags, struct, expected) -> None:
    header, source = render_convolve(convolve_context(kernel_fn, **flags), bias_is_struct=struct)
    calls = _calls(source, kernel_fn)
    assert len(calls) == 2, "the correctness path and the benchmark path both call the kernel"
    assert calls[0] == expected
    assert calls[1] == ["conv_case_input" if a == "input" else "conv_case_output" if a == "output" else a for a in expected]
    assert source.count(f"__typeof__({kernel_fn})") == 1
    assert ("arm_convolve_weight_sum(" in source) == ("&conv_case_weight_sum_ctx" in expected)
    assert ("static const cmsis_nn_bias_data conv_case_bias_data" in source) == struct


def test_no_bias_passes_null_for_both_bias_arguments(contracts) -> None:
    _, source = render_convolve(convolve_context("arm_convolve_wrapper_s8", has_biases=False))
    assert _calls(source, "arm_convolve_wrapper_s8")[0][8:10] == ["NULL", "NULL"]
    assert "static const int32_t* conv_case_biases = NULL;" in render_convolve(
        convolve_context("arm_convolve_wrapper_s8", has_biases=False))[0]


@pytest.mark.parametrize("kernel_fn", ["arm_convolve_s8", "arm_convolve_s8_small_cin", "arm_convolve_s8_3x3_c16_s1"])
def test_direct_s8_entries_get_weight_sums_and_a_null_upscale(contracts, kernel_fn: str) -> None:
    _, source = render_convolve(convolve_context(kernel_fn, sizer="arm_convolve_s8_get_buffer_size"))
    assert _calls(source, kernel_fn)[0][:2] == ["&conv_case_ctx", "&conv_case_weight_sum_ctx"]
    assert _calls(source, kernel_fn)[0][10] == "NULL"
    assert _sizer(source) == ("arm_convolve_s8_get_buffer_size", ["&conv_case_input_dims", "&conv_case_filter_dims"])


@pytest.mark.parametrize("kernel_fn, sizer, flags, expected", [
    ("arm_convolve_wrapper_s8", "arm_convolve_wrapper_s8_get_buffer_size", {}, WRAP4),
    ("arm_convolve_f32", "arm_convolve_f32_get_buffer_size", {"float_kernel": True}, WRAP4 + ["ARM_NN_LAYOUT_NHWC"]),
    ("arm_convolve_s16", "arm_convolve_s16_get_buffer_size", {}, ["&conv_case_input_dims", "&conv_case_filter_dims"]),
    ("arm_convolve_1x1_s8_fast", "arm_convolve_1x1_s8_fast_get_buffer_size", {}, ["&conv_case_input_dims"]),
])
def test_scratch_queries_bind_from_their_prototypes(contracts, kernel_fn, sizer, flags, expected) -> None:
    _, source = render_convolve(convolve_context(kernel_fn, sizer=sizer, **flags),
                                bias_is_struct=kernel_fn == "arm_convolve_s16")
    assert _sizer(source) == (sizer, expected)
    assert f'HELIA_VALIDATE_SIZER("{sizer}", required_buffer_size);' in source


def test_entry_scratch_renders_no_sizer_call(contracts) -> None:
    _, source = render_convolve(convolve_context("arm_convolve_s16", sizer=None, scratch=0), bias_is_struct=True)
    assert "int32_t required_buffer_size = 0;" in source
    assert "HELIA_VALIDATE_SIZER" not in source and _sizer(source) is None


def test_weight_sum_buffers_are_guarded_and_checked(contracts) -> None:
    _, source = render_convolve(convolve_context("arm_convolve_1x1_s8_fast", sizer="arm_convolve_1x1_s8_fast_get_buffer_size"))
    assert "#define CONV_CASE_WEIGHT_SUM_BUFFER_SIZE (4 * sizeof(int32_t))" in source
    assert "HELIA_GUARD_ARM(conv_case_weight_sum_buffer, true" in source
    assert re.search(r'HELIA_GUARD_CHECK\(conv_case_weight_sum_buffer, "[^"]+ weight_sum", failures\);', source)


def _synthetic(tmp_path, *kernels: FunctionDecl) -> ContractSet:
    sizer = FunctionDecl(name="arm_fx_get_buffer_size", header="Include/arm_nnfunctions.h", line=2, guards=(),
                         returns="int32_t", params=(ParamDecl("input_dims", "const cmsis_nn_dims *", "in"),))
    return ContractSet(status=STATUS_PRESENT, root=tmp_path, path=tmp_path / "x.json",
                       functions={d.name: d for d in (*kernels, sizer)})


def test_binding_by_name_never_shifts_arguments(tmp_path) -> None:
    """A wrapper that lost weight_sum_ctx gets a call without it, not a shifted one."""
    dropped = FunctionDecl(name="arm_convolve_wrapper_s8", header="Include/arm_nnfunctions.h", line=1, guards=(),
                           returns="arm_cmsis_nn_status",
                           params=tuple(ParamDecl(n, "const int8_t *", "in") for n in
                                        ("ctx", "conv_params", "quant_params", "input_dims", "input_data", "filter_dims",
                                         "filter_data", "bias_dims", "bias_data", "output_dims", "output_data")))
    contracts = _synthetic(tmp_path, dropped)
    _, source = render_convolve(convolve_context("arm_convolve_wrapper_s8", sizer="arm_fx_get_buffer_size"),
                                contracts=contracts)
    assert "weight_sum" not in _strip(source)


def test_the_pool_refuses_what_it_cannot_supply(tmp_path) -> None:
    lut = FunctionDecl(name="arm_fx_convolve_s8", header="Include/arm_nnfunctions.h", line=1, guards=(),
                       returns="arm_cmsis_nn_status",
                       params=(ParamDecl("ctx", "const cmsis_nn_context *", "in"), ParamDecl("lut", "const int16_t *", "in")))
    contracts = _synthetic(tmp_path, lut)
    with pytest.raises(ContractBindError, match=r"cannot supply \['lut \(const int16_t \*\)'\]"):
        render_convolve(convolve_context("arm_fx_convolve_s8", sizer="arm_fx_get_buffer_size"), contracts=contracts)
    with pytest.raises(ContractError, match="arm_convolve_wrapper_s4: not in the kernel contract"):
        render_convolve(convolve_context("arm_convolve_wrapper_s4", sizer="arm_fx_get_buffer_size"), contracts=contracts)
