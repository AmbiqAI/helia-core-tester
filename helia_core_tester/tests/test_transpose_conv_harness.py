"""TransposeConv renders through the generic harness (iteration 1b, G9): the reverse-conv context is
a provider with its own scratch query that answers to either parameter name, the weight-sum
context is a provider or NULL, and the fault kinds are pool edits."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, HarnessError, Provider, RuleCheck, SizeQuery, plan_harness
from helia_core_tester.generation.ops.ConvolutionFunctions.transpose_conv import (
    transpose_conv_argument_pool,
    transpose_conv_fault,
)
from helia_core_tester.tests.harness_render import TRANSPOSE_CONV_VALIDATION_KEY, render_pool, render_transpose_conv

S8_SIZER = "arm_transpose_conv_s8_get_buffer_size"
S8_REVERSE = "arm_transpose_conv_s8_get_reverse_conv_buffer_size"


def tc_context(kernel_fn: str = "arm_transpose_conv_wrapper_s8", *, float_kernel: bool = False,
               weight_sum: bool = True, has_biases: bool = True, **overrides) -> dict:
    ctype = "float" if float_kernel else "int8_t"
    context = {
        "name": "tc_case", "kernel_fn": kernel_fn, "float_kernel": float_kernel,
        "kernel_get_buffer_size_fn": "arm_transpose_conv_f32_get_buffer_size" if float_kernel else S8_SIZER,
        "kernel_get_reverse_buffer_size_fn": ("arm_transpose_conv_f32_get_reverse_conv_buffer_size" if float_kernel
                                              else S8_REVERSE),
        "input_dims": {"n": 1, "h": 3, "w": 3, "c": 20}, "filter_dims": {"n": 4, "h": 2, "w": 2, "c": 20},
        "output_dims": {"n": 1, "h": 6, "w": 6, "c": 4},
        "transpose_conv_params": {"input_offset": 1, "output_offset": -1, "stride_w": 2, "stride_h": 2,
                                  "dilation_w": 1, "dilation_h": 1, "pad_w": 0, "pad_h": 0, "pad_offset_w": 0,
                                  "pad_offset_h": 0, "activation_min": -128, "activation_max": 127},
        "quant_params": {"per_channel": False, "multiplier": 1073741824, "shift": -2},
        "weights_array": "    1", "biases_array": "    0", "has_biases": has_biases, "has_weight_sum": weight_sum,
        "input_data_array": "    0", "expected_output_array": "    0", "input_dtype": ctype, "output_dtype": ctype,
        "weight_dtype": ctype, "bias_dtype": "float" if float_kernel else "int32_t", "buffer_size_max": 256,
        "reverse_conv_ctx_size": 320, "use_batch_harness": False, "kernel_layout": "ARM_NN_LAYOUT_NHWC",
        "transpose_conv_params_type": "cmsis_nn_transpose_conv_params_f32" if float_kernel else None,
        "transpose_activation_min_literal": "-1.0e+30f", "transpose_activation_max_literal": "1.0e+30f",
    }
    context.update(overrides)
    return context


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def test_wrapper_gets_the_reverse_and_weight_sum_contexts() -> None:
    _, source = render_transpose_conv(tc_context())
    args = _call(source, "arm_transpose_conv_wrapper_s8")
    assert args[:3] == ["&tc_case_ctx", "&tc_case_weight_sum_ctx", "&tc_case_reverse_conv_ctx"]
    assert "int32_t reverse_required_buffer_size = arm_transpose_conv_s8_get_reverse_conv_buffer_size(" in source
    assert f'HELIA_VALIDATE_SIZER_FITS("{S8_REVERSE}", reverse_required_buffer_size, TC_CASE_REVERSE_CONV_CTX_SIZE);' in source
    assert "#define TC_CASE_REVERSE_CONV_CTX_SIZE 320" in source
    assert 'HELIA_GUARD_CHECK(tc_case_reverse_conv_ctx_buffer, "TransposeConv reverse_conv_ctx", failures);' in source
    # Both size answers are validated before any context is populated.
    assert source.index("HELIA_VALIDATE_SIZER_FITS(\"arm_transpose_conv_s8_get_reverse") < source.index("tc_case_ctx.buf = ")


def test_the_main_sizer_binds_its_own_parameter_spellings() -> None:
    _, source = render_transpose_conv(tc_context())
    sizer = re.search(rf"{S8_SIZER}\((.*?)\);", source, flags=re.S).group(1)
    assert "/* transposed_conv_params */" in sizer and "/* out_dims */" in sizer


def test_direct_kernel_takes_the_reverse_context_as_output_ctx_and_no_weight_sums() -> None:
    _, source = render_transpose_conv(tc_context("arm_transpose_conv_s8"))
    assert "&tc_case_reverse_conv_ctx, /* output_ctx */" in source
    assert "arm_convolve_weight_sum" not in source and "tc_case_weight_sum_ctx" not in source


def test_without_weight_sums_the_wrapper_gets_null() -> None:
    _, source = render_transpose_conv(tc_context(weight_sum=False))
    assert _call(source, "arm_transpose_conv_wrapper_s8")[1] == "NULL"


def test_float_kernel_binds_layout_and_its_reverse_query() -> None:
    _, source = render_transpose_conv(tc_context("arm_transpose_conv_f32", float_kernel=True, weight_sum=False))
    assert _call(source, "arm_transpose_conv_f32")[-1] == "ARM_NN_LAYOUT_NHWC"
    assert "arm_transpose_conv_f32_get_reverse_conv_buffer_size(" in source


@pytest.mark.parametrize("kind, context, marker", [
    ("null_reverse_conv_ctx_buf", tc_context(), "tc_case_reverse_conv_ctx.buf = NULL;"),
    ("null_weight_sum_ctx", tc_context(), "tc_case_weight_sum_ctx.buf = NULL;"),
    ("nonunit_dilation", tc_context(), "tc_case_fault_transpose_conv_params.dilation.w = 2;"),
    ("null_ctx_buf", tc_context(), "tc_case_ctx.buf = NULL;"),
    ("null_output", tc_context("arm_transpose_conv_f32", float_kernel=True, weight_sum=False), "NULL, /* output_data */"),
])
def test_faults(kind: str, context: dict, marker: str) -> None:
    pool = transpose_conv_fault(transpose_conv_argument_pool(context), kind, context)
    context = {**context, "fault": kind, "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    _, source = render_pool(context, pool, stem="transpose_conv", validation_key=TRANSPOSE_CONV_VALIDATION_KEY,
                            label="Transpose convolution")
    assert marker in source and f"Fault mode: {kind}." in source


def test_fault_without_an_edit_or_its_context_is_refused() -> None:
    context = tc_context()
    with pytest.raises(ValueError, match="no TransposeConv fault edit for 'zero_stride'"):
        transpose_conv_fault(transpose_conv_argument_pool(context), "zero_stride", context)
    no_sums = tc_context(weight_sum=False)
    with pytest.raises(HarnessError, match="clears tc_case_weight_sum_ctx, but no provider supplies 'weight_sum_ctx'"):
        transpose_conv_fault(transpose_conv_argument_pool(no_sums), "null_weight_sum_ctx", no_sums)
    direct = tc_context("arm_transpose_conv_s8")
    pool = transpose_conv_fault(transpose_conv_argument_pool(direct), "null_weight_sum_ctx", direct)
    with pytest.raises(HarnessError, match="edits 'weight_sum_ctx', which arm_transpose_conv_s8 does not take"):
        render_pool(direct, pool, stem="transpose_conv", validation_key=TRANSPOSE_CONV_VALIDATION_KEY,
                    label="Transpose convolution")


# --- harness mechanics this family introduced --------------------------------------------------

def _decl(name: str, *params: str, returns: str = "arm_cmsis_nn_status") -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns=returns,
                        params=tuple(ParamDecl(p, "const int8_t *", "in") for p in params))


KERNEL = _decl("arm_fx_kernel_s8", "ctx", "other_ctx", "input_data", "output_data")
QUERY = _decl("arm_fx_other_get_buffer_size", "input_dims", returns="int32_t")
CONTRACTS = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None,
                        functions={d.name: d for d in (KERNEL, QUERY)})


def _pool(query: SizeQuery | None, **overrides) -> ArgumentPool:
    provider = Provider(param="aux_ctx", aliases=("other_ctx",), expr="&x_aux", size_query=query)
    fields = {"name": "x", "values": {"ctx": "&x_ctx", "input_dims": "&x_dims"}, "providers": (provider,), **overrides}
    return ArgumentPool(**fields)


def test_a_provider_answers_to_its_aliases_and_runs_its_size_query() -> None:
    plan = plan_harness(_pool(SizeQuery("arm_fx_other_get_buffer_size", "other_size", "X_OTHER_MAX")),
                        kernel_fn="arm_fx_kernel_s8", sizer_fn=None, scratch_bytes=0, contracts=CONTRACTS, indent="")
    assert "&x_aux, /* other_ctx */" in plan.run_call and len(plan.providers) == 1
    assert plan.size_queries == [("other_size", "arm_fx_other_get_buffer_size(\n&x_dims /* input_dims */\n)",
                                  "arm_fx_other_get_buffer_size", "X_OTHER_MAX")]


@pytest.mark.parametrize("query, overrides, error, message", [
    (SizeQuery("arm_fx_kernel_s8", "q", "M"), {}, HarnessError, "is a kernel, not a scratch-size query"),
    (SizeQuery("arm_fx_other_get_buffer_size", "required_buffer_size", "M"), {}, HarnessError, "not a free C identifier"),
    (SizeQuery("arm_fx_other_get_buffer_size", "2q", "M"), {}, HarnessError, "not a free C identifier"),
    (None, {"values": {"ctx": "&x_ctx", "other_ctx": "&y"}}, HarnessError, "other_ctx is both a pool value and a provider"),
    (SizeQuery("arm_fx_other_get_buffer_size", "q", "M"), {"checks": (RuleCheck("arm_fx_rule", 1, "q"),)}, HarnessError,
     "rule check variable 'q' is not a free C identifier"),
])
def test_size_query_and_alias_validation(query, overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        plan_harness(_pool(query, **overrides), kernel_fn="arm_fx_kernel_s8", sizer_fn=None, scratch_bytes=0,
                     contracts=CONTRACTS)
