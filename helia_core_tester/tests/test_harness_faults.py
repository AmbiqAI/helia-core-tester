"""Fault cases as edits of an operator's pool (iteration 1b, G7): the kernel call alone sees the
faulted argument, the sizer and rule checks keep the passing case's values, and an edit the
kernel cannot receive is refused at generation time."""

from __future__ import annotations

import re

import pytest

from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, Declaration, FaultEdit, HarnessError, plan_harness
from helia_core_tester.generation.harness.faults import common_fault, null_context_buffer, struct_copy, with_fault
from helia_core_tester.generation.kernel_dispatch import DEPTHWISE_CONV_S8_PLANAR_RULE
from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import convolve_argument_pool, convolve_fault
from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import depthwise_argument_pool, depthwise_fault
from helia_core_tester.tests.harness_render import (
    CONVOLVE_VALIDATION_KEY,
    DEPTHWISE_VALIDATION_KEY,
    convolve_context,
    depthwise_context,
    render_pool,
)


def _decl(name: str, *params: str, returns: str = "arm_cmsis_nn_status") -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns=returns,
                        params=tuple(ParamDecl(p, "const int8_t *", "in") for p in params))


KERNEL = _decl("arm_fx_kernel_s8", "ctx", "params", "input_data", "output_data", "layout")
PLAIN = _decl("arm_fx_plain_s8", "ctx", "input_data", "output_data")
SIZER = _decl("arm_fx_kernel_s8_get_buffer_size", "params", "layout", returns="int32_t")
CONTRACTS = ContractSet(status=STATUS_PRESENT, root=None, path=None,
                        functions={d.name: d for d in (KERNEL, PLAIN, SIZER)})


def _pool(**overrides) -> ArgumentPool:
    fields = dict(name="c", values={"ctx": "&c_ctx", "params": "&c_params", "layout": "ARM_NN_LAYOUT_NHWC"},
                  header=(Declaration("c_params", "cmsis_nn_conv_params", {"stride": {"w": 1, "h": 1}}),),
                  output_count="(4)")
    fields.update(overrides)
    return ArgumentPool(**fields)


def _plan(pool: ArgumentPool, kernel: str = "arm_fx_kernel_s8", sizer: str | None = "arm_fx_kernel_s8_get_buffer_size"):
    return plan_harness(pool, kernel_fn=kernel, sizer_fn=sizer, scratch_bytes=None if sizer else 0,
                        contracts=CONTRACTS, indent="    ")


def test_a_fault_edits_the_kernel_call_but_not_the_sizer() -> None:
    pool = with_fault(_pool(), struct_copy(_pool(), "zero_stride", "params", "cmsis_nn_conv_params", "c_params",
                                           {"stride.w": 0}))
    plan = _plan(pool)
    assert "&c_fault_params, /* params */" in plan.run_call and "&c_fault_params" in plan.bench_call
    assert "&c_params /* params */" not in plan.run_call
    assert plan.sizer_call.startswith("arm_fx_kernel_s8_get_buffer_size(\n    &c_params, /* params */")
    assert plan.fault_kind == "zero_stride" and plan.fault_declarations == []
    assert plan.fault_setup.splitlines()[1:] == ["    cmsis_nn_conv_params c_fault_params = c_params;",
                                                 "    c_fault_params.stride.w = 0;"]
    assert pool.benchmark is False


def test_shared_kinds() -> None:
    pool = _pool()
    assert "NULL, /* input_data */" in _plan(with_fault(pool, common_fault(pool, "null_input"))).run_call
    assert "NULL, /* output_data */" in _plan(with_fault(pool, common_fault(pool, "null_output"))).run_call
    plan = _plan(with_fault(pool, common_fault(pool, "null_ctx_buf")))
    assert plan.no_scratch and "&c_ctx, /* ctx */" in plan.run_call
    layout = _plan(with_fault(pool, common_fault(pool, "invalid_layout", layout="ARM_NN_LAYOUT_NHWC")))
    assert "(arm_nn_tensor_layout)(ARM_NN_LAYOUT_NHWC + 1) /* layout */" in layout.run_call
    assert layout.sizer_call.endswith("ARM_NN_LAYOUT_NHWC /* layout */\n)")
    assert common_fault(pool, "zero_stride") is None


@pytest.mark.parametrize("fault, message", [
    (FaultEdit("noop"), "fault 'noop' edits nothing"),
    (FaultEdit("x", values={"bias_data": "NULL"}), "edits 'bias_data', which the pool does not supply"),
    (FaultEdit("x", values={"params": " "}), "value for 'params' is empty"),
    (FaultEdit("x", values={"params": "&c_params"}, declarations=(Declaration("c_params", "int"),)),
     "fault declaration 'c_params' is not a free C identifier"),
])
def test_fault_validation(fault: FaultEdit, message: str) -> None:
    with pytest.raises(HarnessError, match=message):
        with_fault(_pool(), fault)


def test_a_fault_the_kernel_cannot_receive_is_refused() -> None:
    pool = _pool()
    with pytest.raises(HarnessError, match="edits 'params', which arm_fx_plain_s8 does not take"):
        _plan(with_fault(pool, FaultEdit("x", values={"params": "NULL"})), kernel="arm_fx_plain_s8", sizer=None)
    with pytest.raises(HarnessError, match="clears c_ws_ctx, but no provider supplies 'weight_sum_ctx'"):
        null_context_buffer(pool, "null_weight_sum_ctx", "weight_sum_ctx", "c_ws_ctx")
    with pytest.raises(HarnessError, match="copies c_params without changing a field"):
        struct_copy(pool, "zero_stride", "params", "cmsis_nn_conv_params", "c_params", {})
    with pytest.raises(HarnessError, match="needs the case's layout"):
        common_fault(pool, "invalid_layout")


def _convolve(kind: str, kernel_fn: str, **flags) -> str:
    context = convolve_context(kernel_fn, **flags)
    pool = convolve_fault(convolve_argument_pool(context, has_biases=True, bias_is_struct=False), kind, context)
    context.update(fault=kind, expected_status="ARM_CMSIS_NN_ARG_ERROR")
    return render_pool(context, pool, stem="convolve", validation_key=CONVOLVE_VALIDATION_KEY, label="Convolution")[1]


def _depthwise(kind: str, kernel_fn: str, sizer: str, **flags) -> str:
    context = depthwise_context(kernel_fn, sizer, **flags)
    pool = depthwise_fault(depthwise_argument_pool(context), kind, context)
    context.update(fault=kind, expected_status="ARM_CMSIS_NN_ARG_ERROR")
    return render_pool(context, pool, stem="depthwise_conv", validation_key=DEPTHWISE_VALIDATION_KEY,
                       label="Depthwise convolution")[1]


def _before_call(source: str, kernel_fn: str) -> str:
    return source[:source.index(f"return {kernel_fn}(")]


def test_convolve_faults_render_as_status_only_cases() -> None:
    source = _convolve("channel_group_mismatch", "arm_convolve_wrapper_s8")
    assert "Fault mode: channel_group_mismatch." in source
    assert "conv_case_fault_input_dims.c = 9;" in source  # 2 * filter c (4) + 1
    assert "&conv_case_fault_input_dims, /* input_dims */" in source
    assert "arm_convolve_wrapper_s8_get_buffer_size(" in source and "&conv_case_input_dims, /* input_dims */" in source
    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in source and "HELIA_VALIDATE_OUTPUTS" not in source
    assert "HELIA_BENCHMARK_MODE" not in source
    null_ws = _before_call(_convolve("null_weight_sum_ctx", "arm_convolve_wrapper_s8"), "arm_convolve_wrapper_s8")
    # The weight sums are still computed; the fault nulls the buffer afterwards.
    assert null_ws.index("arm_convolve_weight_sum(") < null_ws.index("conv_case_weight_sum_ctx.buf = NULL;")
    no_scratch = _convolve("null_ctx_buf", "arm_convolve_wrapper_s4")
    assert "conv_case_ctx.buf = NULL;" in no_scratch and "conv_case_ctx.buf = conv_case_buffer;" not in no_scratch


def test_convolve_fault_kind_without_an_edit_is_refused() -> None:
    context = convolve_context("arm_convolve_wrapper_s8")
    with pytest.raises(ValueError, match="no Convolve fault edit for 'channel_mismatch'"):
        convolve_fault(convolve_argument_pool(context, has_biases=True, bias_is_struct=False), "channel_mismatch", context)


def test_depthwise_fault_kind_without_an_edit_is_refused() -> None:
    context = depthwise_context("arm_depthwise_conv_wrapper_s8", "arm_depthwise_conv_wrapper_s8_get_buffer_size")
    with pytest.raises(ValueError, match="no DepthwiseConv fault edit for 'zero_stride'"):
        depthwise_fault(depthwise_argument_pool(context), "zero_stride", context)


def test_null_ctx_buf_needs_a_kernel_that_takes_a_context() -> None:
    bare = _decl("arm_fx_bare_s8", "input_data", "output_data")
    contracts = ContractSet(status=STATUS_PRESENT, root=None, path=None, functions={bare.name: bare})
    pool = with_fault(_pool(), common_fault(_pool(), "null_ctx_buf"))
    with pytest.raises(HarnessError, match="fault 'null_ctx_buf' edits 'ctx', which arm_fx_bare_s8 does not take"):
        plan_harness(pool, kernel_fn="arm_fx_bare_s8", sizer_fn=None, scratch_bytes=0, contracts=contracts, indent="    ")


def test_depthwise_faults_render_as_status_only_cases() -> None:
    wrapper = ("arm_depthwise_conv_wrapper_s8", "arm_depthwise_conv_wrapper_s8_get_buffer_size")
    source = _depthwise("channel_mismatch", *wrapper, weight_sum=True)
    assert "dw_case_fault_output_dims.c = 5;" in source  # input c (4) + 1
    assert "&dw_case_fault_output_dims, /* output_dims */" in source
    null_ws = _before_call(_depthwise("null_weight_sum_ctx", *wrapper, weight_sum=True), wrapper[0])
    assert null_ws.index("arm_depthwise_convolve_weight_sum(") < null_ws.index("dw_case_weight_sum_ctx.buf = NULL;")
    layout = _depthwise("invalid_layout", "arm_depthwise_conv_f32", "arm_depthwise_conv_f32_get_buffer_size",
                        float_kernel=True)
    assert "(arm_nn_tensor_layout)(ARM_NN_LAYOUT_NHWC + 1) /* layout */" in layout


def test_depthwise_fault_keeps_the_planar_rule_on_the_unedited_values() -> None:
    source = _depthwise("channel_mismatch", "arm_depthwise_conv_s8_opt_planar", "arm_depthwise_conv_s8_opt_get_buffer_size",
                        weight_sum=True, planar_supported=True, planar_rule_fn=DEPTHWISE_CONV_S8_PLANAR_RULE)
    rule = re.search(rf"{DEPTHWISE_CONV_S8_PLANAR_RULE}\((.*?)\);", source, flags=re.S).group(1)
    assert "&dw_case_output_dims" in rule and "fault" not in rule


def test_null_weight_sum_fault_on_a_kernel_without_weight_sums_is_refused() -> None:
    context = depthwise_context("arm_depthwise_conv_s16", None, scratch=0, bias_dtype="int64_t")
    with pytest.raises(HarnessError, match="no provider supplies 'weight_sum_ctx'"):
        depthwise_fault(depthwise_argument_pool(context), "null_weight_sum_ctx", context)
