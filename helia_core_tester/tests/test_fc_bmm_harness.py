"""FullyConnected and BatchMatMul render through the generic harness (iteration 1b, G8): several
inputs, a context that carries precomputed kernel sums, cases without a scratch query, the
quant argument each prototype asks for, and their fault edits."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, Declaration, HarnessError, HarnessInput, plan_harness
from helia_core_tester.generation.harness.model import render_declaration
from helia_core_tester.generation.ops.FullyConnectedFunctions.batch_matmul import bmm_argument_pool, bmm_fault
from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import (
    PER_CHANNEL_S16_SIZER,
    fc_argument_pool,
    fc_fault,
    fc_quant_argument,
    fc_sizer,
)
from helia_core_tester.tests.harness_render import (
    BMM_VALIDATION_KEY,
    FC_VALIDATION_KEY,
    render_batch_matmul,
    render_fully_connected,
    render_pool,
)

DIMS = {"n": 1, "h": 1, "w": 1, "c": 8}


def fc_context(kernel_fn: str, sizer: str | None = "default", *, per_channel: bool = False, float_kernel: bool = False,
               weight_sum: bool = False, has_biases: bool = True, **overrides) -> dict:
    ctype = "float" if float_kernel else ("int16_t" if "s16" in kernel_fn else "int8_t")
    context = {
        "name": "fc_case", "kernel_fn": kernel_fn, "float_kernel": float_kernel,
        "kernel_get_buffer_size_fn": ("arm_fully_connected_s8_get_buffer_size" if sizer == "default" else sizer),
        "input_dims": {"n": 2, "h": 1, "w": 1, "c": 8}, "filter_dims": {"n": 8, "h": 1, "w": 1, "c": 5},
        "output_dims": {"n": 2, "h": 1, "w": 1, "c": 5},
        "fc_params": {"input_offset": 3, "filter_offset": 0, "output_offset": -2, "activation_min": -128,
                      "activation_max": 127},
        "quant_params": ({"per_channel": True, "multiplier_array": "    1, 2, 3, 4, 5", "shift_array": "    0, 0, 0, 0, 0"}
                         if per_channel else {"per_channel": False, "multiplier": 1073741824, "shift": -3}),
        "weights_array": "    1", "biases_array": "    7", "has_biases": has_biases, "has_bias_array": has_biases or weight_sum,
        "weight_sum": weight_sum, "has_weight_sum": weight_sum, "weight_sum_array": "    9, 9, 9, 9, 9",
        "input_data_array": "    0", "expected_output_array": "    0", "input_dtype": ctype, "output_dtype": ctype,
        "bias_dtype": "int64_t" if "s16" in kernel_fn else ("float" if float_kernel else "int32_t"),
        "buffer_size_max": 64, "use_batch_harness": False,
        "fc_params_type": "cmsis_nn_fc_params_f32" if float_kernel else None,
        "fc_activation_min_literal": "-1.0e+30f", "fc_activation_max_literal": "1.0e+30f",
        "kernel_layout": "ARM_NN_LAYOUT_NHWC",
    }
    context.update(overrides)
    return context


def bmm_context(kernel_fn: str = "arm_batch_matmul_s8", *, float_kernel: bool = False, **overrides) -> dict:
    ctype = "float" if float_kernel else "int8_t"
    context = {
        "name": "bmm_case", "kernel_fn": kernel_fn, "float_kernel": float_kernel,
        "kernel_get_buffer_size_fn": (f"{kernel_fn}_get_buffer_size" if float_kernel
                                      else "arm_fully_connected_s8_get_buffer_size"),
        "input_lhs_dims": {"n": 1, "h": 2, "w": 3, "c": 4}, "input_rhs_dims": {"n": 1, "h": 2, "w": 5, "c": 4},
        "output_dims": {"n": 1, "h": 2, "w": 3, "c": 5},
        "bmm_params": {"adj_x": False, "adj_y": True, "fc_params": {"input_offset": 1, "filter_offset": 2,
                                                                     "output_offset": 3, "activation_min": -128,
                                                                     "activation_max": 127}},
        "quant_params": {"multiplier": 1073741824, "shift": -1},
        "input_lhs_array": "    1", "input_rhs_array": "    2", "expected_output_array": "    0",
        "input_dtype": ctype, "input_rhs_dtype": ctype, "output_dtype": ctype, "buffer_size_max": 64,
        "use_batch_harness": False, "bmm_params_type": "cmsis_nn_bmm_params_f32" if float_kernel else None,
        "bmm_activation_min_literal": "-1.0e+30f", "bmm_activation_max_literal": "1.0e+30f",
    }
    context.update(overrides)
    return context


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


# --- FullyConnected ----------------------------------------------------------------------------

def test_wrapper_takes_a_file_scope_generic_quant_wrapper() -> None:
    _, source = render_fully_connected(fc_context("arm_fully_connected_wrapper_s8", per_channel=True))
    assert _call(source, "arm_fully_connected_wrapper_s8")[2] == "&fc_case_quant_params_wrapper"
    assert re.search(r"static const cmsis_nn_quant_params fc_case_quant_params_wrapper = \{\s*"
                     r"\.multiplier = \(int32_t\*\)fc_case_multiplier,\s*\.shift = \(int32_t\*\)fc_case_shift,\s*"
                     r"\.is_per_channel = 1\s*\};", source)
    header, per_tensor = render_fully_connected(fc_context("arm_fully_connected_wrapper_s8"))
    assert "(int32_t*)&fc_case_multiplier_val" in per_tensor and ".is_per_channel = 0" in per_tensor
    assert "static int32_t fc_case_multiplier_val = 1073741824;" in header


def test_direct_kernels_take_the_header_struct_and_refuse_the_wrong_granularity() -> None:
    _, source = render_fully_connected(fc_context("arm_fully_connected_s8"))
    assert _call(source, "arm_fully_connected_s8")[2] == "&fc_case_quant_params"
    assert "quant_params_wrapper" not in source
    with pytest.raises(ValueError, match=r"arm_fully_connected_s8 takes const cmsis_nn_per_tensor_quant_params \*, "
                                         r"but this case's quantization is per-channel"):
        fc_quant_argument(fc_context("arm_fully_connected_s8", per_channel=True))
    with pytest.raises(ValueError, match="takes const cmsis_nn_per_channel_quant_params"):
        fc_quant_argument(fc_context("arm_fully_connected_per_channel_s8"))


def test_weight_sums_become_the_context() -> None:
    header, source = render_fully_connected(fc_context("arm_fully_connected_wrapper_s8", weight_sum=True, has_biases=False))
    assert "static const int32_t fc_case_weight_sum[5] = {\n    9, 9, 9, 9, 9\n};" in header
    assert "fc_case_ctx.buf = (uint8_t *)fc_case_weight_sum;" in source and "fc_case_ctx.size = 5 * 4;" in source
    assert "fc_case_ctx.buf = fc_case_buffer;" not in source
    # The folded bias stays in the header for consumers, and the kernel gets NULL.
    assert "static const int32_t fc_case_biases[] = {" in header
    assert _call(source, "arm_fully_connected_wrapper_s8")[8] == "NULL"
    assert "HELIA_VALIDATE_SIZER(\"arm_fully_connected_s8_get_buffer_size\"" in source


@pytest.mark.parametrize("context, sizer", [
    (fc_context("arm_fully_connected_s4", None), None),
    (fc_context("arm_fully_connected_wrapper_s16", "arm_fully_connected_s16_get_buffer_size", per_channel=True),
     PER_CHANNEL_S16_SIZER),
    (fc_context("arm_fully_connected_wrapper_s16", "arm_fully_connected_s16_get_buffer_size"),
     "arm_fully_connected_s16_get_buffer_size"),
    (fc_context("arm_fully_connected_s16", "arm_fully_connected_s16_get_buffer_size_mve", per_channel=True),
     "arm_fully_connected_s16_get_buffer_size_mve"),
    (fc_context("arm_fully_connected_f32", "arm_fully_connected_f32_get_buffer_size", float_kernel=True),
     "arm_fully_connected_f32_get_buffer_size"),
])
def test_fc_sizer(context: dict, sizer: str | None) -> None:
    assert fc_sizer(context) == sizer


def test_s4_has_no_scratch_query_and_no_unused_size() -> None:
    _, source = render_fully_connected(fc_context("arm_fully_connected_s4", None))
    assert "required_buffer_size" not in source and "HELIA_VALIDATE_SIZER" not in source
    assert "fc_case_ctx.buf = NULL;" in source and "fc_case_ctx.size = 0;" in source


def test_float_fc_binds_layout_into_kernel_and_sizer() -> None:
    _, source = render_fully_connected(fc_context("arm_fully_connected_f32", "arm_fully_connected_f32_get_buffer_size",
                                                  float_kernel=True))
    assert _call(source, "arm_fully_connected_f32")[-1] == "ARM_NN_LAYOUT_NHWC"
    assert re.search(r"arm_fully_connected_f32_get_buffer_size\([^;]*ARM_NN_LAYOUT_NHWC", source, flags=re.S)


@pytest.mark.parametrize("kind, context, marker", [
    ("small_ctx_size", fc_context("arm_fully_connected_wrapper_s16", per_channel=True), "fc_case_ctx.size = 1;"),
    ("null_ctx_buf", fc_context("arm_fully_connected_wrapper_s8", weight_sum=True), "fc_case_ctx.buf = NULL;"),
    ("filter_n_mismatch", fc_context("arm_fully_connected_f32", "arm_fully_connected_f32_get_buffer_size",
                                     float_kernel=True), "fc_case_fault_filter_dims.n = 9;"),
])
def test_fc_faults(kind: str, context: dict, marker: str) -> None:
    pool = fc_fault(fc_argument_pool(context), kind, context)
    context = {**context, "fault": kind, "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    _, source = render_pool(context, pool, stem="fully_connected", validation_key=FC_VALIDATION_KEY,
                            label="Fully connected", sizer_fn=fc_sizer(context))
    assert marker in source and f"Fault mode: {kind}." in source and "HELIA_VALIDATE_EXPECTED_STATUS(" in source
    if kind == "small_ctx_size":
        # The shrunk context is re-stamped, so its slack check matches what the kernel was handed.
        tail = source[source.index("fc_case_ctx.size = 1;"):]
        assert tail.index("HELIA_GUARD_STAMP_SLACK(fc_case_buffer") < tail.index("return arm_fully_connected_wrapper_s16(")


def test_fc_fault_kind_without_an_edit_is_refused() -> None:
    context = fc_context("arm_fully_connected_wrapper_s8")
    with pytest.raises(ValueError, match="no FullyConnected fault edit for 'zero_stride'"):
        fc_fault(fc_argument_pool(context), "zero_stride", context)


# --- BatchMatMul -------------------------------------------------------------------------------

def test_bmm_run_takes_both_inputs_and_the_test_passes_both_arrays() -> None:
    header, source = render_batch_matmul(bmm_context())
    assert re.search(r"int32_t bmm_case_run\(\s*const int8_t\* __restrict input_lhs,\s*const int8_t\* __restrict input_rhs,"
                     r"\s*int8_t\* __restrict output\s*\)", source)
    assert "bmm_case_run(bmm_case_input_lhs, bmm_case_input_rhs, bmm_case_output);" in source
    args = _call(source, "arm_batch_matmul_s8")
    assert args[4] == "input_lhs" and args[6] == "input_rhs" and args[-1] == "output"
    assert "static const cmsis_nn_dims bmm_case_filter_dims_for_buffer = {" in source
    assert re.search(r"arm_fully_connected_s8_get_buffer_size\(\s*&bmm_case_filter_dims_for_buffer", source)
    assert ".adj_y = true" in header and ".fc_params = {.input_offset = 1," in header


def test_float_bmm_has_no_quant_and_sizes_from_its_params() -> None:
    header, source = render_batch_matmul(bmm_context("arm_batch_matmul_f32", float_kernel=True))
    assert "quant_params" not in header and "filter_dims_for_buffer" not in source
    assert ".rhs_format = ARM_NN_WEIGHT_FORMAT_STANDARD" in header


@pytest.mark.parametrize("kind, kernel, marker", [
    ("null_input", "arm_batch_matmul_f32", "NULL, /* input_lhs */"),
    ("null_output", "arm_batch_matmul_f32", "NULL /* output */"),
    ("negative_dim", "arm_batch_matmul_s8", "bmm_case_fault_input_rhs_dims.w = -1;"),
    ("packed_rhs_adjoint", "arm_batch_matmul_f32", ".rhs_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;"),
])
def test_bmm_faults(kind: str, kernel: str, marker: str) -> None:
    context = bmm_context(kernel, float_kernel=kernel.endswith("f32"))
    pool = bmm_fault(bmm_argument_pool(context), kind, context)
    context = {**context, "fault": kind, "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    _, source = render_pool(context, pool, stem="batch_matmul", validation_key=BMM_VALIDATION_KEY, label="Batch matmul")
    assert marker in source
    if kind == "negative_dim":  # the scratch query still sees the passing dims
        assert re.search(r"arm_fully_connected_s8_get_buffer_size\(\s*&bmm_case_filter_dims_for_buffer", source)


# --- harness mechanics these operators introduced ----------------------------------------------

def _decl(name: str, *params: str, returns: str = "arm_cmsis_nn_status") -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns=returns,
                        params=tuple(ParamDecl(p, "const int8_t *", "in") for p in params))


TWO = _decl("arm_fx_two_s8", "ctx", "a", "b", "output")
CONTRACTS = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None, functions={TWO.name: TWO})
INPUTS = (HarnessInput("a", "a", "x_a"), HarnessInput("b", "b", "x_b"))


@pytest.mark.parametrize("overrides, message", [
    ({"inputs": (HarnessInput("a", "same", "x_a"), HarnessInput("b", "same", "x_b"))}, "must be distinct C identifiers"),
    ({"inputs": (HarnessInput("a", "a", "x_a"), HarnessInput("a", "b", "x_b"))}, "call-site parameters"),
    ({"inputs": (HarnessInput("a", "output", "x_a"),)}, "must be distinct C identifiers"),
    ({"inputs": INPUTS, "values": {"ctx": "&x_ctx", "a": "x"}}, "a is supplied per call site"),
    ({"inputs": INPUTS, "no_scratch": True, "context_setup": "    x_ctx.buf = 0;"}, "both set the context"),
])
def test_pool_validation_for_inputs_and_context(overrides: dict, message: str) -> None:
    fields = {"name": "x", "values": {"ctx": "&x_ctx"}, "output_param": "output", **overrides}
    with pytest.raises(HarnessError, match=message):
        ArgumentPool(**fields).validate()


def test_several_inputs_bind_in_prototype_order() -> None:
    plan = plan_harness(ArgumentPool(name="x", values={"ctx": "&x_ctx"}, output_param="output", inputs=INPUTS),
                        kernel_fn="arm_fx_two_s8", sizer_fn=None, scratch_bytes=0, contracts=CONTRACTS, indent="")
    assert plan.inputs == [("a", "", "x_a"), ("b", "", "x_b")]
    assert plan.run_call == "arm_fx_two_s8(\n&x_ctx, /* ctx */\na, /* a */\nb, /* b */\noutput /* output */\n)"
    assert "x_a, /* a */" in plan.bench_call and "x_output /* output */" in plan.bench_call


def test_declaration_extent() -> None:
    assert render_declaration(Declaration("w", "int32_t", "{ 1 }", array=True, extent="3")) == \
        "static const int32_t w[3] = { 1 };"
