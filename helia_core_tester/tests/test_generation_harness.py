"""The generic harness: ArgumentPool validation, declaration rendering, and the plan that
binds a kernel and its scratch query from a pool against the kernel contract."""

from __future__ import annotations

import pytest

from helia_core_tester.contract.bind import ContractBindError
from helia_core_tester.contract.ir import STATUS_ABSENT, STATUS_PRESENT, ContractError, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.contract.render import ContractRenderError
from helia_core_tester.generation.harness import (
    ArgumentPool,
    ArrayLiteral,
    Declaration,
    GuardedBuffer,
    HarnessError,
    Provider,
    plan_harness,
    render_declaration,
)


def _decl(name: str, *params: str, returns: str = "arm_cmsis_nn_status") -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns=returns,
                        params=tuple(ParamDecl(p, "const int8_t *", "in") for p in params))


KERNEL = _decl("arm_fx_kernel_s8", "ctx", "weight_sum_ctx", "input_dims", "input_data", "output_data")
PLAIN = _decl("arm_fx_plain_s8", "ctx", "input_data", "output")
SIZER = _decl("arm_fx_kernel_s8_get_buffer_size", "input_dims", returns="int32_t")
CONTRACTS = ContractSet(status=STATUS_PRESENT, root=None, path=None,
                        functions={d.name: d for d in (KERNEL, PLAIN, SIZER)})
WEIGHT_SUM = Provider(param="weight_sum_ctx", expr="&c_ws_ctx",
                      declarations=(Declaration("c_ws_ctx", "cmsis_nn_context", storage="static"),),
                      buffers=(GuardedBuffer("c_ws_buffer", "uint8_t", "C_WS_SIZE", "(4 * sizeof(int32_t))", "weight_sum"),),
                      setup="    c_ws_ctx.buf = c_ws_buffer;")


def _pool(**overrides) -> ArgumentPool:
    fields = dict(name="c", values={"ctx": "&c_ctx", "input_dims": "&c_in_dims"},
                  header=(Declaration("c_in_dims", "cmsis_nn_dims", {"n": 1, "h": 2, "w": 3, "c": 4}),),
                  providers=(WEIGHT_SUM,), output_count="(24)")
    fields.update(overrides)
    return ArgumentPool(**fields)


def test_plan_binds_both_call_sites_and_the_sizer() -> None:
    plan = plan_harness(_pool(), kernel_fn="arm_fx_kernel_s8", sizer_fn="arm_fx_kernel_s8_get_buffer_size",
                        scratch_bytes=None, contracts=CONTRACTS, indent="    ")
    assert plan.run_call == ("arm_fx_kernel_s8(\n    &c_ctx, /* ctx */\n    &c_ws_ctx, /* weight_sum_ctx */\n"
                             "    &c_in_dims, /* input_dims */\n    input, /* input_data */\n    output /* output_data */\n)")
    assert "c_input, /* input_data */" in plan.bench_call and "c_output /* output_data */" in plan.bench_call
    assert plan.sizer_call == "arm_fx_kernel_s8_get_buffer_size(\n    &c_in_dims /* input_dims */\n)"
    assert [block.declarations for block in plan.providers] == [["static cmsis_nn_context c_ws_ctx;"]]
    assert plan.header_declarations == ["static const cmsis_nn_dims c_in_dims = {\n    .n = 1,\n    .h = 2,\n    .w = 3,\n    .c = 4\n};"]


def test_providers_appear_only_when_the_prototype_takes_them() -> None:
    plan = plan_harness(_pool(), kernel_fn="arm_fx_plain_s8", sizer_fn=None, scratch_bytes=0, contracts=CONTRACTS)
    assert plan.providers == [] and plan.sizer_call is None and plan.scratch_bytes == 0
    assert "output /* output */" in plan.run_call  # the output_data value answers to its alias


@pytest.mark.parametrize("sizer, scratch, message", [
    ("arm_fx_kernel_s8_get_buffer_size", 0, "not both or neither"),
    (None, None, "not both or neither"),
    ("arm_fx_plain_s8", None, "is a kernel, not a scratch-size query"),
])
def test_plan_scratch_errors(sizer, scratch, message: str) -> None:
    with pytest.raises(HarnessError, match=message):
        plan_harness(_pool(), kernel_fn="arm_fx_kernel_s8", sizer_fn=sizer, scratch_bytes=scratch, contracts=CONTRACTS)


def test_a_checkout_without_the_export_names_the_ns_cmsis_nn_it_needs(tmp_path) -> None:
    absent = ContractSet(status=STATUS_ABSENT, root=tmp_path, path=None)
    with pytest.raises(ContractRenderError, match=r"arm_fx_kernel_s8: .*needs an ns-cmsis-nn that carries the export "
                                                  r"\(AmbiqAI/ns-cmsis-nn#549 or later\)"):
        plan_harness(_pool(), kernel_fn="arm_fx_kernel_s8", sizer_fn=None, scratch_bytes=0, contracts=absent)


def test_plan_fails_closed_on_the_contract() -> None:
    with pytest.raises(ContractError, match="arm_fx_missing: not in the kernel contract"):
        plan_harness(_pool(), kernel_fn="arm_fx_missing", sizer_fn=None, scratch_bytes=0, contracts=CONTRACTS)
    with pytest.raises(ContractBindError, match=r"cannot supply \['input_dims"):
        plan_harness(_pool(values={"ctx": "&c_ctx"}), kernel_fn="arm_fx_kernel_s8", sizer_fn=None, scratch_bytes=0,
                     contracts=CONTRACTS)


@pytest.mark.parametrize("overrides, message", [
    ({"header": (Declaration("x", "int"), Declaration("x", "int"))}, "x is declared twice"),
    ({"header": (Declaration("c_ws_buffer", "int"),)}, "c_ws_buffer is declared twice"),
    ({"header": (Declaration("c_ws_ctx", "int"),)}, "c_ws_ctx is declared twice"),
    ({"header": (Declaration("2bad", "int"),)}, "is not a C identifier"),
    ({"values": {"weight_sum_ctx": "&x"}}, "both a pool value and a provider"),
    ({"values": {"input_data": "x"}}, "supplied per call site"),
    ({"values": {"ctx": "  "}}, "pool value 'ctx' is empty"),
])
def test_pool_validation(overrides: dict, message: str) -> None:
    with pytest.raises(HarnessError, match=message):
        _pool(**overrides).validate()


@pytest.mark.parametrize("decl, text", [
    (Declaration("w", "int8_t", ArrayLiteral("    1, 2"), array=True), "static const int8_t w[] = {\n    1, 2\n};"),
    (Declaration("b", "int32_t*", "NULL", comment="No biases"), "// No biases\nstatic const int32_t* b = NULL;"),
    (Declaration("m", "int32_t", "{ 7 }", storage="static", array=True), "static int32_t m[] = { 7 };"),
    (Declaration("ctx", "cmsis_nn_context", storage="static"), "static cmsis_nn_context ctx;"),
    (Declaration("p", "cmsis_nn_conv_params", {"stride": {"w": 1, "h": 2}, "flag": True}),
     "static const cmsis_nn_conv_params p = {\n    .stride = {.w = 1, .h = 2},\n    .flag = true\n};"),
])
def test_render_declaration(decl: Declaration, text: str) -> None:
    assert render_declaration(decl) == text
