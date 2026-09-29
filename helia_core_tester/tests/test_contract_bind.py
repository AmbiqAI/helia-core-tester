"""contract_bind: a kernel call bound by parameter name from what a harness supplies."""

from __future__ import annotations

import os
import re

import jinja2
import pytest

from helia_core_tester.contract.bind import (
    ALIAS_GROUPS,
    ContractBindError,
    bind,
    check_types,
    param_type,
    pointer_element,
    takes,
)
from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl, load_contract_set
from helia_core_tester.contract.render import contract_globals, render_call
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root


def _decl(name: str, *params: tuple[str, str, str]) -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(),
                        returns="arm_cmsis_nn_status",
                        params=tuple(ParamDecl(n, t, d) for n, t, d in params))


DW_S16 = _decl(
    "arm_fx_depthwise_s16",
    ("ctx", "const cmsis_nn_context *", "in,out"), ("dw_conv_params", "const cmsis_nn_dw_conv_params *", "in"),
    ("quant_params", "const cmsis_nn_per_channel_quant_params *", "in"), ("input_dims", "const cmsis_nn_dims *", "in"),
    ("input_data", "const int16_t *", "in"), ("filter_dims", "const cmsis_nn_dims *", "in"),
    ("filter_data", "const int8_t *", "in"), ("bias_dims", "const cmsis_nn_dims *", "in"),
    ("bias_data", "const int64_t *", "in"), ("output_dims", "const cmsis_nn_dims *", "in"),
    ("output_data", "int16_t *", "out"),
)
LEGACY_S4 = _decl(
    "arm_fx_depthwise_s4",
    ("ctx", "const cmsis_nn_context *", "in,out"), ("input", "const int8_t *", "in"),
    ("kernel", "const int8_t *", "in"), ("bias", "const int32_t *", "in"), ("output", "int8_t *", "out"),
)
POOL = {
    "ctx": "&c_ctx", "dw_conv_params": "&c_params", "quant_params": "&c_quant", "input_dims": "&c_in_dims",
    "input_data": "c_input", "filter_dims": "&c_f_dims", "filter_data": "c_weights", "bias_dims": "&c_b_dims",
    "bias_data": "c_biases", "output_dims": "&c_out_dims", "output_data": "c_output",
    "weight_sum_ctx": "&c_ws_ctx", "upscale_dims": "NULL", "layout": "ARM_NN_LAYOUT_NHWC",
}


def test_binds_exactly_the_prototype_parameters_in_order() -> None:
    bound = bind(DW_S16, POOL)
    assert list(bound) == [p.name for p in DW_S16.params]
    assert bound["input_data"] == "c_input" and "weight_sum_ctx" not in bound and "layout" not in bound


def test_aliases_answer_to_either_spelling() -> None:
    assert bind(LEGACY_S4, POOL) == {"ctx": "&c_ctx", "input": "c_input", "kernel": "c_weights",
                                     "bias": "c_biases", "output": "c_output"}
    legacy_pool = {"ctx": "&x", "input": "in", "kernel": "k", "bias": "b", "output": "out"}
    renamed = _decl("arm_fx_new", ("input_data", "const int8_t *", "in"), ("filter_data", "const int8_t *", "in"),
                    ("bias_data", "const int32_t *", "in"), ("output_data", "int8_t *", "out"))
    assert list(bind(renamed, legacy_pool).values()) == ["in", "k", "b", "out"]


def test_the_exact_name_wins_over_an_alias() -> None:
    assert bind(LEGACY_S4, {**POOL, "input": "exact_input"})["input"] == "exact_input"


def test_missing_values_are_named_with_their_types() -> None:
    pool = {k: v for k, v in POOL.items() if k not in ("quant_params", "bias_dims")}
    with pytest.raises(ContractBindError) as info:
        bind(DW_S16, pool)
    message = str(info.value)
    assert "quant_params (const cmsis_nn_per_channel_quant_params *)" in message
    assert "bias_dims (const cmsis_nn_dims *)" in message and "it offers" in message


def test_two_spellings_of_one_value_are_refused() -> None:
    with pytest.raises(ContractBindError, match=r"several spellings \['input_1_data', 'input1_data'\]"):
        bind(_decl("arm_fx_x", ("input_1_vect", "const int8_t *", "in")), {"input_1_data": "a", "input1_data": "b"})


@pytest.mark.parametrize("value", ["", "   ", None, 3])
def test_empty_or_non_string_values_are_refused(value) -> None:
    with pytest.raises(ContractBindError, match="is empty"):
        bind(_decl("arm_fx_x", ("ctx", "const cmsis_nn_context *", "in")), {"ctx": value})


def test_a_parameterless_function_binds_to_nothing() -> None:
    assert bind(_decl("arm_fx_void"), POOL) == {}


def test_param_type_follows_aliases() -> None:
    assert param_type(DW_S16, "bias_data") == "const int64_t *"
    assert param_type(LEGACY_S4, "bias_data") == "const int32_t *"
    assert param_type(LEGACY_S4, "weight_sum_ctx") == ""


def test_takes_follows_aliases() -> None:
    assert takes(LEGACY_S4, "input_data") and takes(LEGACY_S4, "input") and not takes(LEGACY_S4, "weight_sum_ctx")


@pytest.mark.parametrize("c_type, element", [
    ("const int16_t *", "int16_t"), ("int8_t *", "int8_t"), ("const float16_t *const", "float16_t"),
    ("const   int64_t  *", "int64_t"), ("const cmsis_nn_bias_data *", "cmsis_nn_bias_data"),
    ("int32_t", None), ("const int8_t **", None), ("void (*)(int32_t)", None),
])
def test_pointer_element(c_type: str, element) -> None:
    assert pointer_element(c_type) == element


def test_type_check_accepts_the_matching_precision() -> None:
    check_types(DW_S16, {"input": "S16", "output": "S16", "filter": "S8", "bias": "S64"})
    check_types(LEGACY_S4, {"input": "s8", "filter": "S4", "output": "S8"})


def test_type_check_names_every_mismatched_parameter() -> None:
    with pytest.raises(ContractBindError) as info:
        check_types(DW_S16, {"input": "S8", "output": "S8", "filter": "S8"})
    message = str(info.value)
    assert "input_data is 'const int16_t *' but the input dtype is S8" in message
    assert "output_data is 'int16_t *' but the output dtype is S8" in message
    assert "filter_data" not in message


def test_type_check_skips_struct_and_unchecked_roles() -> None:
    struct_bias = _decl("arm_fx_x", ("bias_data", "const cmsis_nn_bias_data *", "in"))
    check_types(struct_bias, {"bias": "S64"})
    check_types(DW_S16, {"input": "S16"})


@pytest.mark.parametrize("roles, message", [({"weights": "S8"}, "unknown tensor role"), ({"input": "S12"}, "unknown dtype")])
def test_type_check_refuses_unknown_roles_and_dtypes(roles, message: str) -> None:
    with pytest.raises(ContractBindError, match=message):
        check_types(DW_S16, roles)


def _contracts(*decls: FunctionDecl) -> ContractSet:
    return ContractSet(status=STATUS_PRESENT, root=None, path=None, functions={d.name: d for d in decls})


def test_template_global_renders_like_contract_call() -> None:
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.globals.update(contract_globals(lambda: _contracts(DW_S16, LEGACY_S4)))
    rendered = env.from_string("return {{ contract_bind('arm_fx_depthwise_s16', pool, indent='    ') }};").render(pool=POOL)
    assert rendered == "return " + render_call(DW_S16, bind(DW_S16, POOL), indent="    ") + ";"
    assert env.from_string("{{ contract_takes('arm_fx_depthwise_s4', 'input_data') }}").render() == "True"
    assert env.from_string("{{ contract_param_type('arm_fx_depthwise_s4', 'bias_data') }}").render() == "const int32_t *"
    with pytest.raises(ContractBindError, match="cannot supply"):
        env.from_string("{{ contract_bind('arm_fx_depthwise_s16', {'ctx': '&c'}) }}").render()


def test_alias_groups_are_unambiguous_in_the_real_tree() -> None:
    """No exported function takes two names from one alias group, so binding never guesses."""
    root = resolve_cmsis_nn_root()
    real = load_contract_set(root)
    if not real.present:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {root} has no kernel contract")
        pytest.skip("no ns-cmsis-nn checkout with a kernel contract")
    clashes = []
    for decl in real.functions.values():
        names = {p.name for p in decl.params}
        for group in ALIAS_GROUPS:
            if len(names & set(group)) > 1:
                clashes.append(f"{decl.name}: {sorted(names & set(group))}")
    assert not clashes, clashes
    assert all(re.fullmatch(r"\w+", name) for group in ALIAS_GROUPS for name in group)
