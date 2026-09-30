"""Single-call operators render through the generic harness (iteration 1b, G11): no context, no
scratch buffer, the kernel's scalar parameters bound by name from literals, and the Activation
families' pools, PReLU's included."""

from __future__ import annotations

import re

import pytest

from helia_core_tester.contract.bind import ContractBindError
from helia_core_tester.generation.harness import HarnessError, HarnessInput
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool
from helia_core_tester.generation.ops.ActivationFunctions.prelu import prelu_argument_pool
from helia_core_tester.tests.harness_render import render_pool

DIMS = {"n": 1, "h": 2, "w": 3, "c": 4}


def act_context(kernel_fn: str, ctype: str = "int8_t", **overrides) -> dict:
    context = {"name": "act_case", "kernel_fn": kernel_fn, "input_dims": DIMS, "output_dims": DIMS, "output_size": 24,
               "input_data_array": "    1", "expected_output_array": "    2", "input_dtype": ctype,
               "output_dtype": ctype, "use_batch_harness": False}
    context.update(overrides)
    return context


def _source(context: dict, values: dict, **pool_kw) -> str:
    return render_pool(context, tensor_case_pool(context, values, **pool_kw), stem="act",
                       validation_key="ActivationFunctions/relu/relu.c.j2", label="ReLU")[1]


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def test_relu_binds_its_scalars_in_prototype_order_with_no_context_or_scratch() -> None:
    source = _source(act_context("arm_relu_s8"), {"input_offset": -3, "output_offset": 5, "output_multiplier": 1073741824,
                                                   "output_shift": -2, "output_size": 24})
    assert _call(source, "arm_relu_s8") == ["input", "-3", "5", "1073741824", "-2", "output", "24"]
    assert "cmsis_nn_context" not in source and "_buffer" not in source and "required_buffer_size" not in source
    assert "#define ACT_CASE_OUTPUT_SIZE 24" in source
    assert "HELIA_VALIDATE_OUTPUTS(" in source and "HELIA_BENCHMARK_MODE" not in source


@pytest.mark.parametrize("kernel_fn, values, expected", [
    ("arm_clamp_s8", {"act_min": -10, "act_max": 20, "output_size": 24}, ["input", "-10", "20", "output", "24"]),
    ("arm_logistic_s16", {"input_size": 24, "input_multiplier": 7, "input_left_shift": 3},
     ["input", "output", "24", "7", "3"]),
    ("arm_nn_activation_s16", {"size": 24, "left_shift": 1, "type": "ARM_SIGMOID"},
     ["input", "output", "24", "1", "ARM_SIGMOID"]),
    ("arm_hard_swish_f32", {"size": 24}, ["input", "output", "24"]),
])
def test_other_activation_kernels(kernel_fn: str, values: dict, expected: list[str]) -> None:
    ctype = "int16_t" if "s16" in kernel_fn else ("float" if kernel_fn.endswith("f32") else "int8_t")
    assert _call(_source(act_context(kernel_fn, ctype), values), kernel_fn) == expected


def test_a_missing_scalar_is_refused_by_name() -> None:
    with pytest.raises(ContractBindError, match=r"arm_relu_s8: the harness cannot supply \['output_shift"):
        _source(act_context("arm_relu_s8"), {"input_offset": 0, "output_offset": 0, "output_multiplier": 1,
                                             "output_size": 24})


def test_header_carries_dims_extra_arrays_input_and_golden() -> None:
    header, _ = render_pool(act_context("arm_relu_s8"),
                            tensor_case_pool(act_context("arm_relu_s8"), {"input_offset": 0, "output_offset": 0,
                                                                          "output_multiplier": 1, "output_shift": 0,
                                                                          "output_size": 24}),
                            stem="relu", validation_key="ActivationFunctions/relu/relu.c.j2", label="ReLU")
    for decl in ("static const cmsis_nn_dims act_case_input_dims", "static const cmsis_nn_dims act_case_output_dims",
                 "static const int8_t act_case_input[] = {\n    1\n};",
                 "static const int8_t act_case_expected_output[] = {\n    2\n};"):
        assert decl in header


def test_tensor_case_pool_options() -> None:
    context = act_context("arm_hard_swish_f32", "float")
    pool = tensor_case_pool(context, {"size": 24}, dims=(), output_count=dims_count(DIMS))
    assert pool.output_count == "(1 * 2 * 3 * 4)" and not pool.scratch_buffer and not pool.benchmark
    assert [d.name for d in pool.header] == ["act_case_input", "act_case_expected_output"]
    assert pool.harness_inputs == (HarnessInput("input_data", "input", "act_case_input"),)
    with pytest.raises(HarnessError, match="input_data is supplied per call site"):
        tensor_case_pool(context, {"input_data": "x"}).validate()


def prelu_context(kernel_fn: str = "arm_prelu_s8", *, float_kernel: bool = False, **overrides) -> dict:
    ctype = "float" if float_kernel else "int8_t"
    context = act_context(kernel_fn, ctype, alpha_dims={"n": 1, "h": 1, "w": 1, "c": 4}, alpha_array="    3",
                          alpha_dtype=ctype, input_offset=1, alpha_offset=2, output_offset=-3, output_mult_identity=11,
                          output_shift_identity=-1, output_mult_alpha=22, output_shift_alpha=-2)
    context.update(overrides)
    return context


def test_prelu_int_binds_alpha_and_both_requantisations() -> None:
    context = prelu_context()
    header, source = render_pool(context, prelu_argument_pool(context), stem="prelu",
                                 validation_key="ActivationFunctions/prelu/prelu.c.j2", label="PReLU")
    assert _call(source, "arm_prelu_s8") == ["&act_case_input_dims", "input", "&act_case_alpha_dims", "act_case_alpha",
                                             "1", "2", "-3", "11", "-1", "22", "-2", "&act_case_output_dims", "output"]
    assert "static const int8_t act_case_alpha[] = {\n    3\n};" in header
    assert "#define ACT_CASE_OUTPUT_SIZE (1 * 2 * 3 * 4)" in source


def test_prelu_float_ignores_the_integer_scalars() -> None:
    context = prelu_context("arm_prelu_f32", float_kernel=True)
    _, source = render_pool(context, prelu_argument_pool(context), stem="prelu",
                            validation_key="ActivationFunctions/prelu/prelu.c.j2", label="PReLU")
    assert _call(source, "arm_prelu_f32") == ["&act_case_input_dims", "input", "&act_case_alpha_dims", "act_case_alpha",
                                              "&act_case_output_dims", "output"]


def test_prelu_needs_the_alpha_dtype() -> None:
    context = prelu_context()
    del context["alpha_dtype"]
    with pytest.raises(KeyError, match="alpha_dtype"):
        prelu_argument_pool(context)


# Same-typed scalar pairs that a swapped context key would silently exchange: each op's
# context-to-parameter map is pinned with distinct sentinels in prototype order.
@pytest.mark.parametrize("kernel_fn, values, expected", [
    ("arm_leaky_relu_s8", {"input_offset": 11, "output_offset": 12, "output_multiplier_alpha": 13,
                           "output_shift_alpha": 14, "output_multiplier_identity": 15, "output_shift_identity": 16,
                           "output_size": 24},
     ["input", "11", "12", "13", "14", "15", "16", "output", "24"]),
    ("arm_relu_generic_s8", {"input_offset": 11, "output_offset": 12, "output_multiplier": 13, "output_shift": 14,
                      "act_min": 15, "act_max": 16, "output_size": 24},
     ["input", "11", "12", "13", "14", "15", "16", "output", "24"]),
    ("arm_hard_swish_precise_s8", {"input_offset": 11, "output_offset": 12, "output_multiplier": 13, "output_shift": 14,
                                   "relu_q3": 15, "relu_q6": 16, "prescale": 17, "output_size": 24},
     ["input", "11", "12", "13", "14", "15", "16", "17", "output", "24"]),
    ("arm_hard_swish_compat_s8", {"input_offset": 11, "output_offset": 12, "output_multiplier_fp": 13,
                                  "output_multiplier_exp": 14, "relu_multiplier_fp": 15, "relu_multiplier_exp": 16,
                                  "output_size": 24},
     ["input", "11", "12", "13", "14", "15", "16", "output", "24"]),
])
def test_same_typed_scalar_pairs_bind_in_prototype_order(kernel_fn: str, values: dict, expected: list[str]) -> None:
    assert _call(_source(act_context(kernel_fn), values), kernel_fn) == expected


def test_prelu_arg_error_case_checks_status_and_an_untouched_output() -> None:
    context = prelu_context(expected_status="ARM_CMSIS_NN_ARG_ERROR")
    _, source = render_pool(context, prelu_argument_pool(context), stem="prelu",
                            validation_key="ActivationFunctions/prelu/prelu.c.j2", label="PReLU")
    assert "HELIA_GUARD_ARM(act_case_output, true" in source and "HELIA_GUARD_CHECK_UNTOUCHED(act_case_output" in source
    assert re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_ARG_ERROR", source)
    assert "HELIA_VALIDATE_OUTPUTS(" not in source
