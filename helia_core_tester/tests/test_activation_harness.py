"""Single-call operators render through the generic harness (iteration 1b, G11): no context, no
scratch buffer, the kernel's scalar parameters bound by name from literals, and the Activation
families' pools, PReLU's included, and PReLUScalar's per-pixel call list."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.bind import ContractBindError
from helia_core_tester.generation.harness import ArgumentPool, FaultEdit, HarnessError, HarnessInput
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool
from helia_core_tester.generation.ops._shared.hard_swish_base import hard_swish_values
from helia_core_tester.generation.ops.ActivationFunctions.leaky_relu import leaky_relu_values
from helia_core_tester.generation.ops.ActivationFunctions.prelu import prelu_argument_pool
from helia_core_tester.generation.ops.ActivationFunctions.relu6 import relu6_values
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


# Same-typed scalar pairs that a swapped context key would silently exchange: each op's own
# context-to-parameter map is driven with distinct sentinels under the op's context keys and
# the rendered call must list them in prototype order.
@pytest.mark.parametrize("kernel_fn, values, expected", [
    ("arm_leaky_relu_s8", leaky_relu_values({"input_offset": 11, "output_offset": 12, "output_mult_alpha": 13,
                                             "output_shift_alpha": 14, "output_mult_identity": 15,
                                             "output_shift_identity": 16, "output_size": 24}),
     ["input", "11", "12", "13", "14", "15", "16", "output", "24"]),
    ("arm_relu_generic_s8", relu6_values({"input_offset": 11, "output_offset": 12, "output_mult": 13, "output_shift": 14,
                                          "act_min": 15, "act_max": 16, "output_size": 24}),
     ["input", "11", "12", "13", "14", "15", "16", "output", "24"]),
    ("arm_hard_swish_precise_s8", hard_swish_values({"input_offset": 11, "output_offset": 12, "output_mult": 13,
                                                     "output_shift": 14, "relu_q3": 15, "relu_q6": 16, "prescale": 17,
                                                     "output_size": 24}, "precise"),
     ["input", "11", "12", "13", "14", "15", "16", "17", "output", "24"]),
    ("arm_hard_swish_compat_s8", hard_swish_values({"input_offset": 11, "output_offset": 12, "output_multiplier_fp": 13,
                                                    "output_multiplier_exp": 14, "relu_multiplier_fp": 15,
                                                    "relu_multiplier_exp": 16, "output_size": 24}, "compat"),
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


# --- PReLUScalar: one kernel call per pixel (iteration 1b, G15) --------------------------------

def prelu_scalar_context(num_pixels: int = 3, block_size: int = 4, **overrides) -> dict:
    context = {"name": "ps", "kernel_fn": "arm_prelu_scalar_s8", "num_pixels": num_pixels, "block_size": block_size,
               "scalar_array": "    1", "alpha_array": "    2", "expected_output_array": "    3", "input_dtype": "int8_t",
               "output_dtype": "int8_t", "input_offset": 1, "alpha_offset": 2, "output_offset": -3,
               "output_mult_identity": 11, "output_shift_identity": -1, "output_mult_alpha": 22, "output_shift_alpha": -2,
               "use_batch_harness": False}
    context.update(overrides)
    return context


def _prelu_scalar(context: dict) -> tuple[str, str]:
    from helia_core_tester.generation.ops.ActivationFunctions.prelu_scalar import prelu_scalar_argument_pool

    return render_pool(context, prelu_scalar_argument_pool(context), stem="prelu_scalar",
                       validation_key="ActivationFunctions/prelu_scalar/prelu_scalar.c.j2", label="PReLUScalar")


def test_prelu_scalar_calls_the_kernel_once_per_pixel_and_stops_on_the_first_failure() -> None:
    _, source = _prelu_scalar(prelu_scalar_context())
    run = source[source.index("ps_run("):source.index("ps_test_case_run(void)")]
    calls = re.findall(r"kernel_status = arm_prelu_scalar_s8\((.*?)\);", run, flags=re.S)
    assert len(calls) == 3 and run.count("if (kernel_status != ARM_CMSIS_NN_SUCCESS) {\n        return kernel_status;") == 3
    args = [[a.strip() for a in re.sub(r"/\*.*?\*/", "", c).split(",")] for c in calls]
    assert args[0] == ["scalar_input + 0", "alpha + 0", "true", "1", "2", "-3", "11", "-1", "22", "-2", "output + 0", "4"]
    assert args[2][:2] == ["scalar_input + 2", "alpha + 8"] and args[2][-2:] == ["output + 8", "4"]
    assert "const int8_t* __restrict scalar_input,\n    const int8_t* __restrict alpha,\n    int8_t* __restrict output" in run
    assert "ps_run(ps_scalar_input, ps_alpha, ps_output);" in source and "#define PS_OUTPUT_SIZE (3 * 4)" in source
    assert "HELIA_BENCHMARK_MODE" not in source


def test_prelu_scalar_single_pixel_keeps_the_status_check() -> None:
    _, source = _prelu_scalar(prelu_scalar_context(num_pixels=1, block_size=5))
    assert source.count("kernel_status = arm_prelu_scalar_s8(") == 1 and "return kernel_status;" in source
    assert "#define PS_OUTPUT_SIZE (1 * 5)" in source


@pytest.mark.parametrize("field, value", [("num_pixels", 0), ("block_size", 0), ("block_size", -1)])
def test_prelu_scalar_refuses_empty_shapes(field: str, value: int) -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.prelu_scalar import prelu_scalar_argument_pool

    with pytest.raises(ValueError, match="at least one pixel and a positive block size"):
        prelu_scalar_argument_pool(prelu_scalar_context(**{field: value}))


def _bare_pool(**fields) -> ArgumentPool:
    return ArgumentPool(name="x", values={}, scratch_buffer=False, benchmark=False, **fields)


@pytest.mark.parametrize("fields, message", [
    ({"calls": ()}, "needs at least one call"),
    ({"calls": ({"output": "output + 1"},), "benchmark": True}, "no benchmark or fault form"),
    ({"calls": ({"output": ""},)}, "call 0 gives 'output' an empty expression"),
    ({"calls": ({"output": "output + 1"},), "fault": FaultEdit("x", setup="    x;")}, "no benchmark or fault form"),
])
def test_call_lists_are_validated(fields: dict, message: str) -> None:
    pool = ArgumentPool(name="x", values={}, scratch_buffer=False, **{"benchmark": False, **fields})
    with pytest.raises(HarnessError, match=message):
        pool.validate()


def test_a_call_may_only_override_parameters_the_kernel_takes() -> None:
    from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
    from helia_core_tester.generation.harness import plan_harness

    def decl(name, returns="arm_cmsis_nn_status"):
        return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns=returns,
                            params=(ParamDecl("input", "const int8_t *", "in"), ParamDecl("output", "int8_t *", "out")))

    contracts = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None,
                            functions={d.name: d for d in (decl("arm_fx_s8"), decl("arm_fx_void", "void"))})
    plan = plan_harness(_bare_pool(calls=({"output": "output + 0"}, {"output": "output + 8"})), kernel_fn="arm_fx_s8",
                        sizer_fn=None, scratch_bytes=0, contracts=contracts)
    assert [c.split("\n")[-2].strip() for c in plan.run_calls] == ["output + 0 /* output */", "output + 8 /* output */"]
    with pytest.raises(HarnessError, match=re.escape("overrides 'stride', which arm_fx_s8 does not take; spell overrides "
                                                    "as the kernel's parameters ['input', 'output']")):
        plan_harness(_bare_pool(calls=({"stride": "1"},)), kernel_fn="arm_fx_s8", sizer_fn=None, scratch_bytes=0,
                     contracts=contracts)
    # An alias spelling passes `takes` but bind prefers the kernel's own name, so the override
    # would be shadowed by the call-site value: refused, naming the spellings that work.
    for alias in ("input_data", "output_data"):
        with pytest.raises(HarnessError, match=re.escape(f"overrides '{alias}', which arm_fx_s8 does not take under that "
                                                        "name; spell overrides as the kernel's parameters ['input', 'output']")):
            plan_harness(_bare_pool(calls=({alias: "output + 1"},)), kernel_fn="arm_fx_s8", sizer_fn=None,
                         scratch_bytes=0, contracts=contracts)
    void_plan = plan_harness(_bare_pool(calls=({"output": "output"}, {"output": "output + 1"})), kernel_fn="arm_fx_void",
                             sizer_fn=None, scratch_bytes=0, contracts=contracts)
    assert void_plan.void_return and len(void_plan.run_calls) == 2
