"""Quantize and Dequantize render through the generic harness (iteration 1b, G15c): a fused
activation becomes a pre-pass over a guarded float copy of the input (Quantize, with the legacy
activation kernel probed on a copy of the output) or a post-pass over the float output
(Dequantize); both force the status-checked call form."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, HarnessError, plan_harness
from helia_core_tester.generation.ops.QuantizationFunctions.pools import dequantize_argument_pool, quantize_argument_pool
from helia_core_tester.tests.harness_render import render_pool


def q_context(**overrides) -> dict:
    context = {"name": "q", "kernel_fn": "arm_quantize_f32_s8", "input_size": 5, "zero_point": -128, "scale": 0.25,
               "input_data_array": "    1.0f", "expected_output_array": "    2", "input_dtype": "float",
               "output_dtype": "int8_t", "has_activation": False, "activation_kernel_fn": None,
               "activation_type": "NONE", "use_batch_harness": False}
    context.update(overrides)
    return context


def dq_context(**overrides) -> dict:
    context = {"name": "dq", "kernel_fn": "arm_dequantize_s8_f32", "input_size": 5, "zero_point": 3, "scale": 0.5,
               "input_data_array": "    1", "expected_output_array": "    2.0f", "input_dtype": "int8_t",
               "output_dtype": "float", "kernel_style": "scale_offset", "has_activation": False,
               "activation_type": "NONE", "use_batch_harness": False}
    context.update(overrides)
    return context


def _quantize(context):
    return render_pool(context, quantize_argument_pool(context), stem="quantize",
                       validation_key="QuantizationFunctions/quantize/quantize.c.j2", label="Quantize")[1]


def _dequantize(context):
    return render_pool(context, dequantize_argument_pool(context), stem="dequantize",
                       validation_key="QuantizationFunctions/dequantize/dequantize.c.j2", label="Dequantize")[1]


def _run(source: str, name: str) -> str:
    return source[source.index(f"{name}_run("):source.index(f"{name}_test_case_run(void)")]


def _args(call: str) -> list[str]:
    return [a.strip() for a in re.sub(r"/\*.*?\*/", "", call).split(",")]


def test_plain_quantize_is_a_single_call() -> None:
    run = _run(_quantize(q_context()), "q")
    assert _args(re.search(r"return arm_quantize_f32_s8\((.*?)\);", run, flags=re.S).group(1)) == [
        "input", "output", "5", "-128", "0.25f"]
    assert "kernel_status" not in run and "activated_input" not in run


@pytest.mark.parametrize("kind, probe, loop_line", [
    ("RELU", "arm_relu_q7", "q_activated_input[i] = (input[i] < 0.0f) ? 0.0f : input[i];"),
    ("RELU6", "arm_relu6_q7", "if (val > 6.0f) val = 6.0f;"),
    ("RELU6", None, "if (val > 6.0f) val = 6.0f;"),
    ("RELU", "arm_relu_q15", "q_activated_input[i] = (input[i] < 0.0f) ? 0.0f : input[i];"),
])
def test_quantize_activation_runs_before_the_call_on_a_guarded_copy(kind, probe, loop_line) -> None:
    s16 = probe == "arm_relu_q15"
    source = _quantize(q_context(has_activation=True, activation_type=kind, activation_kernel_fn=probe,
                                 **({"kernel_fn": "arm_quantize_f32_s16", "output_dtype": "int16_t"} if s16 else {})))
    if s16:
        assert "int16_t body[5];" in source[source.index("q_activation_probe_guard") - 200:source.index("q_activation_probe_guard")]
    run = _run(source, "q")
    assert "float q_activated_input" in source or "q_activated_input_guard" in source
    assert loop_line in run
    call = re.search(r"kernel_status = arm_quantize_f32_s(8|16)\((.*?)\);", run, flags=re.S).group(2)
    assert _args(call)[0] == "q_activated_input"
    assert run.index("HELIA_GUARD_ARM(q_activated_input, false") < run.index("kernel_status = arm_quantize")
    assert 'HELIA_GUARD_CHECK(q_activated_input, "Quantize activated_input", failures);' in source
    if probe:
        assert run.index("kernel_status = arm_quantize") < run.index(f"{probe}(q_activation_probe, (uint16_t)5);")
        assert "q_activation_probe[i] = output[i];" in run
        assert 'HELIA_GUARD_CHECK(q_activation_probe, "Quantize activation_probe", failures);' in source
    else:
        assert "activation_probe" not in source


@pytest.mark.parametrize("kind, marker, widen", [("RELU", "if (output[i] < 0.0f) output[i] = 0.0f;", False),
                                                 ("RELU6", "if (output[i] > 6.0f) output[i] = 6.0f;", False),
                                                 ("RELU6", "if (output[i] > 6.0f) output[i] = 6.0f;", True)])
def test_dequantize_activation_runs_after_a_successful_call(kind, marker, widen) -> None:
    overrides = ({"kernel_fn": "arm_dequantize_f16_f32", "kernel_style": "widen", "input_dtype": "float16_t",
                  "output_dtype": "float32_t"} if widen else {})
    run = _run(_dequantize(dq_context(has_activation=True, activation_type=kind, **overrides)), "dq")
    call = run.index("kernel_status = arm_dequantize_f16_f32(" if widen else "kernel_status = arm_dequantize_s8_f32(")
    check = run.index("if (kernel_status != ARM_CMSIS_NN_SUCCESS) {\n        return kernel_status;")
    assert call < check < run.index(marker) < run.rindex("return kernel_status;")


def test_widening_dequantize_takes_only_the_block_size() -> None:
    run = _run(_dequantize(dq_context(kernel_fn="arm_dequantize_f16_f32", kernel_style="widen", input_dtype="float16_t",
                                      output_dtype="float32_t")), "dq")
    assert _args(re.search(r"return arm_dequantize_f16_f32\((.*?)\);", run, flags=re.S).group(1)) == [
        "input", "output", "5"]


@pytest.mark.parametrize("build, context", [
    (quantize_argument_pool, q_context(input_size=0)),
    (dequantize_argument_pool, dq_context(input_size=0)),
])
def test_an_empty_block_is_refused(build, context) -> None:
    with pytest.raises(ValueError, match="at least one element"):
        build(context)


@pytest.mark.parametrize("build, context, kind", [
    (dequantize_argument_pool, dq_context(), "tanh"), (dequantize_argument_pool, dq_context(), "NONE"),
    (quantize_argument_pool, q_context(), "tanh"), (quantize_argument_pool, q_context(), "NONE"),
])
def test_an_unknown_activation_is_refused_by_name(build, context, kind) -> None:
    with pytest.raises(ValueError, match=f"fused activation '{kind.upper()}' is not one of"):
        build({**context, "has_activation": True, "activation_type": kind})


def test_passes_refuse_the_benchmark_form_and_void_kernels() -> None:
    with pytest.raises(HarnessError, match="pre- and post-passes have no benchmark or fault form"):
        ArgumentPool(name="x", values={}, scratch_buffer=False, post_call="    x;").validate()
    decl = FunctionDecl(name="arm_fx_void", header="Include/arm_nnfunctions.h", line=1, guards=(), returns="void",
                        params=(ParamDecl("input", "const int8_t *", "in"), ParamDecl("output", "int8_t *", "out")))
    contracts = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None, functions={decl.name: decl})
    with pytest.raises(HarnessError, match="post-pass needs a status to stop on"):
        plan_harness(ArgumentPool(name="x", values={}, scratch_buffer=False, benchmark=False, post_call="    x;"),
                     kernel_fn="arm_fx_void", sizer_fn=None, scratch_bytes=0, contracts=contracts)
