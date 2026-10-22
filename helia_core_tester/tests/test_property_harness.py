"""Property cases render through the generic harness (iteration 1b, G16a): the pool supplies the
whole test body and every guarded buffer, `_run` may take extra scalar parameters, and the
harness contributes the contract-bound kernel call and the parity assert."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, Define, FaultEdit, HarnessError, plan_harness
from helia_core_tester.generation.harness.model import render_declaration
from helia_core_tester.generation.ops._shared.sqrt_float import sqrt_float_argument_pool
from helia_core_tester.generation.ops.BasicMathFunctions.chunked_equivalence import chunked_equivalence_argument_pool
from helia_core_tester.tests.harness_render import render_pool

CHUNKED_KEY = "BasicMathFunctions/chunked_equivalence/chunked_equivalence.c.j2"
SQRT_KEY = "BasicMathFunctions/sqrt_float/sqrt_float.c.j2"


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def chunked_context(kernel_fn: str, style: str, operands: int = 2, **overrides) -> dict:
    context = {"name": "ce", "kernel_fn": kernel_fn, "call_style": style, "operand_count": operands,
               "input_dtype": "int8_t", "output_dtype": "int8_t", "element_count": 19, "chunk_count": 3,
               "chunk_sizes_array": "7, 5, 7", "input1_data_array": "    1", "input2_data_array": "    2",
               "validation_mode": "exact_int", "validation_label": f"ChunkedEquivalence {kernel_fn}",
               "use_batch_harness": False}
    context.update(overrides)
    return context


ADDSUB = {"input_1_offset": 1, "input_1_mult": 2, "input_1_shift": 3, "input_2_offset": 4, "input_2_mult": 5,
          "input_2_shift": 6, "left_shift": 7, "out_offset": 8, "out_mult": 9, "out_shift": 10,
          "out_activation_min": -128, "out_activation_max": 127}


def _chunked(context: dict, quant: dict) -> tuple[str, str]:
    return render_pool(context, chunked_equivalence_argument_pool(context, quant), stem="chunked_equivalence",
                       validation_key=CHUNKED_KEY, label="ChunkedEquivalence")


def test_chunked_run_takes_the_block_size_and_the_body_calls_it_twice() -> None:
    header, source = _chunked(chunked_context("arm_elementwise_add_s8", "addsub"), ADDSUB)
    assert _call(source, "arm_elementwise_add_s8") == ["input1", "input2", "1", "2", "3", "4", "5", "6", "7", "output",
                                                       "8", "9", "10", "-128", "127", "block_size"]
    assert re.search(r"int32_t ce_run\(\n    const int8_t\* __restrict input1,\n    const int8_t\* __restrict input2,\n"
                     r"    int8_t\* __restrict output,\n    int32_t block_size\n\) \{", source)
    assert "ce_input1, ce_input2,\n        ce_output_full, CE_ELEMENT_COUNT);" in source
    assert "ce_input1 + offset, ce_input2 + offset,\n            ce_output_chunked + offset, chunk_size);" in source
    assert 'HELIA_VALIDATE_STATUS("ChunkedEquivalence arm_elementwise_add_s8 full", status);' in source
    assert "        ce_output_chunked,\n        ce_output_full,\n        CE_ELEMENT_COUNT," in source
    assert "ce_output;" not in source and "CE_OUTPUT_SIZE" not in source and "HELIA_BENCHMARK_MODE" not in source
    assert "#define CE_ELEMENT_COUNT (19)" in header and "#define CE_CHUNK_COUNT (3)" in header
    assert "static const int32_t ce_chunk_sizes[CE_CHUNK_COUNT] = {\n    7, 5, 7\n};" in header
    assert "static const int8_t ce_input1[CE_ELEMENT_COUNT]" in header


def test_minmax_slices_by_dims_built_from_the_block_size() -> None:
    _, source = _chunked(chunked_context("arm_maximum_s8", "minmax"), {})
    assert "const cmsis_nn_context ctx = {NULL, 0};\n    const cmsis_nn_dims dims = {1, 1, 1, block_size};" in source
    assert _call(source, "arm_maximum_s8") == ["&ctx", "input1", "&dims", "input2", "&dims", "output", "&dims"]
    assert "static cmsis_nn_context ce_ctx" not in source


def test_requantize_has_one_operand_and_binds_size() -> None:
    quant = {"effective_scale_multiplier": 11, "effective_scale_shift": 0, "input_zeropoint": -3, "output_zeropoint": 2}
    header, source = _chunked(chunked_context("arm_requantize_s8_s8", "requantize", 1), quant)
    assert _call(source, "arm_requantize_s8_s8") == ["input1", "output", "block_size", "11", "0", "-3", "2"]
    assert "ce_input2" not in header and "ce_input1,\n        ce_output_full, CE_ELEMENT_COUNT);" in source


@pytest.mark.parametrize("style, operands, message", [("addsub", 1, "does not take 1 operand"),
                                                      ("vector", 2, "call_style 'vector' does not take")])
def test_chunked_refuses_a_style_and_operand_count_that_disagree(style, operands, message) -> None:
    with pytest.raises(ValueError, match=message):
        chunked_equivalence_argument_pool(chunked_context("arm_elementwise_add_s8", style, operands), ADDSUB)


def sqrt_context(**overrides) -> dict:
    context = {"name": "sq", "kernel_fn": "arm_nn_sqrt_f32", "op_suffix": "sqrt", "input_dtype": "float",
               "output_dtype": "float", "word_type": "uint32_t", "half": False, "block_size": 5,
               "input_bits": ["0x3f800000"] * 5, "expected_bits": ["0x3f800000"] * 5, "api_error": "", "in_place": False,
               "call_size": 5, "expected_status": "ARM_CMSIS_NN_SUCCESS", "max_ulp": 0, "flushed_bits": "0",
               "validation_mode": "float", "use_batch_harness": False}
    context.update(overrides)
    return context


def _sqrt(context: dict) -> tuple[str, str]:
    return render_pool(context, sqrt_float_argument_pool(context), stem="sqrt", validation_key=SQRT_KEY, label="Sqrt")


def test_float_sqrt_runs_over_a_fenced_copy_and_compares_bits() -> None:
    header, source = _sqrt(sqrt_context())
    assert _call(source, "arm_nn_sqrt_f32") == ["input", "output", "5"]
    assert "int32_t status = sq_run(\n        sq_input + 1,\n        result);" in source
    assert "float *result = sq_output + 1;" in source and "float body[5 + 2];" in source
    assert "static int sq_flushes_inputs(void)" in source and "const int flush_inputs = sq_flushes_inputs();" in source
    assert "HELIA_VALIDATE_FLOAT_BITS(actual, expected, 0x7f800000u, allowed, i, 8, failures);" in source
    assert "memcmp(sq_input + 1, sq_input_bits, sizeof(sq_input_bits))" in source
    assert "static const uint32_t sq_input_bits[] = {\n    0x3f800000, 0x3f800000" in header


def test_in_place_sqrt_writes_back_into_the_input_block() -> None:
    _, source = _sqrt(sqrt_context(in_place=True))
    assert "float *result = sq_input + 1;" in source and "memcmp(sq_input + 1" not in source


@pytest.mark.parametrize("error, call_size, first, second", [
    ("null_input", 5, "NULL,", "result);"), ("null_output", 5, "sq_input + 1,", "NULL);"),
    ("zero_block", 0, "sq_input + 1,", "result);"), ("negative_block", -1, "sq_input + 1,", "result);"),
])
def test_a_rejected_sqrt_call_must_leave_the_block_untouched(error, call_size, first, second) -> None:
    _, source = _sqrt(sqrt_context(api_error=error, call_size=call_size, expected_status="ARM_CMSIS_NN_ARG_ERROR"))
    assert f"sq_run(\n        {first}\n        {second}" in source
    assert _call(source, "arm_nn_sqrt_f32")[-1] == str(call_size)
    assert "const uint32_t expected = 0xa5a5a5a5;" in source and "const uint32_t allowed = 0;" in source
    assert 'HELIA_VALIDATE_EXPECTED_STATUS("sq", status, ARM_CMSIS_NN_ARG_ERROR);' in source
    assert "sq_flushes_inputs" not in source


def test_half_sqrt_has_no_flush_probe() -> None:
    context = sqrt_context(kernel_fn="arm_nn_sqrt_f16", input_dtype="float16_t", output_dtype="float16_t",
                           word_type="uint16_t", half=True, input_bits=["0x3c00"] * 5, expected_bits=["0x3c00"] * 5)
    _, source = _sqrt(context)
    assert "flushes_inputs" not in source and "0x7c00u" in source and "uint16_t actual;" in source


def test_a_sqrt_case_needs_an_element() -> None:
    with pytest.raises(ValueError, match="at least one element"):
        sqrt_float_argument_pool(sqrt_context(block_size=0))


# --- harness mechanics the property cases introduced -------------------------------------------

def _pool(**fields) -> ArgumentPool:
    return ArgumentPool(**{"name": "x", "values": {}, "scratch_buffer": False, "benchmark": False,
                           "test_body": "    int failures = 0;", **fields})


@pytest.mark.parametrize("fields, message", [
    ({"test_body": "  "}, "a property case needs a test body"),
    ({"benchmark": True}, "no benchmark, fault or output-slot form"),
    ({"fault": FaultEdit("k", setup="    x;")}, "no benchmark, fault or output-slot form"),
    ({"extra_checks": "    x;"}, "extra_checks belongs to the harness's own test body"),
    ({"output_poison": True}, "output_poison belongs to the harness's own test body"),
    ({"test_body": None, "run_params": (("n", "int32_t"),)}, "run_params need a test body"),
    ({"run_params": (("n", ""),)}, "run parameter 'n' has no C type"),
    ({"run_params": (("output", "int32_t"),)}, "must be distinct C identifiers"),
    ({"run_params": (("2n", "int32_t"),)}, "must be distinct C identifiers"),
])
def test_property_case_validation(fields: dict, message: str) -> None:
    with pytest.raises(HarnessError, match=message):
        _pool(**fields).validate()


def test_a_run_parameter_reaches_the_bound_call() -> None:
    decl = FunctionDecl(name="arm_fx_s8", header="Include/arm_nnfunctions.h", line=1, guards=(),
                        returns="arm_cmsis_nn_status",
                        params=(ParamDecl("input", "const int8_t *", "in"), ParamDecl("output", "int8_t *", "out"),
                                ParamDecl("block_size", "int32_t", "in")))
    contracts = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None, functions={decl.name: decl})
    plan = plan_harness(_pool(values={"block_size": "n"}, run_params=(("n", "int32_t"),)), kernel_fn="arm_fx_s8",
                        sizer_fn=None, scratch_bytes=0, contracts=contracts, indent="")
    assert plan.run_call.endswith("n /* block_size */\n)")


def test_define_renders_as_a_header_macro() -> None:
    assert render_declaration(Define("X_COUNT", "(4)", comment="how many")) == "// how many\n#define X_COUNT (4)"
