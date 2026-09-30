"""BasicMath, Comparison and BroadcastTo render through the generic harness (iteration 1b, G12):
pools registered by former template path, binary prototypes' two spellings, a pool that owns its
context, test prologues, extra checks, custom validation, output poisoning, no-input cases and
output storage that differs from the validated size."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation.harness import ArgumentPool, GuardedBuffer, HarnessError, plan_harness
from helia_core_tester.generation.harness import registry
from helia_core_tester.generation.harness.simple import binary_case_pool
from helia_core_tester.generation.ops.BasicMathFunctions.fill import fill_argument_pool
from helia_core_tester.generation.ops.BasicMathFunctions.minmax import minmax_argument_pool
from helia_core_tester.generation.ops.BasicMathFunctions.squared_difference import (
    squared_difference_argument_pool,
    squared_difference_fault_pool,
)
from helia_core_tester.generation.ops.BroadcastFunctions.broadcast_to import broadcast_to_argument_pool
from helia_core_tester.generation.ops._shared.reduce_extrema_float import _reduce_extrema_pool
from helia_core_tester.tests.harness_render import render_pool

DIMS = {"n": 1, "h": 1, "w": 2, "c": 3}


def binary_context(kernel_fn: str, *, float_kernel: bool = False, **overrides) -> dict:
    ctype = "float" if float_kernel else "int8_t"
    context = {"name": "bin", "kernel_fn": kernel_fn, "float_kernel": float_kernel, "input1_dims": DIMS,
               "input2_dims": DIMS, "output_dims": DIMS, "input1_data_array": "    1", "input2_data_array": "    2",
               "expected_output_array": "    3", "input_dtype": ctype, "output_dtype": ctype, "use_batch_harness": False,
               "input1_offset": 1, "input1_mult": 2, "input1_shift": 3, "input2_offset": 4, "input2_mult": 5,
               "input2_shift": 6, "left_shift": 7, "out_offset": 8, "out_mult": 9, "out_shift": 10,
               "out_activation_min": -128, "out_activation_max": 127, "block_size": 6,
               "out_activation_min_literal": "-1.0e+30f", "out_activation_max_literal": "1.0e+30f"}
    context.update(overrides)
    return context


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def _render(context: dict, pool: ArgumentPool, stem: str = "case", key: str = "BasicMathFunctions/add/add.c.j2"):
    return render_pool(context, pool, stem=stem, validation_key=key, label="Case")


@pytest.mark.parametrize("kernel_fn, expected", [
    ("arm_add_s8", ["input1", "&bin_input1_dims", "input2", "&bin_input2_dims", "1", "2", "3", "4", "5", "6", "7",
                    "output", "&bin_output_dims", "8", "9", "10", "-128", "127"]),
    ("arm_elementwise_add_s8", ["input1", "input2", "1", "2", "3", "4", "5", "6", "7", "output", "8", "9", "10",
                                "-128", "127", "6"]),
    ("arm_mul_s8", ["input1", "&bin_input1_dims", "input2", "&bin_input2_dims", "1", "4", "output",
                    "&bin_output_dims", "8", "9", "10", "-128", "127"]),
])
def test_binary_pool_offers_both_spellings(kernel_fn: str, expected: list[str]) -> None:
    context = binary_context(kernel_fn)
    _, source = _render(context, binary_case_pool(context))
    assert _call(source, kernel_fn) == expected
    assert "bin_run(bin_input1, bin_input2, bin_output);" in source


def test_binary_float_takes_the_activation_literals() -> None:
    context = binary_context("arm_elementwise_add_f32", float_kernel=True)
    _, source = _render(context, binary_case_pool(context))
    assert _call(source, "arm_elementwise_add_f32") == ["input1", "input2", "output", "-1.0e+30f", "1.0e+30f", "6"]


def test_minmax_owns_an_empty_static_context() -> None:
    context = binary_context("arm_maximum_s8")
    _, source = _render(context, minmax_argument_pool(context))
    assert re.search(r"static const cmsis_nn_context bin_ctx = \{\s*\.buf = NULL,\s*\.size = 0\s*\};", source) or \
        re.search(r"static cmsis_nn_context bin_ctx = \{\s*\.buf = NULL,\s*\.size = 0\s*\};", source)
    assert _call(source, "arm_maximum_s8")[0] == "&bin_ctx"
    assert "bin_ctx.buf = " not in source and "_buffer" not in source


def test_squared_difference_bit_exact_validation_uses_the_report_limit() -> None:
    context = binary_context("arm_elementwise_squared_difference_f16", float_kernel=True, bit_exact=True,
                             expected_bits_array="    0x0000", input_dtype="float16_t", output_dtype="float16_t")
    header, source = _render(context, squared_difference_argument_pool(context),
                             key="BasicMathFunctions/squared_difference/squared_difference.c.j2")
    assert "static const uint16_t bin_expected_bits[] = {" in header
    assert "HELIA_VALIDATE_FLOAT_BITS(actual, bin_expected_bits[i], 0x7c00u, 0, i," in source
    assert re.search(r"0x7c00u, 0, i,\s*\d+, failures\);", source), "the report limit is rendered, not left as Jinja"
    assert "HELIA_VALIDATE_OUTPUTS" not in source and "#include <string.h>" in source


@pytest.mark.parametrize("kind, marker", [("null_input_1", "NULL, /* input_1_vect */"),
                                          ("zero_block", "0 /* block_size */"),
                                          ("negative_block", "-1 /* block_size */")])
def test_squared_difference_faults(kind: str, marker: str) -> None:
    context = binary_context("arm_elementwise_squared_difference_f16", float_kernel=True, fault=kind,
                             expected_status="ARM_CMSIS_NN_ARG_ERROR", input_dtype="float16_t", output_dtype="float16_t")
    _, source = _render(context, squared_difference_fault_pool(context),
                        key="BasicMathFunctions/squared_difference/squared_difference_fault.c.j2")
    assert marker in source and "HELIA_GUARD_CHECK_UNTOUCHED(bin_output" in source
    with pytest.raises(ValueError, match="no SquaredDifference fault edit for 'bogus'"):
        squared_difference_fault_pool({**context, "fault": "bogus"})


def test_fill_zero_block_keeps_one_element_of_poisoned_untouched_storage() -> None:
    context = {"name": "fill", "kernel_fn": "arm_nn_fill_f32", "output_dtype": "float", "block_size": 0,
               "fill_value_array": "    1.0f", "fill_value_repr": "1.0", "expected_output_array": "    0",
               "use_batch_harness": False}
    _, source = _render(context, fill_argument_pool(context), key="BasicMathFunctions/fill/fill.c.j2")
    assert "#define FILL_OUTPUT_SIZE (0)" in source and "float body[1];" in source
    assert "HELIA_GUARD_ARM(fill_output, true" in source and "HELIA_GUARD_CHECK_UNTOUCHED(fill_output" in source
    assert "fill_run(fill_output);" in source
    assert _call(source, "arm_nn_fill_f32") == ["fill_fill_value[0]", "output", "0"]


def test_reduce_float_prologue_guarded_input_and_bit_validation() -> None:
    context = {"name": "rmax", "kernel_fn": "arm_reduce_max_f32", "float_kernel": True, "input_dtype": "float",
               "output_dtype": "float", "input_dims": DIMS, "output_dims": {"n": 1, "h": 1, "w": 1, "c": 3},
               "axis_dims": {"n": 0, "h": 0, "w": 1, "c": 0}, "word_type": "uint32_t", "infinity_bits": "0x7f800000u",
               "input_count": 6, "input_bits": ["0x1"] * 6, "expected_bits": ["0x1"] * 3, "use_batch_harness": False}
    _, source = _render(context, _reduce_extrema_pool("ReduceMax")(context), key="BasicMathFunctions/reduce_max/reduce_max.c.j2")
    text = source[source.index("int32_t rmax_test_case_run"):]
    assert text.index("HELIA_GUARD_ARM(rmax_input, false") < text.index("memcpy(rmax_input, rmax_input_bits")
    assert text.index("memcpy(rmax_input") < text.index("HELIA_GUARD_ARM(rmax_output, true")
    assert text.index("HELIA_GUARD_CHECK(rmax_input") < text.index("HELIA_VALIDATE_STATUS(")
    assert "memcmp(rmax_input, rmax_input_bits" in text and "HELIA_VALIDATE_OUTPUTS" not in text
    assert "rmax_input_guard" in source  # the input copy is a guarded file-scope buffer


def test_broadcast_to_null_arguments_are_pool_edits() -> None:
    base = {"name": "bt", "c_type": "int8_t", "kernel_fn": "arm_broadcast_to_s8", "output_size": 4, "rank": 1,
            "input_shape": [1], "output_shape": [4], "input_data_array": "    0", "expected_output_array": "    0",
            "input_arg": "bt_input", "params_arg": "&bt_params", "output_arg": "bt_output", "use_batch_harness": False}
    pool = broadcast_to_argument_pool(base)
    assert pool.fault is None and pool.output_ctype == "int8_t"
    nulled = broadcast_to_argument_pool({**base, "output_arg": "NULL", "params_arg": "NULL",
                                         "expected_status": "ARM_CMSIS_NN_ARG_ERROR"})
    assert nulled.fault.values == {"output_data": "NULL"} and nulled.values["params"] == "NULL"


# --- harness mechanics ------------------------------------------------------------------------

def _decl(name: str, *params: str) -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(), returns="arm_cmsis_nn_status",
                        params=tuple(ParamDecl(p, "const int8_t *", "in") for p in params))


CONTRACTS = ContractSet(status=STATUS_PRESENT, root=Path("."), path=None,
                        functions={d.name: d for d in (_decl("arm_fx_ctx_s8", "ctx", "input", "output"),
                                                       _decl("arm_fx_out_s8", "output"))})


@pytest.mark.parametrize("overrides, message", [
    ({"owns_ctx": True}, "owns_ctx needs the pool to supply ctx"),
    ({"output_untouched": True}, "untouched-output check needs the output poisoned first"),
    ({"includes": ("string.h",)}, "is not <header> or"),
    ({"guarded": (GuardedBuffer("x_ctx", "int8_t", "4"),), "source": ()}, None),
])
def test_pool_validation_for_the_new_fields(overrides: dict, message) -> None:
    pool = ArgumentPool(name="x", values={}, **overrides)
    if message is None:
        pool.validate()
    else:
        with pytest.raises(HarnessError, match=message):
            pool.validate()


def test_a_case_without_inputs_and_an_owned_context() -> None:
    plan = plan_harness(ArgumentPool(name="x", values={}, inputs=(), scratch_buffer=False),
                        kernel_fn="arm_fx_out_s8", sizer_fn=None, scratch_bytes=0, contracts=CONTRACTS, indent="")
    assert plan.inputs == [] and plan.run_call == "arm_fx_out_s8(\noutput /* output */\n)"
    owned = plan_harness(ArgumentPool(name="x", values={"ctx": "NULL"}, owns_ctx=True, scratch_buffer=False),
                         kernel_fn="arm_fx_ctx_s8", sizer_fn=None, scratch_bytes=0, contracts=CONTRACTS, indent="")
    assert not owned.uses_ctx and "NULL, /* ctx */" in owned.run_call


def test_registry_refuses_a_second_builder_for_a_template() -> None:
    registry.harness_pool("Fx/fx/fx.c.j2", label="Fx")(fill_argument_pool)
    registry.harness_pool("Fx/fx/fx.c.j2", label="Fx")(fill_argument_pool)  # the same builder again is fine
    with pytest.raises(HarnessError, match="Fx/fx/fx.c.j2 already has a harness pool"):
        registry.harness_pool("Fx/fx/fx.c.j2", label="Fx")(minmax_argument_pool)
    assert registry.lookup("Fx/fx/fx.c.j2")[1] == "Fx" and registry.lookup("Fx/missing.c.j2") is None
