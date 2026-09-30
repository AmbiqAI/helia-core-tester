"""Concatenation renders through the generic harness (iteration 1b, G15d): the any-rank kernels
take every input through a pointer array in one call; the legacy per-axis kernels return void
and are called once per input from the call list, as statements."""

from __future__ import annotations

import re

import pytest

from helia_core_tester.generation.ops.ConcatenationFunctions.concat_pool import concatenation_argument_pool
from helia_core_tester.tests.harness_render import render_pool


def concat_context(kernel_fn: str = "arm_concatenation_s8", style: str = "per_tensor", count: int = 2, **overrides) -> dict:
    context = {"name": "cc", "kernel_fn": kernel_fn, "output_dims": {"n": 1, "h": 2, "w": 3, "c": 4}, "output_rank": 4,
               "num_inputs": count, "axis": 2, "input_concat_dims_array": "    1, 2", "output_shape_array": "    1, 2, 3, 4",
               "input_data_arrays": ["    1"] * count, "expected_output_array": "    2", "input_dtype": "int8_t",
               "output_dtype": "int8_t", "call_style": style, "input_x_array": "    1, 2", "input_y_array": "    2, 2",
               "input_z_array": "    4, 4", "input_w_array": "    1, 1", "output_x": 3, "output_y": 2, "output_z": 4,
               "output_w": 1, "offsets_array": "    0, 1", "use_batch_harness": False}
    context.update(overrides)
    return context


def _render(context: dict) -> tuple[str, str]:
    return render_pool(context, concatenation_argument_pool(context), stem="concatenation",
                       validation_key="ConcatenationFunctions/concatenation/concatenation.c.j2", label="Concatenation")


def _calls(source: str, fn: str) -> list[list[str]]:
    run = source[source.index("cc_run("):source.index("cc_test_case_run(void)")]
    return [[a.strip() for a in re.sub(r"/\*.*?\*/", "", body).split(",")]
            for body in re.findall(rf"{re.escape(fn)}\((.*?)\);", run, flags=re.S)]


@pytest.mark.parametrize("style", ["per_tensor", "any_rank", None])
def test_any_rank_kernels_take_every_input_in_one_call(style) -> None:
    header, source = _render(concat_context(style=style))
    assert _calls(source, "return arm_concatenation_s8") == [
        ["input_ptrs", "2", "cc_input_concat_dims", "2", "output", "4", "cc_output_shape"]]
    assert "static const int8_t* cc_input_ptrs[] = {\n    cc_input1,\n    cc_input2,\n};" in source
    assert "const int8_t* const* __restrict input_ptrs" in source and "cc_run(cc_input_ptrs, cc_output);" in source
    for name in ("cc_input_x", "cc_input_y", "cc_input_z", "cc_input_w", "cc_offsets", "cc_output_dims"):
        assert name in header


def test_float_any_rank_kernel_spells_its_parameters_its_own_way() -> None:
    _, source = _render(concat_context("arm_concatenation_f32", "any_rank", input_dtype="float", output_dtype="float"))
    assert _calls(source, "return arm_concatenation_f32") == [
        ["input_ptrs", "2", "cc_input_concat_dims", "4", "cc_output_shape", "2", "output"]]


@pytest.mark.parametrize("axis, extra", [("x", ["(uint16_t)3"]), ("y", ["(uint16_t)2"]), ("z", ["(uint16_t)4"]), ("w", [])])
def test_axis_kernels_are_called_once_per_input_as_statements(axis: str, extra: list[str]) -> None:
    _, source = _render(concat_context(f"arm_concatenation_s8_{axis}", f"axis_{axis}", count=3,
                                       input_data_arrays=["    1"] * 3))
    calls = _calls(source, f"arm_concatenation_s8_{axis}")
    assert len(calls) == 3
    for i, call in enumerate(calls):
        assert call == [f"input_ptrs[{i}]", f"(uint16_t)cc_input_x[{i}]", f"(uint16_t)cc_input_y[{i}]",
                        f"(uint16_t)cc_input_z[{i}]", f"(uint16_t)cc_input_w[{i}]", "output", *extra,
                        f"(uint32_t)cc_offsets[{i}]"]
    run = source[source.index("cc_run("):source.index("cc_test_case_run(void)")]
    body = run[run.index(") {"):run.index("\n}\n")]
    assert "kernel_status" not in body and body.rstrip().endswith("return ARM_CMSIS_NN_SUCCESS;")


@pytest.mark.parametrize("overrides, message", [
    ({"num_inputs": 0, "input_data_arrays": []}, "at least one input"),
    ({"call_style": "axis_q"}, "call_style 'axis_q' is not one of"),
    ({"num_inputs": 3}, "3 inputs declared but 2 input arrays supplied"),
])
def test_bad_shapes_are_refused(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        concatenation_argument_pool(concat_context(**overrides))


def test_bridge_skips_the_parity_asserts_status_type_when_naming_the_kernel() -> None:
    from helia_core_tester.hardware.generated_test_bridge import (
        UnsupportedGeneratedTestError,
        _extract_first_cmsis_function_name,
    )

    source = ("_Static_assert(__builtin_types_compatible_p(__typeof__(arm_concatenation_s8), "
              "arm_cmsis_nn_status (const int8_t * const *)), \"x\");\n    return arm_concatenation_s8(\n")
    assert _extract_first_cmsis_function_name(source) == "arm_concatenation_s8"
    with pytest.raises(UnsupportedGeneratedTestError, match="Could not find"):
        _extract_first_cmsis_function_name("arm_cmsis_nn_status (int);")
