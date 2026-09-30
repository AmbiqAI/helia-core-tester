"""Split and Unpack render through the generic harness (iteration 1b, G15b): a case with several
outputs reached through a pointer array has no `output` local, and the harness guards, checks and
validates each slot, keeping a zero-extent slice untouched."""

from __future__ import annotations

import re

import pytest

from helia_core_tester.generation.harness import ArgumentPool, HarnessError, OutputSlot
from helia_core_tester.generation.ops.ConcatenationFunctions.slices import split_argument_pool, unpack_argument_pool
from helia_core_tester.tests.harness_render import render_pool


def _outputs(sizes):
    return [{"name": f"sp_out_{i}", "expected_output_array": "    1", "size": size} for i, size in enumerate(sizes)]


def split_context(sizes=(4, 4), **overrides) -> dict:
    context = {"name": "sp", "kernel_fn": "arm_split_s8", "input_dims": {"n": 1, "h": 1, "w": 2, "c": 4},
               "input_dims_count": 4, "axis": 3, "num_splits": len(sizes), "split_dims_array": "    2, 2",
               "input_shape_array": "    1, 1, 2, 4", "input_data_array": "    0", "outputs": _outputs(sizes),
               "input_dtype": "int8_t", "output_dtype": "int8_t", "use_batch_harness": False}
    context.update(overrides)
    return context


def _split(context: dict) -> tuple[str, str]:
    return render_pool(context, split_argument_pool(context), stem="split",
                       validation_key="ConcatenationFunctions/split/split.c.j2", label="Split")


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def test_split_reaches_its_outputs_through_a_pointer_array() -> None:
    header, source = _split(split_context())
    assert _call(source, "arm_split_s8") == ["input", "4", "sp_input_shape", "3", "2", "sp_split_dims", "sp_output_ptrs"]
    assert "static int8_t* sp_output_ptrs[] = {\n    sp_out_0_output,\n    sp_out_1_output,\n};" in source
    assert re.search(r"int32_t sp_run\(\n    const int8_t\* __restrict input\n\) \{", source)
    assert "int32_t status = sp_run(sp_input);" in source and "sp_output" not in source.replace("sp_output_ptrs", "")
    for i in (0, 1):
        assert f"#define SP_OUT_{i}_OUTPUT_SIZE (4)" in source
        assert f"static const int8_t sp_out_{i}_expected_output[] = {{\n    1\n}};" in header
        assert f'HELIA_GUARD_ARM(sp_out_{i}_output, false' in source
        assert f'HELIA_GUARD_CHECK(sp_out_{i}_output, "Split sp_out_{i} output", failures);' in source
        assert f"        sp_out_{i}_output,\n        sp_out_{i}_expected_output,\n        SP_OUT_{i}_OUTPUT_SIZE," in source
    assert "HELIA_GUARD_CHECK_UNTOUCHED" not in source and "HELIA_BENCHMARK_MODE" not in source
    assert "static const int32_t sp_split_dims[] = {\n    2, 2\n};" in header and "sp_input_dims" in header


def test_a_zero_extent_slice_keeps_one_element_and_must_stay_untouched() -> None:
    _, source = _split(split_context(sizes=(8, 0)))
    assert "#define SP_OUT_1_OUTPUT_SIZE (0)" in source
    assert "HELIA_GUARD_ARM(sp_out_1_output, true /* zero-extent slice must not be written: poison */);" in source
    assert 'HELIA_GUARD_CHECK_UNTOUCHED(sp_out_1_output, "Split sp_out_1 output", failures);' in source
    assert re.search(r"sp_out_1_output_body\[1\]|int8_t body\[1\]", source.replace("\n", " ")) or "body[1]" in source
    # The empty slot is still validated over zero elements, as the template did.
    assert source.count("HELIA_VALIDATE_OUTPUTS(") == 2


def test_unpack_takes_no_split_dims() -> None:
    context = {"name": "up", "kernel_fn": "arm_unpack_f32", "input_dtype": "float", "output_dtype": "float",
               "input_dims_count": 2, "axis": 0, "num_outputs": 2, "input_shape_array": "    2, 3",
               "input_data_array": "    0.0f", "outputs": [{"name": f"up_out_{i}", "expected_output_array": "    1.0f",
                                                            "size": 3} for i in range(2)],
               "validation_mode": "float", "use_batch_harness": False}
    header, source = render_pool(context, unpack_argument_pool(context), stem="unpack",
                                 validation_key="ConcatenationFunctions/unpack/unpack.c.j2", label="Unpack")
    assert _call(source, "arm_unpack_f32") == ["input", "2", "up_input_shape", "0", "up_output_ptrs"]
    assert "split_dims" not in header and "input_dims" not in header
    assert "static float* up_output_ptrs[] = {" in source


def test_a_slicing_case_needs_a_slice() -> None:
    with pytest.raises(ValueError, match="at least one output slice"):
        split_argument_pool(split_context(sizes=()))


@pytest.mark.parametrize("fields, message", [
    ({"benchmark": True}, "no benchmark, fault or call-list form"),
    ({"calls": ({"axis": "1"},)}, "no benchmark, fault or call-list form"),
    ({"output_poison": True}, "output_poison describes the single output"),
    ({"validation": "    x;"}, "validation describes the single output"),
    ({"outputs": (OutputSlot("a", 1), OutputSlot("a", 2))}, "must be distinct C identifiers"),
    ({"outputs": (OutputSlot("1a", 1),)}, "must be distinct C identifiers"),
    ({"outputs": (OutputSlot("a", -1),)}, "cannot hold a negative element count"),
])
def test_output_slots_are_validated(fields: dict, message: str) -> None:
    pool = ArgumentPool(name="x", values={}, scratch_buffer=False, **{"benchmark": False,
                                                                     "outputs": (OutputSlot("a", 1),), **fields})
    with pytest.raises(HarnessError, match=message):
        pool.validate()


def test_output_slots_let_the_pool_supply_the_output_pointer_array() -> None:
    pool = ArgumentPool(name="x", values={"output_data": "x_ptrs"}, scratch_buffer=False, benchmark=False,
                        outputs=(OutputSlot("a", 1),))
    pool.validate()
    with pytest.raises(HarnessError, match="output_data is supplied per call site"):
        ArgumentPool(name="x", values={"output_data": "x_ptrs"}, scratch_buffer=False, benchmark=False).validate()
