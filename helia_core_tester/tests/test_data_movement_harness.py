"""The single-kernel data-movement families render through the generic harness (iteration 1b, G13):
shape-array/params-struct cases, a pointer-array input, a guarded scratch context the case owns,
a count returned through an extra output, a zero-seeded accumulator and NULL-argument edits."""

from __future__ import annotations

import re

import pytest

from helia_core_tester.generation.harness import HarnessError
from helia_core_tester.generation.harness.simple import int_list, shaped_case_pool
from helia_core_tester.generation.ops.ConcatenationFunctions.pack import pack_argument_pool
from helia_core_tester.generation.ops.DynamicUpdateSliceFunctions.dynamic_update_slice import (
    dynamic_update_slice_argument_pool,
)
from helia_core_tester.generation.ops.ReshapeFunctions.resize_nearest_neighbor import resize_nearest_neighbor_argument_pool
from helia_core_tester.generation.ops.ScatterFunctions.scatter_nd import scatter_nd_argument_pool
from helia_core_tester.generation.ops.SelectFunctions.where import where_argument_pool
from helia_core_tester.tests.harness_render import render_pool

D = {"n": 1, "h": 1, "w": 2, "c": 2}


def _call(source: str, fn: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {re.escape(fn)}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


def _render(context, pool, stem, key):
    return render_pool({"use_batch_harness": False, **context}, pool, stem=stem, validation_key=key, label=stem)


def test_int_list() -> None:
    assert int_list([1, 2, 3]) == "{ 1, 2, 3 }" and int_list([]) == "{  }"


def test_shaped_case_pool_declares_shapes_params_inputs_and_golden() -> None:
    context = {"name": "t", "c_type": "int8_t", "input_shape": [2, 3], "multiples": [1, 2], "input_data_array": "    1",
               "expected_output_array": "    2"}
    pool = shaped_case_pool(context, shapes=(("input_shape", "input_shape"), ("multiples", "multiples")),
                            params_type="cmsis_nn_tile_params",
                            params={"rank": 2, "input_shape": "t_input_shape", "multiples": "t_multiples"},
                            inputs=(("input", "input", "input_data_array"),), output_count="12")
    assert [d.name for d in pool.header] == ["t_input_shape", "t_multiples", "t_params", "t_input", "t_expected_output"]
    assert pool.values["params"] == "&t_params" and pool.output_ctype == "int8_t" and not pool.scratch_buffer
    header, source = _render({**context, "kernel_fn": "arm_tile_s8"}, pool, "tile", "TileFunctions/tile/tile.c.j2")
    assert "static const int32_t t_input_shape[] = { 2, 3 };" in header
    assert _call(source, "arm_tile_s8") == ["input", "&t_params", "output"]


def test_pack_passes_its_pointer_array() -> None:
    context = {"name": "p", "input_dtype": "float", "output_dtype": "float", "kernel_fn": "arm_pack_f32", "num_inputs": 2,
               "input_dims_count": 0, "input_shape_array": "", "output_shape_array": "    2",
               "input_data_arrays": ["    1.0f", "    2.0f"], "expected_output_array": "    1.0f, 2.0f", "axis": 0,
               "output_size": 2}
    _, source = _render(context, pack_argument_pool(context), "pack", "ConcatenationFunctions/pack/pack.c.j2")
    assert "static const float* p_input_ptrs[] = {\n    p_input1,\n    p_input2,\n};" in source
    assert _call(source, "arm_pack_f32") == ["input_ptrs", "2", "0", "NULL", "0", "output"]
    assert "const float* const* __restrict input_ptrs" in source and "p_run(p_input_ptrs, p_output);" in source


def test_resize_owns_a_guarded_scratch_context() -> None:
    context = {"name": "r", "input_dtype": "int8_t", "output_dtype": "int8_t", "kernel_fn": "arm_resize_nearest_neighbor_s8",
               "input_dims": D, "output_dims": D, "output_size_dims": {"n": 1, "h": 1, "w": 1, "c": 2},
               "output_size_array": "    1, 2", "input_data_array": "    0", "expected_output_array": "    0",
               "output_size": 4, "buffer_size": 8, "align_corners": "false", "half_pixel_centers": "true"}
    _, source = _render(context, resize_nearest_neighbor_argument_pool(context), "resize_nearest_neighbor",
                        "ReshapeFunctions/resize_nearest_neighbor/resize_nearest_neighbor.c.j2")
    assert re.search(r"int32_t body\[8\];", source) and "static cmsis_nn_context r_ctx" in source
    assert ".buf = r_buffer" in source and "r_ctx.buf = " not in source
    assert "HELIA_GUARD_ARM(r_buffer, true" in source and 'HELIA_GUARD_CHECK(r_buffer, "' in source
    assert _call(source, "arm_resize_nearest_neighbor_s8")[:2] == ["&r_ctx", "&r_params"]


def test_where_validates_the_count_before_the_variable_output() -> None:
    context = {"name": "w", "cond_c_type": "bool", "output_c_type": "int64_t", "kernel_fn": "arm_where_s8",
               "input_shape": [2, 2], "rank": 2, "condition_array": "    true", "expected_output_array": "    0",
               "max_output_size": 8, "num_true": 3}
    _, source = _render(context, where_argument_pool(context), "where", "SelectFunctions/where/where.c.j2")
    assert "static int32_t w_num_true = 0;" in source
    assert _call(source, "arm_where_s8") == ["condition", "&w_params", "output", "&w_num_true"]
    text = source[source.index("w_test_case_run"):]
    assert text.index('HELIA_VALIDATE_SCALAR_EQ_INT("Where", "num_true", 3, w_num_true);') < text.index("output_count,")
    assert "#define W_OUTPUT_SIZE 8" in source and "int64_t body[W_OUTPUT_SIZE]" in source


def test_where_with_nothing_true_compares_nothing() -> None:
    context = {"name": "w", "cond_c_type": "bool", "output_c_type": "int64_t", "kernel_fn": "arm_where_s8",
               "input_shape": [2, 2], "rank": 2, "condition_array": "    false", "expected_output_array": "    0",
               "max_output_size": 8, "num_true": 0}
    _, source = _render(context, where_argument_pool(context), "where", "SelectFunctions/where/where.c.j2")
    assert 'HELIA_VALIDATE_SCALAR_EQ_INT("Where", "num_true", 0, w_num_true);' in source
    assert "#define W_OUTPUT_SIZE 8" in source


def test_scatter_zero_seeds_its_accumulator() -> None:
    context = {"name": "s", "c_type": "int8_t", "kernel_fn": "arm_scatter_nd_s8", "output_strides": [1], "num_updates": 2,
               "index_depth": 1, "slice_size": 1, "output_size": 4, "indices_array": "    0, 1", "updates_array": "    5, 6",
               "expected_output_array": "    5, 6, 0, 0"}
    _, source = _render(context, scatter_nd_argument_pool(context), "scatter_nd", "ScatterFunctions/scatter_nd/scatter_nd.c.j2")
    assert "memset(s_output, 0, sizeof(s_output));" in source and "#include <string.h>" in source
    assert source.index("memset(s_output, 0, sizeof(s_output));") < source.index("int32_t status = s_run(")
    assert _call(source, "arm_scatter_nd_s8") == ["indices", "updates", "&s_params", "output"]


def test_data_movement_operators_keep_their_log_labels() -> None:
    context = {"name": "d", "c_type": "int8_t", "kernel_fn": "arm_dynamic_update_slice_s8", "rank": 1, "operand_shape": [4],
               "update_shape": [2], "operand_strides": [1], "operand_size": 4, "update_size": 2,
               "operand_data_array": "    0", "update_data_array": "    0", "start_indices_array": "    0",
               "expected_output_array": "    0", "operand_arg": "d_operand", "update_arg": "d_update",
               "start_indices_arg": "d_start_indices", "params_arg": "&d_params", "output_arg": "d_output"}
    _, source = _render(context, dynamic_update_slice_argument_pool(context), "dynamic_update_slice",
                        "DynamicUpdateSliceFunctions/dynamic_update_slice/dynamic_update_slice.c.j2")
    assert 'HELIA_GUARD_CHECK(d_output, "DynamicUpdateSlice output", failures);' in source


@pytest.mark.parametrize("arg, param", [("operand_arg", "operand"), ("start_indices_arg", "start_indices"),
                                        ("output_arg", "output_data")])
def test_dynamic_update_slice_null_arguments_are_edits(arg: str, param: str) -> None:
    context = {"name": "d", "c_type": "int8_t", "kernel_fn": "arm_dynamic_update_slice_s8", "rank": 1, "operand_shape": [4],
               "update_shape": [2], "operand_strides": [1], "operand_size": 4, "update_size": 2,
               "operand_data_array": "    0", "update_data_array": "    0", "start_indices_array": "    0",
               "expected_output_array": "    0", "operand_arg": "d_operand", "update_arg": "d_update",
               "start_indices_arg": "d_start_indices", "params_arg": "&d_params", "output_arg": "d_output",
               "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    assert dynamic_update_slice_argument_pool(context).fault is None
    pool = dynamic_update_slice_argument_pool({**context, arg: "NULL"})
    assert pool.fault.values == {param: "NULL"}
    _, source = _render({**context, arg: "NULL"}, pool, "dynamic_update_slice",
                        "DynamicUpdateSliceFunctions/dynamic_update_slice/dynamic_update_slice.c.j2")
    assert "HELIA_GUARD_CHECK_UNTOUCHED(d_output" in source


def test_shaped_case_pool_refuses_an_input_named_like_a_pool_value() -> None:
    context = {"name": "t", "c_type": "int8_t", "input_shape": [2], "input_data_array": "    1",
               "expected_output_array": "    2"}
    with pytest.raises(HarnessError, match="params is supplied per call site"):
        shaped_case_pool(context, shapes=(("input_shape", "input_shape"),), params_type="p", params={"rank": 1},
                         inputs=(("params", "input", "input_data_array"),), output_count="1").validate()
