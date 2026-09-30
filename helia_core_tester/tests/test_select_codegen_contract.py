from __future__ import annotations

from pathlib import Path

from helia_core_tester.generation.ops.SelectFunctions.select_v2 import select_v2_argument_pool
from helia_core_tester.generation.ops.SelectFunctions.where import where_argument_pool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_select_v2_uses_bool_condition() -> None:
    source = (_repo_root() / "helia_core_tester" / "generation" / "ops" / "SelectFunctions" / "select_v2.py").read_text()
    context = {"name": "s", "c_type": "int8_t", "condition_c_type": "bool", "rank": 1, "output_shape": [2],
               "cond_strides": [1], "x_strides": [1], "y_strides": [1], "condition_array": "    true, false",
               "x_data_array": "    1, 2", "y_data_array": "    3, 4", "expected_output_array": "    1, 4", "output_size": 2}
    pool = select_v2_argument_pool(context)
    condition = next(d for d in pool.header if d.name == "s_condition")
    assert condition.ctype == "bool" and pool.harness_inputs[0].ctype == "bool"
    assert '"condition_c_type": "bool"' in source
    assert "condition_int8" not in source


def test_where_uses_int64_coordinates() -> None:
    op_source = (_repo_root() / "helia_core_tester" / "generation" / "ops" / "SelectFunctions" / "where.py").read_text()
    context = {"name": "w", "cond_c_type": "bool", "output_c_type": "int64_t", "input_shape": [2], "rank": 1,
               "condition_array": "    true", "expected_output_array": "    0", "max_output_size": 2, "num_true": 1}
    pool = where_argument_pool(context)
    golden = next(d for d in pool.header if d.name == "w_expected_output")
    assert golden.ctype == "int64_t" and pool.output_ctype == "int64_t"
    assert '"output_c_type": "int64_t"' in op_source
    assert "dtype=np.int64" in op_source
