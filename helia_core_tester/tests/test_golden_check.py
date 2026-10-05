"""Degenerate-golden guard and the families it caught."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.golden_check import (
    EDGE_CASE_KEY,
    case_problems,
    check_case_golden,
    golden_problem,
)
from helia_core_tester.generation.io.descriptors import load_all_descriptors


@pytest.mark.parametrize(
    ("values", "c_type", "flagged"),
    [
        ([-117] * 12, "int8_t", True),
        ([-128, 127] * 6, "int8_t", True),
        ([32767] * 9 + [5], "int16_t", True),
        ([127] * 27 + [0, 1, 2], "int8_t", True),
        (list(range(-6, 6)), "int8_t", False),
        ([127] * 8 + [0, 1, 2], "int8_t", False),
        ([1] * 7, "int8_t", False),
        ([1] * 12, "bool", True),
        ([0, 1] * 6, "bool", False),
    ],
)
def test_golden_problem(values, c_type, flagged):
    assert (golden_problem(np.array(values), c_type) is not None) is flagged


def _write_case(tmp_path: Path, body: str) -> Path:
    includes = tmp_path / "includes"
    includes.mkdir()
    (includes / "case_op.h").write_text(
        "static const int8_t case_input[] = {\n    1, 2, 3\n};\n"
        f"static const int8_t case_expected_output[] = {{\n    {body}\n}};\n"
    )
    return tmp_path


def test_check_rejects_flat(tmp_path):
    case_dir = _write_case(tmp_path, ", ".join(["-117"] * 12))
    assert case_problems(case_dir) == ["case_expected_output: 1 distinct value(s) over 12"]
    with pytest.raises(ValueError, match=EDGE_CASE_KEY):
        check_case_golden(case_dir, {"name": "case"})


@pytest.mark.parametrize(
    "desc",
    [
        {EDGE_CASE_KEY: "two-value input"},
        {"fault": "zero_dim"},
        {"expected_status": "ARM_CMSIS_NN_ARG_ERROR"},
    ],
)
def test_check_allows_exempt(tmp_path, desc):
    check_case_golden(_write_case(tmp_path, ", ".join(["0"] * 12)), desc)


def test_labels_carry_reason():
    for desc in load_all_descriptors(str(find_descriptors_dir())):
        if EDGE_CASE_KEY in desc:
            assert str(desc[EDGE_CASE_KEY]).strip(), desc["name"]


# One case per family the guard caught.
FIXED_CASES = [
    "hard_swish_precise_vector_s8",
    "hard_swish_compat_vector_expneg_s8",
    "rsqrt_small_tensor_universal_s16",
    "comparison_equal_s8",
    "comparison_greater_scalar_s16",
    "max_pool_same_pool1x3_stride2x1_s8",
    "avg_pool_valid_pool1x1_stride1x2_s16",
    "softmax_cmsis_s8",
    "minimum_scalar_right_s16",
]


@pytest.mark.parametrize("name", FIXED_CASES)
def test_fixed_family_golden(tmp_path, name):
    pytest.importorskip("tensorflow")
    from helia_core_tester.generation.test_ops import generate_test

    descs = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}
    generate_test(descs[name], str(tmp_path), seed=Config.seed)
    (case_dir,) = tmp_path.glob(f"*/{name}")
    assert case_problems(case_dir) == []
