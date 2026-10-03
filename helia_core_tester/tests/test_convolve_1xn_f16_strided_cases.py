"""The strided 1xN FP16 cases must stay off ns-cmsis-nn's in-place contiguous-K route.

arm_convolve_1_x_n_f16 and its _acc16 entry run the no-padding rows in place when
Cout >= 4 && ((Cout % 8 >= 4 && K >= 80) || K >= 224), with K = kernel_w * Cin, and through
the strided kernel otherwise. These cases exist only to keep the strided kernel covered, so a
shape edited across that boundary would silently drop the coverage they were added for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CASES = [
    desc
    for desc in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors"))
    if "_1xn_" in desc["name"] and "_strided_" in desc["name"]
]


def _rows_in_place(output_c: int, rhs_cols: int) -> bool:
    return output_c >= 4 and ((output_c % 8 >= 4 and rhs_cols >= 80) or rhs_cols >= 224)


def _k_and_cout(desc: dict) -> tuple[int, int]:
    kernel_h, kernel_w, in_c, out_c = desc["filter_shape"]
    return kernel_w * in_c, out_c


@pytest.mark.parametrize("desc", CASES, ids=lambda desc: desc["name"])
def test_case_runs_the_strided_kernel(desc: dict) -> None:
    rhs_cols, output_c = _k_and_cout(desc)
    assert not _rows_in_place(output_c, rhs_cols)
    weight_format = str((desc.get("hint") or {}).get("weight_format", "STANDARD")).upper()
    assert weight_format in {"STANDARD", "ARM_NN_WEIGHT_FORMAT_STANDARD"}
    assert desc["filter_shape"][0] == 1 and desc["input_shape"][1] == 1
    assert desc["strides"] == [1, 1] and desc.get("dilation", [1, 1]) in ([1, 1], 1)
    if "acc16" in desc["name"]:
        assert desc["entry"] == "arm_convolve_1_x_n_f16_acc16"
    else:
        assert desc["hint"]["kernel_variant"] == "direct_1_x_n"
    # At least 16 no-padding output rows reach the strided kernel.
    assert desc["input_shape"][2] - desc["filter_shape"][1] + 1 >= 16


def test_cases_reach_every_strided_block() -> None:
    shapes = {_k_and_cout(desc) for desc in CASES}
    variants = {}
    for desc in CASES:
        variants.setdefault(_k_and_cout(desc), set()).add((desc["padding"], "acc16" in desc["name"]))
    assert all(
        found == {(pad, acc16) for pad in ("SAME", "VALID") for acc16 in (False, True)} for found in variants.values()
    )
    assert {(k <= 32, cout >= 16, cout % 8) for k, cout in shapes} >= {
        (True, True, 0),  # dual block without the fold block
        (False, True, 0),  # dual block with the fold
    }
    assert any(cout % 8 == 0 and cout < 16 and k > 32 for k, cout in shapes)  # 8-row block alone, with the fold
    assert any(cout % 8 >= 4 and k > 32 for k, cout in shapes)  # 4-row sub-block, with the fold
    assert {cout % 4 for _, cout in shapes} >= {1, 3}  # remainder dot, 1 and 3 rows
    assert any(cout < 4 and k >= 224 for k, cout in shapes)  # long K that never routes
    assert any(cout < 4 and k > 256 for k, cout in shapes)  # dot fold over several 256-element blocks
