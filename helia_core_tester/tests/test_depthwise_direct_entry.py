"""Depthwise cases that call a named ns-cmsis-nn entry (`entry:`) instead of the wrapper."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.kernel_dispatch import (
    DEPTHWISE_CONV_S8_DIRECT_ENTRIES,
    DEPTHWISE_CONV_S8_PLANAR_RULE,
    resolve_depthwise_conv_entry,
)
from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path, **overrides) -> str:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


def test_entries_resolve_to_themselves_with_the_opt_scratch_query() -> None:
    for entry in DEPTHWISE_CONV_S8_DIRECT_ENTRIES:
        assert resolve_depthwise_conv_entry(entry, "S8", "S8") == {
            "kernel_fn": entry,
            "kernel_get_buffer_size_fn": "arm_depthwise_conv_s8_opt_get_buffer_size",
        }


@pytest.mark.parametrize(
    ("entry", "act", "weight", "message"),
    [
        ("arm_depthwise_conv_s8_opt_3x4", "S8", "S8", "Unknown DepthwiseConv entry"),
        ("arm_depthwise_conv_s8_opt_3x3", "S16", "S8", "s8 depthwise entry"),
    ],
)
def test_unknown_or_mismatched_entries_are_rejected(entry, act, weight, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_depthwise_conv_entry(entry, act, weight)


def test_entry_and_planar_rule_gate_the_case_on_the_checkout() -> None:
    desc = _descriptor("depthwise_conv_entry_planar_48x48_c8_s8")

    assert _required_kernel_symbols(desc) == ["arm_depthwise_conv_s8_opt_planar", DEPTHWISE_CONV_S8_PLANAR_RULE]
    assert _required_kernel_symbols(_descriptor("depthwise_conv_dilated_1d_k7_d2_c24_s8")) == []


def test_3x3_entry_gates_on_its_sizer() -> None:
    assert _required_kernel_symbols(_descriptor("depthwise_conv_entry_3x3_25x5_c64_s8")) == [
        "arm_depthwise_conv_s8_opt_3x3",
        "arm_depthwise_conv_s8_opt_3x3_get_buffer_size",
    ]


def test_entry_case_calls_the_entry_with_weight_sums_and_its_scratch_query(tmp_path: Path) -> None:
    name = "depthwise_conv_entry_3x3_25x5_c64_s8"
    source = _source(name, tmp_path)

    call = re.search(r"kernel_status = (\w+)\(\s*&\w+_ctx,\s*&(\w+)_weight_sum_ctx,", source)
    assert call and call.group(1) == "arm_depthwise_conv_s8_opt_3x3"
    assert re.search(r"arm_depthwise_conv_s8_opt_get_buffer_size\(\s*&\w+_input_dims,\s*&\w+_filter_dims\s*\)", source)
    assert "arm_depthwise_conv_wrapper_s8(" not in source
    assert "HELIA_VALIDATE_OUTPUTS(" in source


@pytest.mark.parametrize(
    ("name", "calls_3x3_sizer"),
    [
        ("depthwise_conv_entry_3x3_28x28_c64_stride2_s8", True),
        ("depthwise_conv_entry_3x3_c64_s1_14x28_s8", True),
        ("depthwise_conv_entry_channelwise_25x5_c64_s8", False),
    ],
)
def test_3x3_entries_also_query_their_own_size(tmp_path: Path, name: str, calls_3x3_sizer: bool) -> None:
    source = _source(name, tmp_path)

    assert re.search(r"arm_depthwise_conv_s8_opt_get_buffer_size\(", source)
    assert bool(re.search(r"arm_depthwise_conv_s8_opt_3x3_get_buffer_size\(\s*&\w+_input_dims\s*\)", source)) == calls_3x3_sizer


def test_declined_case_checks_the_status_and_an_untouched_output(tmp_path: Path) -> None:
    source = _source("depthwise_conv_entry_3x3_declines_c62_s8", tmp_path)

    assert "HELIA_GUARD_CHECK_UNTOUCHED(" in source
    assert re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_NO_IMPL_ERROR", source)
    assert "HELIA_VALIDATE_OUTPUTS(" not in source


def test_planar_case_checks_the_rule(tmp_path: Path) -> None:
    source = _source("depthwise_conv_entry_planar_declines_25x5_c64_s8", tmp_path)

    assert re.search(rf"{DEPTHWISE_CONV_S8_PLANAR_RULE}\(", source)
    assert "planar_supported != 0" in source


def test_entry_with_a_fault_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supports only fault: invalid_layout"):
        _source(
            "depthwise_conv_entry_3x3_25x5_c64_s8",
            tmp_path,
            fault="null_ctx_buf",
            expected_status="ARM_CMSIS_NN_ARG_ERROR",
        )


@pytest.mark.parametrize("name", ["depthwise_conv_float_2x5_f16", "depthwise_conv_dilated_1d_k7_d2_c24_s8"])
def test_cases_without_an_entry_render_no_entry_code(name: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)

    assert DEPTHWISE_CONV_S8_PLANAR_RULE not in source
    assert "planar_supported" not in source
    assert "HELIA_GUARD_CHECK_UNTOUCHED(" not in source
