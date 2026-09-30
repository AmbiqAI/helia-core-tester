"""Depthwise cases that call a named ns-cmsis-nn entry (`entry:`) instead of the wrapper."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.entry import EntryError, resolve_entry
from helia_core_tester.generation.kernel_dispatch import DEPTHWISE_CONV_S8_PLANAR_RULE

from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPTHWISE_CONV_S8_ENTRIES = ("arm_depthwise_conv_s8_opt_3x3", "arm_depthwise_conv_s8_opt_3x3_c64_s1",
                             "arm_depthwise_conv_s8_opt_planar", "arm_depthwise_conv_s8_opt_channelwise")


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path, **overrides) -> str:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


def _resolve(entry: str, act: str = "S8", weight: str = "S8", **desc) -> dict:
    return resolve_entry("DepthwiseConv", entry, activation_dtype=act, weight_dtype=weight, cpu="cortex-m55",
                         desc={"name": "x", **desc})


def test_entries_resolve_from_the_contract_with_the_opt_scratch_query() -> None:
    for entry in DEPTHWISE_CONV_S8_ENTRIES:
        assert _resolve(entry, entry_sizer="arm_depthwise_conv_s8_opt_get_buffer_size") == {
            "kernel_fn": entry,
            "kernel_get_buffer_size_fn": "arm_depthwise_conv_s8_opt_get_buffer_size",
            "entry_family": "contract",
        }


@pytest.mark.parametrize(
    ("entry", "act", "message"),
    [
        ("arm_depthwise_conv_s8_opt_3x4", "S8", "is not a public function"),
        ("arm_depthwise_conv_s8_opt_3x3", "S16", "input_data"),
    ],
)
def test_unknown_or_mismatched_entries_are_rejected(entry, act, message) -> None:
    with pytest.raises(EntryError, match=message):
        _resolve(entry, act, entry_sizer="arm_depthwise_conv_s8_opt_get_buffer_size")


def test_entry_and_planar_rule_gate_the_case_on_the_checkout() -> None:
    desc = _descriptor("depthwise_conv_entry_planar_48x48_c8_s8")

    assert _required_kernel_symbols(desc) == ["arm_depthwise_conv_s8_opt_planar", "arm_depthwise_conv_s8_opt_get_buffer_size",
                                              DEPTHWISE_CONV_S8_PLANAR_RULE]
    assert _required_kernel_symbols(_descriptor("depthwise_conv_dilated_1d_k7_d2_c24_s8")) == []


def test_entry_case_calls_the_entry_with_weight_sums_and_its_scratch_query(tmp_path: Path) -> None:
    name = "depthwise_conv_entry_3x3_25x5_c64_s8"
    source = _source(name, tmp_path)

    # The calls are bound from the kernel contract, which names each argument in a comment.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    call = re.search(r"return (\w+)\(\s*&\w+_ctx,\s*&(\w+)_weight_sum_ctx,", code)
    assert call and call.group(1) == "arm_depthwise_conv_s8_opt_3x3"
    assert re.search(r"arm_depthwise_conv_s8_opt_get_buffer_size\(\s*&\w+_input_dims,\s*&\w+_filter_dims\s*\)", code)
    assert "arm_depthwise_conv_wrapper_s8(" not in source
    assert "HELIA_VALIDATE_OUTPUTS(" in source


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
    with pytest.raises(ValueError, match="not supported with fault"):
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
