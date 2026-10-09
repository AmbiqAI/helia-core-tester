"""Convolve cases that call a named ns-cmsis-nn entry (`entry:`) instead of the wrapper."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.entry import EntryError, resolve_entry
from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONVOLVE_ENTRIES = ["arm_convolve_s8_small_cin", "arm_convolve_s8_3x3_c16_s1"]
# arm_convolve_1x1_s8_fast's arguments and its one-argument scratch query.
_CONVOLVE_1X1_ENTRIES = ["arm_convolve_1x1_s8_short_k"]


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path, **overrides) -> str:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.reference.json"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


def _resolve(entry: str, act: str = "S8", **desc) -> dict:
    return resolve_entry("Convolve", entry, activation_dtype=act, weight_dtype="S8", cpu="cortex-m55",
                         desc={"name": "x", **desc})


def test_convolve_s8_entries_resolve_from_the_contract_with_the_family_sizer() -> None:
    for entry in _CONVOLVE_ENTRIES:
        assert _resolve(entry, entry_sizer="arm_convolve_s8_get_buffer_size") == {
            "kernel_fn": entry,
            "kernel_get_buffer_size_fn": "arm_convolve_s8_get_buffer_size",
            "entry_family": "contract",
            "kernel_needs_layout": False,
            "buffer_size_needs_layout": False,
        }


def test_an_entry_without_a_sizer_of_its_own_must_declare_one() -> None:
    with pytest.raises(EntryError, match="declares no arm_convolve_s8_small_cin_get_buffer_size; set entry_sizer"):
        _resolve("arm_convolve_s8_small_cin")


def test_an_entry_of_another_precision_is_rejected() -> None:
    with pytest.raises(EntryError, match="input_data"):
        _resolve("arm_convolve_s8_small_cin", act="S16", entry_sizer="arm_convolve_s8_get_buffer_size")


def test_an_entry_of_another_operator_fails_to_bind(tmp_path: Path) -> None:
    from helia_core_tester.contract.bind import ContractBindError

    with pytest.raises((ContractBindError, RuntimeError, ValueError), match="dw_conv_params|cannot supply"):
        _source("convolve_entry_small_cin3_8x8_k3x3_co16_s8", tmp_path, entry="arm_depthwise_conv_s8_opt_3x3",
                entry_sizer="arm_depthwise_conv_s8_opt_get_buffer_size")


def test_entry_gates_the_case_on_the_checkout() -> None:
    assert _required_kernel_symbols(_descriptor("convolve_entry_small_cin3_8x8_k3x3_co16_s8")) == [
        "arm_convolve_s8_small_cin", "arm_convolve_s8_get_buffer_size"
    ]


def test_entry_case_calls_the_entry_with_arm_convolve_s8_arguments(tmp_path: Path) -> None:
    source = _source("convolve_entry_small_cin3_8x8_k3x3_co16_s8", tmp_path)

    # The call is rendered from the kernel contract, which names each argument in a comment.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    call = re.search(r"return (\w+)\(\s*&\w+_ctx,\s*&\w+_weight_sum_ctx,(.*?)\);", code, re.S)
    assert call and call.group(1) == "arm_convolve_s8_small_cin"
    args = [arg.strip() for arg in call.group(2).split(",")]
    assert args[8] == "NULL" and "NULL, /* upscale_dims */" in source
    assert re.search(r"arm_convolve_s8_get_buffer_size\(\s*&\w+_input_dims,\s*&\w+_filter_dims\s*\)", code)
    assert "arm_convolve_weight_sum(" in source
    assert "arm_convolve_wrapper_s8(" not in source


def test_1x1_entries_resolve_with_arm_convolve_1x1_s8_fast_query() -> None:
    for entry in _CONVOLVE_1X1_ENTRIES:
        assert _resolve(entry, entry_sizer="arm_convolve_1x1_s8_fast_get_buffer_size") == {
            "kernel_fn": entry,
            "kernel_get_buffer_size_fn": "arm_convolve_1x1_s8_fast_get_buffer_size",
            "entry_family": "contract",
            "kernel_needs_layout": False,
            "buffer_size_needs_layout": False,
        }


def test_1x1_entry_case_calls_the_entry_with_arm_convolve_1x1_s8_fast_arguments(tmp_path: Path) -> None:
    name = "convolve_entry_1x1_short_k_c8_2x4_co16_s8"
    assert _required_kernel_symbols(_descriptor(name)) == [
        "arm_convolve_1x1_s8_short_k", "arm_convolve_1x1_s8_fast_get_buffer_size"
    ]
    source = _source(name, tmp_path)

    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    call = re.search(r"return (\w+)\(\s*&\w+_ctx,\s*&\w+_weight_sum_ctx,(.*?)\);", code, re.S)
    assert call and call.group(1) == "arm_convolve_1x1_s8_short_k"
    # arm_convolve_1x1_s8_fast's arguments: no upscale_dims between the bias and the output dims.
    assert re.search(r"_biases,\s*&\w+_output_dims,", call.group(2))
    assert "upscale_dims" not in source
    assert re.search(r"arm_convolve_1x1_s8_fast_get_buffer_size\(\s*&\w+_input_dims\s*\)", code)
    assert "arm_convolve_weight_sum(" in source
    assert "arm_convolve_wrapper_s8(" not in source


@pytest.mark.parametrize(
    "name", ["convolve_entry_3x3_c16_declines_stride2_8x8_s8", "convolve_entry_1x1_short_k_declines_c17_2x4_s8"]
)
def test_declined_case_checks_the_status_and_an_untouched_output(name: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)

    assert "HELIA_GUARD_CHECK_UNTOUCHED(" in source
    assert re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_NO_IMPL_ERROR", source)
    assert "HELIA_VALIDATE_OUTPUTS(" not in source


@pytest.mark.parametrize("name", ["convolve_small_cin3_8x8_k3x3_co16_s8", "convolve_float_default_f16"])
def test_cases_without_an_entry_render_no_entry_code(name: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)

    assert "upscale_dims" not in source
    assert "HELIA_CMSIS_NN_INT_AUTOVECTORIZE" not in source
    assert "HELIA_GUARD_CHECK_UNTOUCHED(" not in source
    assert "arm_convolve_s8_get_buffer_size(" not in source


def test_entry_with_a_kernel_variant_hint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported with a kernel_variant hint"):
        _source("convolve_entry_small_cin3_8x8_k3x3_co16_s8", tmp_path, hint={"kernel_variant": "wrapper"})


def test_in_gate_entry_case_expects_a_decline_on_an_autovectorize_build(tmp_path: Path) -> None:
    source = _source("convolve_entry_small_cin3_8x8_k3x3_co16_s8", tmp_path)

    declined, full = re.search(
        r"#if defined\(HELIA_CMSIS_NN_INT_AUTOVECTORIZE\)(.*?)#else(.*?)#endif",
        source[source.index("_test_case_run(void)"):],
        re.S,
    ).groups()
    assert "true /* the entry declines on this build: poison */" in declined
    declined, full = re.findall(
        r"#if defined\(HELIA_CMSIS_NN_INT_AUTOVECTORIZE\)(.*?)#else(.*?)#endif", source, re.S
    )[-1]
    assert "ARM_CMSIS_NN_NO_IMPL_ERROR" in declined and "HELIA_GUARD_CHECK_UNTOUCHED(" in declined
    assert "HELIA_VALIDATE_OUTPUTS(" in full and "HELIA_VALIDATE_STATUS(" in full
