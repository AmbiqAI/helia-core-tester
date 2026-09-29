"""FP16 cases that call a named ns-cmsis-nn entry (`entry:`): _acc16 and layout-free entries."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.kernel_dispatch import DIRECT_ENTRIES, resolve_direct_entry
from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FP16_ENTRIES = {name: spec for name, spec in DIRECT_ENTRIES.items() if spec.activation_dtype == "FP16"}


def _descriptors() -> dict:
    return {d["name"]: d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors"))}


def _source(name: str, tmp_path: Path) -> str:
    generate_test(_descriptors()[name], str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


def _call(source: str, fn: str) -> str:
    match = re.search(rf"\b{fn}\((.*?)\);", source, re.S)
    assert match, f"{fn} is not called"
    return match.group(1)


def test_every_fp16_entry_has_a_case() -> None:
    called = {d.get("entry") for d in _descriptors().values()}

    assert set(_FP16_ENTRIES) <= called


def test_fp16_entries_resolve_with_their_layout_flags() -> None:
    resolved = resolve_direct_entry("Convolve", "arm_convolve_nhwc_f16_acc16", "FP16", "FP16")

    assert resolved["kernel_needs_layout"] is False
    assert resolved["buffer_size_needs_layout"] is True
    with pytest.raises(ValueError, match="fp16 convolve entry"):
        resolve_direct_entry("Convolve", "arm_convolve_nhwc_f16_acc16", "S8", "S8")


def test_acc16_case_is_gated_on_the_checkout() -> None:
    assert _required_kernel_symbols(_descriptors()["fully_connected_float_entry_nhwc_acc16_k24_n17_f16"]) == [
        "arm_fully_connected_nhwc_f16_acc16"
    ]


@pytest.mark.parametrize(
    ("name", "entry", "sizer", "call_has_layout", "sizer_has_layout"),
    [
        ("convolve_float_entry_acc16_8x8_k3x3_f16", "arm_convolve_f16_acc16", "arm_convolve_f16_get_buffer_size", True, True),
        ("convolve_float_entry_nhwc_8x8_k3x3_f16", "arm_convolve_nhwc_f16", "arm_convolve_f16_get_buffer_size", False, True),
        (
            "convolve_float_entry_wrapper_acc16_6x6_k3x3_f16",
            "arm_convolve_wrapper_f16_acc16",
            "arm_convolve_wrapper_f16_get_buffer_size",
            False,
            False,
        ),
        (
            "fully_connected_float_entry_nhwc_acc16_k24_n17_f16",
            "arm_fully_connected_nhwc_f16_acc16",
            "arm_fully_connected_f16_get_buffer_size",
            False,
            True,
        ),
        (
            "depthwise_conv_float_entry_acc16_chmult2_f16",
            "arm_depthwise_conv_f16_acc16",
            "arm_depthwise_conv_f16_get_buffer_size",
            True,
            True,
        ),
        (
            "depthwise_conv_float_entry_nhwc_5x5_c4_f16",
            "arm_depthwise_nhwc_conv_f16",
            "arm_depthwise_conv_f16_get_buffer_size",
            False,
            True,
        ),
    ],
)
def test_entry_case_calls_the_entry_with_its_layout_arguments(
    name, entry, sizer, call_has_layout, sizer_has_layout, tmp_path: Path
) -> None:
    source = _source(name, tmp_path)

    assert ("ARM_NN_LAYOUT_" in _call(source, entry)) is call_has_layout
    assert ("ARM_NN_LAYOUT_" in _call(source, sizer)) is sizer_has_layout


@pytest.mark.parametrize(
    "name", ["fully_connected_float_k24_n17_f16", "depthwise_conv_float_3x3_valid_stride1_f16", "convolve_float_patch_gemm_f16"]
)
def test_cases_without_an_entry_keep_the_layout_argument(name: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)

    kernel = re.search(r"kernel_status = (\w+)\(|return (arm_\w+_f16)\(", source)
    fn = kernel.group(1) or kernel.group(2)
    assert "ARM_NN_LAYOUT_" in _call(source, fn)


@pytest.mark.parametrize(
    ("name", "entry"),
    [
        ("convolve_fault_entry_acc16_invalid_layout_f16", "arm_convolve_f16_acc16"),
        ("convolve_fault_entry_1x1_acc16_invalid_layout_f16", "arm_convolve_1x1_f16_acc16"),
        ("convolve_fault_entry_1xn_acc16_invalid_layout_f16", "arm_convolve_1_x_n_f16_acc16"),
        ("depthwise_conv_fault_entry_acc16_invalid_layout_f16", "arm_depthwise_conv_f16_acc16"),
        ("fully_connected_fault_entry_acc16_invalid_layout_f16", "arm_fully_connected_f16_acc16"),
    ],
)
def test_invalid_layout_fault_calls_the_entry_with_a_bad_layout(name: str, entry: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)

    assert "(arm_nn_tensor_layout)(ARM_NN_LAYOUT_NHWC + 1)" in _call(source, entry)
    assert "ARM_CMSIS_NN_ARG_ERROR" in source


@pytest.mark.parametrize(
    ("name", "fault"),
    [
        ("convolve_float_entry_nhwc_8x8_k3x3_f16", "invalid_layout"),
        ("convolve_float_entry_acc16_8x8_k3x3_f16", "null_input"),
    ],
)
def test_other_entry_faults_are_rejected(name: str, fault: str, tmp_path: Path) -> None:
    desc = {**_descriptors()[name], "fault": fault, "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}

    with pytest.raises(ValueError, match="supports only fault: invalid_layout"):
        generate_test(desc, str(tmp_path))
