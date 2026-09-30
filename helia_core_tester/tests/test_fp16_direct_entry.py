"""FP16 cases that call a named ns-cmsis-nn entry (`entry:`): _acc16 and layout-free entries."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.contract.bind import takes
from helia_core_tester.contract.render import load_current_contracts
from helia_core_tester.generation.entry import EntryError, resolve_entry
from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test
from helia_core_tester.generation.entry import CONTRACT_BOUND_OPERATORS
from helia_core_tester.generation.utils.temp_sizer_probe import probe_header_symbols

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_every_fp16_entry_case_names_its_family_sizer() -> None:
    fp16 = {name: d for name, d in _descriptors().items() if d.get("entry") and str(d["entry"]).endswith(("f16", "acc16"))}
    assert len(fp16) >= 17
    for name, desc in fp16.items():
        assert desc.get("entry_sizer", "").endswith("_f16_get_buffer_size"), f"{name} declares no f16 family sizer"


def test_fp16_entries_take_their_layout_from_the_prototype() -> None:
    contracts = load_current_contracts()
    resolved = resolve_entry("Convolve", "arm_convolve_nhwc_f16", activation_dtype="FP16", weight_dtype="FP16",
                             cpu="cortex-m55", desc={"name": "x", "entry_sizer": "arm_convolve_f16_get_buffer_size"},
                             contracts=contracts)
    assert resolved == {"kernel_fn": "arm_convolve_nhwc_f16", "entry_family": "contract",
                        "kernel_get_buffer_size_fn": "arm_convolve_f16_get_buffer_size"}
    # The nhwc entry takes no layout while its family's sizer does: the harness binds each by name.
    assert not takes(contracts.require("arm_convolve_nhwc_f16"), "layout")
    assert takes(contracts.require("arm_convolve_f16_get_buffer_size"), "layout")
    with pytest.raises(EntryError, match="input_data"):
        resolve_entry("Convolve", "arm_convolve_nhwc_f16", activation_dtype="S8", weight_dtype="S8",
                      cpu="cortex-m55", desc={"name": "x", "entry_sizer": "arm_convolve_f16_get_buffer_size"},
                      contracts=contracts)


def test_an_entry_the_checkout_lacks_gates_the_case_before_it_resolves() -> None:
    # The acc16 entries are not in the current export: the case is skipped by its required
    # symbols and never reaches resolve_entry, which would refuse the unknown kernel.
    desc = _descriptors()["convolve_float_entry_acc16_8x8_k3x3_f16"]
    assert _required_kernel_symbols(desc) == ["arm_convolve_f16_acc16", "arm_convolve_f16_get_buffer_size"]
    with pytest.raises(EntryError, match="is not a public function"):
        resolve_entry("Convolve", desc["entry"], activation_dtype="FP16", weight_dtype="FP16", cpu="cortex-m55",
                      desc=desc, contracts=load_current_contracts())


def test_acc16_case_is_gated_on_the_checkout() -> None:
    assert _required_kernel_symbols(_descriptors()["fully_connected_float_entry_nhwc_acc16_k24_n17_f16"]) == [
        "arm_fully_connected_nhwc_f16_acc16", "arm_fully_connected_f16_get_buffer_size"
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
    # The pipeline skips a case whose entry this ns-cmsis-nn checkout does not declare, and a
    # contract-bound operator's template renders its call from the contract, which cannot name one.
    desc = _descriptors()[name]
    required = _required_kernel_symbols(desc)
    if desc.get("operator") in CONTRACT_BOUND_OPERATORS and required and not probe_header_symbols(required):
        pytest.skip(f"{name}: this ns-cmsis-nn checkout does not declare {required}")
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
