"""Fully connected cases that call arm_fully_connected_per_channel_packed_s8 (`entry:`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.kernel_dispatch import resolve_direct_entry
from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = "arm_fully_connected_per_channel_packed_s8"
_IN_GATE = "fully_connected_entry_packed_k13_c5_b2_bias_s8"
_DECLINED = "fully_connected_entry_packed_declines_filter_offset_k13_c5_s8"


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path, **overrides) -> str:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


def test_packed_entry_resolves_to_its_family_and_gates_the_case() -> None:
    assert resolve_direct_entry("FullyConnected", _ENTRY, "S8", "S8") == {
        "kernel_fn": _ENTRY,
        "kernel_get_buffer_size_fn": f"{_ENTRY}_get_packed_size",
        "entry_family": "fully_connected_packed_s8",
    }
    assert _required_kernel_symbols(_descriptor(_IN_GATE)) == [_ENTRY]


def test_case_packs_once_after_the_gate_and_calls_only_the_entry(tmp_path: Path) -> None:
    source = _source(_IN_GATE, tmp_path)
    body = source[source.index("_test_case_run(void)"):]

    gate = body.index("arm_nn_fc_packed_s8_supported(")
    vector_sum = re.search(
        r"arm_vector_sum_s8\(\s*\w+_kernel_sum,\s*13,\s*5,\s*\w+_weights,\s*\w+_fc_params\.input_offset,\s*0,\s*\w+_biases\s*\)",
        body,
    )
    pack = body.index(f"{_ENTRY}_pack(")
    run = re.search(r"_run\(\w+_input,", body).start()
    assert vector_sum and gate < vector_sum.start() < pack < run
    assert re.search(r"HELIA_VALIDATE_SCALAR_EQ_INT\([^;]*\b1,\s*arm_nn_fc_packed_s8_supported", body, re.S)
    call = re.search(rf"return {_ENTRY}\(\s*&\w+_fc_params,\s*&\w+_input_dims,\s*input,\s*&\w+_filter_dims,"
                     r"\s*\(const int8_t \*\)\w+_buffer,\s*&\w+_output_dims,\s*output\s*\);", source)
    assert call
    assert source.count(f"{_ENTRY}(") == 1
    assert "arm_fully_connected_s8(" not in source and "arm_fully_connected_wrapper_s8(" not in source


def test_declined_case_expects_the_gate_closed_and_an_untouched_output(tmp_path: Path) -> None:
    source = _source(_DECLINED, tmp_path)

    assert re.search(r"HELIA_VALIDATE_SCALAR_EQ_INT\([^;]*\b0,\s*arm_nn_fc_packed_s8_supported", source, re.S)
    assert "HELIA_GUARD_CHECK_UNTOUCHED(" in source
    assert re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_NO_IMPL_ERROR", source)
    assert "HELIA_VALIDATE_OUTPUTS(" not in source


def test_per_tensor_case_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="per-channel quantization only"):
        _source(_IN_GATE, tmp_path, hint={"call_style": "per_tensor", "force_per_tensor": True})
