"""Add and Mul cases that call the s8 row-broadcast entries (`entry:`), bound from the contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract.bind import ContractBindError
from helia_core_tester.generation.entry import EntryError, resolve_entry
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPS = [("Add", "add"), ("Mul", "mul")]


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path, **overrides) -> str:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob("descriptor.yaml") if p.parent.name == name)
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


def _resolve(operator: str, entry: str, dtype: str, **desc) -> dict:
    return resolve_entry(operator, entry, activation_dtype=dtype, weight_dtype=dtype, cpu="cortex-m55",
                         desc={"name": "x", **desc}, extra_roles={"input_1": dtype, "input_2": dtype})


@pytest.mark.parametrize(("operator", "op"), _OPS)
def test_entry_resolves_without_scratch_and_gates_the_case(operator: str, op: str) -> None:
    entry = f"arm_{op}_row_broadcast_s8"
    assert _resolve(operator, entry, "S8") == {
        "kernel_fn": entry,
        "entry_family": "contract",
        "kernel_needs_layout": False,
        "buffer_size_needs_layout": False,
        "kernel_get_buffer_size_fn": None,
        "entry_scratch_bytes": 0,
    }
    assert _required_kernel_symbols(_descriptor(f"{op}_entry_row_broadcast_w2_s8")) == [entry]
    with pytest.raises(EntryError, match="input1_data is 'const int8_t \\*' but the input_1 dtype is S16"):
        _resolve(operator, entry, "S16")
    with pytest.raises(EntryError, match="pass no scratch"):
        _resolve(operator, entry, "S8", entry_scratch="none")


@pytest.mark.parametrize("op", ["add", "mul"])
def test_case_calls_only_the_entry_and_validates_outputs(op: str, tmp_path: Path) -> None:
    source = _source(f"{op}_entry_row_broadcast_w1_s8", tmp_path)

    assert source.count(f"arm_{op}_row_broadcast_s8(") == 1
    assert f"arm_{op}_s8(" not in source
    assert "HELIA_VALIDATE_STATUS(" in source
    assert "HELIA_VALIDATE_OUTPUTS(" in source and "HELIA_GUARD_CHECK_UNTOUCHED(" not in source


@pytest.mark.parametrize("op", ["add", "mul"])
def test_declined_case_expects_no_impl_and_an_untouched_output(op: str, tmp_path: Path) -> None:
    source = _source(f"{op}_entry_row_broadcast_declines_same_shape_s8", tmp_path)

    assert re.search(r"HELIA_GUARD_ARM\(\w+_output, true", source)
    status = re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_NO_IMPL_ERROR", source)
    # The untouched check runs first: a status mismatch returns before anything after it.
    assert status and -1 < source.find("HELIA_GUARD_CHECK_UNTOUCHED(") < status.start()
    assert "HELIA_VALIDATE_OUTPUTS(" not in source


def test_entry_of_another_operator_fails_to_bind(tmp_path: Path) -> None:
    # The Mul pool offers no input shifts or left shift, which arm_add_row_broadcast_s8 takes.
    with pytest.raises((ContractBindError, RuntimeError, ValueError), match="cannot supply"):
        _source("mul_entry_row_broadcast_w2_s8", tmp_path, entry="arm_add_row_broadcast_s8")


def test_other_binary_operators_reject_an_entry_at_load() -> None:
    from helia_core_tester.generation.ops.BasicMathFunctions.sub import OpSub

    desc = {**_descriptor("add_entry_row_broadcast_w2_s8"), "operator": "Sub", "name": "sub_with_entry_s8"}
    with pytest.raises(EntryError, match="Sub does not resolve entries from the kernel contract"):
        OpSub(desc)
