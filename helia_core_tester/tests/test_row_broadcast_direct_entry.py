"""Add and Mul cases that call the s8 row-broadcast entries (`entry:`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.kernel_dispatch import resolve_direct_entry
from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPS = [("Add", "add"), ("Mul", "mul")]


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path, **overrides) -> str:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


@pytest.mark.parametrize(("operator", "op"), _OPS)
def test_entry_resolves_for_its_operator_and_gates_the_case(operator: str, op: str) -> None:
    entry = f"arm_{op}_row_broadcast_s8"
    assert resolve_direct_entry(operator, entry, "S8", "S8")["kernel_fn"] == entry
    assert _required_kernel_symbols(_descriptor(f"{op}_entry_row_broadcast_w2_s8")) == [entry]
    with pytest.raises(ValueError, match="s8"):
        resolve_direct_entry(operator, entry, "S16", "S8")


@pytest.mark.parametrize("op", ["add", "mul"])
def test_case_calls_only_the_entry_and_validates_outputs(op: str, tmp_path: Path) -> None:
    source = _source(f"{op}_entry_row_broadcast_w1_s8", tmp_path)

    assert source.count(f"arm_{op}_row_broadcast_s8(") == 1
    assert f"arm_{op}_s8(" not in source
    assert re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_SUCCESS", source)
    assert "HELIA_VALIDATE_OUTPUTS(" in source and "HELIA_GUARD_CHECK_UNTOUCHED(" not in source


@pytest.mark.parametrize("op", ["add", "mul"])
def test_declined_case_expects_no_impl_and_an_untouched_output(op: str, tmp_path: Path) -> None:
    source = _source(f"{op}_entry_row_broadcast_declines_same_shape_s8", tmp_path)

    assert re.search(r"HELIA_GUARD_ARM\(\w+_output, true", source)
    status = re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_NO_IMPL_ERROR", source)
    # The untouched check runs first: a status mismatch returns before anything after it.
    assert status and -1 < source.find("HELIA_GUARD_CHECK_UNTOUCHED(") < status.start()
    assert "HELIA_VALIDATE_OUTPUTS(" not in source


def test_entry_of_another_operator_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown Mul entry"):
        _source("mul_entry_row_broadcast_w2_s8", tmp_path, entry="arm_add_row_broadcast_s8")


def test_other_binary_operators_reject_an_entry_at_load() -> None:
    from helia_core_tester.generation.ops.BasicMathFunctions.sub import OpSub

    desc = {**_descriptor("add_entry_row_broadcast_w2_s8"), "operator": "Sub", "name": "sub_with_entry_s8"}
    with pytest.raises(ValueError, match="Unknown Sub entry"):
        OpSub(desc)

