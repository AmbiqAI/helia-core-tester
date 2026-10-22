"""The generation driver's contract with reference-backed operators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import helia_core_tester.generation.test_ops as generation_module
from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.reference.call import ReferenceCall


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(find_descriptors_dir())) if d["name"] == name)


@pytest.mark.parametrize("name", ["add_broadcast_batch_s8", "add_default_s16", "add_float_default_f16"])
def test_reference_case_writes_its_record_and_no_model(tmp_path: Path, name: str) -> None:
    desc = _descriptor(name)
    generation_module.generate_test(desc, str(tmp_path), seed=11, run_seed=5)
    case = tmp_path / desc["_family"] / name
    assert not list(case.glob("*.tflite"))
    record = json.loads((case / f"{name}.reference.json").read_text())
    assert record["entry"].startswith("add_") and record["seeds"] == {"run_seed": 5, "case_seed": 11}
    assert record["library_key"]
    sidecar = json.loads(next(case.glob("*.sidecar.json")).read_text())
    assert sidecar["reference"]["entry"] == record["entry"]
    assert sidecar["reference"]["params"] == record["params"]
    entry = generation_module._manifest_entry({**desc}, test_dir=case, generated_tests_dir=tmp_path,
                                              cpu="cortex-m55", reused=False)
    assert entry["reference"] == str(case / f"{name}.reference.json")


@pytest.mark.parametrize("name", ["prelu_arg_error_output_mismatch_s8", "prelu_arg_error_output_mismatch_s16"])
def test_a_status_only_case_records_no_reference(tmp_path: Path, name: str) -> None:
    desc = _descriptor(name)
    generation_module.generate_test(desc, str(tmp_path), seed=11, run_seed=5)
    case = tmp_path / desc["_family"] / name
    assert not list(case.glob("*.tflite")) and not list(case.glob("*.reference.json"))
    source = next(case.glob("*.c")).read_text()
    assert "ARM_CMSIS_NN_ARG_ERROR" in source


class _NoCallOp(OperationBase):
    def uses_reference(self) -> bool:
        return True

    def generate_c_files(self, output_dir: Path) -> None:
        (output_dir / "includes").mkdir(exist_ok=True)


class _NotImplementedOp(_NoCallOp):
    def generate_c_files(self, output_dir: Path) -> None:
        raise NotImplementedError("not yet")


class _UndeclaredOp(OperationBase):
    """Neither reference-backed nor status-only nor self-checking, and records nothing."""

    def generate_c_files(self, output_dir: Path) -> None:
        (output_dir / "includes").mkdir(exist_ok=True)


class _StatusOnlyOp(_UndeclaredOp):
    def status_only(self) -> bool:
        return True


class _SelfCheckingOp(_UndeclaredOp):
    def self_checking(self) -> bool:
        return True


@pytest.mark.parametrize("op_class, match", [(_NoCallOp, "recorded no ReferenceCall"),
                                             (_UndeclaredOp, "recorded no ReferenceCall"),
                                             (_NotImplementedOp, "not yet")])
def test_a_case_without_a_golden_fails_generation(tmp_path, monkeypatch, op_class, match) -> None:
    desc = {**_descriptor("add_default_s8"), "name": "probe_case"}
    monkeypatch.setattr(generation_module, "get_op_map", lambda: {desc["operator"]: op_class})
    failures = []
    with pytest.raises(RuntimeError, match=match):
        generation_module.generate_test(desc, str(tmp_path), seed=1, generation_failures=failures)
    assert failures and failures[0]["stage"] == "c_files"
    assert not list((tmp_path / desc["_family"] / "probe_case").glob("*.reference.json"))


@pytest.mark.parametrize("op_class", [_StatusOnlyOp, _SelfCheckingOp])
def test_only_a_status_only_or_self_checking_case_may_record_no_golden(tmp_path, monkeypatch, op_class) -> None:
    desc = {**_descriptor("add_default_s8"), "name": "probe_case"}
    monkeypatch.setattr(generation_module, "get_op_map", lambda: {desc["operator"]: op_class})
    generation_module.generate_test(desc, str(tmp_path), seed=1)
    assert not list((tmp_path / desc["_family"] / "probe_case").glob("*.reference.json"))


def test_every_descriptor_is_reference_backed_status_only_or_self_checking() -> None:
    """The driver refuses a case that recorded no golden, so every registered operator must say,
    per descriptor, where its verdict comes from."""
    from helia_core_tester.generation.ops import get_op_map

    op_map = get_op_map()
    undeclared = []
    for desc in load_all_descriptors(str(find_descriptors_dir())):
        op = op_map[desc["operator"]](desc, seed=1)
        if not (op.uses_reference() or op.status_only() or op.self_checking()):
            undeclared.append(desc["name"])
    assert undeclared == []


def test_a_golden_can_be_recorded_once_per_case() -> None:
    op = _NoCallOp({"name": "x", "operator": "Add"}, seed=1)
    call = ReferenceCall("add_f32", {"activation_min": -1.0, "activation_max": 1.0},
                         {"input1": np.zeros(2, np.float32), "input2": np.ones(2, np.float32)}, {"output": (2,)})
    np.testing.assert_array_equal(op.reference_golden(call), [1.0, 1.0])
    assert op.reference is call
    with pytest.raises(RuntimeError, match="already recorded"):
        op.reference_golden(call)
    with pytest.raises(TypeError, match="ReferenceCall"):
        _NoCallOp({"name": "y"}, seed=1).reference_golden("add_f32")
    np.testing.assert_array_equal(op.reference_probe([np.full(2, 5, np.float32), np.zeros(2, np.float32)]),
                                  [1.0, 1.0])
    with pytest.raises(ValueError, match="operands"):
        op.reference_probe([np.zeros(2, np.float32)])
    with pytest.raises(RuntimeError, match="needs a recorded"):
        _NoCallOp({"name": "z"}, seed=1).reference_probe([])
