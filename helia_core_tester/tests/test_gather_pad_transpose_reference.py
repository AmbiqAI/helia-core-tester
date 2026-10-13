"""Gather, GatherND, Pad, MirrorPad and Transpose draw their goldens from the C reference.

Each descriptor case is generated through its op class and the recorded golden is checked
against an independent numpy model of the same inputs; ARG_ERROR cases record nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.GatherFunctions.gather import OpGather
from helia_core_tester.generation.ops.GatherFunctions.gather_nd import OpGatherND
from helia_core_tester.generation.ops.PadFunctions.mirror_pad import OpMirrorPad
from helia_core_tester.generation.ops.PadFunctions.pad import OpPad
from helia_core_tester.generation.ops.TransposeFunctions.transpose import OpTranspose

OPS = {"Gather": OpGather, "GatherND": OpGatherND, "Pad": OpPad, "MirrorPad": OpMirrorPad, "Transpose": OpTranspose}
_OP_FILES = ("GatherFunctions/gather.py", "GatherFunctions/gather_nd.py", "PadFunctions/pad.py",
             "PadFunctions/mirror_pad.py", "TransposeFunctions/transpose.py")


def _cases() -> list[dict]:
    descs = load_all_descriptors(str(find_descriptors_dir()))
    return sorted((d for d in descs if d.get("operator") in OPS), key=lambda d: d["name"])


CASES = _cases()


def _numpy_model(call) -> np.ndarray:
    x = call.inputs
    if call.entry == "gather":
        return np.take(x["input"], x["indices"], axis=int(call.params["axis"]))
    if call.entry == "gather_nd":
        idx = x["indices"]
        depth = idx.shape[-1]
        flat = [x["input"][tuple(i)] for i in idx.reshape(-1, depth)]
        return np.array(flat, dtype=x["input"].dtype).reshape(idx.shape[:-1] + x["input"].shape[depth:])
    if call.entry == "pad":
        mode = int(call.params["mode"])
        pads = [tuple(int(v) for v in p) for p in x["paddings"]]
        if mode == 0:
            return np.pad(x["input"], pads, mode="constant", constant_values=x["pad_value"][0])
        return np.pad(x["input"], pads, mode="reflect" if mode == 1 else "symmetric")
    if call.entry == "transpose":
        return np.transpose(x["input"], [int(p) for p in x["perm"]])
    raise AssertionError(f"no numpy model for {call.entry}")


@pytest.mark.parametrize("desc", CASES, ids=[d["name"] for d in CASES])
def test_golden_matches_an_independent_numpy_model(desc: dict, tmp_path: Path) -> None:
    op = OPS[desc["operator"]](desc, seed=1234, target_cpu="cortex-m55")
    assert op.needs_keras_model() is False and op.allow_no_tflite()
    op.generate_c_files(tmp_path)
    if desc.get("expected_status", "ARM_CMSIS_NN_SUCCESS") != "ARM_CMSIS_NN_SUCCESS":
        assert op.status_only() and not op.uses_reference()
        assert op.reference is None
        assert "ARM_CMSIS_NN_ARG_ERROR" in "".join(p.read_text() for p in tmp_path.rglob("*.c"))
        return
    assert op.uses_reference() and not op.status_only()
    expected = _numpy_model(op.reference)
    got = op.reference.output()
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_suite_covers_every_operator_and_both_status_kinds() -> None:
    assert {d["operator"] for d in CASES} == set(OPS)
    arg_errors = {d["name"] for d in CASES if d.get("expected_status") == "ARM_CMSIS_NN_ARG_ERROR"}
    assert {"gather_nd_oob_index_s8", "gather_nd_negative_index_s16",
            "transpose_time_batch_invalid_perm_s8", "transpose_time_batch_invalid_perm_s16"} <= arg_errors


def test_mirror_pad_modes_map_to_reflect_and_symmetric(tmp_path: Path) -> None:
    base = {"operator": "MirrorPad", "activation_dtype": "S8", "weight_dtype": "S8", "activation": "NONE",
            "input_shape": [1, 3, 3, 1], "paddings": [[0, 0], [1, 1], [1, 1], [0, 0]]}
    for mode, ref_mode in (("reflect", 1), ("symmetric", 2)):
        op = OpMirrorPad({**base, "name": f"mp_{mode}", "mode": mode}, seed=3, target_cpu="cortex-m55")
        op.generate_c_files(tmp_path / mode)
        assert op.reference.params == {"mode": ref_mode}


def test_pad_rejects_a_rank_other_than_four(tmp_path: Path) -> None:
    desc = {"operator": "Pad", "name": "pad_rank3", "activation_dtype": "S8", "weight_dtype": "S8",
            "activation": "NONE", "resolved_tensor_dtypes": {"input": "S8", "output": "S8"},
            "input_shape": [2, 3, 4], "paddings": [[0, 0], [1, 1], [1, 1]]}
    op = OpPad(desc, seed=1, target_cpu="cortex-m55")
    with pytest.raises(ValueError, match="rank 4"):
        op.generate_c_files(tmp_path)


def test_gather_out_of_range_index_is_rejected_by_the_reference() -> None:
    from helia_core_tester.generation.reference.call import ReferenceCall

    call = ReferenceCall("gather", {"axis": 0}, {"input": np.arange(6, dtype=np.int8).reshape(3, 2),
                                                 "indices": np.array([3], dtype=np.int32)}, {"output": (1, 2)})
    with pytest.raises(Exception):
        call.output()


def test_op_files_carry_no_model_path() -> None:
    root = Path(__file__).resolve().parents[1] / "generation" / "ops"
    for rel in _OP_FILES:
        text = (root / rel).read_text().lower()
        for banned in ("litert", "tflite_path", "tensorflow", "keras", "interpreter"):
            assert banned not in text, f"{rel} mentions {banned}"
