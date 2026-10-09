"""StridedSlice, Tile, BroadcastTo, ScatterNd, SelectV2, Where, ReverseSequence and DynamicUpdateSlice
draw their goldens from the C reference.

Each descriptor case is generated through its op class and the recorded golden is checked against an
independent numpy model of the same inputs; ARG_ERROR cases record nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.BroadcastFunctions.broadcast_to import OpBroadcastTo
from helia_core_tester.generation.ops.DynamicUpdateSliceFunctions.dynamic_update_slice import OpDynamicUpdateSlice
from helia_core_tester.generation.ops.ReverseSequenceFunctions.reverse_sequence import OpReverseSequence
from helia_core_tester.generation.ops.ScatterFunctions.scatter_nd import OpScatterNd
from helia_core_tester.generation.ops.SelectFunctions.select_v2 import OpSelectV2
from helia_core_tester.generation.ops.SelectFunctions.where import OpWhere
from helia_core_tester.generation.ops.StridedSliceFunctions.strided_slice import OpStridedSlice
from helia_core_tester.generation.ops.TileFunctions.tile import OpTile

OPS = {"StridedSlice": OpStridedSlice, "Tile": OpTile, "BroadcastTo": OpBroadcastTo, "ScatterNd": OpScatterNd,
       "SelectV2": OpSelectV2, "Where": OpWhere, "ReverseSequence": OpReverseSequence,
       "DynamicUpdateSlice": OpDynamicUpdateSlice}


def _cases() -> list[dict]:
    descs = load_all_descriptors(str(find_descriptors_dir()))
    return sorted((d for d in descs if d.get("operator") in OPS), key=lambda d: d["name"])


CASES = _cases()


def _strided_slice(x, begin, end, strides, shrink):
    index = []
    for d, size in enumerate(x.shape):
        b = int(begin[d])
        if shrink & (1 << d):
            index.append(b + size if b < 0 else b)
        else:
            index.append(slice(b, int(end[d]), int(strides[d])))
    return x[tuple(index)]


def _numpy_model(call) -> dict[str, np.ndarray]:
    x, p = call.inputs, call.params
    if call.entry == "strided_slice":
        out = _strided_slice(x["input"], x["begin"], x["end"], x["strides"], int(p["shrink_axis_mask"]))
    elif call.entry == "tile":
        out = np.tile(x["input"], [int(m) for m in x["multiples"]])
    elif call.entry == "broadcast_to":
        out = np.broadcast_to(x["input"], call.output_shapes["output"])
    elif call.entry == "scatter_nd":
        upd = x["updates"]
        out = np.zeros(call.output_shapes["output"], dtype=upd.dtype)
        depth = x["indices"].shape[-1]
        flat_upd = upd.reshape((-1,) + out.shape[depth:])
        for i, idx in enumerate(x["indices"].reshape(-1, depth)):
            out[tuple(idx)] += flat_upd[i]
    elif call.entry == "select_v2":
        out = np.where(x["condition"].astype(bool), x["x"], x["y"])
    elif call.entry == "where":
        coords = np.argwhere(x["condition"]).astype(np.int64)
        full = np.zeros(call.output_shapes["coords"], dtype=np.int64)
        full[: len(coords)] = coords
        return {"coords": full, "count": np.array([len(coords)], dtype=np.int32)}
    elif call.entry == "reverse_sequence":
        seq_dim, batch_dim = int(p["seq_dim"]), int(p["batch_dim"])
        out = x["input"].copy()
        for b, n in enumerate(x["seq_lengths"]):
            src = [slice(None)] * out.ndim
            src[batch_dim], src[seq_dim] = b, slice(0, int(n))
            dst = list(src)
            dst[seq_dim] = slice(int(n) - 1, None, -1) if n > 0 else slice(0, 0)
            out[tuple(src)] = x["input"][tuple(dst)]
    elif call.entry == "dynamic_update_slice":
        operand, update = x["operand"], x["update"]
        out = operand.copy()
        start = [min(max(0, int(s)), o - u) for s, o, u in zip(x["start"], operand.shape, update.shape)]
        out[tuple(slice(s, s + u) for s, u in zip(start, update.shape))] = update
    else:
        raise AssertionError(f"no numpy model for {call.entry}")
    return {"output": np.ascontiguousarray(out)}


def _generate(desc: dict, tmp_path: Path, seed: int = 1234):
    op = OPS[desc["operator"]](desc, seed=seed, target_cpu="cortex-m55")
    assert op.needs_keras_model() is False and op.allow_no_tflite()
    op.generate_c_files(tmp_path)
    return op


@pytest.mark.parametrize("desc", CASES, ids=[d["name"] for d in CASES])
def test_golden_matches_an_independent_numpy_model(desc: dict, tmp_path: Path) -> None:
    op = _generate(desc, tmp_path)
    assert not list(tmp_path.rglob("*.tflite"))
    if desc.get("expected_status", "ARM_CMSIS_NN_SUCCESS") != "ARM_CMSIS_NN_SUCCESS":
        assert op.status_only() and not op.uses_reference()
        assert op.reference is None
        assert "ARM_CMSIS_NN_ARG_ERROR" in "".join(p.read_text() for p in tmp_path.rglob("*.c"))
        return
    assert op.uses_reference() and not op.status_only()
    expected = _numpy_model(op.reference)
    got = op.reference.run()
    assert set(got) == set(expected)
    for name, want in expected.items():
        assert got[name].dtype == want.dtype, name
        np.testing.assert_array_equal(got[name], want, err_msg=name)


def test_suite_covers_every_operator_and_the_status_only_cases() -> None:
    assert {d["operator"] for d in CASES} == set(OPS)
    arg_errors = [d for d in CASES if d.get("expected_status") == "ARM_CMSIS_NN_ARG_ERROR"]
    assert sorted({d["operator"] for d in arg_errors}) == ["BroadcastTo", "DynamicUpdateSlice"]
    assert sum(d["operator"] == "BroadcastTo" for d in arg_errors) == 10
    assert sum(d["operator"] == "DynamicUpdateSlice" for d in arg_errors) == 14


def _desc(name: str) -> dict:
    return next(d for d in CASES if d["name"] == name)


@pytest.mark.parametrize("name", ["strided_slice_crop_width_s8", "strided_slice_stride2_s16"])
def test_strided_slice_int_input_is_quantized_by_the_policy(name, tmp_path: Path) -> None:
    from helia_core_tester.generation.reference import policy

    desc = _desc(name)
    op = _generate(desc, tmp_path)
    kind = desc["activation_dtype"].lower()
    q = policy.activation_quant(kind, (-1.0, 1.0))
    x = op.reference.inputs["input"]
    assert x.dtype == (np.int16 if kind == "s16" else np.int8)
    assert x.min() >= policy.quantize(np.float32([-1.0]), q)[0] and x.max() <= policy.quantize(np.float32([1.0]), q)[0]


def test_strided_slice_honours_the_declared_batch(tmp_path: Path) -> None:
    desc = _desc("strided_slice_case1_whole_slab_s8")
    op = _generate(desc, tmp_path)
    x = op.reference.inputs["input"]
    assert x.shape == (2, 3, 4, 2)
    np.testing.assert_array_equal(op.reference.output(), x[1:2])


def test_strided_slice_shrink_drops_the_axis_from_the_golden_only(tmp_path: Path) -> None:
    op = _generate(_desc("strided_slice_shrink_last_s8"), tmp_path)
    assert op.reference.output().shape == (1, 3, 2)
    np.testing.assert_array_equal(op.reference.output(), op.reference.inputs["input"][:, :, -1, :])


def test_strided_slice_rejects_a_begin_of_the_wrong_rank(tmp_path: Path) -> None:
    desc = {**_desc("strided_slice_crop_width_s8"), "begin": [0, 0, 0]}
    with pytest.raises(ValueError, match="begin/end/strides"):
        _generate(desc, tmp_path)


def test_scatter_nd_sums_duplicates_in_the_element_type(tmp_path: Path) -> None:
    desc = {**_desc("scatter_nd_1d_s8"), "indices": [[2], [2], [5]], "updates": [100, 100, -7]}
    op = _generate(desc, tmp_path)
    want = np.zeros(8, dtype=np.int8)
    want[2], want[5] = np.int8(-56), np.int8(-7)
    np.testing.assert_array_equal(op.reference.output(), want)


def test_where_emits_the_coordinates_up_to_the_count(tmp_path: Path) -> None:
    op = _generate(_desc("where_2d_s8"), tmp_path)
    cond = op.reference.inputs["condition"]
    got = op.reference.run()
    count = int(np.count_nonzero(cond))
    assert int(got["count"][0]) == count and got["coords"].shape == (cond.size, cond.ndim)
    assert not got["coords"][count:].any()
    text = "".join(p.read_text() for p in tmp_path.rglob("*.c"))
    assert f'"num_true", {count},' in text


def test_where_with_nothing_true_records_a_zero_count(tmp_path: Path, monkeypatch) -> None:
    desc = _desc("where_3d_s16")
    op = OpWhere(desc, seed=1)
    monkeypatch.setattr(op, "_seeded_rng", lambda: _ZeroRng())
    op.generate_c_files(tmp_path)
    got = op.reference.run()
    assert int(got["count"][0]) == 0 and not got["coords"].any()
    assert '"num_true", 0,' in "".join(p.read_text() for p in tmp_path.rglob("*.c"))


class _ZeroRng:
    def integers(self, low, high, size, dtype):
        return np.zeros(size, dtype=dtype)


def test_dynamic_update_slice_clamps_the_start(tmp_path: Path) -> None:
    op = _generate(_desc("dynamic_update_slice_clamped_start_s8"), tmp_path)
    out, upd = op.reference.output(), op.reference.inputs["update"]
    np.testing.assert_array_equal(out[0:2, 2:5], upd)
