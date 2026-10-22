"""The dtype-generic data-movement entries on the C reference, against numpy (and loop models of
TFLite's reference_ops where numpy has no counterpart), for every element type."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference.abi import load_spec, resolve_any
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.generation.reference.call import ReferenceCall

DTYPES = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float16]


@pytest.fixture(scope="module")
def lib():
    return get_bindings()


def _status(fn) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        fn()
    return info.value.status


def _draw(rng, shape, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(max(info.min, -1000), min(info.max, 1000) + 1, shape).astype(dtype)
    return rng.uniform(-8, 8, shape).astype(dtype)


def _run(lib, entry, params, inputs, out_shape, out="output"):
    return lib.run(entry, params, inputs, {out: out_shape})[out]


# ------------------------------------------------------------------ `any`


def test_any_outputs_take_the_input_dtype_and_disagreeing_inputs_fail(lib) -> None:
    k = load_spec().kernels["select_v2"]
    x = np.zeros(3, np.int16)
    assert resolve_any(k, {"condition": np.zeros(3, np.uint8), "x": x, "y": x})["output"] == "int16"
    with pytest.raises(TypeError, match="disagree"):
        resolve_any(k, {"condition": np.zeros(3, np.uint8), "x": x, "y": x.astype(np.int8)})
    with pytest.raises(TypeError, match="no reference tensor dtype"):
        lib.run("reshape", {"unused": 0}, {"input": np.zeros(2, np.uint32)}, {"output": (2,)})
    with pytest.raises(TypeError):
        ReferenceCall("reshape", {"unused": 0}, {"input": [1, 2]}, {"output": (2,)})


def test_an_any_output_without_an_any_input_is_a_spec_error(tmp_path) -> None:
    import yaml

    from helia_core_tester.generation.reference import abi

    raw = yaml.safe_load(abi.SPEC_PATH.read_text())
    raw["kernels"] = {"bad": {"params": "HctNoParams", "inputs": {"x": "int8"}, "outputs": {"y": "any"}}}
    path = tmp_path / "entries.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(abi.SpecError, match="needs an `any` input"):
        abi.load_spec(path)


# --------------------------------------------------- concatenation / split


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concatenation_and_split_are_numpy(lib, dtype, axis) -> None:
    rng = np.random.default_rng(1)
    parts = [_draw(rng, s, dtype) for s in ((2, 3, 4),)]
    shapes = []
    for e in (1, 0, 3):
        s = [2, 3, 4]
        s[axis] = e
        shapes.append(tuple(s))
    parts = [_draw(rng, s, dtype) for s in shapes]
    want = np.concatenate(parts, axis=axis)
    data = np.concatenate([p.ravel() for p in parts])
    ext = np.array([p.shape[axis] for p in parts], np.int32)
    got = _run(lib, "concatenation", {"axis": axis}, {"data": data, "extents": ext}, want.shape)
    np.testing.assert_array_equal(got, want)
    back = _run(lib, "split", {"axis": axis}, {"input": want, "extents": ext}, data.shape)
    np.testing.assert_array_equal(back, data)


def test_concatenation_rejects_bad_extents_and_axes(lib) -> None:
    data = np.zeros(12, np.int8)
    run = lambda ext, axis=0, out=(4, 3): _run(lib, "concatenation", {"axis": axis},
                                               {"data": data, "extents": np.array(ext, np.int32)}, out)
    run([1, 3])
    assert _status(lambda: run([1, 2])) == "E_SHAPE"
    assert _status(lambda: run([-1, 5])) == "E_PARAM"
    assert _status(lambda: run([1, 3], axis=2)) == "E_PARAM"
    assert _status(lambda: run([1, 3], out=(4, 4))) == "E_SHAPE"


# ----------------------------------------------------------- gather family


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis,idx_shape", [(0, (5,)), (1, (2, 3)), (2, ()), (-1, (4,))])
def test_gather_is_numpy_take(lib, dtype, axis, idx_shape) -> None:
    rng = np.random.default_rng(2)
    x = _draw(rng, (3, 4, 5), dtype)
    idx = rng.integers(0, x.shape[axis], idx_shape).astype(np.int32)
    want = np.take(x, idx, axis=axis)
    np.testing.assert_array_equal(_run(lib, "gather", {"axis": axis}, {"input": x, "indices": idx}, want.shape), want)


def test_gather_rejects_out_of_range_indices(lib) -> None:
    x = np.zeros((3, 2), np.int8)
    for bad in (-1, 3):
        assert _status(lambda: _run(lib, "gather", {"axis": 0}, {"input": x, "indices": np.array([0, bad], np.int32)},
                                    (2, 2))) == "E_PARAM"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_gather_nd_is_the_tuple_lookup(lib, dtype, depth) -> None:
    rng = np.random.default_rng(3)
    x = _draw(rng, (3, 4, 5), dtype)
    idx = np.stack([rng.integers(0, x.shape[d], (2, 3)) for d in range(depth)], axis=-1).astype(np.int32)
    want = np.array([[x[tuple(idx[i, j])] for j in range(3)] for i in range(2)])
    got = _run(lib, "gather_nd", {"unused": 0}, {"input": x, "indices": idx}, want.shape)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("bad", [-1, 3])
def test_gather_nd_rejects_out_of_range_and_negative_indices(lib, bad) -> None:
    x = np.zeros((3, 2), np.int8)
    assert _status(lambda: _run(lib, "gather_nd", {"unused": 0},
                                {"input": x, "indices": np.array([[bad, 0]], np.int32)}, (1,))) == "E_PARAM"
    assert _status(lambda: _run(lib, "gather_nd", {"unused": 0},
                                {"input": x, "indices": np.zeros((1, 3), np.int32)}, (1,))) == "E_SHAPE"


# --------------------------------------------------------------------- pad


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode,np_mode", [(0, "constant"), (1, "reflect"), (2, "symmetric")])
def test_pad_modes_are_numpy(lib, dtype, mode, np_mode) -> None:
    rng = np.random.default_rng(4)
    x = _draw(rng, (2, 3, 4), dtype)
    pads = np.array([[0, 1], [2, 1], [3, 0]], np.int32)
    value = np.array([5], dtype)
    kw = {"constant_values": value[0]} if mode == 0 else {}
    want = np.pad(x, pads, mode=np_mode, **kw)
    got = _run(lib, "pad", {"mode": mode}, {"input": x, "paddings": pads, "pad_value": value}, want.shape)
    np.testing.assert_array_equal(got, want)


def test_pad_rejects_reach_past_the_mirror_and_negative_pads(lib) -> None:
    x = np.zeros((3,), np.int8)
    v = np.zeros(1, np.int8)
    run = lambda mode, pads, out: _run(lib, "pad", {"mode": mode},
                                       {"input": x, "paddings": np.array([pads], np.int32), "pad_value": v}, (out,))
    run(1, [2, 2], 7)  # REFLECT may reach dim - 1
    assert _status(lambda: run(1, [3, 0], 6)) == "E_PARAM"
    run(2, [3, 3], 9)  # SYMMETRIC may reach dim
    assert _status(lambda: run(2, [4, 0], 7)) == "E_PARAM"
    assert _status(lambda: run(0, [-1, 0], 2)) == "E_PARAM"
    assert _status(lambda: run(3, [0, 0], 3)) == "E_PARAM"
    with pytest.raises(TypeError, match="disagree"):
        _run(lib, "pad", {"mode": 0}, {"input": x, "paddings": np.zeros((1, 2), np.int32),
                                       "pad_value": np.zeros(1, np.int16)}, (3,))


# ------------------------------------------------------------ space family


def b2s_model(x, block, crops):
    """TFLite reference_ops::BatchToSpaceND: input batch = spatial_offset * out_batch + b."""
    bh, bw = block
    ob = x.shape[0] // (bh * bw)
    out = np.empty((ob, x.shape[1] * bh - sum(crops[0]), x.shape[2] * bw - sum(crops[1]), x.shape[3]), x.dtype)
    for ib in range(x.shape[0]):
        b, sp = ib % ob, ib // ob
        for ih in range(x.shape[1]):
            oh = ih * bh + sp // bw - crops[0][0]
            for iw in range(x.shape[2]):
                ow = iw * bw + sp % bw - crops[1][0]
                if 0 <= oh < out.shape[1] and 0 <= ow < out.shape[2]:
                    out[b, oh, ow] = x[ib, ih, iw]
    return out


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.float32])
@pytest.mark.parametrize("batch,block,crops", [(4, (2, 2), ((0, 0), (0, 0))), (8, (2, 2), ((1, 0), (0, 1))),
                                               (12, (2, 3), ((0, 1), (1, 1)))])
def test_batch_and_space_nd_round_trip_tflite(lib, dtype, batch, block, crops) -> None:
    rng = np.random.default_rng(5)
    x = _draw(rng, (batch, 3, 4, 2), dtype)
    want = b2s_model(x, block, crops)
    got = _run(lib, "batch_to_space_nd", {"unused": 0},
               {"input": x, "block_shape": np.array(block, np.int32), "crops": np.array(crops, np.int32)}, want.shape)
    np.testing.assert_array_equal(got, want)
    # SpaceToBatchND with the crops as padding undoes it wherever no pad value lands.
    pad = np.array([3], dtype)
    back = _run(lib, "space_to_batch_nd", {"unused": 0},
                {"input": got, "block_shape": np.array(block, np.int32), "paddings": np.array(crops, np.int32),
                 "pad_value": pad}, x.shape)
    keep = b2s_model(np.ones_like(x), block, crops)  # where the inverse is defined
    mask = _run(lib, "space_to_batch_nd", {"unused": 0},
                {"input": keep, "block_shape": np.array(block, np.int32), "paddings": np.array(crops, np.int32),
                 "pad_value": np.zeros(1, dtype)}, x.shape) != 0
    np.testing.assert_array_equal(back[mask], x[mask])
    assert np.all(back[~mask] == pad[0])


def test_space_to_batch_matches_the_tflite_reshape(lib) -> None:
    rng = np.random.default_rng(6)
    x = _draw(rng, (2, 3, 4, 2), np.int8)
    pads = ((1, 0), (0, 2))
    xp = np.pad(x, ((0, 0), *pads, (0, 0)), constant_values=-7)
    b, h, w, c = xp.shape
    want = xp.reshape(b, h // 2, 2, w // 3, 3, c).transpose(2, 4, 0, 1, 3, 5).reshape(-1, h // 2, w // 3, c)
    got = _run(lib, "space_to_batch_nd", {"unused": 0},
               {"input": x, "block_shape": np.array([2, 3], np.int32), "paddings": np.array(pads, np.int32),
                "pad_value": np.array([-7], np.int8)}, want.shape)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bs", [1, 2, 3])
def test_depth_and_space_to_depth_are_tf_dcr(lib, dtype, bs) -> None:
    rng = np.random.default_rng(7)
    x = _draw(rng, (2, 2, 3, 2 * bs * bs), dtype)
    n, h, w, c = x.shape
    oc = c // (bs * bs)
    want = x.reshape(n, h, w, bs, bs, oc).transpose(0, 1, 3, 2, 4, 5).reshape(n, h * bs, w * bs, oc)
    got = _run(lib, "depth_to_space", {"block_size": bs}, {"input": x}, want.shape)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(_run(lib, "space_to_depth", {"block_size": bs}, {"input": got}, x.shape), x)


def test_block_ops_reject_indivisible_shapes(lib) -> None:
    x = np.zeros((1, 2, 2, 3), np.int8)
    assert _status(lambda: _run(lib, "depth_to_space", {"block_size": 2}, {"input": x}, (1, 4, 4, 0))) == "E_PARAM"
    assert _status(lambda: _run(lib, "space_to_depth", {"block_size": 3}, {"input": x}, (1, 0, 0, 27))) == "E_PARAM"
    assert _status(lambda: _run(lib, "depth_to_space", {"block_size": 0}, {"input": x}, (1, 2, 2, 3))) == "E_PARAM"
    assert _status(lambda: _run(lib, "batch_to_space_nd", {"unused": 0},
                                {"input": np.zeros((3, 1, 1, 1), np.int8), "block_shape": np.array([2, 1], np.int32),
                                 "crops": np.zeros((2, 2), np.int32)}, (1, 2, 1, 1))) == "E_PARAM"


def nearest_model(n_out, n_in, align, half):
    scale = np.float32(n_in - 1) / np.float32(n_out - 1) if align and n_out > 1 else np.float32(n_in) / np.float32(n_out)
    off = np.float32(0.5 if half else 0.0)
    out = []
    for x in range(n_out):
        v = (np.float32(x) + off) * scale
        r = int(math.floor(v + 0.5)) if align else int(math.floor(v))  # v >= 0: round = half up
        r = min(r, n_in - 1)
        out.append(max(r, 0) if half else r)
    return out


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.float32])
@pytest.mark.parametrize("in_hw,out_hw", [((3, 4), (6, 8)), ((5, 5), (3, 2)), ((1, 1), (4, 3)), ((4, 7), (9, 7))])
@pytest.mark.parametrize("align,half", [(0, 0), (1, 0), (0, 1)])
def test_resize_nearest_follows_get_nearest_neighbor(lib, dtype, in_hw, out_hw, align, half) -> None:
    rng = np.random.default_rng(8)
    x = _draw(rng, (2, *in_hw, 2), dtype)
    ys = nearest_model(out_hw[0], in_hw[0], align, half)
    xs = nearest_model(out_hw[1], in_hw[1], align, half)
    want = x[:, ys][:, :, xs]
    got = _run(lib, "resize_nearest_neighbor", {"align_corners": align, "half_pixel_centers": half}, {"input": x},
               (2, *out_hw, 2))
    np.testing.assert_array_equal(got, want)


def test_resize_rejects_both_flags(lib) -> None:
    x = np.zeros((1, 2, 2, 1), np.int8)
    assert _status(lambda: _run(lib, "resize_nearest_neighbor", {"align_corners": 1, "half_pixel_centers": 1},
                                {"input": x}, (1, 4, 4, 1))) == "E_PARAM"


@pytest.mark.parametrize("dtype", DTYPES)
def test_reshape_is_a_copy_and_checks_the_count(lib, dtype) -> None:
    x = _draw(np.random.default_rng(9), (2, 3, 4), dtype)
    np.testing.assert_array_equal(_run(lib, "reshape", {"unused": 0}, {"input": x}, (4, 6)), x.reshape(4, 6))
    assert _status(lambda: _run(lib, "reshape", {"unused": 0}, {"input": x}, (5, 5))) == "E_SHAPE"
    assert _run(lib, "reshape", {"unused": 0}, {"input": np.zeros((0, 3), dtype)}, (3, 0)).size == 0


# --------------------------------------------------------------- index maps


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("perm", [(1, 0), (0, 2, 1), (2, 0, 1), (3, 1, 0, 2), (4, 3, 2, 1, 0)])
def test_transpose_is_numpy(lib, dtype, perm) -> None:
    x = _draw(np.random.default_rng(10), (2, 3, 4, 5, 2)[: len(perm)], dtype)
    want = np.transpose(x, perm)
    got = _run(lib, "transpose", {"unused": 0}, {"input": x, "perm": np.array(perm, np.int32)}, want.shape)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("perm", [(0, 0, 1), (0, 1, 3), (-1, 0, 1)])
def test_transpose_rejects_a_non_permutation(lib, perm) -> None:
    x = np.zeros((2, 3, 4), np.int8)
    assert _status(lambda: _run(lib, "transpose", {"unused": 0}, {"input": x, "perm": np.array(perm, np.int32)},
                                (2, 3, 4))) == "E_PARAM"


def strided_slice_model(x, begin, end, strides, begin_mask, end_mask, shrink):
    """Python slicing reproduces TFLite's start/stop rules for these masks."""
    sl = []
    for d in range(x.ndim):
        if (shrink >> d) & 1:
            b = begin[d] + (x.shape[d] if begin[d] < 0 else 0)
            sl.append(b)
            continue
        b = None if (begin_mask >> d) & 1 else begin[d]
        e = None if (end_mask >> d) & 1 else end[d]
        sl.append(slice(b, e, strides[d]))
    return x[tuple(sl)]


SLICES = [
    ((0, 0, 0), (2, 3, 4), (1, 1, 1), 0, 0, 0),
    ((1, -2, 0), (2, 3, 4), (1, 1, 2), 0, 0, 0),
    ((1, 2, 3), (0, 0, 0), (-1, -1, -2), 0, 0, 0),
    ((0, 1, 0), (2, 100, -1), (1, 2, 1), 0b001, 0b100, 0),
    ((-1, 0, -5), (0, 3, 9), (1, 1, 1), 0, 0, 0b001),
    ((1, 2, 3), (2, 3, 4), (1, 1, 1), 0, 0, 0b111),
    ((0, 0, 0), (0, 0, 0), (1, 1, -1), 0b111, 0b111, 0),
    ((5, -100, 2), (9, 100, 2), (1, 1, 1), 0, 0, 0),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("begin,end,strides,bm,em,shrink", SLICES)
def test_strided_slice_follows_tflite(lib, dtype, begin, end, strides, bm, em, shrink) -> None:
    x = _draw(np.random.default_rng(11), (2, 3, 4), dtype)
    want = np.array(strided_slice_model(x, begin, end, strides, bm, em, shrink), dtype=dtype)
    got = _run(lib, "strided_slice",
               {"begin_mask": bm, "end_mask": em, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": shrink},
               {"input": x, "begin": np.array(begin, np.int32), "end": np.array(end, np.int32),
                "strides": np.array(strides, np.int32)}, want.shape)
    np.testing.assert_array_equal(got, want)


def test_strided_slice_rejects_unsupported_masks_and_zero_strides(lib) -> None:
    x = np.zeros((2, 3), np.int8)
    p = {"begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0}
    v = lambda *a: np.array(a, np.int32)
    run = lambda params, strides=(1, 1), out=(2, 3): _run(
        lib, "strided_slice", params, {"input": x, "begin": v(0, 0), "end": v(2, 3), "strides": v(*strides)}, out)
    run(p)
    assert _status(lambda: run(p, (1, 0))) == "E_PARAM"
    assert _status(lambda: run(dict(p, ellipsis_mask=1))) == "E_PARAM"
    assert _status(lambda: run(dict(p, new_axis_mask=2))) == "E_PARAM"
    assert _status(lambda: run(dict(p, begin_mask=4))) == "E_PARAM"
    assert _status(lambda: run(p, out=(2, 2))) == "E_SHAPE"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mult", [(1, 1, 1), (2, 1, 3), (1, 0, 2)])
def test_tile_is_numpy(lib, dtype, mult) -> None:
    x = _draw(np.random.default_rng(12), (2, 3, 1), dtype)
    want = np.tile(x, mult)
    got = _run(lib, "tile", {"unused": 0}, {"input": x, "multiples": np.array(mult, np.int32)}, want.shape)
    np.testing.assert_array_equal(got, want)
    assert _status(lambda: _run(lib, "tile", {"unused": 0}, {"input": x, "multiples": np.array([1, -1, 1], np.int32)},
                                (2, 3, 1))) == "E_PARAM"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_shape,out_shape", [((3, 1), (2, 3, 4)), ((1,), (5,)), ((), (2, 2)), ((2, 3), (2, 3))])
def test_broadcast_to_is_numpy(lib, dtype, in_shape, out_shape) -> None:
    x = _draw(np.random.default_rng(13), in_shape, dtype)
    got = _run(lib, "broadcast_to", {"unused": 0}, {"input": x}, out_shape)
    np.testing.assert_array_equal(got, np.broadcast_to(x, out_shape))


def test_broadcast_to_rejects_incompatible_shapes(lib) -> None:
    x = np.zeros((2, 3), np.int8)
    assert _status(lambda: _run(lib, "broadcast_to", {"unused": 0}, {"input": x}, (2, 4))) == "E_SHAPE"
    assert _status(lambda: _run(lib, "broadcast_to", {"unused": 0}, {"input": x}, (3,))) == "E_SHAPE"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_dim,batch_dim", [(1, 0), (0, 1), (2, 0)])
def test_reverse_sequence_reverses_each_prefix(lib, dtype, seq_dim, batch_dim) -> None:
    rng = np.random.default_rng(14)
    x = _draw(rng, (3, 4, 5), dtype)
    lens = rng.integers(0, x.shape[seq_dim] + 1, x.shape[batch_dim]).astype(np.int32)
    want = x.copy()
    for b, l in enumerate(lens):
        src = [slice(None)] * 3
        src[batch_dim] = b
        src[seq_dim] = slice(0, l)
        want[tuple(src)] = np.flip(x[tuple(src)], axis=seq_dim - (1 if batch_dim < seq_dim else 0))
    got = _run(lib, "reverse_sequence", {"seq_dim": seq_dim, "batch_dim": batch_dim},
               {"input": x, "seq_lengths": lens}, x.shape)
    np.testing.assert_array_equal(got, want)


def test_reverse_sequence_rejects_long_sequences_and_equal_dims(lib) -> None:
    x = np.zeros((2, 3), np.int8)
    run = lambda p, lens: _run(lib, "reverse_sequence", p, {"input": x, "seq_lengths": np.array(lens, np.int32)}, x.shape)
    run({"seq_dim": 1, "batch_dim": 0}, [3, 0])
    assert _status(lambda: run({"seq_dim": 1, "batch_dim": 0}, [4, 0])) == "E_PARAM"
    assert _status(lambda: run({"seq_dim": 0, "batch_dim": 0}, [1, 1])) == "E_PARAM"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("start", [(0, 0, 0), (1, 2, 1), (-1, 99, 2), (5, -3, -1)])
def test_dynamic_update_slice_clamps_the_start(lib, dtype, start) -> None:
    rng = np.random.default_rng(15)
    op = _draw(rng, (3, 4, 5), dtype)
    upd = _draw(rng, (2, 2, 3), dtype)
    s = [min(max(v, 0), op.shape[d] - upd.shape[d]) for d, v in enumerate(start)]
    want = op.copy()
    want[s[0]:s[0] + 2, s[1]:s[1] + 2, s[2]:s[2] + 3] = upd
    got = _run(lib, "dynamic_update_slice", {"unused": 0},
               {"operand": op, "update": upd, "start": np.array(start, np.int32)}, op.shape)
    np.testing.assert_array_equal(got, want)
    assert _status(lambda: _run(lib, "dynamic_update_slice", {"unused": 0},
                                {"operand": op, "update": _draw(rng, (4, 1, 1), dtype),
                                 "start": np.zeros(3, np.int32)}, op.shape)) == "E_SHAPE"


# ---------------------------------------------------- select / scatter / fill


@pytest.mark.parametrize("dtype", DTYPES)
def test_scatter_nd_sums_duplicates_in_the_element_type(lib, dtype) -> None:
    rng = np.random.default_rng(16)
    idx = np.array([[0, 1], [2, 0], [0, 1], [1, 3]], np.int32)
    upd = _draw(rng, (4, 5), dtype)
    want = np.zeros((3, 4, 5), dtype)
    for k, (i, j) in enumerate(idx):
        if dtype == np.float16:
            want[i, j] = (want[i, j].astype(np.float32) + upd[k].astype(np.float32)).astype(np.float16)
        else:
            want[i, j] = want[i, j] + upd[k]
    got = _run(lib, "scatter_nd", {"unused": 0}, {"indices": idx, "updates": upd}, want.shape)
    np.testing.assert_array_equal(got, want)


def test_scatter_nd_wraps_integers_and_rejects_bad_indices(lib) -> None:
    idx = np.array([[0], [0]], np.int32)
    got = _run(lib, "scatter_nd", {"unused": 0}, {"indices": idx, "updates": np.array([100, 100], np.int8)}, (2,))
    assert got.tolist() == [-56, 0]
    for bad in (-1, 2):
        assert _status(lambda: _run(lib, "scatter_nd", {"unused": 0},
                                    {"indices": np.array([[bad]], np.int32), "updates": np.array([1], np.int8)},
                                    (2,))) == "E_PARAM"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cs,xs,ys", [((2, 3), (2, 3), (2, 3)), ((1, 3), (2, 3), (2, 1)), ((), (4,), (1,)),
                                      ((2, 1, 1), (3, 1), (1, 4))])
def test_select_v2_broadcasts_all_three(lib, dtype, cs, xs, ys) -> None:
    rng = np.random.default_rng(17)
    c = rng.integers(0, 2, cs).astype(np.uint8)
    x, y = _draw(rng, xs, dtype), _draw(rng, ys, dtype)
    want = np.where(c.astype(bool), x, y)
    got = _run(lib, "select_v2", {"unused": 0}, {"condition": c, "x": x, "y": y}, want.shape)
    np.testing.assert_array_equal(got, want)


def test_select_v2_rejects_an_output_no_operand_spans(lib) -> None:
    c = np.ones((2, 1), np.uint8)
    x = np.zeros((2, 1), np.int8)
    assert _status(lambda: _run(lib, "select_v2", {"unused": 0}, {"condition": c, "x": x, "y": x}, (2, 3))) == "E_SHAPE"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4), (0, 3)])
def test_where_lists_nonzero_coordinates(lib, dtype, shape) -> None:
    rng = np.random.default_rng(18)
    x = (rng.integers(-1, 2, shape)).astype(dtype)
    want = np.argwhere(x != 0)
    r = lib.run("where", {"unused": 0}, {"condition": x}, {"coords": (x.size, x.ndim), "count": (1,)})
    assert r["count"][0] == len(want)
    np.testing.assert_array_equal(r["coords"][: len(want)], want)
    assert not r["coords"][len(want):].any()


def test_where_treats_negative_zero_as_zero_and_nan_as_true(lib) -> None:
    for dtype in (np.float32, np.float16):
        x = np.array([-0.0, np.nan, 0.0, 1.0], dtype)
        r = lib.run("where", {"unused": 0}, {"condition": x}, {"coords": (4, 1), "count": (1,)})
        assert r["count"][0] == 2 and r["coords"][:2, 0].tolist() == [1, 3]
    assert _status(lambda: lib.run("where", {"unused": 0}, {"condition": np.ones(3, np.int8)},
                                   {"coords": (2, 1), "count": (1,)})) == "E_SHAPE"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(3, 4), (0,), ()])
def test_fill_repeats_the_value_bit_exactly(lib, dtype, shape) -> None:
    v = np.array([-0.0 if np.issubdtype(dtype, np.floating) else -7], dtype)
    got = _run(lib, "fill", {"unused": 0}, {"value": v}, shape)
    assert got.shape == shape and got.dtype == dtype
    assert np.all(got.reshape(-1).view(np.uint8).reshape(-1, v.itemsize) == v.view(np.uint8))
    assert _status(lambda: _run(lib, "fill", {"unused": 0}, {"value": np.zeros(2, dtype)}, shape)) == "E_SHAPE"
