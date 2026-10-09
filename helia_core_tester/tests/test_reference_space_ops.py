"""Reshape, BatchToSpaceND, SpaceToBatchND, DepthToSpace, SpaceToDepth, ResizeNearestNeighbor and
Squeeze generators: every emitted golden is the recorded reference call, and equals an independent
numpy model of the TFLite op on the emitted input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.ops.ReshapeFunctions.batch_to_space_nd import OpBatchToSpaceND
from helia_core_tester.generation.ops.ReshapeFunctions.depth_to_space import OpDepthToSpace
from helia_core_tester.generation.ops.ReshapeFunctions.reshape import OpReshape
from helia_core_tester.generation.ops.ReshapeFunctions.resize_nearest_neighbor import OpResizeNearestNeighbor
from helia_core_tester.generation.ops.ReshapeFunctions.space_to_batch_nd import OpSpaceToBatchND
from helia_core_tester.generation.ops.ReshapeFunctions.space_to_depth import OpSpaceToDepth
from helia_core_tester.generation.ops.TesterExtensions.squeeze import OpSqueeze, squeezed_shape
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

DESCRIPTORS = Path(__file__).resolve().parents[2] / "assets" / "descriptors"
OP_FILES = Path(__file__).resolve().parents[1] / "generation" / "ops"
SOURCES = {
    "Reshape": (OpReshape, ["ReshapeFunctions/reshape.yaml", "ReshapeFunctions/reshape_float.yaml"],
                "ReshapeFunctions/reshape.py"),
    "BatchToSpaceND": (OpBatchToSpaceND, ["ReshapeFunctions/batch_to_space_nd.yaml"],
                       "ReshapeFunctions/batch_to_space_nd.py"),
    "SpaceToBatchND": (OpSpaceToBatchND, ["ReshapeFunctions/space_to_batch_nd.yaml"],
                       "ReshapeFunctions/space_to_batch_nd.py"),
    "DepthToSpace": (OpDepthToSpace, ["ReshapeFunctions/depth_to_space.yaml"], "ReshapeFunctions/depth_to_space.py"),
    "SpaceToDepth": (OpSpaceToDepth, ["ReshapeFunctions/space_to_depth.yaml"], "ReshapeFunctions/space_to_depth.py"),
    "ResizeNearestNeighbor": (OpResizeNearestNeighbor, ["ReshapeFunctions/resize_nearest_neighbor.yaml"],
                              "ReshapeFunctions/resize_nearest_neighbor.py"),
    "Squeeze": (OpSqueeze, ["TesterExtensions/squeeze.yaml"], "TesterExtensions/squeeze.py"),
}


def _descriptors():
    out = []
    for operator, (_, files, _) in SOURCES.items():
        for rel in files:
            for desc in yaml.safe_load_all((DESCRIPTORS / rel).read_text()):
                if desc and desc.get("operator") == operator:
                    out.append(pytest.param(operator, desc, id=desc["name"]))
    return out


# ------------------------------------------------------------------ numpy models of the TFLite ops


def batch_to_space_nd(x, block, crops):
    # TFLite: input batch index = (b0_offset * block_w + b1_offset) * out_batch + b.
    n, h, w, c = x.shape
    b0, b1 = block
    out_batch = n // (b0 * b1)
    y = x.reshape(b0, b1, out_batch, h, w, c).transpose(2, 3, 0, 4, 1, 5).reshape(out_batch, h * b0, w * b1, c)
    (t, b), (l, r) = crops
    return y[:, t:h * b0 - b, l:w * b1 - r, :]


def space_to_batch_nd(x, block, pads, pad_value):
    (t, b), (l, r) = pads
    p = np.pad(x, ((0, 0), (t, b), (l, r), (0, 0)), constant_values=pad_value)
    n, ph, pw, c = p.shape
    bh, bw = block
    return (p.reshape(n, ph // bh, bh, pw // bw, bw, c).transpose(2, 4, 0, 1, 3, 5)
            .reshape(n * bh * bw, ph // bh, pw // bw, c))


def depth_to_space(x, bs):
    n, h, w, c = x.shape
    return x.reshape(n, h, w, bs, bs, c // (bs * bs)).transpose(0, 1, 3, 2, 4, 5).reshape(n, h * bs, w * bs, -1)


def space_to_depth(x, bs):
    n, h, w, c = x.shape
    return x.reshape(n, h // bs, bs, w // bs, bs, c).transpose(0, 1, 3, 2, 4, 5).reshape(n, h // bs, w // bs, -1)


def _nearest(out_idx, in_size, out_size, align, half):
    f32 = np.float32
    scale = f32(in_size - 1) / f32(out_size - 1) if align and out_size > 1 else f32(in_size) / f32(out_size)
    scaled = (f32(out_idx) + (f32(0.5) if half else f32(0.0))) * scale
    whole = np.floor(scaled)
    idx = int(whole) + (1 if align and scaled - whole >= f32(0.5) else 0)
    idx = min(idx, in_size - 1)
    return max(0, idx) if half else idx


def resize_nearest(x, out_h, out_w, align, half):
    ys = [_nearest(i, x.shape[1], out_h, align, half) for i in range(out_h)]
    xs = [_nearest(i, x.shape[2], out_w, align, half) for i in range(out_w)]
    return x[:, ys][:, :, xs]


def _model(operator, desc, x):
    if operator == "Reshape":
        return x.reshape(desc["target_shape"])
    if operator == "BatchToSpaceND":
        return batch_to_space_nd(x, desc["block_shape"], desc["crops"])
    if operator == "SpaceToBatchND":
        return space_to_batch_nd(x, desc["block_shape"], desc["paddings"], 0)
    if operator == "DepthToSpace":
        return depth_to_space(x, desc["block_size"])
    if operator == "SpaceToDepth":
        return space_to_depth(x, desc["block_size"])
    if operator == "ResizeNearestNeighbor":
        return resize_nearest(x, *desc["size"], bool(desc.get("align_corners")), bool(desc.get("half_pixel_centers")))
    if operator == "Squeeze":
        axes = desc.get("axes")
        return np.squeeze(x, axis=None if axes is None else tuple(axes))
    raise AssertionError(operator)


@pytest.mark.parametrize(("operator", "desc"), _descriptors())
def test_emitted_golden_is_the_reference_and_matches_numpy(tmp_path, operator, desc) -> None:
    cls = SOURCES[operator][0]
    op = cls(dict(desc), seed=500, target_cpu="cortex-m55")
    assert op.uses_reference() and not op.needs_keras_model() and op.allow_no_tflite()
    op.generate_c_files(tmp_path)
    call = op.reference
    assert call is not None
    x = call.inputs["input"]
    golden = call.output()
    want = _model(operator, desc, x)
    np.testing.assert_array_equal(golden.ravel(), want.ravel())
    assert golden.dtype == x.dtype
    header = next(tmp_path.rglob("*.h")).read_text()
    assert TemplateContextBuilder.format_array_as_c_literal(want) in header
    assert not list(tmp_path.rglob("*.tflite"))


def test_descriptor_batch_to_space_cases_have_one_output_batch() -> None:
    # The pre-reference numpy model put out_batch first; it matched TFLite only for out_batch == 1.
    for p in _descriptors():
        operator, desc = p.values
        if operator == "BatchToSpaceND":
            b0, b1 = desc["block_shape"]
            assert desc["input_shape"][0] == b0 * b1


def test_batch_to_space_follows_tflite_batch_order_beyond_one_output_batch(tmp_path) -> None:
    desc = {"name": "b2s_two_batches", "operator": "BatchToSpaceND", "activation_dtype": "S16",
            "input_shape": [8, 1, 2, 1], "block_shape": [2, 2], "crops": [[0, 0], [0, 1]]}
    op = OpBatchToSpaceND(desc, seed=3)
    op.generate_c_files(tmp_path)
    x = op.reference.inputs["input"]
    want = batch_to_space_nd(x, [2, 2], [[0, 0], [0, 1]])
    assert want.shape == (2, 2, 3, 1)
    np.testing.assert_array_equal(op.reference.output(), want)
    # The out_batch-first order differs here, so the golden really pins TFLite's order.
    old = x.reshape(2, 2, 2, 1, 2, 1).transpose(0, 3, 1, 4, 2, 5).reshape(2, 2, 4, 1)[:, :, :3, :]
    assert not np.array_equal(old, want)


# ------------------------------------------------------------------ negative and boundary cases


@pytest.mark.parametrize(("shape", "axes", "want"), [
    ((1, 1, 4, 1), None, (4,)),
    ((1, 1, 3, 2), [1], (1, 3, 2)),
    ((1, 1, 3, 2), [-4, 1], (3, 2)),
    ((1, 1, 1, 1), None, ()),
    ((2, 3), [], (2, 3)),
])
def test_squeezed_shape(shape, axes, want) -> None:
    assert squeezed_shape(shape, axes) == want


@pytest.mark.parametrize(("shape", "axes", "match"), [
    ((1, 1, 3, 2), [2], "has size 3"),
    ((1, 1, 3, 2), [4], "out of range"),
    ((1, 1, 3, 2), [-5], "out of range"),
])
def test_squeezed_shape_rejects_bad_axes(shape, axes, match) -> None:
    with pytest.raises(ValueError, match=match):
        squeezed_shape(shape, axes)


def test_squeeze_to_a_scalar_is_carried_as_one_element(tmp_path) -> None:
    op = OpSqueeze({"name": "sq_scalar", "operator": "Squeeze", "activation_dtype": "S8", "input_shape": [1, 1, 1, 1]},
                   seed=7)
    op.generate_c_files(tmp_path)
    assert op.reference.output_shapes["output"] == (1,)
    np.testing.assert_array_equal(op.reference.output().ravel(), op.reference.inputs["input"].ravel())


def test_reshape_rejects_a_target_with_another_element_count(tmp_path) -> None:
    op = OpReshape({"name": "r_bad", "operator": "Reshape", "activation_dtype": "S8",
                    "input_shape": [1, 2, 3, 1], "target_shape": [1, 7]}, seed=1)
    with pytest.raises(ValueError, match="does not hold"):
        op.generate_c_files(tmp_path)
    assert op.reference is None


@pytest.mark.parametrize("cls", [OpDepthToSpace, OpSpaceToDepth])
def test_block_ops_require_a_block_size(tmp_path, cls) -> None:
    op = cls({"name": "blk", "operator": "X", "activation_dtype": "S8", "input_shape": [1, 2, 2, 4]}, seed=1)
    with pytest.raises(ValueError, match="block_size"):
        op.generate_c_files(tmp_path)
    assert op.reference is None


@pytest.mark.parametrize("rel", sorted(v[2] for v in SOURCES.values()))
def test_op_files_carry_no_model_path(rel) -> None:
    text = (OP_FILES / rel).read_text().lower()
    for word in ("litert", "tflite_path", "tensorflow", "keras", "interpreter"):
        assert word not in text, f"{rel} mentions {word}"
