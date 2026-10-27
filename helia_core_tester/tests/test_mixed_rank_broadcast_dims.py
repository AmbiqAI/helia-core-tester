"""Add, Sub and Mul hand their kernels right-aligned dims for mixed-rank operands (#482)."""
from pathlib import Path
import re

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops._shared.binary_basic_math_base import broadcast_cmsis_dims
from helia_core_tester.generation.ops.BasicMathFunctions.add import OpAdd
from helia_core_tester.generation.ops.BasicMathFunctions.mul import OpMul
from helia_core_tester.generation.ops.BasicMathFunctions.sub import OpSub
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

ROOT = Path(__file__).resolve().parents[2]
OPS = {"Add": OpAdd, "Sub": OpSub, "Mul": OpMul}
SEED = 482


def _descriptors(stem):
    return list(load_descriptor(str(ROOT / f"assets/descriptors/BasicMathFunctions/{stem}.yaml")))


MIXED = [
    d
    for op in ("add", "sub", "mul")
    for stem in (op, f"{op}_float")
    for d in _descriptors(stem)
    if "_mixed_rank2" in d["name"]
]


def _nhwc(dims):
    return (dims["n"], dims["h"], dims["w"], dims["c"])


def _right_aligned(shape):
    return (1,) * (4 - len(shape)) + tuple(shape)


def _header_dims(header, name, field):
    body = header.split(f"{name}_{field}_dims = {{", 1)[1].split("};", 1)[0]
    return tuple(int(v) for v in re.findall(r"\.[nhwc] = (\d+)", body))


def _header_array(header, name, dtype):
    body = header.split(name + "[] = {", 1)[1].split("};", 1)[0]
    values = [re.sub(r"\(float16_t\)", "", x).strip().rstrip("f") for x in body.split(",") if x.strip()]
    return np.array([float(v) for v in values], dtype=dtype)


def _generate(desc, out_dir):
    OPS[desc["operator"]](dict(desc), seed=SEED, target_cpu="cortex-m55").generate_c_files(out_dir)
    return next((out_dir / "includes").glob(f"{desc['name']}*.h")).read_text()


@pytest.mark.parametrize(
    "a,b,want_a,want_b,want_out",
    [
        ([1, 2, 3, 4], [3, 4], (1, 2, 3, 4), (1, 1, 3, 4), (1, 2, 3, 4)),
        ([3, 1], [1, 2, 3, 4], (1, 1, 3, 1), (1, 2, 3, 4), (1, 2, 3, 4)),
        ([2, 3, 4], [4], (1, 2, 3, 4), (1, 1, 1, 4), (1, 2, 3, 4)),
        ([5, 1], [1, 7], (1, 1, 5, 1), (1, 1, 1, 7), (1, 1, 5, 7)),
        ([1], [1, 2, 2, 1], (1, 1, 1, 1), (1, 2, 2, 1), (1, 2, 2, 1)),
        ([1, 1, 1, 1], [1, 1, 1, 1], (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)),
        ([65535], [65535], (1, 1, 1, 65535), (1, 1, 1, 65535), (1, 1, 1, 65535)),
    ],
)
def test_shapes_right_align(a, b, want_a, want_b, want_out):
    out = np.broadcast_shapes(tuple(a), tuple(b))
    dims = broadcast_cmsis_dims("Add", a, b, out)
    assert tuple(_nhwc(d) for d in dims) == (want_a, want_b, want_out)


@pytest.mark.parametrize("shape", [[1, 2, 3, 4], [2, 3, 4], [4]])
def test_ranks_other_than_two_keep_the_legacy_reading(shape):
    # Only rank 2 changed meaning; every existing descriptor uses rank 1 or 4.
    dims, _, _ = broadcast_cmsis_dims("Mul", shape, shape, shape)
    assert dims == TemplateContextBuilder.nhwc_to_cmsis_dims(tuple(shape))


@pytest.mark.parametrize(
    "a,b,out,match",
    [
        ([], [1], [1], "rank 1 through 4"),
        ([1, 1, 1, 1, 2], [2], [1, 1, 1, 1, 2], "rank 1 through 4"),
        ([2, 0], [2, 1], [2, 0], "positive extents"),
        ([-1, 3], [3], [-1, 3], "positive extents"),
        ([2.0, 3], [3], [2, 3], "positive extents"),
        ([True, 3], [3], [1, 3], "positive extents"),
        ([2, 3], [4], [2, 4], "do not broadcast"),
        ([1, 2, 3, 4], [3, 4], [2, 3, 4], "is not the broadcast"),
        ([1, 2, 3, 4], [3, 4], [1, 2, 3, 1], "is not the broadcast"),
        ([3, 4], [3, 4], [1, 3, 4], "is not the broadcast"),
        (None, [3], [3], "has no shape"),
    ],
)
def test_bad_shapes_fail_closed(a, b, out, match):
    with pytest.raises(ValueError, match=match):
        broadcast_cmsis_dims("Sub", a, b, out)


def test_every_op_and_dtype_has_a_mixed_rank_case():
    covered = {(d["operator"], (d.get("tensor_dtypes") or {}).get("input", d.get("activation_dtype"))) for d in MIXED}
    assert covered == {(op, dt) for op in OPS for dt in ("S8", "S16", "FP32", "FP16")}
    assert all(min(len(d["input_1_shape"]), len(d["input_2_shape"])) == 2 for d in MIXED)


@pytest.mark.parametrize("desc", MIXED, ids=lambda d: d["name"])
def test_generated_dims_are_right_aligned(desc, tmp_path, monkeypatch):
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(ROOT))
    name = desc["name"]
    header = _generate(desc, tmp_path)
    a, b = tuple(desc["input_1_shape"]), tuple(desc["input_2_shape"])
    out = np.broadcast_shapes(a, b)
    assert _header_dims(header, name, "input1") == _right_aligned(a)
    assert _header_dims(header, name, "input2") == _right_aligned(b)
    assert _header_dims(header, name, "output") == _right_aligned(out)
    dtype_name = (desc.get("tensor_dtypes") or {}).get("input")
    if dtype_name not in ("FP32", "FP16"):
        return
    # The broadcast entry gets the operands as drawn; the golden is their NumPy broadcast.
    dtype = np.float16 if dtype_name == "FP16" else np.float32
    x = _header_array(header, f"{name}_input1", dtype).reshape(a)
    y = _header_array(header, f"{name}_input2", dtype).reshape(b)
    fn = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply}[desc["operator"]]
    want = fn(x.astype(np.float32), y.astype(np.float32)).astype(dtype)
    np.testing.assert_array_equal(_header_array(header, f"{name}_expected_output", dtype), want.ravel())


@pytest.mark.parametrize("op", sorted(OPS))
def test_non_broadcastable_descriptor_fails_closed(op, tmp_path, monkeypatch):
    # The output shape is derived from the inputs (no model to disagree with), so the
    # only inconsistent descriptor left is one whose inputs do not broadcast.
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(ROOT))
    desc = dict(next(d for d in MIXED if d["operator"] == op and d.get("activation_dtype") == "S8"))
    desc["input_2_shape"] = [5, 4]
    with pytest.raises(ValueError, match="broadcast"):
        _generate(desc, tmp_path)
