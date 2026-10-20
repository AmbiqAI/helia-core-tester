"""DIV uses compact operands and an exact finite quotient assertion."""
from pathlib import Path
import json
import re

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.div import OpDiv
from helia_core_tester.generation.ops.catalog import get_operator_spec

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = list(load_descriptor(str(ROOT / "assets/descriptors/BasicMathFunctions/div_float.yaml")))


def array(header, name, dtype):
    body = header.split(name + "[] = {", 1)[1].split("};", 1)[0]
    return np.array([float(re.sub(r"\(float16_t\)", "", x).strip().rstrip("f")) for x in body.split(",") if x.strip()], dtype=dtype)


@pytest.mark.parametrize("desc", DESCRIPTORS, ids=lambda d: d["name"])
def test_reference_generation(desc, tmp_path, monkeypatch):
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(ROOT))
    op = OpDiv(desc, seed=831, target_cpu="cortex-m55")
    name = desc["name"]
    op.generate_c_files(tmp_path)
    assert op.reference is not None and op.reference.entry in ("div_f16", "div_f32")
    assert not list(tmp_path.rglob("*.tflite"))
    text = (tmp_path / (name + "_mul.c")).read_text()
    header = (tmp_path / "includes" / (name + "_mul.h")).read_text()
    sidecar = json.loads((tmp_path / (name + "_mul.sidecar.json")).read_text())
    assert desc["required_kernel_symbols"][0] in text
    assert sidecar["operator"] == "Div"
    assert sidecar["scalars"]["validation_atol"] == 0
    assert sidecar["scalars"]["validation_rtol"] == 0
    dtype = np.float16 if desc["tensor_dtypes"]["input"] == "FP16" else np.float32
    a = array(header, name + "_input1", dtype).reshape(desc["input_1_shape"])
    b = array(header, name + "_input2", dtype).reshape(desc["input_2_shape"])
    got = array(header, name + "_expected_output", dtype)
    expected = (a.astype(np.float64) / b.astype(np.float64)).astype(dtype)
    expected = np.clip(expected, desc.get("act_min", -1000), desc.get("act_max", 1000)).astype(dtype)
    np.testing.assert_array_equal(got, expected.ravel())
    for field, shape in (("input1", a.shape), ("input2", b.shape), ("output", expected.shape)):
        dims = header.split(name + "_" + field + "_dims = {", 1)[1].split("};", 1)[0]
        actual = tuple(int(value) for value in re.findall(r"\.[nhwc] = (\d+)", dims))
        assert actual == (1,) * (4 - len(shape)) + shape
    assert "HELIA_GUARD_CHECK" in text
    if a.shape != b.shape:
        assert "&" + name + "_input1_dims" in text
        assert a.size == np.prod(desc["input_1_shape"])
        assert b.size == np.prod(desc["input_2_shape"])


def test_catalog_and_descriptors():
    assert get_operator_spec("Div").descriptor_relpaths == ("BasicMathFunctions/div_float.yaml",)
    assert len(DESCRIPTORS) == 26
    assert {d["tensor_dtypes"]["input"] for d in DESCRIPTORS} == {"FP16", "FP32"}
    assert all(d["comparison"] == {"atol": 0, "rtol": 0} for d in DESCRIPTORS)


@pytest.mark.parametrize("input_type,output_type", [("S8", "S8"), ("S16", "S16"), ("FP16", "FP32")])
def test_unsupported_storage_rejected(input_type, output_type):
    desc = dict(DESCRIPTORS[0], tensor_dtypes={"input": input_type, "output": output_type})
    desc.pop("resolved_tensor_dtypes", None)
    with pytest.raises(ValueError, match="matching FP16 or FP32"):
        OpDiv(desc, seed=831, target_cpu="cortex-m55")._dtype()


def test_oracle_failure_is_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(ROOT))
    op = OpDiv(DESCRIPTORS[0], seed=831, target_cpu="cortex-m55")
    def unavailable(*args):
        raise RuntimeError("oracle unavailable")
    monkeypatch.setattr(op, "reference_golden", unavailable)
    with pytest.raises(RuntimeError, match="oracle unavailable"):
        op.generate_c_files(tmp_path)


@pytest.mark.parametrize("shape", [[], [1, 1, 1, 1, 1], [2, 0], [-1, 3]])
def test_invalid_rank_or_extent_rejected(shape):
    desc = dict(DESCRIPTORS[0], input_1_shape=shape)
    with pytest.raises(ValueError, match="positive shapes of rank 1 through 4"):
        OpDiv(desc).validate_descriptor()


@pytest.mark.parametrize("lo,hi", [(2, 1), (float("nan"), 1), (0, float("inf"))])
def test_invalid_clamp_rejected(lo, hi):
    desc = dict(DESCRIPTORS[0], act_min=lo, act_max=hi)
    with pytest.raises(ValueError, match="finite ordered activation bounds"):
        OpDiv(desc).validate_descriptor()
