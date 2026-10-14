"""FullyConnected filter_shape [OutCh, InFeatures] is honored exactly.

A 4-D input is flattened to prod(input_shape[1:]) features and Dense takes its
input width from that, so a descriptor InFeatures that differed from the
flattened count used to be replaced without notice.
"""

import copy
import math
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from helia_core_tester.core.config import Config  # noqa: E402
from helia_core_tester.core.discovery import find_descriptors_dir  # noqa: E402
from helia_core_tester.generation.io import descriptors as desc_io  # noqa: E402
from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import (  # noqa: E402
    OpFullyConnected,
)
from helia_core_tester.generation.test_ops import generate_test  # noqa: E402

BASE = "fully_connected_default_s8"  # input [3, 1, 2, 10]
S4_BASE = "fully_connected_bias_s4"  # input [1, 1, 1, 8]

# Declared [OutCh, last input dim] before the fix; the kernel always ran on the flattened width.
FORMERLY_MISMATCHED = {
    "fully_connected_default_s8": ([3, 1, 2, 10], [6, 20]),
    "fully_connected_weight_offset_s8": ([3, 1, 2, 10], [6, 20]),
    "fully_connected_default_s16": ([2, 3, 3, 7], [11, 63]),
    "fully_connected_slow_s16": ([3, 8, 10, 7], [11, 560]),
    "fully_connected_per_tensor_s16": ([1, 2, 2, 4], [3, 16]),
}


def _load(file_name: str) -> dict:
    path = find_descriptors_dir() / "FullyConnectedFunctions" / file_name
    return {
        expanded["name"]: expanded
        for desc in desc_io.load_descriptor(str(path))
        for expanded in desc_io.expand_descriptor_variations(desc)
    }


@pytest.fixture(scope="module")
def shipped() -> dict:
    return {**_load("fully_connected.yaml"), **_load("fully_connected_float.yaml")}


@pytest.fixture
def base_desc(shipped) -> dict:
    return copy.deepcopy(shipped[BASE])


def test_every_shipped_descriptor_declares_the_flattened_feature_count(shipped):
    assert len(shipped) >= 70
    assert any(d["activation_dtype"].startswith("FP") for d in shipped.values())
    bad = {
        n: (d["input_shape"], d["filter_shape"])
        for n, d in shipped.items()
        if d["filter_shape"][1] != math.prod(d["input_shape"][1:])
    }
    assert bad == {}


def test_every_shipped_descriptor_passes_validation(shipped):
    for desc in shipped.values():
        assert OpFullyConnected(desc, 1).declared_filter_shape() == tuple(desc["filter_shape"])


def test_formerly_mismatched_cases_declare_the_flattened_width(shipped):
    for name, (input_shape, filter_shape) in FORMERLY_MISMATCHED.items():
        assert shipped[name]["input_shape"] == input_shape, name
        assert shipped[name]["filter_shape"] == filter_shape, name


@pytest.mark.parametrize("in_features", [1, 2, 19, 21, 200])
def test_mismatched_in_features_is_rejected(base_desc, in_features):
    base_desc["filter_shape"][1] = in_features
    with pytest.raises(
        ValueError,
        match=r"^%s: FullyConnected filter_shape InFeatures \(%d\) must equal the flattened input feature count \(20\)"
        % (BASE, in_features),
    ):
        OpFullyConnected(base_desc, 1).build_keras_model()


def test_last_input_dim_gets_a_flatten_hint(base_desc):
    base_desc["filter_shape"] = [6, 10]
    with pytest.raises(ValueError, match=re.escape("flattened over H*W*C, did you mean [6, 20]?")):
        OpFullyConnected(base_desc, 1).declared_filter_shape()


def test_no_flatten_hint_for_a_2d_input(shipped):
    desc = copy.deepcopy(shipped["fully_connected_mve_case_01_s8"])  # [1, 20], [4, 20]
    desc["filter_shape"] = [4, 21]
    with pytest.raises(ValueError, match=r"InFeatures \(21\) must equal input_shape\[1\] \(20\)") as excinfo:
        OpFullyConnected(desc, 1).declared_filter_shape()
    assert "did you mean" not in str(excinfo.value)
    assert "flattened" not in str(excinfo.value)


@pytest.mark.parametrize("name", ["fully_connected_float_4d_input_f32", "fully_connected_float_4d_input_f16"])
def test_float_4d_case_with_last_dim_width_is_rejected(shipped, name):
    desc = copy.deepcopy(shipped[name])  # [1, 2, 3, 4], [6, 24]
    desc["filter_shape"] = [6, 4]
    with pytest.raises(ValueError, match=rf"^{name}: .*did you mean \[6, 24\]\?"):
        OpFullyConnected(desc, 1).build_keras_model()


def test_numpy_integer_dims_are_accepted(base_desc):
    base_desc["input_shape"] = [np.int64(v) for v in base_desc["input_shape"]]
    base_desc["filter_shape"] = [np.int32(v) for v in base_desc["filter_shape"]]
    shape = OpFullyConnected(base_desc, 1).declared_filter_shape()
    assert shape == (6, 20)
    assert all(type(dim) is int for dim in shape)


def test_boundary_single_feature_is_accepted(base_desc):
    base_desc["input_shape"] = [1, 1, 1, 1]
    base_desc["filter_shape"] = [1, 1]
    assert OpFullyConnected(base_desc, 1).declared_filter_shape() == (1, 1)


MALFORMED = [
    # (field, value, also rejected by schema.json at load time)
    ("filter_shape", [6], True),
    ("filter_shape", [6, 20, 1], True),
    ("filter_shape", None, True),
    ("filter_shape", "6,20", True),
    ("filter_shape", [0, 20], True),
    ("filter_shape", [6, -20], True),
    ("filter_shape", [True, 20], True),
    # JSON Schema's integer admits 20.0; only generation rejects it.
    ("filter_shape", [6, 20.0], False),
    ("input_shape", [3], True),
    ("input_shape", [3, 2, 10], True),
    ("input_shape", [3, 1, 2, 10, 1], True),
    ("input_shape", [3, 1, 0, 10], True),
]


@pytest.mark.parametrize("field, value, _schema", MALFORMED)
def test_malformed_shapes_are_rejected(base_desc, field, value, _schema):
    base_desc[field] = value
    with pytest.raises(ValueError, match=rf"^{BASE}: FullyConnected"):
        OpFullyConnected(base_desc, 1).declared_filter_shape()


@pytest.mark.parametrize("field, value, schema_rejects", MALFORMED)
def test_schema_rejects_the_same_shapes_at_load_time(field, value, schema_rejects):
    pytest.importorskip("jsonschema")
    doc = {
        "operator": "FullyConnected",
        "name": "fully_connected_case_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "input_shape": [3, 1, 2, 10],
        "filter_shape": [6, 20],
    }
    desc_io.validate_descriptor_schema(doc, "fully_connected")
    if value is None:
        doc.pop(field)
    else:
        doc[field] = value
    if schema_rejects:
        with pytest.raises(ValueError, match="violates schema.json"):
            desc_io.validate_descriptor_schema(doc, "fully_connected")
    else:
        desc_io.validate_descriptor_schema(doc, "fully_connected")


def test_s4_path_rejects_a_mismatched_filter(shipped, tmp_path):
    """S4 builds its LiteRT model directly from filter_shape, without Keras."""
    desc = copy.deepcopy(shipped[S4_BASE])
    desc["input_shape"] = [1, 1, 2, 8]  # 16 features, filter still [4, 8]
    out = tmp_path / f"{S4_BASE}.tflite"
    with pytest.raises(ValueError, match=rf"^{S4_BASE}: .*did you mean \[4, 16\]\?"):
        OpFullyConnected(desc, Config.seed).convert_to_tflite(None, str(out), Config.seed)
    assert not out.exists()


def test_generate_test_fails_closed_on_a_mismatch(base_desc, tmp_path):
    base_desc["filter_shape"] = [6, 10]
    failures = []
    with pytest.raises(ValueError, match=rf"^{BASE}: FullyConnected filter_shape InFeatures \(10\)"):
        generate_test(base_desc, str(tmp_path), seed=Config.seed, generation_failures=failures)
    assert [f["stage"] for f in failures] == ["validate_descriptor"]
    assert not any(tmp_path.rglob("*")), "nothing may be written for a rejected descriptor"


def _weights_shape_from_tflite(tflite_path: Path) -> list:
    op = OpFullyConnected({"name": "probe"}, 1)
    model, subgraph = op.load_litert_model(str(tflite_path))
    from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert

    tensors = get_operator_tensors_from_litert(model, subgraph, op._find_fully_connected_op_index(model, subgraph))
    return [int(v) for v in tensors["weights"].shape]


def _header_dims(header: str, name: str, which: str) -> dict:
    match = re.search(rf"{name}_{which}_dims = \{{([^}}]*)\}}", header)
    assert match, f"{which}_dims not found"
    return {k: int(v) for k, v in re.findall(r"\.(\w) = (-?\d+)", match.group(1))}


@pytest.mark.parametrize("name", ["fully_connected_default_s8", "fully_connected_per_tensor_s16"])
def test_converted_weights_and_header_match_the_declared_filter(shipped, tmp_path, name):
    desc = shipped[name]
    out_ch, in_features = desc["filter_shape"]
    assert in_features != desc["input_shape"][-1]
    generate_test(desc, str(tmp_path), seed=Config.seed)
    (tflite,) = tmp_path.rglob(f"{name}.tflite")
    assert _weights_shape_from_tflite(tflite) == [out_ch, in_features]
    header = next(tflite.parent.rglob(f"{name}_fully_connected.h")).read_text()
    assert _header_dims(header, name, "filter") == {"n": in_features, "h": 1, "w": 1, "c": out_ch}
    assert _header_dims(header, name, "input")["c"] == in_features
    assert _header_dims(header, name, "output")["c"] == out_ch


def _convert(desc: dict, out: Path) -> None:
    op = OpFullyConnected(desc, Config.seed)
    out.mkdir()
    op.convert_to_tflite(op.build_keras_model(), str(out / f"{desc['name']}.tflite"), Config.seed)


def test_converted_weights_disagreeing_with_descriptor_are_rejected(base_desc, tmp_path):
    """Fault injection: the model on disk no longer matches the descriptor's filter."""
    out = tmp_path / "case"
    _convert(base_desc, out)
    tampered = copy.deepcopy(base_desc)
    tampered["filter_shape"][0] = 7  # OutCh in the model stays 6
    with pytest.raises(ValueError, match=re.escape(f"{BASE}: converted FullyConnected weights [OutCh, InFeatures]=[6, 20]")):
        OpFullyConnected(tampered, Config.seed).generate_c_files(out)


def test_weightless_model_fails_closed(base_desc, tmp_path, monkeypatch):
    """Fault injection: no weight tensor in the model stops generation before any C is written."""
    out = tmp_path / "case"
    _convert(base_desc, out)
    import helia_core_tester.generation.utils.litert_utils as litert_utils

    real = litert_utils.get_operator_tensors_from_litert

    def without_weights(*args, **kwargs):
        tensors = dict(real(*args, **kwargs))
        tensors["weights"] = None
        return tensors

    monkeypatch.setattr(litert_utils, "get_operator_tensors_from_litert", without_weights)
    with pytest.raises(RuntimeError, match="weight quantization could not be recovered"):
        OpFullyConnected(base_desc, Config.seed).generate_c_files(out)
    assert not list(out.rglob("*.c"))


def test_s4_c_files_reject_a_mismatched_filter(shipped, tmp_path):
    """The S4 C path takes filter dims from the descriptor, not the model; it validates them."""
    desc = copy.deepcopy(shipped[S4_BASE])
    out = tmp_path / "case"
    out.mkdir()
    OpFullyConnected(desc, Config.seed).convert_to_tflite(None, str(out / f"{S4_BASE}.tflite"), Config.seed)
    tampered = copy.deepcopy(desc)
    tampered["filter_shape"] = [4, 16]
    with pytest.raises(ValueError, match=r"InFeatures \(16\) must equal the flattened input feature count \(8\)"):
        OpFullyConnected(tampered, Config.seed).generate_c_files(out)
    assert not list(out.rglob("*.c"))
