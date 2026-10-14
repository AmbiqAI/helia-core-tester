"""Issue #144: TransposeConv filter_shape [KH, KW, OutCh, InCh] is honored exactly.

Keras takes InCh from the input tensor, so a descriptor InCh that differed from
the input channel count used to be replaced without notice.
"""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from helia_core_tester.core.config import Config  # noqa: E402
from helia_core_tester.core.discovery import find_descriptors_dir  # noqa: E402
from helia_core_tester.generation.io import descriptors as desc_io  # noqa: E402
from helia_core_tester.generation.ops.ConvolutionFunctions.transpose_conv import (  # noqa: E402
    OpTransposeConv,
)
from helia_core_tester.generation.test_ops import generate_test  # noqa: E402

BASE = "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8"

# Written in Conv2D order [KH, KW, InCh, OutCh] before #144; now [KH, KW, OutCh, InCh].
CHANNEL_CHANGING = {
    "transpose_conv_same_kernel6x6_stride2x2_bias_s8": (9, 3),
    "transpose_conv_valid_kernel3x3_stride1x3_no_bias_s8": (15, 4),
    "transpose_conv_same_kernel1x5_stride5x2_bias_s8": (7, 5),
    "transpose_conv_valid_kernel3x3_stride1x3_no_bias_case_02_s8": (32, 5),
    "transpose_conv_reverse_same_kernel2x2_stride1x1_bias_s8": (32, 2),
    "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8": (17, 5),
    "transpose_conv_reverse_same_kernel3x3_stride1x1_bias_s8": (16, 8),
    "transpose_conv_valid_kernel2x2_stride3x1_no_bias_s8": (32, 8),
    "transpose_conv_same_kernel4x6_stride2x2_bias_padrem_h_s8": (8, 4),
    "transpose_conv_same_kernel6x4_stride2x2_no_bias_padrem_w_s8": (8, 4),
    "transpose_conv_same_kernel4x4_stride2x2_bias_padrem_hw_s8": (8, 4),
}


@pytest.fixture(scope="module")
def shipped() -> dict:
    path = find_descriptors_dir() / "ConvolutionFunctions" / "transpose_conv.yaml"
    return {
        expanded["name"]: expanded
        for desc in desc_io.load_descriptor(str(path))
        for expanded in desc_io.expand_descriptor_variations(desc)
    }


@pytest.fixture
def base_desc(shipped) -> dict:
    return copy.deepcopy(shipped[BASE])


def test_every_shipped_descriptor_declares_input_channels_as_filter_inch(shipped):
    assert len(shipped) >= 30
    bad = {n: (d["input_shape"], d["filter_shape"]) for n, d in shipped.items() if d["filter_shape"][3] != d["input_shape"][3]}
    assert bad == {}


def test_every_shipped_descriptor_passes_validation(shipped):
    for desc in shipped.values():
        assert OpTransposeConv(desc, 1).declared_filter_shape() == tuple(desc["filter_shape"])


def test_formerly_mismatched_cases_change_the_channel_count(shipped):
    for name, (in_ch, out_ch) in CHANNEL_CHANGING.items():
        assert shipped[name]["input_shape"][3] == in_ch, name
        assert shipped[name]["filter_shape"][2:] == [out_ch, in_ch], name


@pytest.mark.parametrize("in_ch", [1, 5, 16, 18])
def test_mismatched_inch_is_rejected(base_desc, in_ch):
    base_desc["filter_shape"][3] = in_ch
    with pytest.raises(ValueError, match=r"InCh \(%d\) must equal the input channel count \(17\)" % in_ch):
        OpTransposeConv(base_desc, 1).build_keras_model()


def test_conv2d_order_gets_a_swap_hint(base_desc):
    base_desc["filter_shape"] = [1, 1, 17, 5]  # [KH, KW, InCh, OutCh]
    with pytest.raises(ValueError, match=re.escape("did you mean [1, 1, 5, 17] (OutCh 5)?")):
        OpTransposeConv(base_desc, 1).declared_filter_shape()


def test_no_swap_hint_when_outch_differs_too(base_desc):
    base_desc["filter_shape"] = [1, 1, 6, 5]
    with pytest.raises(ValueError) as excinfo:
        OpTransposeConv(base_desc, 1).declared_filter_shape()
    assert "did you mean" not in str(excinfo.value)


def test_numpy_integer_dims_are_accepted(base_desc):
    base_desc["input_shape"] = [np.int64(v) for v in base_desc["input_shape"]]
    base_desc["filter_shape"] = [np.int32(v) for v in base_desc["filter_shape"]]
    shape = OpTransposeConv(base_desc, 1).declared_filter_shape()
    assert shape == (1, 1, 5, 17)
    assert all(type(dim) is int for dim in shape)


MALFORMED = [
    # (field, value, also rejected by schema.json at load time)
    ("filter_shape", [1, 1, 17], True),
    ("filter_shape", [1, 1, 5, 17, 1], True),
    ("filter_shape", None, True),
    ("filter_shape", "1,1,5,17", True),
    ("filter_shape", [0, 1, 5, 17], True),
    ("filter_shape", [1, -1, 5, 17], True),
    ("filter_shape", [1, 1, True, 17], True),
    # JSON Schema's integer admits 5.0; only generation rejects it.
    ("filter_shape", [1, 1, 5.0, 17], False),
    ("input_shape", [1, 2, 17], True),
    ("input_shape", [1, 1, 2, 0], True),
]


@pytest.mark.parametrize("field, value, _schema", MALFORMED)
def test_malformed_shapes_are_rejected(base_desc, field, value, _schema):
    base_desc[field] = value
    with pytest.raises(ValueError, match="TransposeConv"):
        OpTransposeConv(base_desc, 1).declared_filter_shape()


@pytest.mark.parametrize("field, value, schema_rejects", MALFORMED)
def test_schema_rejects_the_same_shapes_at_load_time(field, value, schema_rejects):
    pytest.importorskip("jsonschema")
    doc = {
        "operator": "TransposeConv",
        "name": "transpose_conv_case_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "input_shape": [1, 1, 2, 17],
        "filter_shape": [1, 1, 5, 17],
    }
    desc_io.validate_descriptor_schema(doc, "transpose_conv")
    if value is None:
        doc.pop(field)
    else:
        doc[field] = value
    if schema_rejects:
        with pytest.raises(ValueError, match="violates schema.json"):
            desc_io.validate_descriptor_schema(doc, "transpose_conv")
    else:
        desc_io.validate_descriptor_schema(doc, "transpose_conv")


def _filter_shape_from_tflite(tflite_path: Path) -> list:
    """Filter tensor shape [OutCh, KH, KW, InCh] of the model's TRANSPOSE_CONV op."""
    op = OpTransposeConv({"name": "probe"}, 1)
    tensors = op.load_primary_operator_tensors(str(tflite_path))
    return [int(v) for v in tensors["weights"].shape]


def _header_dims(header: str, name: str, which: str) -> dict:
    match = re.search(rf"{name}_{which}_dims = \{{([^}}]*)\}}", header)
    assert match, f"{which}_dims not found"
    return {k: int(v) for k, v in re.findall(r"\.(\w) = (-?\d+)", match.group(1))}


@pytest.mark.parametrize(
    "name",
    [
        "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8",
        "transpose_conv_same_kernel6x6_stride2x2_bias_s8",
        "transpose_conv_valid_kernel2x2_stride3x1_no_bias_s8",
        "transpose_conv_same_kernel1x5_stride5x2_bias_s8",
    ],
)
def test_converted_filter_matches_all_four_declared_dims(shipped, tmp_path, name):
    desc = shipped[name]
    kh, kw, out_ch, in_ch = desc["filter_shape"]
    assert out_ch != in_ch
    generate_test(desc, str(tmp_path), seed=Config.seed)
    (tflite,) = tmp_path.rglob(f"{name}.tflite")
    assert _filter_shape_from_tflite(tflite) == [out_ch, kh, kw, in_ch]
    header = next(tflite.parent.rglob(f"{name}_transpose_conv.h")).read_text()
    assert _header_dims(header, name, "filter") == {"n": out_ch, "h": kh, "w": kw, "c": in_ch}
    assert _header_dims(header, name, "input")["c"] == in_ch
    assert _header_dims(header, name, "output")["c"] == out_ch


def _convert(desc: dict, out: Path) -> OpTransposeConv:
    op = OpTransposeConv(desc, Config.seed)
    out.mkdir()
    op.convert_to_tflite(op.build_keras_model(), str(out / f"{desc['name']}.tflite"), Config.seed)
    return op


def test_converted_filter_disagreeing_with_descriptor_is_rejected(base_desc, tmp_path):
    """Fault injection: the model on disk no longer matches the descriptor's filter."""
    out = tmp_path / "case"
    _convert(base_desc, out)
    tampered = copy.deepcopy(base_desc)
    tampered["filter_shape"][2] = 8  # OutCh in the model stays 5
    with pytest.raises(ValueError, match="does not match the descriptor's filter_shape"):
        OpTransposeConv(tampered, Config.seed).generate_c_files(out)


def test_no_weights_fallback_rejects_a_mismatched_inch(base_desc, tmp_path, monkeypatch):
    """The descriptor-only fallback (no weight tensor in the model) validates too."""
    out = tmp_path / "case"
    _convert(base_desc, out)
    real = OpTransposeConv.load_primary_operator_tensors

    def without_weights(self, path):
        tensors = dict(real(self, path))
        tensors["weights"] = None
        return tensors

    monkeypatch.setattr(OpTransposeConv, "load_primary_operator_tensors", without_weights)
    tampered = copy.deepcopy(base_desc)
    tampered["filter_shape"][3] = 16
    with pytest.raises(ValueError, match=r"InCh \(16\) must equal the input channel count \(17\)"):
        OpTransposeConv(tampered, Config.seed).generate_c_files(out)
