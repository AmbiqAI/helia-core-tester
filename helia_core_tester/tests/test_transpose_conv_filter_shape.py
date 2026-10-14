"""Issue #144: TransposeConv filter_shape [KH, KW, OutCh, InCh] is honored exactly.

Keras takes InCh from the input tensor, so a descriptor InCh that differed from
the input channel count used to be replaced without notice.
"""

import copy
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from helia_core_tester.core.config import Config  # noqa: E402
from helia_core_tester.core.discovery import find_descriptors_dir  # noqa: E402
from helia_core_tester.generation.io.descriptors import load_all_descriptors  # noqa: E402
from helia_core_tester.generation.ops.ConvolutionFunctions.transpose_conv import (  # noqa: E402
    OpTransposeConv,
)
from helia_core_tester.generation.test_ops import generate_test  # noqa: E402


def _transpose_conv_descs() -> dict:
    return {
        d["name"]: d
        for d in load_all_descriptors(str(find_descriptors_dir()))
        if d.get("operator") == "TransposeConv"
    }


def _base_desc() -> dict:
    return copy.deepcopy(_transpose_conv_descs()["transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8"])


def test_every_shipped_descriptor_declares_input_channels_as_filter_inch():
    descs = _transpose_conv_descs()
    assert len(descs) >= 50
    bad = {n: (d["input_shape"], d["filter_shape"]) for n, d in descs.items() if d["filter_shape"][3] != d["input_shape"][3]}
    assert bad == {}


def test_every_shipped_descriptor_passes_validation():
    for desc in _transpose_conv_descs().values():
        assert OpTransposeConv(desc, 1).declared_filter_shape() == tuple(desc["filter_shape"])


@pytest.mark.parametrize("in_ch", [1, 5, 16, 18])
def test_mismatched_inch_is_rejected(in_ch):
    desc = _base_desc()
    desc["filter_shape"][3] = in_ch
    with pytest.raises(ValueError, match=r"InCh \(%d\) must equal the input channel count \(17\)" % in_ch):
        OpTransposeConv(desc, 1).build_keras_model()


@pytest.mark.parametrize(
    "field, value",
    [
        ("filter_shape", [1, 1, 17]),
        ("filter_shape", [1, 1, 17, 17, 1]),
        ("filter_shape", None),
        ("filter_shape", "1,1,17,17"),
        ("filter_shape", [0, 1, 17, 17]),
        ("filter_shape", [1, -1, 17, 17]),
        ("filter_shape", [1, 1, 17.0, 17]),
        ("filter_shape", [1, 1, True, 17]),
        ("input_shape", [1, 2, 17]),
        ("input_shape", [1, 1, 2, 0]),
    ],
)
def test_malformed_shapes_are_rejected(field, value):
    desc = _base_desc()
    desc[field] = value
    with pytest.raises(ValueError, match="TransposeConv"):
        OpTransposeConv(desc, 1).declared_filter_shape()


def _filter_shape_from_tflite(tflite_path: Path) -> list:
    """Filter tensor shape [OutCh, KH, KW, InCh] of the model's TRANSPOSE_CONV op."""
    op = OpTransposeConv({"name": "probe"}, 1)
    tensors = op.load_primary_operator_tensors(str(tflite_path))
    return [int(v) for v in tensors["weights"].shape]


@pytest.mark.parametrize(
    "name",
    [
        # Previously mismatched (declared InCh 5 / 3 / 8 against input C 17 / 9 / 32).
        "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8",
        "transpose_conv_same_kernel6x6_stride2x2_bias_s8",
        "transpose_conv_valid_kernel2x2_stride3x1_no_bias_s8",
        "transpose_conv_same_kernel1x5_stride5x2_bias_s8",
    ],
)
def test_converted_filter_matches_all_four_declared_dims(tmp_path, name):
    desc = _transpose_conv_descs()[name]
    kh, kw, out_ch, in_ch = desc["filter_shape"]
    generate_test(desc, str(tmp_path), seed=Config.seed)
    (tflite,) = tmp_path.rglob(f"{name}.tflite")
    assert _filter_shape_from_tflite(tflite) == [out_ch, kh, kw, in_ch]
    c_text = next(tflite.parent.rglob("*.c")).read_text()
    assert c_text  # generation completed through the post-conversion check


def test_converted_filter_disagreeing_with_descriptor_is_rejected(tmp_path):
    """Fault injection: the model on disk no longer matches the descriptor's filter."""
    desc = _base_desc()
    op = OpTransposeConv(desc, Config.seed)
    out = tmp_path / "case"
    out.mkdir()
    op.convert_to_tflite(op.build_keras_model(), str(out / f"{desc['name']}.tflite"), Config.seed)

    tampered = copy.deepcopy(desc)
    tampered["filter_shape"][2] = 8  # OutCh in the model stays 17
    with pytest.raises(ValueError, match="does not match the descriptor's filter_shape"):
        OpTransposeConv(tampered, Config.seed).generate_c_files(out)
