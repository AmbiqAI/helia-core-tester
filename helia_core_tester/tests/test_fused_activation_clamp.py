"""Kernel clamps follow the descriptor's fused activation (#266)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import expand_descriptor_variations, load_descriptor
from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import OpFullyConnected
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

clamp = TemplateContextBuilder.activation_clamp

S16_OUT = {"scale": 0.0005, "zero_point": 0}


def _conv(desc: dict, output_quant: dict) -> dict:
    return TemplateContextBuilder.build_conv_params(desc, (1, 3, 5, 8), (1, 1), (1, 3, 5, 7), {}, output_quant)


def _dw(desc: dict, output_quant: dict) -> dict:
    return TemplateContextBuilder.build_dw_conv_params(
        {**desc, "depth_multiplier": 1}, (1, 5, 9, 4), (3, 3), (1, 5, 9, 4), {}, output_quant
    )


def _tconv(desc: dict, output_quant: dict) -> dict:
    return TemplateContextBuilder.build_transpose_conv_params(desc, (1, 4, 4, 2), (3, 3), (1, 8, 8, 3), {}, output_quant)


def _fc(desc: dict, output_quant: dict) -> dict:
    return TemplateContextBuilder.build_fc_params(desc, {}, {}, output_quant)


BUILDERS = pytest.mark.parametrize("build", [_conv, _dw, _tconv, _fc], ids=["conv", "dw", "tconv", "fc"])


@BUILDERS
@pytest.mark.parametrize(("activation", "expected"), [("RELU", (0, 32767)), ("RELU6", (0, 12000)), ("NONE", (-32768, 32767))])
def test_s16_builders_clamp_at_the_fused_activation(build, activation: str, expected: tuple[int, int]) -> None:
    params = build({"activation_dtype": "S16", "activation": activation, "padding": "same", "strides": [1, 1]}, S16_OUT)
    assert (params["activation_min"], params["activation_max"]) == expected


@BUILDERS
def test_s8_builders_clamp_at_the_output_zero_point(build) -> None:
    params = build({"activation_dtype": "S8", "activation": "RELU", "padding": "same", "strides": [1, 1]}, {"scale": 0.02, "zero_point": -17})
    assert (params["activation_min"], params["activation_max"]) == (-17, 127)


@BUILDERS
@pytest.mark.parametrize(("activation", "expected"), [("RELU", (0.0, 1.0e30)), ("RELU6", (0.0, 6.0)), ("NONE", (-1.0e30, 1.0e30))])
def test_float_builders_clamp_at_the_fused_activation(build, activation: str, expected: tuple[float, float]) -> None:
    desc = {"tensor_dtypes": {"input": "FP32", "output": "FP32"}, "activation": activation, "padding": "same", "strides": [1, 1]}
    params = build(desc, {})
    assert (params["activation_min"], params["activation_max"]) == expected


@pytest.mark.parametrize(
    ("dtype", "quant", "activation", "expected"),
    [
        ("S8", {"scale": 6 / 255, "zero_point": -128}, "RELU6", (-128, 127)),
        ("S8", {"scale": 0.1, "zero_point": 0}, "RELU6", (0, 60)),
        ("S8", {"scale": 0.1, "zero_point": 100}, "RELU6", (100, 127)),
        ("S8", {"scale": 0.1, "zero_point": -200}, "RELU", (-128, 127)),
        ("S8", {"scale": 0.1, "zero_point": 127}, "RELU", (127, 127)),
        ("S16", {"scale": 6 / 32767, "zero_point": 0}, "RELU6", (0, 32767)),
        ("S16", {"scale": 1e-6, "zero_point": 0}, "RELU6", (0, 32767)),
        # 6 / 2.4 = 2.5: TfLiteRound goes away from zero, Python round() would give 2.
        ("S8", {"scale": 2.4, "zero_point": 0}, "RELU6", (0, 3)),
        ("S8", {"scale": 2.4, "zero_point": -10}, "RELU6", (-10, -7)),
        # The FlatBuffer stores 2.400000095...: 6 / scale is 2.4999999 in float64 but exactly 2.5 in float32.
        ("S8", {"scale": np.float32(2.4), "zero_point": 0}, "RELU6", (0, 3)),
        ("S8", {"scale": np.array([2.4], dtype=np.float32), "zero_point": np.array([0])}, "RELU6", (0, 3)),
        ("S8", {"scale": float(np.float32(2.4)), "zero_point": 0}, "RELU6", (0, 3)),
    ],
    ids=[
        "s8-full", "s8-zp0", "s8-zp-high", "s8-zp-below-qmin", "s8-zp-qmax", "s16-full", "s16-cap",
        "half-away", "half-away-zp", "float32-scale", "float32-array-scale", "float32-widened-scale",
    ],
)
def test_clamp_matches_tflite_activation_range(dtype: str, quant: dict, activation: str, expected: tuple[int, int]) -> None:
    assert clamp({"activation": activation}, dtype, quant) == expected


@pytest.mark.parametrize("activation", [None, "NONE", "none", "TANH", "SIGMOID"])
def test_unfused_activations_keep_the_dtype_range(activation) -> None:
    # No output quantization needed when nothing is fused.
    assert clamp({"activation": activation}, "S16", None) == (-32768, 32767)
    assert clamp({"activation": activation}, "S8", None) == (-128, 127)
    assert clamp({"activation": activation}, "FP32", None) == (-1.0e30, 1.0e30)
    assert clamp({}, "S8", None) == (-128, 127)


def test_lowercase_activation_names_are_accepted() -> None:
    assert clamp({"activation": "relu6"}, "S8", {"scale": 0.1, "zero_point": 0}) == (0, 60)


def test_quant_params_may_be_arrays() -> None:
    quant = {"scale": np.array([0.1], dtype=np.float32), "zero_point": np.array([-3], dtype=np.int64)}
    assert clamp({"activation": "RELU"}, "S8", quant) == (-3, 127)
    assert clamp({"activation": "RELU"}, "S8", {"scale": [0.1], "zero_point": [4]}) == (4, 127)


def test_explicit_bounds_win() -> None:
    quant = {"scale": 0.1, "zero_point": 0}
    assert clamp({"activation": "RELU6", "activation_min": 5}, "S8", quant) == (5, 60)
    assert clamp({"activation": "RELU6", "activation_max": 30}, "S8", quant) == (0, 30)
    # Both explicit: no output quantization is read.
    assert clamp({"activation": "RELU", "activation_min": -7, "activation_max": 9}, "S16", None) == (-7, 9)
    assert clamp({"activation": "RELU", "activation_min": -2.0}, "FP32", None) == (-2.0, 1.0e30)


@pytest.mark.parametrize("activation", ["GELU", "RELU_N1_TO_1", "", 3])
def test_unsupported_activation_is_rejected(activation) -> None:
    with pytest.raises(ValueError, match=r"^Unsupported activation: "):
        clamp({"activation": activation}, "S8", {"scale": 0.1, "zero_point": 0})


@pytest.mark.parametrize(
    "quant",
    [
        None, {}, {"scale": 0.0}, {"scale": -0.1}, {"scale": math.nan}, {"scale": math.inf}, {"scale": []},
        {"scale": 1e300}, {"scale": 1e-300}, {"scale": "abc"}, {"scale": object()}, {"scale": True},
    ],
    ids=[
        "none", "empty", "zero", "negative", "nan", "inf", "empty-list",
        "float32-overflow", "float32-underflow", "string", "object", "bool",
    ],
)
@pytest.mark.parametrize("activation", ["RELU", "RELU6"])
def test_fused_int_clamp_needs_a_valid_output_scale(quant, activation: str) -> None:
    with pytest.raises(ValueError, match=rf"^{activation} needs a positive finite output scale, got "):
        clamp({"activation": activation}, "S16", quant)


@pytest.mark.parametrize(
    "zero_point",
    [0.5, True, math.inf, -math.inf, math.nan, "abc", object(), None],
    ids=["fraction", "bool", "inf", "neg-inf", "nan", "string", "object", "none"],
)
def test_fused_int_clamp_needs_an_integer_zero_point(zero_point) -> None:
    with pytest.raises(ValueError, match=r"^RELU needs an integer output zero point, got "):
        clamp({"activation": "RELU"}, "S8", {"scale": 0.1, "zero_point": zero_point})


@pytest.mark.parametrize(
    ("desc", "dtype"),
    [
        ({"activation": "RELU", "activation_max": -1}, "S16"),
        ({"activation": "NONE", "activation_min": 10, "activation_max": 9}, "S8"),
        ({"activation": "RELU6", "activation_min": 7.0}, "FP32"),
    ],
    ids=["relu-below-max", "explicit-inverted", "float-relu6-inverted"],
)
def test_empty_clamp_is_rejected(desc: dict, dtype: str) -> None:
    with pytest.raises(ValueError, match=r" clamp is empty for "):
        clamp(desc, dtype, {"scale": 0.1, "zero_point": 0})


def test_conv_builder_propagates_a_bad_output_scale() -> None:
    with pytest.raises(ValueError, match="positive finite output scale"):
        _conv({"activation_dtype": "S16", "activation": "RELU"}, {"scale": 0.0, "zero_point": 0})


def test_fully_connected_int_path_uses_the_shared_clamp() -> None:
    op = OpFullyConnected.__new__(OpFullyConnected)
    op.desc = {"activation": "RELU6"}
    quant = {"scale": 2.4, "zero_point": -10}
    assert op._compute_activation_range(quant, np.dtype(np.int8)) == clamp(op.desc, "S8", quant) == (-10, -7)
    assert op._compute_activation_range(S16_OUT, np.dtype(np.int16)) == (0, 12000)


CORPUS_FILES = (
    "ConvolutionFunctions/convolve.yaml",
    "ConvolutionFunctions/depthwise_conv.yaml",
    "FullyConnectedFunctions/fully_connected.yaml",
)


def test_corpus_covers_fused_s16_activations() -> None:
    root = find_descriptors_dir()
    present = {
        (d.get("operator"), d.get("activation"))
        for rel in CORPUS_FILES
        for doc in load_descriptor(str(root / rel))
        for d in expand_descriptor_variations(doc)
        if d.get("activation_dtype") == "S16" and "activation_min" not in d
    }
    for operator in ("Convolve", "DepthwiseConv", "FullyConnected"):
        for activation in ("RELU", "RELU6"):
            assert (operator, activation) in present


@pytest.mark.parametrize("dtype", ["S32", "U8", "U16", "S4", "", "int8"])
def test_unknown_int_activation_dtype_is_rejected(dtype: str) -> None:
    for activation in ("NONE", "RELU"):
        with pytest.raises(ValueError, match=r"^No kernel activation range for activation dtype "):
            clamp({"activation": activation}, dtype, {"scale": 0.1, "zero_point": 0})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("activation_min", 0.5, "must be an integer"),
        ("activation_max", True, "must be an integer"),
        ("activation_min", math.nan, "must be an integer"),
        ("activation_max", math.inf, "must be an integer"),
        ("activation_min", "0", "must be an integer"),
        ("activation_min", -129, r"outside the S8 range \[-128, 127\]"),
        ("activation_max", 128, r"outside the S8 range \[-128, 127\]"),
    ],
)
def test_explicit_int_bounds_are_validated(key: str, value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        clamp({"activation": "NONE", key: value}, "S8", None)


@pytest.mark.parametrize(
    ("desc", "expected"),
    [
        ({"activation_min": -128, "activation_max": 127}, (-128, 127)),
        ({"activation_min": np.int16(-5), "activation_max": np.int64(7)}, (-5, 7)),
        ({"activation_min": -5.0}, (-5, 32767)),
        ({"activation_min": -32768, "activation_max": 32767}, (-32768, 32767)),
    ],
    ids=["s8-limits", "numpy-ints", "integral-float", "s16-limits"],
)
def test_explicit_int_bounds_at_the_limits_are_accepted(desc: dict, expected) -> None:
    dtype = "S8" if expected[1] == 127 else "S16"
    assert clamp({"activation": "NONE", **desc}, dtype, None) == expected


def test_explicit_s16_bound_outside_the_range_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"outside the S16 range"):
        clamp({"activation": "RELU", "activation_max": 40000}, "S16", S16_OUT)


@BUILDERS
def test_builders_read_zero_points_from_arrays_and_empty_lists(build) -> None:
    desc = {"activation_dtype": "S8", "activation": "NONE", "padding": "same", "strides": [1, 1]}
    params = build(desc, {"scale": 0.1, "zero_point": np.array([-3])})
    assert params["output_offset"] == -3
    assert build(desc, {"scale": 0.1, "zero_point": []})["output_offset"] == 0


def test_zero_points_are_zero_for_float_kernels() -> None:
    assert TemplateContextBuilder._zero_points("FP32", {"zero_point": 5}, {"zero_point": -7}) == (0, 0)
    assert TemplateContextBuilder._zero_points("S8", {"zero_point": [5]}, {"zero_point": np.array([-7])}) == (5, -7)


def _bmm(activation) -> "OpBatchMatMul":
    from helia_core_tester.generation.ops.FullyConnectedFunctions.batch_matmul import OpBatchMatMul

    desc = {
        "operator": "BatchMatMul",
        "name": "batch_matmul_case_s8",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "input_1_shape": [1, 2, 3],
        "input_2_shape": [1, 3, 4],
        "activation": activation,
    }
    return OpBatchMatMul(desc, 1)


@pytest.mark.parametrize("activation", ["RELU", "RELU6", "relu", "TANH"])
def test_batch_matmul_rejects_a_fused_activation(activation, tmp_path) -> None:
    for call in (lambda op: op.build_keras_model(), lambda op: op.generate_c_files(tmp_path)):
        with pytest.raises(ValueError, match="BatchMatMul has no fused activation"):
            call(_bmm(activation))


@pytest.mark.parametrize("activation", ["NONE", None])
def test_batch_matmul_without_activation_builds(activation) -> None:
    assert _bmm(activation).build_keras_model() is not None


def _layer_kinds(model) -> list[str]:
    return [type(layer).__name__ for layer in model.layers]


@pytest.mark.parametrize(("activation", "relu_max"), [("relu", None), ("relu6", 6.0), ("Relu6", 6.0)])
def test_model_builders_accept_the_names_the_clamp_accepts(activation: str, relu_max) -> None:
    from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import OpDepthwiseConv

    base = {"activation_dtype": "S8", "weight_dtype": "S8", "padding": "SAME", "strides": [1, 1]}
    dw = OpDepthwiseConv(
        {**base, "operator": "DepthwiseConv", "name": "dw_case_s8", "input_shape": [1, 4, 4, 2],
         "filter_shape": [3, 3, 2, 1], "depth_multiplier": 1, "activation": activation},
        1,
    ).build_keras_model()
    relus = [layer for layer in dw.layers if type(layer).__name__ == "ReLU"]
    assert len(relus) == 1
    assert (None if relus[0].max_value is None else float(relus[0].max_value)) == relu_max

    fc = OpFullyConnected(
        {**base, "operator": "FullyConnected", "name": "fc_case_s8", "input_shape": [1, 8],
         "filter_shape": [4, 8], "activation": activation},
        1,
    ).build_keras_model()
    assert "ReLU" in _layer_kinds(fc)
