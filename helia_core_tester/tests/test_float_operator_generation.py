from __future__ import annotations

from pathlib import Path

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.catalog import get_operator_spec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_float_descriptor_suite_is_registered_in_catalog() -> None:
    expected = {
        "NNActivationFloat": "ActivationFunctions/nn_activation_float.yaml",
        "BatchNorm": "NNSupportFunctions/batch_norm_float.yaml",
        "Add": "BasicMathFunctions/add_float.yaml",
        "Mul": "BasicMathFunctions/mul_float.yaml",
        "Maximum": "BasicMathFunctions/maximum_float.yaml",
        "Minimum": "BasicMathFunctions/minimum_float.yaml",
        "Concatenation": "ConcatenationFunctions/concatenation_float.yaml",
        "Convolve": "ConvolutionFunctions/convolve_float.yaml",
        "DepthwiseConv": "ConvolutionFunctions/depthwise_conv_float.yaml",
        "FullyConnected": "FullyConnectedFunctions/fully_connected_float.yaml",
        "BatchMatMul": "FullyConnectedFunctions/batch_matmul_float.yaml",
        "TransposeConv": "ConvolutionFunctions/transpose_conv_float.yaml",
        "SVDF": "SVDFunctions/svdf_float.yaml",
        "LSTMUnidirectional": "LSTMFunctions/lstm_unidirectional_float.yaml",
        "GRUUnidirectional": "LSTMFunctions/gru_unidirectional_float.yaml",
        "Pad": "PadFunctions/pad_float.yaml",
        "AvgPool": "PoolingFunctions/avg_pool_float.yaml",
        "MaxPool": "PoolingFunctions/max_pool_float.yaml",
        "Reshape": "ReshapeFunctions/reshape_float.yaml",
        "Softmax": "SoftmaxFunctions/softmax_float.yaml",
        "Transpose": "TransposeFunctions/transpose_float.yaml",
    }

    for operator, descriptor_relpath in expected.items():
        spec = get_operator_spec(operator)
        assert descriptor_relpath in spec.descriptor_relpaths


def test_float_descriptors_load_and_are_tagged_as_float_suite() -> None:
    descriptors = load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    float_names = {desc["name"] for desc in descriptors if desc.get("_descriptor_suite") == "float"}

    expected = {
        "nn_activation_float_sigmoid_f32",
        "nn_activation_float_tanh_f32",
        "nn_activation_float_hardswish_f32",
        "nn_activation_float_leaky_relu_f32",
        "batch_norm_default_f32",
        "add_float_default_f32",
        "add_float_tail_f32",
        "add_float_legacy_fp16_tail",
        "mul_float_default_f32",
        "mul_float_tail_f32",
        "maximum_float_default_f32",
        "minimum_float_default_f32",
        "concatenation_axis_x_f32",
        "convolve_float_default_f32",
        "depthwise_conv_float_default_f32",
        "fully_connected_float_default_f32",
        "batch_matmul_float_default_f32",
        "transpose_conv_float_default_f32",
        "svdf_float_default_f32",
        "lstm_unidirectional_float_default_f32",
        "gru_unidirectional_float_reset_after_f32",
        "pad_float_default_f32",
        "avg_pool_float_default_f32",
        "max_pool_float_default_f32",
        "reshape_float_default_f32",
        "softmax_float_default_f32",
        "transpose_float_default_f32",
    }

    assert expected.issubset(float_names)


def test_templates_use_umbrella_public_headers_only() -> None:
    templates_root = _repo_root() / "assets" / "templates"
    texts = "\n".join(path.read_text() for path in templates_root.rglob("*.j2"))

    assert "arm_nnfunctions_flt.h" not in texts
    assert "arm_nn_types_flt.h" not in texts
    assert "arm_nnfunctions.h" in texts
