from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.io.dtypes import descriptor_matches_dtype_filter
from helia_core_tester.generation.utils.litert_builder import (
    LITERT_AVAILABLE,
    build_unary_same_shape_op,
    litert,
)
from helia_core_tester.generation.utils.litert_utils import load_litert_model
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def _write_descriptor(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "descriptor.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_legacy_quantize_descriptor_resolves_float_io_foundation(tmp_path: Path) -> None:
    path = _write_descriptor(
        tmp_path,
        {
            "name": "quantize_contract_s8",
            "operator": "Quantize",
            "activation_dtype": "S8",
            "weight_dtype": "S8",
            "input_shape": [1, 4],
        },
    )

    desc = load_descriptor(str(path))[0]

    assert desc["resolved_tensor_dtypes"] == {"input": "FP32", "output": "S8", "weights": "S8"}
    assert desc["resolved_comparison"] == {"mode": "exact_int"}
    assert descriptor_matches_dtype_filter(desc, "FP32") is True
    assert descriptor_matches_dtype_filter(desc, "S8") is True


def test_tensor_dtypes_only_descriptor_derives_legacy_quantized_side(tmp_path: Path) -> None:
    path = _write_descriptor(
        tmp_path,
        {
            "name": "dequantize_contract_fp32",
            "operator": "Dequantize",
            "tensor_dtypes": {
                "input": "S8",
                "output": "FP32",
            },
            "input_shape": [1, 4],
        },
    )

    desc = load_descriptor(str(path))[0]

    assert desc["activation_dtype"] == "S8"
    assert desc.get("weight_dtype") is None
    assert desc["resolved_tensor_dtypes"] == {"input": "S8", "output": "FP32"}
    assert desc["resolved_comparison"] == {"mode": "float", "atol": 5.0e-5, "rtol": 2.0e-5}


def test_tensor_dtypes_only_descriptor_uses_fp16_default_comparison(tmp_path: Path) -> None:
    path = _write_descriptor(
        tmp_path,
        {
            "name": "abs_fp16_default_contract",
            "operator": "Abs",
            "tensor_dtypes": {
                "input": "FP16",
                "output": "FP16",
            },
            "input_shape": [1, 4],
        },
    )

    desc = load_descriptor(str(path))[0]

    assert desc["activation_dtype"] == "FP16"
    assert desc["resolved_tensor_dtypes"] == {"input": "FP16", "output": "FP16"}
    assert desc["resolved_comparison"] == {"mode": "float", "atol": 1.0e-3, "rtol": 1.0e-3}


def test_tensor_dtypes_accept_fp16_and_comparison_override(tmp_path: Path) -> None:
    path = _write_descriptor(
        tmp_path,
        {
            "name": "abs_fp16_contract",
            "operator": "Abs",
            "tensor_dtypes": {
                "input": "FP16",
                "output": "FP16",
            },
            "comparison": {
                "atol": 0.125,
                "rtol": 0.25,
            },
            "input_shape": [1, 4],
        },
    )

    desc = load_descriptor(str(path))[0]

    assert desc["activation_dtype"] == "FP16"
    assert desc["resolved_tensor_dtypes"] == {"input": "FP16", "output": "FP16"}
    assert desc["resolved_comparison"] == {"mode": "float", "atol": 0.125, "rtol": 0.25}
    assert descriptor_matches_dtype_filter(desc, "fp16") is True


def test_template_context_formats_float16_literals() -> None:
    rendered = TemplateContextBuilder.format_array_as_c_literal(
        np.array([0.5, -1.25], dtype=np.float16)
    )

    assert "(float16_t)0.500000f" in rendered
    assert "(float16_t)-1.250000f" in rendered


def test_template_context_formats_standalone_float_literals_for_c() -> None:
    assert TemplateContextBuilder.format_float_literal(1.0e30) == "1.0e+30f"
    assert TemplateContextBuilder.format_float_literal(-1.0e30) == "-1.0e+30f"
    assert TemplateContextBuilder.format_float_literal(0.125) == "0.125f"
    assert TemplateContextBuilder.format_float_literal(0.0) == "0.0f"


def _load_nn_activation_float_module(monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    fake_tf = SimpleNamespace(
        keras=SimpleNamespace(
            Model=object,
            activations=SimpleNamespace(sigmoid=lambda x: x, tanh=lambda x: x, linear=lambda x: x),
        ),
        nn=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "generation"
        / "ops"
        / "ActivationFunctions"
        / "nn_activation_float.py"
    )
    spec = importlib.util.spec_from_file_location("nn_activation_float_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nn_activation_float_fp16_tanh_reference_matches_scalar_fallback(monkeypatch) -> None:
    module = _load_nn_activation_float_module(monkeypatch)

    inputs = np.array(
        [
            0.708008,
            0.642578,
            -0.375000,
            0.572266,
            -0.504883,
            0.793945,
            -0.447998,
            0.749512,
            -0.481689,
            3.500000,
            -4.000000,
        ],
        dtype=np.float16,
    )
    expected = np.array(
        [
            0.617676,
            0.573242,
            -0.360107,
            0.521973,
            -0.469482,
            0.670898,
            -0.423096,
            0.644043,
            -0.450928,
            1.000000,
            -1.000000,
        ],
        dtype=np.float16,
    )

    actual = module._activation_reference(inputs, "ARM_NN_FLT_ACT_TANH", 0.0, "FP16")

    np.testing.assert_array_equal(actual, expected)


def test_nn_activation_float_fp16_tanh_reference_mve_matches_helium_lut(monkeypatch) -> None:
    module = _load_nn_activation_float_module(monkeypatch)

    # Inputs paired with the exact float16 output of the Cortex-M55 MVE tanh kernel
    # (arm_nn_vtanh_lut_direct_mve_f16), captured from FVP.
    inputs = np.array(
        [
            0.708008,
            0.642578,
            -0.375000,
            0.572266,
            -0.504883,
            0.793945,
            -0.447998,
            0.749512,
            -0.481689,
        ],
        dtype=np.float16,
    )
    expected = np.array(
        [
            0.609375,
            0.566895,
            -0.358398,
            0.517090,
            -0.466064,
            0.660645,
            -0.420410,
            0.634766,
            -0.447510,
        ],
        dtype=np.float16,
    )

    actual = module._activation_reference(
        inputs, "ARM_NN_FLT_ACT_TANH", 0.0, "FP16", use_mve_tanh=True
    )

    np.testing.assert_array_equal(actual, expected)


def test_nn_activation_float_fp32_tanh_reference_uses_numpy_tanh(monkeypatch) -> None:
    module = _load_nn_activation_float_module(monkeypatch)
    inputs = np.array([-0.75, -0.1, 0.0, 0.5, 0.8], dtype=np.float32)

    actual = module._activation_reference(inputs, "ARM_NN_FLT_ACT_TANH", 0.0, "FP32")

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.tanh(inputs.astype(np.float32)))


def test_build_pool_params_uses_float_activation_defaults_for_fp32() -> None:
    params = TemplateContextBuilder.build_pool_params(
        {
            "tensor_dtypes": {"input": "FP32", "output": "FP32"},
            "padding": "same",
            "strides": [2, 2],
        },
        (1, 6, 6, 3),
        (2, 2),
        (1, 3, 3, 3),
        {},
    )

    assert params["pad_h"] == 0
    assert params["pad_w"] == 0
    assert params["activation_min"] == -1.0e30
    assert params["activation_max"] == 1.0e30


def test_build_pool_params_keeps_quantized_activation_defaults() -> None:
    s8_params = TemplateContextBuilder.build_pool_params(
        {"activation_dtype": "S8", "padding": "valid", "strides": [1, 1]},
        (1, 6, 6, 3),
        (2, 2),
        (1, 5, 5, 3),
        {},
    )
    s16_params = TemplateContextBuilder.build_pool_params(
        {"activation_dtype": "S16", "padding": "valid", "strides": [1, 1]},
        (1, 6, 6, 3),
        (2, 2),
        (1, 5, 5, 3),
        {},
    )

    assert s8_params["activation_min"] == -128
    assert s8_params["activation_max"] == 127
    assert s16_params["activation_min"] == -32768
    assert s16_params["activation_max"] == 32767


def test_float_param_builders_use_public_float_activation_defaults() -> None:
    conv = TemplateContextBuilder.build_conv_params(
        {"tensor_dtypes": {"input": "FP32", "output": "FP32"}, "padding": "same", "strides": [1, 1]},
        (1, 6, 6, 3),
        (3, 3),
        (1, 6, 6, 5),
        {},
        {},
    )
    dw = TemplateContextBuilder.build_dw_conv_params(
        {"tensor_dtypes": {"input": "FP32", "output": "FP32"}, "padding": "same", "strides": [1, 1], "depth_multiplier": 1},
        (1, 6, 6, 3),
        (3, 3),
        (1, 6, 6, 3),
        {},
        {},
    )
    fc = TemplateContextBuilder.build_fc_params(
        {"tensor_dtypes": {"input": "FP32", "output": "FP32"}},
        {},
        {},
        {},
    )
    tconv = TemplateContextBuilder.build_transpose_conv_params(
        {"tensor_dtypes": {"input": "FP32", "output": "FP32"}, "padding": "same", "strides": [2, 2]},
        (1, 4, 4, 2),
        (3, 3),
        (1, 8, 8, 3),
        {},
        {},
    )

    for params in (conv, dw, fc, tconv):
        assert params["activation_min"] == -1.0e30
        assert params["activation_max"] == 1.0e30


@pytest.mark.skipif(not LITERT_AVAILABLE, reason="ai_edge_litert is required for float LiteRT round-trips")
@pytest.mark.parametrize(
    ("dtype", "expected_tensor_type_name"),
    [
        ("FP32", "FLOAT32"),
        ("FP16", "FLOAT16"),
    ],
)
def test_future_fp_ops_can_use_shared_litert_builder_without_infra_changes(
    tmp_path: Path,
    dtype: str,
    expected_tensor_type_name: str,
) -> None:
    model_bytes = build_unary_same_shape_op(
        op_name="ABS",
        input_shape=(1, 4),
        dtype=dtype,
        output_dtype=dtype,
    )
    model_path = tmp_path / f"abs_{dtype.lower()}.tflite"
    model_path.write_bytes(model_bytes)

    model, subgraph = load_litert_model(str(model_path))
    input_tensor = subgraph.tensors[subgraph.inputs[0]]
    output_tensor = subgraph.tensors[subgraph.outputs[0]]

    assert getattr(litert.TensorType, expected_tensor_type_name) == input_tensor.type
    assert input_tensor.type == output_tensor.type
