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
    assert desc["resolved_comparison"] == {"mode": "float", "atol": 0.01, "rtol": 0.001}


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
