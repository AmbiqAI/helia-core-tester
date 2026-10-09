"""Quantize/Dequantize generation on the C reference: where the quantization comes from (the
policy or an explicit descriptor block, never a silent default), and what drives the activation."""

import math
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.ops.QuantizationFunctions.dequantize import OpDequantize
from helia_core_tester.generation.ops.QuantizationFunctions.quantize import OpQuantize
from helia_core_tester.generation.reference import policy


def _quantize_desc(name: str, activation: str, dtype: str, **extra) -> dict:
    return {
        "operator": "Quantize",
        "name": name,
        "tensor_dtypes": {"input": "FP32", "output": dtype},
        "activation_dtype": dtype,
        "activation": activation,
        "input_shape": [1, 4],
        "resolved_tensor_dtypes": {"input": "FP32", "output": dtype},
        "resolved_comparison": {"mode": "exact_int"},
        **extra,
    }


def _dequantize_desc(name: str, activation: str, dtype: str, **extra) -> dict:
    return {
        "operator": "Dequantize",
        "name": name,
        "tensor_dtypes": {"input": dtype, "output": "FP32"},
        "activation_dtype": dtype,
        "activation": activation,
        "input_shape": [1, 4],
        "resolved_tensor_dtypes": {"input": dtype, "output": "FP32"},
        "resolved_comparison": {"mode": "float", "atol": 1.0e-5, "rtol": 1.0e-5},
        **extra,
    }


@pytest.mark.parametrize(("activation", "dtype", "value_range"), [
    ("NONE", "S8", (-1.0, 1.0)), ("RELU", "S8", (0.0, 1.0)), ("RELU6", "S8", (0.0, 1.0)), ("RELU", "S16", (0.0, 1.0)),
])
def test_quantize_takes_the_policy_quantization(tmp_path: Path, activation, dtype, value_range) -> None:
    op = OpQuantize(_quantize_desc(f"quantize_policy_{dtype.lower()}", activation, dtype), seed=1,
                    target_cpu="cortex-m55")
    assert op.uses_reference() and not op.needs_keras_model()
    op.generate_c_files(tmp_path)
    call = op.reference
    want = policy.activation_quant(dtype.lower(), value_range)
    assert (call.params["scale"], call.params["zero_point"]) == (want.scale, want.zero_point)
    lo, hi = {"NONE": (-math.inf, math.inf), "RELU": (0.0, math.inf), "RELU6": (0.0, 6.0)}[activation]
    assert (call.params["activation_min"], call.params["activation_max"]) == (lo, hi)
    source = (tmp_path / f"quantize_policy_{dtype.lower()}_quantize.c").read_text()
    assert f"{want.scale}f" in source and f"{want.zero_point}," in source


def test_quantize_takes_an_explicit_descriptor_quantization(tmp_path: Path) -> None:
    desc = _quantize_desc("quantize_explicit_s8", "RELU", "S8",
                          quantization={"output": {"scale": 0.0625, "zero_point": 3}})
    op = OpQuantize(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    assert (op.reference.params["scale"], op.reference.params["zero_point"]) == (0.0625, 3)
    source = (tmp_path / "quantize_explicit_s8_quantize.c").read_text()
    assert "0.0625f" in source and "3," in source
    expected = np.clip(np.floor(np.maximum(op.reference.inputs["input"], 0) / 0.0625 + 0.5) + 3, -128, 127)
    np.testing.assert_array_equal(op.reference.output(), expected)


@pytest.mark.parametrize("block", [{"output": {"scale": 0.1, "range": [0, 1]}}, {"output": {"zero_point": 3}},
                                   {"output": {"scale": -0.1}}])
def test_quantize_refuses_a_malformed_quantization_block(tmp_path: Path, block) -> None:
    op = OpQuantize(_quantize_desc("quantize_bad_block_s8", "NONE", "S8", quantization=block), seed=1,
                    target_cpu="cortex-m55")
    with pytest.raises(ValueError):
        op.generate_c_files(tmp_path)


@pytest.mark.parametrize("dtype", ["S8", "S16"])
def test_dequantize_takes_the_policy_quantization(tmp_path: Path, dtype) -> None:
    op = OpDequantize(_dequantize_desc(f"dequantize_policy_{dtype.lower()}", "NONE", dtype), seed=1,
                      target_cpu="cortex-m55")
    assert op.uses_reference() and not op.needs_keras_model()
    op.generate_c_files(tmp_path)
    want = policy.activation_quant(dtype.lower(), (-1.0, 1.0))
    assert (op.reference.params["scale"], op.reference.params["zero_point"]) == (want.scale, want.zero_point)


def test_dequantize_name_does_not_drive_activation(tmp_path: Path) -> None:
    op = OpDequantize(_dequantize_desc("dequantize_relu_name_only_s8", "NONE", "S8"), seed=1,
                      target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    assert op.reference.params["activation_min"] == -math.inf
    assert op.reference.params["activation_max"] == math.inf
    assert np.any(op.reference.output() < 0)
