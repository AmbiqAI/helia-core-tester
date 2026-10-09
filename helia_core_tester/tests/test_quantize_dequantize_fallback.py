from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.ops.QuantizationFunctions.quantize import OpQuantize
from helia_core_tester.generation.ops.QuantizationFunctions.dequantize import OpDequantize


def _quantize_desc(name: str, activation: str, dtype: str) -> dict:
    return {
        "operator": "Quantize",
        "name": name,
        "tensor_dtypes": {
            "input": "FP32",
            "output": dtype,
        },
        "activation_dtype": dtype,
        "activation": activation,
        "input_shape": [1, 4],
        "resolved_tensor_dtypes": {
            "input": "FP32",
            "output": dtype,
        },
        "resolved_comparison": {
            "mode": "exact_int",
        },
    }


def _dequantize_desc(name: str, activation: str, dtype: str) -> dict:
    return {
        "operator": "Dequantize",
        "name": name,
        "tensor_dtypes": {
            "input": dtype,
            "output": "FP32",
        },
        "activation_dtype": dtype,
        "activation": activation,
        "input_shape": [1, 4],
        "resolved_tensor_dtypes": {
            "input": dtype,
            "output": "FP32",
        },
        "resolved_comparison": {
            "mode": "float",
            "atol": 1.0e-5,
            "rtol": 1.0e-5,
        },
    }


def _array(text: str, name: str) -> np.ndarray:
    import re

    body = re.search(rf"{name}\[[^]]*\]\s*=\s*\{{([^}}]+)", text).group(1)
    return np.array([float(v.strip().rstrip("f")) for v in body.replace("\n", " ").split(",") if v.strip()])


def _render(op, tmp_path: Path, stem: str) -> str:
    op.generate_c_files(tmp_path)
    name = op.desc["name"]
    return (tmp_path / f"{name}_{stem}.c").read_text() + (tmp_path / "includes" / f"{name}_{stem}.h").read_text()


@pytest.mark.parametrize(
    ("name", "activation", "dtype", "scale", "zero_point"),
    [
        ("quantize_relu_s8", "RELU", "S8", 0.0625, 3),
        ("quantize_relu_s16", "RELU", "S16", 0.00390625, 0),
        ("quantize_relu6_vec_s8", "RELU6", "S8", 0.125, -5),
        ("quantize_none_s8", "NONE", "S8", 0.03125, 7),
    ],
)
def test_quantize_golden_is_affine_quantize_of_the_activated_input(tmp_path, name, activation, dtype, scale, zero_point):
    desc = {**_quantize_desc(name, activation, dtype), "quantization": {"output": {"scale": scale, "zero_point": zero_point}}}
    text = _render(OpQuantize(desc, seed=1, target_cpu="cortex-m55"), tmp_path, "quantize")
    assert f"{zero_point}," in text and f"{scale}f" in text
    x = _array(text, f"{name}_input").astype(np.float32)
    if activation != "NONE":
        x = np.clip(x, 0.0, 6.0 if activation == "RELU6" else np.inf).astype(np.float32)
    info = np.iinfo(np.int8 if dtype == "S8" else np.int16)
    q = x / np.float32(scale)
    expected = np.clip(np.sign(q) * np.floor(np.abs(q) + 0.5) + zero_point, info.min, info.max)
    np.testing.assert_array_equal(_array(text, f"{name}_expected_output"), expected)


def test_quantize_rejects_a_malformed_quantization_block(tmp_path) -> None:
    desc = {**_quantize_desc("quantize_bad_block_s8", "NONE", "S8"),
            "quantization": {"output": {"scale": 0.1, "range": [-1.0, 1.0]}}}
    with pytest.raises(ValueError, match="exactly one of"):
        OpQuantize(desc, seed=1, target_cpu="cortex-m55").generate_c_files(tmp_path)


def test_dequantize_honours_explicit_input_quantization(tmp_path) -> None:
    name = "dequantize_explicit_s8"
    desc = {**_dequantize_desc(name, "NONE", "S8"), "quantization": {"input": {"scale": 0.05, "zero_point": -9}}}
    text = _render(OpDequantize(desc, seed=1, target_cpu="cortex-m55"), tmp_path, "dequantize")
    q = _array(text, f"{name}_input")
    np.testing.assert_allclose(_array(text, f"{name}_expected_output"), (q + 9) * np.float32(0.05), rtol=1e-6)


def test_dequantize_name_does_not_drive_activation(tmp_path) -> None:
    name = "dequantize_relu_name_only_s8"
    text = _render(OpDequantize(_dequantize_desc(name, "NONE", "S8"), seed=1, target_cpu="cortex-m55"),
                   tmp_path, "dequantize")
    assert (_array(text, f"{name}_expected_output") < 0).any()
