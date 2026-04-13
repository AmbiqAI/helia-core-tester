from pathlib import Path

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
            "atol": 0.01,
            "rtol": 0.001,
        },
    }


def _assert_generated(output_dir: Path, name: str, suffix: str) -> None:
    header = output_dir / "includes" / f"{name}_{suffix}.h"
    source = output_dir / f"{name}_{suffix}.c"
    assert header.exists()
    assert source.exists()


def test_quantize_generates_without_tflite(tmp_path: Path) -> None:
    desc = _quantize_desc("quantize_fp32_to_s8_basic", "NONE", "S8")
    op = OpQuantize(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    _assert_generated(tmp_path, desc["name"], "quantize")


def test_dequantize_generates_without_tflite(tmp_path: Path) -> None:
    desc = _dequantize_desc("dequantize_s8_to_fp32_basic", "NONE", "S8")
    op = OpDequantize(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    _assert_generated(tmp_path, desc["name"], "dequantize")


def test_dequantize_name_does_not_drive_activation() -> None:
    desc = _dequantize_desc("dequantize_relu_name_only_s8", "NONE", "S8")
    op = OpDequantize(desc, seed=1, target_cpu="cortex-m55")

    model = op.build_keras_model()

    assert all(layer.__class__.__name__ != "ReLU" for layer in model.layers)
