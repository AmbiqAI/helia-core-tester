from pathlib import Path

from helia_core_tester.generation.ops.quantize import OpQuantize
from helia_core_tester.generation.ops.dequantize import OpDequantize


def _quantize_desc(name: str, activation: str, dtype: str) -> dict:
    return {
        "operator": "Quantize",
        "name": name,
        "activation_dtype": dtype,
        "weight_dtype": "S8",
        "activation": activation,
        "input_shape": [1, 4],
    }


def _dequantize_desc(name: str, activation: str, dtype: str) -> dict:
    return {
        "operator": "Dequantize",
        "name": name,
        "activation_dtype": dtype,
        "weight_dtype": "S8",
        "activation": activation,
        "input_shape": [1, 4],
    }


def _assert_generated(output_dir: Path, name: str, suffix: str) -> None:
    header = output_dir / "includes" / f"{name}_{suffix}.h"
    source = output_dir / f"{name}_{suffix}.c"
    assert header.exists()
    assert source.exists()


def test_quantize_generates_without_tflite(tmp_path: Path) -> None:
    desc = _quantize_desc("quantize_relu_s8", "RELU", "S8")
    op = OpQuantize(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    _assert_generated(tmp_path, desc["name"], "quantize")


def test_dequantize_generates_without_tflite(tmp_path: Path) -> None:
    desc = _dequantize_desc("dequantize_relu_s16", "RELU", "S16")
    op = OpDequantize(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    _assert_generated(tmp_path, desc["name"], "dequantize")
