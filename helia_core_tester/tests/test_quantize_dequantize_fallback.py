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


class _FakeQuantizeInterpreter:
    def __init__(self, *, input_shape, output_shape, output_data, scales, zero_points):
        self._input_details = [{"shape": np.array(input_shape), "index": 0}]
        self._output_details = [
            {
                "shape": np.array(output_shape),
                "index": 1,
                "quantization_parameters": {
                    "scales": scales,
                    "zero_points": zero_points,
                },
            }
        ]
        self._output_data = np.array(output_data)
        self._input_tensor = None

    def get_input_details(self):
        return self._input_details

    def get_output_details(self):
        return self._output_details

    def set_tensor(self, index, value):
        self._input_tensor = (index, value)

    def invoke(self):
        return None

    def get_tensor(self, index):
        return self._output_data


@pytest.mark.parametrize(
    ("name", "activation", "dtype", "scale", "zero_point"),
    [
        ("quantize_relu_s8", "RELU", "S8", np.array([0.0625], dtype=np.float32), np.array([3], dtype=np.int32)),
        ("quantize_relu_s16", "RELU", "S16", np.array([0.00390625], dtype=np.float32), np.array([0], dtype=np.int32)),
        ("quantize_relu6_vec_s8", "RELU6", "S8", np.array([0.125], dtype=np.float32), np.array([-5], dtype=np.int32)),
        ("quantize_relu_tail_vec31_s8", "RELU", "S8", np.array([0.03125], dtype=np.float32), np.array([7], dtype=np.int32)),
    ],
)
def test_quantize_extracts_scalar_params_from_numpy_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    activation: str,
    dtype: str,
    scale: np.ndarray,
    zero_point: np.ndarray,
) -> None:
    desc = _quantize_desc(name, activation, dtype)
    op = OpQuantize(desc, seed=1, target_cpu="cortex-m55")

    tflite_path = tmp_path / f"{name}.tflite"
    tflite_path.write_bytes(b"")

    output_dtype = np.int8 if dtype == "S8" else np.int16
    fake_interpreter = _FakeQuantizeInterpreter(
        input_shape=(1, 4),
        output_shape=(1, 4),
        output_data=np.array([[1, 2, 3, 4]], dtype=output_dtype),
        scales=scale,
        zero_points=zero_point,
    )
    monkeypatch.setattr(op, "load_litert_interpreter", lambda _: fake_interpreter)

    op.generate_c_files(tmp_path)

    source = (tmp_path / f"{name}_quantize.c").read_text()
    assert f"{int(zero_point[0])}," in source
    assert f"{float(scale[0])}f" in source


def test_quantize_rejects_empty_per_tensor_quantization_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "quantize_empty_quant_params_s8"
    desc = _quantize_desc(name, "NONE", "S8")
    op = OpQuantize(desc, seed=1, target_cpu="cortex-m55")

    tflite_path = tmp_path / f"{name}.tflite"
    tflite_path.write_bytes(b"")

    fake_interpreter = _FakeQuantizeInterpreter(
        input_shape=(1, 4),
        output_shape=(1, 4),
        output_data=np.array([[1, 2, 3, 4]], dtype=np.int8),
        scales=np.array([], dtype=np.float32),
        zero_points=np.array([0], dtype=np.int32),
    )
    monkeypatch.setattr(op, "load_litert_interpreter", lambda _: fake_interpreter)

    with pytest.raises(ValueError, match="missing per-tensor scale"):
        op.generate_c_files(tmp_path)
