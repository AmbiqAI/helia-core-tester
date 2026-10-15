"""An FP16 golden is computed from the float16 weights and bias the kernel receives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.generation.utils.litert_utils import load_litert_model

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FLOAT32 = 0  # LiteRT TensorType.FLOAT32


def _golden_constants(name: str, out_dir: Path) -> list[np.ndarray]:
    """Every float32 constant tensor of the case's golden model."""
    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(out_dir))
    model, _ = load_litert_model(str(next(out_dir.rglob(f"{name}.tflite"))))
    constants = []
    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            data = model.buffers[tensor.buffer].data
            if tensor.type == _FLOAT32 and data is not None and len(data):
                constants.append(np.frombuffer(bytes(bytearray(data)), dtype=np.float32))
    assert constants
    return constants


@pytest.mark.parametrize(
    "name",
    [
        "convolve_float_small_c3_k3_same_f16",
        "depthwise_conv_float_default_f16",
        "transpose_conv_float_default_f16",
        "fully_connected_float_default_f16",
    ],
)
def test_fp16_golden_model_holds_float16_weights_and_bias(name: str, tmp_path: Path) -> None:
    for values in _golden_constants(name, tmp_path):
        assert np.count_nonzero(values) == values.size
        assert np.array_equal(values, values.astype(np.float16).astype(np.float32))


@pytest.mark.parametrize(
    "name",
    [
        "convolve_float_default_f32",
        "depthwise_conv_float_default_f32",
        "transpose_conv_float_default_f32",
        "fully_connected_float_default_f32",
    ],
)
def test_fp32_golden_model_keeps_its_float32_draw(name: str, tmp_path: Path) -> None:
    assert any(
        not np.array_equal(values, values.astype(np.float16).astype(np.float32))
        for values in _golden_constants(name, tmp_path)
    )
