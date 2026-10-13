"""An FP16 golden is computed from the float16 weights and bias the kernel receives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _record(name: str, out_dir: Path) -> dict:
    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(out_dir))
    return json.loads(next(out_dir.rglob(f"{name}.reference.json")).read_text())


@pytest.mark.parametrize(
    "name",
    [
        "convolve_float_small_c3_k3_same_f16",
        "depthwise_conv_float_default_f16",
        "transpose_conv_float_default_f16",
        "fully_connected_float_default_f16",
    ],
)
def test_fp16_golden_takes_float16_weights_and_bias(name: str, tmp_path: Path) -> None:
    record = _record(name, tmp_path)
    assert record["entry"].endswith("_f16")
    assert all(record["inputs"][role]["dtype"] == "float16" for role in ("input", "filter", "bias"))


@pytest.mark.parametrize(
    "name",
    [
        "convolve_float_default_f32",
        "depthwise_conv_float_default_f32",
        "transpose_conv_float_default_f32",
        "fully_connected_float_default_f32",
    ],
)
def test_fp32_golden_takes_float32_weights(name: str, tmp_path: Path) -> None:
    record = _record(name, tmp_path)
    assert record["entry"].endswith("_f32")
    assert all(record["inputs"][role]["dtype"] == "float32" for role in ("input", "filter", "bias"))
