"""An FP16 golden is computed from the float16 weights and bias the kernel receives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops import get_op_map

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _reference_constants(name: str, out_dir: Path) -> list[np.ndarray]:
    """The weights and bias the case's f32 reference call computed its golden from."""
    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    op = get_op_map()[desc["operator"]](desc, 500, target_cpu="cortex-m55")
    out_dir.mkdir(parents=True, exist_ok=True)
    op.generate_c_files(out_dir)
    call = op.reference
    assert call.kernel.endswith("_f32")
    constants = [np.asarray(call.tensors[k]) for k in ("filter", "bias") if call.tensors.get(k) is not None]
    assert constants and all(c.dtype == np.float32 for c in constants)
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
def test_fp16_golden_holds_float16_weights_and_bias(name: str, tmp_path: Path) -> None:
    for values in _reference_constants(name, tmp_path):
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
def test_fp32_golden_keeps_its_float32_draw(name: str, tmp_path: Path) -> None:
    assert any(
        not np.array_equal(values, values.astype(np.float16).astype(np.float32))
        for values in _reference_constants(name, tmp_path)
    )
