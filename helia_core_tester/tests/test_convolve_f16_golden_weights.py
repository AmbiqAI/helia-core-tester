"""An FP16 Convolve golden is computed from the float16 weights and bias the kernel receives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _golden_constants(name: str, out_dir: Path) -> list[np.ndarray]:
    """The golden model's weight and bias tensors for a descriptor with OHWI [5, 3, 3, 3] weights."""
    from ai_edge_litert.interpreter import Interpreter

    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(out_dir))
    interpreter = Interpreter(model_path=str(next(out_dir.rglob(f"{name}.tflite"))))
    interpreter.allocate_tensors()
    constants = {
        tuple(t["shape"]): interpreter.get_tensor(t["index"])
        for t in interpreter.get_tensor_details()
        if tuple(t["shape"]) in {(5, 3, 3, 3), (5,)}
    }
    assert set(constants) == {(5, 3, 3, 3), (5,)}
    return list(constants.values())


@pytest.mark.parametrize(
    ("name", "float16_exact"),
    [("convolve_float_small_c3_k3_same_f16", True), ("convolve_float_default_f32", False)],
)
def test_only_fp16_goldens_hold_float16_weights_and_bias(name: str, float16_exact: bool, tmp_path: Path) -> None:
    for values in _golden_constants(name, tmp_path):
        assert np.count_nonzero(values) == values.size
        assert np.array_equal(values, values.astype(np.float16).astype(values.dtype)) == float16_exact
