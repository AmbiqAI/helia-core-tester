"""An FP16 Convolve golden is computed from the float16 weights and bias the kernel receives."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_fp16_golden_model_holds_float16_weights_and_bias(tmp_path: Path) -> None:
    from ai_edge_litert.interpreter import Interpreter

    name = "convolve_float_small_c3_k3_same_f16"
    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(tmp_path))
    interpreter = Interpreter(model_path=str(next(tmp_path.rglob(f"{name}.tflite"))))
    interpreter.allocate_tensors()

    # The golden model's constant tensors: OHWI weights [5, 3, 3, 3] and bias [5].
    constants = {
        tuple(t["shape"]): interpreter.get_tensor(t["index"])
        for t in interpreter.get_tensor_details()
        if tuple(t["shape"]) in {(5, 3, 3, 3), (5,)}
    }
    assert set(constants) == {(5, 3, 3, 3), (5,)}
    for values in constants.values():
        np.testing.assert_array_equal(values, values.astype(np.float16).astype(values.dtype))
