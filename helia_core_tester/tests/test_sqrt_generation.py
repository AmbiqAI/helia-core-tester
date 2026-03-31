from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.sqrt import OpSqrt
from helia_core_tester.generation.utils.litert_builder import LITERT_AVAILABLE
from helia_core_tester.generation.utils.litert_utils import (
    get_operator_tensors_from_litert,
    load_litert_model,
)

TESTER_ROOT = Path(__file__).resolve().parents[2]
SQRT_DESCRIPTOR_PATH = TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "sqrt.yaml"
SQRT_PARITY_CASES = (
    ("sqrt_small_tensor_s8", (1, 2, 3, 4)),
    ("sqrt_long_row_s8", (1, 1, 64, 1)),
    ("sqrt_multi_batch_s8", (2, 3, 5, 3)),
)


def _sqrt_descriptor_map() -> dict[str, dict]:
    return {desc["name"]: desc for desc in load_descriptor(str(SQRT_DESCRIPTOR_PATH))}


def test_sqrt_descriptors_match_unit_test_parity() -> None:
    descriptors = load_descriptor(str(SQRT_DESCRIPTOR_PATH))

    assert [desc["name"] for desc in descriptors] == [name for name, _ in SQRT_PARITY_CASES]
    assert len(descriptors) == 3

    for desc, (_, shape) in zip(descriptors, SQRT_PARITY_CASES):
        assert desc["operator"] == "Sqrt"
        assert desc["activation_dtype"] == "S8"
        assert desc["weight_dtype"] == "S8"
        assert tuple(desc["input_shape"]) == shape


@pytest.mark.parametrize(("name", "shape"), SQRT_PARITY_CASES)
def test_sqrt_parity_descriptors_generate_litert_and_c(
    name: str,
    shape: tuple[int, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for sqrt LiteRT generation")

    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqrt_descriptor_map()[name]
    op = OpSqrt(desc, seed=1, target_cpu="cortex-m55")
    tflite_path = tmp_path / f"{name}.tflite"

    assert op.needs_keras_model() is False
    with pytest.raises(NotImplementedError, match="LiteRT-only"):
        op.build_keras_model()

    op.convert_to_tflite(None, str(tflite_path), 1)

    model, subgraph = load_litert_model(str(tflite_path))
    op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
    assert tuple(op_tensors["inputs"][0]["shape"]) == shape
    assert tuple(op_tensors["outputs"][0]["shape"]) == shape

    fake_output = np.zeros(shape, dtype=np.int8)
    monkeypatch.setattr(op, "run_inference", lambda *_args, **_kwargs: fake_output)
    op.generate_c_files(tmp_path)

    c_path = tmp_path / f"{name}_sqrt.c"
    h_path = tmp_path / "includes" / f"{name}_sqrt.h"
    assert c_path.exists()
    assert h_path.exists()

    content = c_path.read_text()
    assert "arm_sqrt_s8" in content
