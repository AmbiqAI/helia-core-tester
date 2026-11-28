"""Contract and LiteRT oracle checks for the direct FP32 predicate entries."""

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.ComparisonFunctions.comparison import OpComparison
from helia_core_tester.generation.ops.SelectFunctions.select_v2 import OpSelectV2
from helia_core_tester.generation.utils import litert_utils

ROOT = Path(__file__).resolve().parents[2] / "assets" / "descriptors"


@pytest.mark.parametrize(
    "op_class,relpath,symbol",
    [
        (OpComparison, "ComparisonFunctions/comparison_float.yaml", "arm_greater_f32"),
        (OpSelectV2, "SelectFunctions/select_v2_float.yaml", "arm_select_v2_f32"),
    ],
)
def test_mixed_float_predicate(op_class, relpath, symbol, tmp_path, monkeypatch):
    cases = load_descriptor(str(ROOT / relpath))
    assert all(c["required_kernel_symbols"] == [symbol] for c in cases)
    desc = next(c for c in cases if "_mixed_" in c["name"])
    op = op_class(desc, seed=13, target_cpu="cortex-m55")
    model = tmp_path / (desc["name"] + ".tflite")
    op.convert_to_tflite(None, str(model), 13)
    interpreter = litert_utils.load_builtin_reference(str(model))
    monkeypatch.setattr(litert_utils, "load_builtin_reference", lambda _: interpreter)
    op.generate_c_files(tmp_path)
    values = [interpreter.get_tensor(t["index"]) for t in interpreter.get_input_details()]
    result = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    source = next(tmp_path.glob("*.c")).read_text()
    assert symbol + "(" in source
    if op_class is OpComparison:
        np.testing.assert_array_equal(result, np.greater(*values))
        assert result.dtype == np.bool_
    else:
        np.testing.assert_array_equal(result.view(np.uint32), np.where(values[0], values[1].view(np.uint32), values[2].view(np.uint32)))
        assert "memcmp(" in source


@pytest.mark.parametrize("side,seed", [("left", 5252), ("right", 125)])
def test_scalar_broadcast_produces_both_boolean_values(tmp_path, side, seed):
    from helia_core_tester.generation.test_ops import generate_test
    cases = load_descriptor(str(ROOT / "ComparisonFunctions/comparison_float.yaml"))
    desc = next(c for c in cases if f"scalar_{side}" in c["name"])
    generate_test(desc, str(tmp_path), seed=seed, cpu="cortex-m55")
    assert list(tmp_path.rglob(desc["name"] + "_comparison.c"))
