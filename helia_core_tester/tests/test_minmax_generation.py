from __future__ import annotations

from pathlib import Path

import pytest
from ai_edge_litert import schema_py_generated as litert

from helia_core_tester.generation.ops.BasicMathFunctions.minmax import OpMinMax
from helia_core_tester.generation.utils.litert_builder import LITERT_AVAILABLE
from helia_core_tester.generation.utils.litert_utils import (
    get_operator_tensors_from_litert,
    load_litert_model,
)


@pytest.mark.parametrize(
    ("operator", "dtype", "expected_builtin", "expected_kernel"),
    [
        ("Minimum", "S8", litert.BuiltinOperator.MINIMUM, "arm_minimum_s8"),
        ("Maximum", "S8", litert.BuiltinOperator.MAXIMUM, "arm_maximum_s8"),
        ("Minimum", "S16", litert.BuiltinOperator.MINIMUM, "arm_minimum_s16"),
        ("Maximum", "S16", litert.BuiltinOperator.MAXIMUM, "arm_maximum_s16"),
        ("Minimum", "FP16", litert.BuiltinOperator.MINIMUM, "arm_minimum_f16"),
        ("Maximum", "FP16", litert.BuiltinOperator.MAXIMUM, "arm_maximum_f16"),
        ("Minimum", "FP32", litert.BuiltinOperator.MINIMUM, "arm_minimum_f32"),
        ("Maximum", "FP32", litert.BuiltinOperator.MAXIMUM, "arm_maximum_f32"),
    ],
)
def test_minmax_generates_direct_litert_model_and_c(
    operator: str,
    dtype: str,
    expected_builtin: int,
    expected_kernel: str,
    tmp_path: Path,
) -> None:
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for min/max LiteRT generation")

    name = f"{operator.lower()}_{dtype.lower()}_broadcast"
    input_1_shape = (1, 2, 3, 4)
    input_2_shape = (1, 1, 3, 1)
    desc = {
        "operator": operator,
        "name": name,
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_1_shape": list(input_1_shape),
        "input_2_shape": list(input_2_shape),
    }
    op = OpMinMax(desc, seed=1, target_cpu="cortex-m55")
    tflite_path = tmp_path / f"{name}.tflite"

    assert op.needs_keras_model() is False
    with pytest.raises(NotImplementedError, match="LiteRT-only"):
        op.build_keras_model()

    op.convert_to_tflite(None, str(tflite_path), 1)

    model, subgraph = load_litert_model(str(tflite_path))
    op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
    assert len(subgraph.operators) == 1
    assert model.operatorCodes[0].builtinCode == expected_builtin
    assert tuple(op_tensors["inputs"][0]["shape"]) == input_1_shape
    assert tuple(op_tensors["inputs"][1]["shape"]) == input_2_shape
    assert tuple(op_tensors["outputs"][0]["shape"]) == (1, 2, 3, 4)

    op.generate_c_files(tmp_path)
    generated_c = tmp_path / f"{name}_minmax.c"
    assert generated_c.exists()
    assert expected_kernel in generated_c.read_text()


def test_minmax_rejects_unsupported_dtype(tmp_path: Path) -> None:
    op = OpMinMax(
        {
            "operator": "Maximum",
            "name": "maximum_s32",
            "tensor_dtypes": {"input": "S32", "output": "S32"},
            "input_1_shape": [1, 1, 1, 1],
            "input_2_shape": [1, 1, 1, 1],
        },
        seed=1,
        target_cpu="cortex-m55",
    )

    with pytest.raises(NotImplementedError, match="Unsupported MinMax dtype"):
        op.convert_to_tflite(None, str(tmp_path / "maximum_s32.tflite"), 1)
