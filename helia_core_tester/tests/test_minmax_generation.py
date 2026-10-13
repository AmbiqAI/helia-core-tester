from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.generation.ops.BasicMathFunctions.minmax import OpMinMax


@pytest.mark.parametrize(
    ("operator", "dtype", "expected_kernel"),
    [
        ("Minimum", "S8", "arm_minimum_s8"),
        ("Maximum", "S8", "arm_maximum_s8"),
        ("Minimum", "S16", "arm_minimum_s16"),
        ("Maximum", "S16", "arm_maximum_s16"),
        ("Minimum", "FP16", "arm_minimum_f16"),
        ("Maximum", "FP16", "arm_maximum_f16"),
        ("Minimum", "FP32", "arm_minimum_f32"),
        ("Maximum", "FP32", "arm_maximum_f32"),
    ],
)
def test_minmax_generates_c_from_the_reference(operator: str, dtype: str, expected_kernel: str,
                                               tmp_path: Path) -> None:
    name = f"{operator.lower()}_{dtype.lower()}_broadcast"
    desc = {
        "operator": operator,
        "name": name,
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_1_shape": [1, 2, 3, 4],
        "input_2_shape": [1, 1, 3, 1],
    }
    op = OpMinMax(desc, seed=1, target_cpu="cortex-m55")
    assert op.uses_reference() and not op.needs_keras_model()

    op.generate_c_files(tmp_path)
    assert op.reference.entry == f"{operator.lower()}_{dtype.lower().replace('fp', 'f')}"
    assert op.reference.output_shapes == {"output": (1, 2, 3, 4)}
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

    with pytest.raises((NotImplementedError, ValueError), match="S32|Unsupported"):
        op.generate_c_files(tmp_path)
