"""Emitted requantization multipliers match TFLite's double-precision QuantizeMultiplier.

TFLite derives each output channel's multiplier from input_scale * filter_scale /
output_scale evaluated in double. The model's filter scales are float32, so a generator
that multiplies them without widening keeps the product in float32 and emits Q31
multipliers that differ from the ones the golden output was computed with.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _q31_multiplier(scale: float) -> int:
    """TFLite QuantizeMultiplier on a double-precision scale."""
    fraction, _ = math.frexp(scale)
    multiplier = int(math.floor(fraction * (1 << 31) + 0.5))
    return multiplier // 2 if multiplier == 1 << 31 else multiplier


@pytest.mark.parametrize(
    "name",
    [
        "convolve_case_02_s8",
        # A single-channel filter still takes the per-channel path.
        "convolve_case_03_s8",
        "convolve_int16xint8_s16",
        "transpose_conv_same_kernel6x6_stride2x2_bias_s8",
        "fully_connected_per_channel_rows3_cols9_bias_s8",
    ],
)
def test_multipliers_come_from_double_precision_scales(name: str, tmp_path: Path) -> None:
    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    header = "".join(p.read_text() for p in (case_dir / "includes").glob("*.h"))
    emitted = [int(v) for v in re.findall(r"-?\d+", re.search(r"_multiplier\[[0-9]*\]\s*=\s*\{([^}]*)\}", header).group(1))]

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(case_dir / f"{name}.tflite"))
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    input_scale = float(details[input_index]["quantization_parameters"]["scales"][0])
    output_scale = float(details[output_index]["quantization_parameters"]["scales"][0])
    weights = [
        d
        for i, d in details.items()
        if i not in (input_index, output_index) and d["dtype"] == np.int8 and len(d["shape"]) >= 2
    ]
    assert len(weights) == 1
    weight_scales = weights[0]["quantization_parameters"]["scales"]
    assert len(weight_scales) == len(emitted)
    expected = [_q31_multiplier(input_scale * float(s) / output_scale) for s in weight_scales]

    assert emitted == expected
