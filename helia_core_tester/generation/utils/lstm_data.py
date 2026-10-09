"""
Integer LSTM test data: one case drawn per descriptor and run on TFLM's own
LSTM through the reference shim (generation/reference/lstm.py). No model is
converted and nothing falls back to the ns-cmsis-nn UnitTest vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift


@dataclass
class LstmGeneratedData:
    params: Dict[str, int]
    tensors: Dict[str, np.ndarray]
    scales: Dict[str, float]
    effective_scales: Dict[str, float]
    reference_call: Optional[Any] = field(default=None)


_MULTIPLIER_KEYS = [
    "forget_to_cell",
    "input_to_cell",
    "output",
    "output_gate_hidden",
    "cell_gate_hidden",
    "forget_gate_hidden",
    "input_gate_hidden",
    "output_gate_input",
    "cell_gate_input",
    "forget_gate_input",
    "input_gate_input",
]


def generate_lstm_data(
    *,
    rng: np.random.Generator,
    activation_dtype: str,
    batch_size: int,
    time_steps: int,
    input_size: int,
    hidden_size: int,
    time_major: bool,
    input_zero_point_override: int | None = None,
    output_zero_point_override: int | None = None,
) -> LstmGeneratedData:
    """Draw one case and its TFLM golden. The harness takes INPUT_ZERO_POINT as the
    input offset (-zp) and OUTPUT_ZERO_POINT as +zp; the multipliers come from the
    shim's QuantizeMultiplier on the same float32 scales the reference prepare uses."""
    from helia_core_tester.generation.reference import lstm

    kind = {"S8": "s8", "S16": "s16"}.get(str(activation_dtype).upper())
    if kind is None:
        raise ValueError(f"Unsupported activation_dtype: {activation_dtype}")
    quant = lstm.default_quant(
        kind, input_size, hidden_size,
        input_zero_point=int(input_zero_point_override or 0),
        output_zero_point=int(output_zero_point_override or 0),
    )
    case = lstm.build_lstm_case(
        rng, kind=kind, batch=batch_size, time_steps=time_steps, input_size=input_size,
        hidden_size=hidden_size, time_major=time_major, quant=quant,
    )
    params: Dict[str, int] = {
        "input_zero_point": -quant.input_zero_point,
        "output_zero_point": quant.output_zero_point,
        "cell_scale_power": quant.cell_scale_power,
        "cell_clip": lstm.CELL_CLIP_LSB,
        **quant.multipliers(),
    }
    tensors: Dict[str, np.ndarray] = {"input_tensor": case.input, "output": case.output}
    for key, weights in case.weights.items():
        tensors[f"{key}_weights"] = weights
    tensors.update(case.biases)
    scales = {"input_scale": quant.input_scale, "output_scale": quant.output_scale, "cell_scale": quant.cell_scale}
    scales.update({f"{k}_scale": v for k, v in quant.weight_scales.items()})
    return LstmGeneratedData(params=params, tensors=tensors, scales=scales, effective_scales={},
                             reference_call=case.call)


def build_lstm_context(
    *,
    name: str,
    dataset: str,
    activation_dtype: str,
    batch_size: int,
    time_steps: int,
    input_size: int,
    hidden_size: int,
    time_major: bool,
    data: LstmGeneratedData,
) -> Dict[str, Any]:
    # Quantize scales to multipliers/shifts
    mult_shift = {}
    if data.effective_scales:
        for key, scale in data.effective_scales.items():
            mult, shift = calculate_multiplier_shift(scale)
            mult_shift[key + "_multiplier"] = int(mult)
            mult_shift[key + "_shift"] = int(shift)
    else:
        for key in _MULTIPLIER_KEYS:
            mult = data.params.get(key + "_multiplier")
            shift = data.params.get(key + "_shift")
            if mult is not None and shift is not None:
                mult_shift[key + "_multiplier"] = int(mult)
                mult_shift[key + "_shift"] = int(shift)

    macro_prefix = dataset.upper() + "_"
    data_prefix = dataset.lower() + "_"

    return {
        "name": name,
        "dataset": dataset,
        "macro_prefix": macro_prefix,
        "data_prefix": data_prefix,
        "dtype": "s16" if activation_dtype == "S16" else "s8",
        "output_dtype": "int16_t" if activation_dtype == "S16" else "int8_t",
        "time_major": int(bool(time_major)),
        "batch_size": int(batch_size),
        "time_steps": int(time_steps),
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "input_zero_point": int(data.params["input_zero_point"]),
        "output_zero_point": int(data.params["output_zero_point"]),
        "cell_scale_power": int(data.params["cell_scale_power"]),
        "cell_clip": int(data.params["cell_clip"]),
        "mult_shift": mult_shift,
        "tensors": data.tensors,
    }
