"""Integer LSTM cases on the C reference: the draw, the quantization TFLM's prepare derives from it,
and the golden.

Scales: the input and the hidden state (output) come from the policy (s8 asymmetric over the
draw's real range, s16 symmetric); weights are symmetric int8 per gate; the cell state is a power
of two (TFLM requires it) with `cell_scale_power` from the descriptor, else -11 (range +-16).
Effective scales are TFLM's: gate matmuls to the gate scale 2^-12, forget*cell at 2^-15,
input*cell-gate at 2^-30 / cell scale, output*tanh(cell) at 2^-30 / hidden scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from helia_core_tester.generation.reference import policy
from helia_core_tester.generation.reference.quant import quantize_multiplier

GATES = ("forget", "input", "cell", "output")
DEFAULT_CELL_SCALE_POWER = -11
GATE_SCALE = 2.0 ** -12


@dataclass(frozen=True)
class LstmIntCase:
    params: Dict[str, int]
    inputs: Dict[str, np.ndarray]
    output_shape: tuple
    context: Dict[str, Any]


def _zero_point(extras: Mapping[str, Any], key: str) -> Optional[int]:
    v = extras.get(key)
    return None if v is None else int(v)


def lstm_int_case(rng: np.random.Generator, *, kind: str, batch: int, steps: int, input_size: int, hidden: int,
                  time_major: bool, extras: Mapping[str, Any], cell_scale_power: Optional[int] = None) -> LstmIntCase:
    """Draw one integer LSTM case and its TFLM quantization parameters.

    `extras` may pin `input_zero_point` / `output_zero_point` (the stored zero points; the
    kernel's input offset is the negated input zero point)."""
    if kind not in ("s8", "s16"):
        raise ValueError(f"integer LSTM takes s8 or s16, got {kind!r}")
    for label, v in (("batch", batch), ("time_steps", steps), ("input_size", input_size), ("hidden", hidden)):
        if int(v) < 1:
            raise ValueError(f"LSTM {label} must be positive, got {v}")
    act = np.int16 if kind == "s16" else np.int8
    info = np.iinfo(act)
    power = DEFAULT_CELL_SCALE_POWER if cell_scale_power is None else int(cell_scale_power)
    if not -30 <= power <= 0:
        raise ValueError(f"cell_scale_power must be in [-30, 0], got {power}")

    in_quant = policy.activation_quant(kind, (-1.0, 1.0))
    hid_quant = policy.activation_quant(kind, (-1.0, 1.0))
    in_zp = _zero_point(extras, "input_zero_point")
    out_zp = _zero_point(extras, "output_zero_point")
    in_zp = in_quant.zero_point if in_zp is None else in_zp
    out_zp = hid_quant.zero_point if out_zp is None else out_zp
    for label, zp in (("input_zero_point", in_zp), ("output_zero_point", out_zp)):
        if not info.min <= zp <= info.max:
            raise ValueError(f"LSTM {label} {zp} is outside {act.__name__}")

    shape_in = (steps, batch, input_size) if time_major else (batch, steps, input_size)
    x = rng.integers(int(info.min), int(info.max) + 1, size=shape_in).astype(act)
    w = rng.integers(-127, 128, size=(4, hidden, input_size)).astype(np.int8)
    r = rng.integers(-127, 128, size=(4, hidden, hidden)).astype(np.int8)
    # Per-gate weight scales around Glorot's limit, so gate pre-activations span the sigmoid's range.
    w_scales = np.sqrt(6.0 / (input_size + hidden)) / 127.0 * rng.uniform(0.75, 1.25, size=4)
    r_scales = np.sqrt(6.0 / (2 * hidden)) / 127.0 * rng.uniform(0.75, 1.25, size=4)
    w_scales = w_scales.astype(np.float32)
    r_scales = r_scales.astype(np.float32)
    # Biases of up to half a unit of gate pre-activation, at the input matmul's scale.
    bias_real = rng.uniform(-0.5, 0.5, size=(4, hidden))
    bias_scale = np.float64(in_quant.scale) * w_scales.astype(np.float64)[:, None]
    bias = np.round(bias_real / bias_scale).astype(np.int64 if kind == "s16" else np.int32)

    params: Dict[str, int] = {"time_major": int(time_major), "input_offset": -int(in_zp), "output_offset": int(out_zp)}
    mult_shift: Dict[str, int] = {}
    for g, gate in enumerate(GATES):
        for src, s_in, s_w in (("input", in_quant.scale, w_scales[g]), ("hidden", hid_quant.scale, r_scales[g])):
            m, sh = quantize_multiplier(float(s_in) * float(s_w) / GATE_SCALE)
            params[f"{gate}_{src}_multiplier"], params[f"{gate}_{src}_shift"] = m, sh
            mult_shift[f"{gate}_gate_{src}_multiplier"], mult_shift[f"{gate}_gate_{src}_shift"] = m, sh
    cell_scale = 2.0 ** power
    for key, real in (("forget_to_cell", 2.0 ** -15), ("input_to_cell", 2.0 ** -30 / cell_scale),
                      ("hidden", 2.0 ** -30 / float(hid_quant.scale))):
        m, sh = quantize_multiplier(real)
        params[f"{key}_multiplier"], params[f"{key}_shift"] = m, sh
        ctx_key = "output" if key == "hidden" else key
        mult_shift[f"{ctx_key}_multiplier"], mult_shift[f"{ctx_key}_shift"] = m, sh
    params["cell_clip"] = 0
    params["cell_scale_power"] = power

    shape_out = (steps, batch, hidden) if time_major else (batch, steps, hidden)
    tensors = {"input_tensor": x}
    for g, gate in enumerate(GATES):
        tensors[f"{gate}_gate_input_weights"] = w[g]
        tensors[f"{gate}_gate_hidden_weights"] = r[g]
        tensors[f"{gate}_gate_bias"] = bias[g]
    context = {
        "input_zero_point": -int(in_zp),  # the kernel's input offset, as the template's macro is used
        "output_zero_point": int(out_zp),
        "cell_scale_power": power,
        "cell_clip": 32767,  # no clip: the int16 maximum
        "mult_shift": mult_shift,
        "tensors": tensors,
        "scales": {"input": float(in_quant.scale), "hidden": float(hid_quant.scale), "cell": cell_scale},
    }
    inputs = {"input": x, "input_weights": w, "recurrent_weights": r, "bias": bias}
    return LstmIntCase(params, inputs, shape_out, context)
