"""Integer LSTM cases on TFLM's own LSTM (shim/hct_ref_lstm.cc).

One case is: per-tensor quantization chosen from the shapes (or the descriptor),
int8 Glorot-like weights, biases worth up to half a gate pre-activation, a full
range input draw, and the reference call that produces the golden. The kernel
multipliers come from the same float32 scales through the shim's
QuantizeMultiplier, so the C harness and the reference prepare cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference import params as ref_params
from helia_core_tester.generation.reference.case import ReferenceCall

GATES = ("input", "forget", "cell", "output")
WEIGHT_KEYS = tuple(f"{g}_gate_{k}" for g in GATES for k in ("input", "hidden"))

CELL_CLIP_LSB = 32767

_Q12 = 2.0 ** -12
_Q15 = 2.0 ** -15


def _f32(value: float) -> float:
    return float(np.float32(value))


@dataclass(frozen=True)
class LstmQuant:
    """Per-tensor scales (float32 values) and zero points of one integer LSTM."""

    kind: str
    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int
    cell_scale: float
    weight_scales: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.kind not in ("s8", "s16"):
            raise ValueError(f"kind must be s8 or s16, got {self.kind!r}")
        for name in ("input_scale", "output_scale", "cell_scale"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if math.log2(self.cell_scale) != round(math.log2(self.cell_scale)):
            raise ValueError(f"cell_scale must be a power of two, got {self.cell_scale}")
        lo, hi = (-128, 127) if self.kind == "s8" else (0, 0)
        for name in ("input_zero_point", "output_zero_point"):
            zp = getattr(self, name)
            if not lo <= zp <= hi:
                raise ValueError(f"{name} {zp} outside [{lo}, {hi}] for {self.kind}")
        missing = [k for k in WEIGHT_KEYS if not (math.isfinite(self.weight_scales.get(k, float("nan"))) and self.weight_scales[k] > 0)]
        if missing:
            raise ValueError(f"weight scales missing or invalid: {missing}")

    @property
    def cell_scale_power(self) -> int:
        return int(round(math.log2(self.cell_scale)))

    def effective_scales(self) -> Dict[str, float]:
        """The real multipliers of TFLM's integer LSTM prepare (lstm_eval_common.cc)."""
        scales = {
            "forget_to_cell": _Q15 * self.cell_scale / self.cell_scale,
            "input_to_cell": _Q15 * _Q15 / self.cell_scale,
            "output": _Q15 * _Q15 / self.output_scale,
        }
        for gate in GATES:
            scales[f"{gate}_gate_hidden"] = self.output_scale * self.weight_scales[f"{gate}_gate_hidden"] / _Q12
            scales[f"{gate}_gate_input"] = self.input_scale * self.weight_scales[f"{gate}_gate_input"] / _Q12
        return scales

    def multipliers(self) -> Dict[str, int]:
        """`<name>_multiplier` / `<name>_shift` for every effective scale, via the shim."""
        out: Dict[str, int] = {}
        for name, scale in self.effective_scales().items():
            mult, shift = ref_params.quantize_multiplier(scale)
            out[f"{name}_multiplier"], out[f"{name}_shift"] = int(mult), int(shift)
        return out


def default_quant(kind: str, input_size: int, hidden_size: int, input_zero_point: int = 0,
                  output_zero_point: int = 0) -> LstmQuant:
    """Activations in [-1, 1], hidden state in (-1, 1), cell state within +/-16 (2^-11),
    weights whose gate pre-activations stay O(1)."""
    if input_size < 1 or hidden_size < 1:
        raise ValueError("input_size and hidden_size must be positive")
    act = 1.0 / 128 if kind == "s8" else 1.0 / 32768
    base = 1.5 / math.sqrt(input_size + hidden_size) / 127
    # A distinct scale per weight tensor, so a kernel that swaps two gates' multipliers fails.
    return LstmQuant(
        kind=kind, input_scale=act, input_zero_point=int(input_zero_point), output_scale=act,
        output_zero_point=int(output_zero_point), cell_scale=2.0 ** -11,
        weight_scales={k: _f32(base * (0.7 + 0.08 * i)) for i, k in enumerate(WEIGHT_KEYS)},
    )


def shapes(batch: int, time_steps: int, input_size: int, hidden_size: int, time_major: bool) -> Dict[str, Tuple[int, ...]]:
    if min(batch, time_steps, input_size, hidden_size) < 1:
        raise ValueError("LSTM dimensions must be positive")
    lead = (time_steps, batch) if time_major else (batch, time_steps)
    return {
        "input": lead + (input_size,),
        "output": lead + (hidden_size,),
        "input_weights": (hidden_size, input_size),
        "hidden_weights": (hidden_size, hidden_size),
        "bias": (hidden_size,),
    }


@dataclass
class LstmCase:
    quant: LstmQuant
    input: np.ndarray
    weights: Dict[str, np.ndarray]  # WEIGHT_KEYS
    biases: Dict[str, np.ndarray]   # "<gate>_gate_bias"
    output: np.ndarray
    call: ReferenceCall


def _params_struct(quant: LstmQuant, batch: int, time_steps: int, input_size: int, hidden_size: int,
                   time_major: bool) -> Dict[str, object]:
    return {
        "batch": batch, "time_steps": time_steps, "input_size": input_size, "hidden_size": hidden_size,
        "time_major": int(bool(time_major)), "input_scale": quant.input_scale,
        "input_zero_point": quant.input_zero_point, "output_scale": quant.output_scale,
        "output_zero_point": quant.output_zero_point, "cell_scale": quant.cell_scale,
        # The CMSIS kernels clamp the cell state to +/-CELL_CLIP_LSB; TFLM clips at cell_clip / scale.
        "cell_clip": CELL_CLIP_LSB * quant.cell_scale,
        "weight_scales": [quant.weight_scales[k] for k in b.LSTM_WEIGHT_ORDER],
    }


def build_lstm_case(rng: np.random.Generator, *, kind: str, batch: int, time_steps: int, input_size: int,
                    hidden_size: int, time_major: bool, quant: Optional[LstmQuant] = None) -> LstmCase:
    """Draw one integer LSTM case and run it on the TFLM reference."""
    quant = quant or default_quant(kind, input_size, hidden_size)
    if quant.kind != kind:
        raise ValueError(f"quantization is for {quant.kind}, case is {kind}")
    shp = shapes(batch, time_steps, input_size, hidden_size, time_major)
    weights = {
        k: rng.integers(-127, 128, size=shp["input_weights" if k.endswith("_input") else "hidden_weights"]).astype(np.int8)
        for k in WEIGHT_KEYS
    }
    # Up to half a unit of gate pre-activation, in each gate's accumulator scale.
    bias_dtype = np.int32 if kind == "s8" else np.int64
    biases = {}
    for gate in GATES:
        acc_scale = quant.input_scale * quant.weight_scales[f"{gate}_gate_input"]
        reach = max(1, int(0.5 / acc_scale))
        biases[f"{gate}_gate_bias"] = rng.integers(-reach, reach + 1, size=shp["bias"]).astype(bias_dtype)
    info = np.iinfo(np.int8 if kind == "s8" else np.int16)
    x = rng.integers(info.min, info.max + 1, size=shp["input"]).astype(info.dtype)
    params = _params_struct(quant, batch, time_steps, input_size, hidden_size, time_major)
    call = ReferenceCall(
        f"lstm_{kind}", params, {"input": x, **weights, **biases}, shp["output"], info.dtype.name,
        quant={"input": {"scale": quant.input_scale, "zero_point": quant.input_zero_point},
               "output": {"scale": quant.output_scale, "zero_point": quant.output_zero_point},
               "cell": {"scale": quant.cell_scale}, "weights": dict(quant.weight_scales)},
    )
    from helia_core_tester.generation.reference.run import run_reference

    return LstmCase(quant, x, weights, biases, run_reference(call), call)


def lstm_struct(params: Mapping[str, object]) -> b.HctLstmParams:
    scales = [float(v) for v in params["weight_scales"]]  # type: ignore[union-attr]
    if len(scales) != 8:
        raise ValueError(f"weight_scales needs 8 entries, got {len(scales)}")
    return b.HctLstmParams(
        int(params["batch"]), int(params["time_steps"]), int(params["input_size"]), int(params["hidden_size"]),
        int(params["time_major"]), float(params["input_scale"]), int(params["input_zero_point"]),
        float(params["output_scale"]), int(params["output_zero_point"]), float(params["cell_scale"]),
        float(params["cell_clip"]), (b.ctypes.c_float * 8)(*scales),
    )
