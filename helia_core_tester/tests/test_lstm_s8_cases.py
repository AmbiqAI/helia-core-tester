"""Model-style s8 LSTM cases: DTLN and hidden draws."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation import random_shapes as rs
from helia_core_tester.generation.utils import lstm_data as ld
from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = PROJECT_ROOT / "assets" / "descriptors" / "LSTMFunctions" / "lstm_unidirectional.yaml"
DTLN = ("lstm_unidirectional_dtln_l0_s8", "lstm_unidirectional_dtln_l1_s8",
        "lstm_unidirectional_dtln_l0_t4_s8", "lstm_unidirectional_dtln_l1_t4_s8")



def _dtln() -> dict[str, dict]:
    docs = [d for d in yaml.safe_load_all(DESCRIPTORS.read_text()) if d]
    return {d["name"]: d for d in docs if d["name"] in DTLN}


def _draw(case: dict, rng: np.random.Generator, stateful: bool | None = None) -> ld.LstmGeneratedData:
    return ld.explicit_lstm_s8(
        rng=rng, batch_size=case["batch_size"], time_steps=case["time_steps"], input_size=case["feature_size"],
        hidden_size=case["units"], time_major=case["time_major"], quant=ld.LstmQuant.from_desc(case["lstm_quant"]),
        stateful=case.get("stateful", False) if stateful is None else stateful,
    )


def _golden(case: dict, data: ld.LstmGeneratedData, biases: list, recurrent: list | None = None) -> np.ndarray:
    """TFLM golden for edited operands."""
    b, t, i, h = (case[k] for k in ("batch_size", "time_steps", "feature_size", "units"))
    quant = ld.LstmQuant.from_desc(case["lstm_quant"])
    weights = [data.tensors[f"{g}_gate_input_weights"].reshape(h, i) for g in ld.LSTM_GATES]
    weights += recurrent or [data.tensors[f"{g}_gate_hidden_weights"].reshape(h, h) for g in ld.LSTM_GATES]
    mult_shift = {k: calculate_multiplier_shift(v) for k, v in ld.effective_scales(quant).items()}
    shape = (t, b, i) if case["time_major"] else (b, t, i)
    state = (data.tensors["hidden_state"], data.tensors["cell_state"]) if "hidden_state" in data.tensors else None
    return ld.tflm_lstm_s8(data.tensors["input_tensor"].reshape(shape), weights, biases, quant, mult_shift,
                           case["time_major"], state)


def test_dtln_cases_match_model() -> None:
    cases = _dtln()
    assert set(cases) == set(DTLN)
    for name, case in cases.items():
        size = 257 if "_l0" in name else 128
        assert (case["batch_size"], case["feature_size"], case["units"], case["time_major"]) == (1, size, 128, False)
        assert case["time_steps"] == (4 if "_t4" in name else 1) and "dataset" not in case
        # helia-rt passes the state tensors.
        assert case["stateful"]
    # Quantized clip: 10.0 over the cell scale.
    l0, l1 = (ld.LstmQuant.from_desc(cases[n]["lstm_quant"]) for n in DTLN[:2])
    assert (ld.quantized_clip(l0), ld.quantized_clip(l1)) == (20480, 32767)
    assert (l0.input_zero_point, l1.output_zero_point) == (-128, -4)


def test_dtln_multipliers_tflm() -> None:
    scales = ld.effective_scales(ld.LstmQuant.from_desc(_dtln()[DTLN[0]]["lstm_quant"]))
    # float32 product, as TFLM's Prepare.
    assert calculate_multiplier_shift(scales["forget_gate_input"]) == (1781429632, 3)
    assert calculate_multiplier_shift(scales["forget_to_cell"]) == (1 << 30, -14)
    assert calculate_multiplier_shift(scales["input_to_cell"]) == (1 << 30, -18)


@pytest.mark.parametrize("name", DTLN[:2])
def test_state_drives_recurrence(name: str) -> None:
    case = _dtln()[name]
    other = [np.random.default_rng(k).integers(-127, 128, (128, 128)).astype(np.int8) for k in range(4)]
    for stateful in (False, True):
        data = _draw(case, np.random.default_rng(3), stateful)
        biases = [data.tensors[f"{g}_gate_bias"] for g in ld.LSTM_GATES]
        assert len(np.unique(data.tensors["output"])) > 32
        # Zero-point state: the kernel's NULL hidden.
        same = (_golden(case, data, biases, other) == data.tensors["output"]).all()
        assert same != stateful
        assert ("hidden_state" in data.tensors) == stateful


def test_lut_activations() -> None:
    q312 = np.array([-32768, -4096, -1, 0, 1, 4096, 32767])
    sigmoid, tanh = ld._sigmoid_q15(q312), ld._tanh_q15(q312, -12)
    assert sigmoid[3] == 16384 and tanh[3] == 0
    assert (np.diff(sigmoid.astype(int)) >= 0).all() and (np.diff(tanh.astype(int)) >= 0).all()
    # sigmoid(1) and tanh(1) in Q0.15.
    assert abs(int(sigmoid[5]) - 23955) <= 2 and abs(int(tanh[5]) - 24956) <= 2
    # sigmoid(8) is 0.99966; tanh saturates.
    assert (sigmoid[-1], tanh[-1], tanh[0]) == (32757, 32767, -32767)


def test_explicit_params() -> None:
    case = rs.sample_cases(12, 5, ops=(("LSTMUnidirectional", "S8"),))[-2]
    data = _draw(case, np.random.default_rng(1))
    biases = [data.tensors[f"{g}_gate_bias"] for g in ld.LSTM_GATES]
    assert (_golden(case, data, biases) == data.tensors["output"]).all()
    assert data.params["input_zero_point"] == -case["lstm_quant"]["input_zero_point"]


@pytest.mark.parametrize("seed", (0, 1, 7, 12345))
def test_hidden_sees_bias_lsb(seed: int) -> None:
    caught = 0
    for case in rs.sample_cases(12, seed, ops=(("LSTMUnidirectional", "S8"),)):
        data = _draw(case, np.random.default_rng(seed))
        biases = [data.tensors[f"{g}_gate_bias"] + 1 for g in ld.LSTM_GATES]
        caught += bool((_golden(case, data, biases) != data.tensors["output"]).any())
    # +1 on every gate bias.
    assert caught >= 3
