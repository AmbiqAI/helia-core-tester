"""GRUUnidirectional draws its golden from the C reference (gru_f32 / gru_f16).

Every descriptor case is generated through OpGRUUnidirectional. The weights the harness
emits are parsed back per gate, checked against the stacked reference inputs (z, r, n
order), and the emitted golden is compared bit for bit with an independent Python model
of the reference's rounding contract: each matmul-plus-biases is an exact sum rounded
once to the element type, sigmoid/tanh are binary64 rounded once, every other step is
one IEEE operation rounded to the element type.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared import kernel_faithful as kf
from helia_core_tester.generation.ops.LSTMFunctions.gru_unidirectional import OpGRUUnidirectional
from helia_core_tester.generation.reference.call import ReferenceCall

_OP_FILE = Path(__file__).resolve().parents[1] / "generation" / "ops" / "LSTMFunctions" / "gru_unidirectional.py"
_GATES = ("update", "reset", "candidate")
_SUCCESS = "ARM_CMSIS_NN_SUCCESS"


def _cases() -> list[dict]:
    descs = load_all_descriptors(str(find_descriptors_dir()))
    return sorted((d for d in descs if d.get("operator") == "GRUUnidirectional"), key=lambda d: d["name"])


CASES = _cases()


def _parse_literal(text: str, dtype) -> np.ndarray:
    values = []
    for token in text.replace("\n", " ").split(","):
        token = token.strip().replace("(float16_t)", "").replace("(float32_t)", "").strip()
        if not token:
            continue
        sign = -1.0 if token.startswith("-") else 1.0
        body = token.lstrip("+-")
        if body == "NAN":
            values.append(math.nan)
        elif body == "INFINITY":
            values.append(sign * math.inf)
        else:
            values.append(float(token.rstrip("f")))
    return np.array(values, dtype=np.float64).astype(dtype)


def _emitted(tmp_path: Path, name: str) -> dict:
    """The emitted header's array literals, keyed by the suffix after `<name>_`."""
    header = next(tmp_path.rglob("includes/*.h")).read_text()
    pattern = re.compile(r"static const \w+ " + re.escape(name) + r"_(\w+)\[\] = \{(.*?)\};", re.S)
    return {key: body for key, body in pattern.findall(header)}


def _round_exact(total: Fraction, dtype) -> float:
    """The element-type value nearest the exact `total`, ties to even."""
    guess = dtype(float(total))
    if not np.isfinite(guess):
        return float(guess)
    candidates = {guess, np.nextafter(guess, dtype(np.inf)), np.nextafter(guess, dtype(-np.inf))}
    finite = [c for c in candidates if np.isfinite(c)]
    bits = np.int16 if dtype == np.float16 else np.int32

    def key(c):
        return (abs(Fraction(float(c)) - total), int(np.array(c, dtype=dtype).view(bits)) & 1)

    return float(min(finite, key=key))


def _dot(terms: list[float], dtype) -> float:
    """Exact sum of products and biases, rounded once to the element type."""
    if not all(math.isfinite(t) for t in terms):
        if any(math.isnan(t) for t in terms) or (math.inf in terms and -math.inf in terms):
            return math.nan
        return math.inf if math.inf in terms else -math.inf
    return _round_exact(sum((Fraction(t) for t in terms), Fraction(0)), dtype)


def _sigmoid(v: float) -> float:
    try:
        e = math.exp(-v)
    except OverflowError:
        e = math.inf
    return 1.0 / (1.0 + e)


def _model(x: np.ndarray, gates: dict, *, time_major: bool, reset_after: bool, dtype) -> np.ndarray:
    def rnd(v: float) -> float:
        return float(dtype(v))

    seq = x if time_major else np.transpose(x, (1, 0, 2))
    steps, batch, _ = seq.shape
    hidden_size = gates["update"][0].shape[0]
    out = np.zeros((steps, batch, hidden_size), dtype=dtype)

    def pre(gate: str, xs, hs, u: int, split: bool):
        w_in, w_h, b_in, b_h = gates[gate]
        left = [float(a) * float(b) for a, b in zip(xs, w_in[u])] + ([float(b_in[u])] if b_in is not None else [])
        right = [float(a) * float(b) for a, b in zip(hs, w_h[u])] + ([float(b_h[u])] if b_h is not None else [])
        return (_dot(left, dtype), _dot(right, dtype)) if split else _dot(left + right, dtype)

    for b in range(batch):
        h = [0.0] * hidden_size
        for t in range(steps):
            xs = [float(v) for v in seq[t, b]]
            z = [rnd(_sigmoid(pre("update", xs, h, u, False))) for u in range(hidden_size)]
            r = [rnd(_sigmoid(pre("reset", xs, h, u, False))) for u in range(hidden_size)]
            n = []
            for u in range(hidden_size):
                if reset_after:
                    a, c = pre("candidate", xs, h, u, True)
                    p = rnd(a + rnd(r[u] * c))
                else:
                    p = pre("candidate", xs, [rnd(r[k] * h[k]) for k in range(hidden_size)], u, False)
                n.append(rnd(math.tanh(p)))
            h = [rnd(rnd(z[u] * h[u]) + rnd(rnd(1.0 - z[u]) * n[u])) for u in range(hidden_size)]
            out[t, b] = h
    return out if time_major else np.transpose(out, (1, 0, 2))


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    both_nan = np.isnan(a) & np.isnan(b)
    return a.shape == b.shape and bool(np.all(both_nan | (a == b)))


def test_op_file_has_no_converter_path() -> None:
    text = _OP_FILE.read_text().lower()
    for word in ("litert", "tflite_path", "tensorflow", "keras", "interpreter"):
        assert word not in text, word


@pytest.mark.parametrize("desc", CASES, ids=[d["name"] for d in CASES])
def test_golden_matches_an_independent_model(desc: dict, tmp_path: Path) -> None:
    op = OpGRUUnidirectional(desc, seed=1234, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    if desc.get("expected_status", _SUCCESS) != _SUCCESS:
        assert op.status_only() and not op.uses_reference()
        assert op.reference is None
        assert not list(tmp_path.rglob("*.reference.json"))
        return

    call = op.reference
    assert call is not None and op.uses_reference()
    dtype = np.float16 if desc["name"].endswith("_f16") else np.float32
    # cortex-m55 runs arm_nn_gru_step_f16/_f32 on Helium, so the golden is that route's arithmetic (gru_f16_mve,
    # gru_f32_cmsis target 0); the independent model below checks the exact entry on the same operands.
    assert call.entry == ("gru_f16_mve" if dtype == np.float16 else "gru_f32_cmsis")
    time_major = bool(desc.get("time_major", False))
    reset_after = bool(desc.get("reset_after", True))
    use_bias = bool(desc.get("use_bias", True))
    exact_params = {"time_major": int(time_major), "reset_after": int(reset_after)}
    assert dict(call.params) == (exact_params if dtype == np.float16 else {**exact_params, "target": 0})

    emitted = _emitted(tmp_path, desc["name"])
    x = call.inputs["input"]
    assert x.dtype == dtype
    assert _same_bits(_parse_literal(emitted["input_tensor"], dtype), x)
    hidden = int(desc.get("units", desc.get("hidden_size", 1)))
    feature = int(desc.get("feature_size", desc.get("input_size", 1)))

    gates = {}
    for index, gate in enumerate(_GATES):
        w_in = _parse_literal(emitted[f"{gate}_gate_input_weights"], dtype).reshape(hidden, feature)
        w_h = _parse_literal(emitted[f"{gate}_gate_hidden_weights"], dtype).reshape(hidden, hidden)
        assert np.array_equal(call.inputs["input_weights"][index], w_in)
        assert np.array_equal(call.inputs["recurrent_weights"][index], w_h)
        if use_bias:
            b_in = _parse_literal(emitted[f"{gate}_gate_input_bias"], dtype)
            b_h = _parse_literal(emitted[f"{gate}_gate_hidden_bias"], dtype)
            assert np.array_equal(call.inputs["input_bias"][index], b_in)
            assert np.array_equal(call.inputs["recurrent_bias"][index], b_h)
        else:
            assert f"{gate}_gate_input_bias" not in emitted and f"{gate}_gate_hidden_bias" not in emitted
            b_in = b_h = None
        gates[gate] = (w_in, w_h, b_in, b_h)
    if not use_bias:
        assert call.inputs["input_bias"].shape == (0,) and call.inputs["recurrent_bias"].shape == (0,)

    golden = call.output()
    assert golden.shape == (x.shape[0], x.shape[1], hidden)
    assert _same_bits(_parse_literal(emitted["expected_output"], dtype), golden)
    expected = _model(x, gates, time_major=time_major, reset_after=reset_after, dtype=dtype)
    exact = ReferenceCall("gru_f16" if dtype == np.float16 else "gru_f32", exact_params, call.inputs,
                          call.output_shapes)
    assert _same_bits(exact.output(), expected)
    finite = np.isfinite(golden) & np.isfinite(expected)
    atol = kf.RNN_F16_ATOL if dtype == np.float16 else kf.GRU_F32_ATOL
    assert np.all(np.abs(golden[finite].astype(np.float64) - expected[finite]) <= atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_reference_matches_the_model_on_wide_random_draws(dtype) -> None:
    rng = np.random.default_rng(7)
    for reset_after in (True, False):
        for time_major in (True, False):
            x = rng.uniform(-3, 3, size=(3, 2, 5)).astype(dtype)
            gates = {g: (rng.uniform(-2, 2, size=(4, 5)).astype(dtype), rng.uniform(-2, 2, size=(4, 4)).astype(dtype),
                         rng.uniform(-1, 1, size=4).astype(dtype), rng.uniform(-1, 1, size=4).astype(dtype))
                     for g in _GATES}
            golden = OpGRUUnidirectional.gru_call(x, gates, time_major=time_major, reset_after=reset_after).output()
            assert _same_bits(golden, _model(x, gates, time_major=time_major, reset_after=reset_after, dtype=dtype))


def test_gru_call_stacks_gates_in_update_reset_candidate_order() -> None:
    gates = {g: (np.full((2, 3), i, np.float32), np.full((2, 2), 10 + i, np.float32),
                 np.full(2, 20 + i, np.float32), np.full(2, 30 + i, np.float32)) for i, g in enumerate(_GATES)}
    call = OpGRUUnidirectional.gru_call(np.zeros((1, 1, 3), np.float32), gates, time_major=False, reset_after=True)
    assert [float(call.inputs["input_weights"][i, 0, 0]) for i in range(3)] == [0.0, 1.0, 2.0]
    assert [float(call.inputs["recurrent_weights"][i, 0, 0]) for i in range(3)] == [10.0, 11.0, 12.0]
    assert [float(call.inputs["input_bias"][i, 0]) for i in range(3)] == [20.0, 21.0, 22.0]
    assert [float(call.inputs["recurrent_bias"][i, 0]) for i in range(3)] == [30.0, 31.0, 32.0]
    assert call.output_shapes == {"output": (1, 1, 2)}


def test_gru_call_rejects_partial_biases_bad_dtype_and_rank() -> None:
    w = (np.zeros((2, 3), np.float32), np.zeros((2, 2), np.float32))
    full = {g: (*w, np.zeros(2, np.float32), np.zeros(2, np.float32)) for g in _GATES}
    partial = dict(full, reset=(*w, None, np.zeros(2, np.float32)))
    x = np.zeros((1, 1, 3), np.float32)
    with pytest.raises(ValueError, match="present on every gate"):
        OpGRUUnidirectional.gru_call(x, partial, time_major=False, reset_after=True)
    with pytest.raises(ValueError, match="float32 or float16"):
        OpGRUUnidirectional.gru_call(x.astype(np.int8), full, time_major=False, reset_after=True)
    with pytest.raises(ValueError, match="rank 3"):
        OpGRUUnidirectional.gru_call(x[0], full, time_major=False, reset_after=True)


def test_zero_time_steps_yields_an_empty_golden() -> None:
    gates = {g: (np.ones((2, 3), np.float32), np.ones((2, 2), np.float32), None, None) for g in _GATES}
    golden = OpGRUUnidirectional.gru_call(np.zeros((1, 0, 3), np.float32), gates, time_major=False,
                                          reset_after=True).output()
    assert golden.shape == (1, 0, 2)
