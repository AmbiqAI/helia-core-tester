"""SVDF goldens come from the C reference (svdf_s8 / svdf_state_s16_s8 / svdf_f32 / svdf_f16).

Each emitted golden is checked against an independent model on the emitted tensors: an
integer model with TFLM's fixed-point rescaling (exact) and a float64 model (within the
rounding the reference applies per tensor). Fault cases assert a status only and record no
reference call. The entries' own argument validation is pinned by the negative tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.SVDFunctions.svdf import OpSVDF
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.generation.test_ops import default_seed_for_case

_DESC_DIR = Path(__file__).resolve().parents[2] / "assets" / "descriptors"
_OP_FILE = Path(__file__).resolve().parents[1] / "generation" / "ops" / "SVDFunctions" / "svdf.py"


def _descriptors() -> dict:
    return {d["name"]: d for d in load_all_descriptors(str(_DESC_DIR)) if d["operator"] == "SVDF"}


DESCRIPTORS = _descriptors()
INT_CASES = sorted(n for n, d in DESCRIPTORS.items() if d.get("activation_dtype") == "S8" and "fault" not in d)
FLOAT_CASES = sorted(n for n, d in DESCRIPTORS.items()
                     if d.get("activation_dtype") in ("FP32", "FP16") and "fault" not in d and "input_mode" not in d)
FAULT_CASES = sorted(n for n, d in DESCRIPTORS.items() if "fault" in d)


def _generate(name: str, tmp_path: Path) -> tuple[OpSVDF, str]:
    desc = DESCRIPTORS[name]
    op = OpSVDF(desc, default_seed_for_case(name), target_cpu="cortex-m55")
    out = tmp_path / name
    op.generate_c_files(out)
    return op, (out / "includes" / f"{name}_svdf.h").read_text()


def _emitted_golden(header: str, name: str, dtype) -> np.ndarray:
    # The integer header names the golden `_output_ref`, the float one `_expected_output`.
    body = re.search(rf"{name}_(?:output_ref|expected_output)\[\] = \{{(.*?)\}};", header, re.S).group(1)
    values = re.findall(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?)", body.replace("(float16_t)", ""))
    return np.array([float(v) for v in values], dtype=np.float64).astype(dtype)


# ----------------------------------------------------------- independent models


def _srdhm(a: int, b: int) -> int:
    if a == b == -(1 << 31):
        return (1 << 31) - 1
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    q = abs(ab + nudge) >> 31
    return q if ab + nudge >= 0 else -q


def _rdbp(x: int, exponent: int) -> int:
    mask = (1 << exponent) - 1
    remainder, threshold = x & mask, (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def _mbqm(x: int, multiplier: int, shift: int) -> int:
    scaled = ((x << max(shift, 0)) + (1 << 31)) % (1 << 32) - (1 << 31)
    return _rdbp(_srdhm(scaled, multiplier), max(-shift, 0))


def _svdf_int_model(call) -> tuple[np.ndarray, np.ndarray]:
    p, t = call.params, call.inputs
    x, wf, wt, st = (t[k].astype(np.int64) for k in ("input", "weights_feature", "weights_time", "state"))
    bias = t["bias"].astype(np.int64)
    steps, batches, _ = x.shape
    features, memory = wt.shape
    units = features // p["rank"]
    state = st.copy()
    out = np.zeros((steps, batches, units), np.int64)
    for s in range(steps):
        for b in range(batches):
            for f in range(features):
                state[b, f, :-1] = state[b, f, 1:].copy()
                acc = int(np.dot(wf[f], x[s, b] + p["input_offset"]))
                v = _mbqm(acc, p["input_multiplier"], p["input_shift"])
                state[b, f, -1] = min(max(v, p["input_activation_min"]), p["input_activation_max"])
            for u in range(units):
                total = int(bias[u]) if bias.size else 0
                for r in range(p["rank"]):
                    f = u * p["rank"] + r
                    total += int(np.dot(wt[f], state[b, f]))
                o = _mbqm(total, p["output_multiplier"], p["output_shift"]) + p["output_offset"]
                out[s, b, u] = min(max(o, p["output_activation_min"]), p["output_activation_max"])
    return out, state


def _svdf_float_model(call) -> tuple[np.ndarray, np.ndarray]:
    """float64 SVDF without intermediate rounding, and the per-lane magnitude that bounds rounding."""
    p, t = call.params, call.inputs
    x, wf, wt, st = (t[k].astype(np.float64) for k in ("input", "weights_feature", "weights_time", "state"))
    bias = t["bias"].astype(np.float64)
    steps, batches, _ = x.shape
    features, memory = wt.shape
    units = features // p["rank"]
    state, mag_state = st.copy(), np.abs(st)
    for s in range(steps):
        state = np.concatenate([state[:, :, 1:], np.einsum("fi,bi->bf", wf, x[s])[:, :, None]], axis=2)
        mag_state = np.concatenate(
            [mag_state[:, :, 1:], np.einsum("fi,bi->bf", np.abs(wf), np.abs(x[s]))[:, :, None]], axis=2)
        per_filter = np.einsum("fm,bfm->bf", wt, state).reshape(batches, units, p["rank"]).sum(axis=2)
        magnitude = np.einsum("fm,bfm->bf", np.abs(wt), mag_state).reshape(batches, units, p["rank"]).sum(axis=2)
        out = per_filter + (bias if bias.size else 0.0)
        magnitude = magnitude + (np.abs(bias) if bias.size else 0.0)
    return np.clip(out, p["activation_min"], p["activation_max"]), magnitude


# ------------------------------------------------------------------ goldens


@pytest.mark.parametrize("name", INT_CASES)
def test_integer_golden_equals_the_independent_model(name: str, tmp_path: Path) -> None:
    op, header = _generate(name, tmp_path)
    call = op.reference
    assert call.entry == ("svdf_s8" if DESCRIPTORS[name]["hint"]["state_dtype"] == "S8" else "svdf_state_s16_s8")
    out, state = _svdf_int_model(call)
    np.testing.assert_array_equal(_emitted_golden(header, name, np.int8), out[-1].reshape(-1).astype(np.int8))
    produced = call.run()
    np.testing.assert_array_equal(produced["output"], out)
    np.testing.assert_array_equal(produced["state_out"], state)


@pytest.mark.parametrize("state_dtype", ["S8", "S16"])
def test_integer_offsets_follow_the_kernel_convention(state_dtype: str, tmp_path: Path) -> None:
    """cmsis_nn_svdf_params.input_offset is subtracted from the input (the kernel passes -zp_in)."""
    base = DESCRIPTORS["svdf_bias_rank2_s8" if state_dtype == "S8" else "svdf_state_s16_bias_rank2_s8"]
    desc = {**base, "name": f"svdf_offsets_{state_dtype.lower()}",
            "hint": {**base["hint"], "input_offset": 3, "output_offset": -2, "output_multiplier": 1 << 30,
                     "output_shift": -1}}
    op = OpSVDF(desc, 7, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    header = (tmp_path / "includes" / f"{desc['name']}_svdf.h").read_text()
    call = op.reference
    assert call.params["input_offset"] == -3 and call.params["output_offset"] == -2
    x = call.inputs["input"].astype(np.int64)
    wf = call.inputs["weights_feature"].astype(np.int64)
    wt = call.inputs["weights_time"].astype(np.int64)
    state = call.inputs["state"].astype(np.int64)
    state = np.concatenate([state[:, :, 1:], np.einsum("fi,bi->bf", wf, x[0] - 3)[:, :, None]], axis=2)
    state = np.clip(state, call.params["input_activation_min"], call.params["input_activation_max"])
    rank = call.params["rank"]
    totals = np.einsum("fm,bfm->bf", wt, state).reshape(state.shape[0], -1, rank).sum(axis=2) + call.inputs["bias"]
    expected = [min(max(_mbqm(int(v), 1 << 30, -1) - 2, -128), 127) for v in totals.reshape(-1)]
    np.testing.assert_array_equal(_emitted_golden(header, desc["name"], np.int8), np.array(expected, np.int8))


@pytest.mark.parametrize("name", FLOAT_CASES)
def test_float_golden_is_the_float64_model_within_rounding(name: str, tmp_path: Path) -> None:
    op, header = _generate(name, tmp_path)
    call = op.reference
    f16 = DESCRIPTORS[name]["activation_dtype"] == "FP16"
    assert call.entry == ("svdf_f16" if f16 else "svdf_f32")
    assert call.inputs["input"].shape[0] == DESCRIPTORS[name]["hint"]["sequence_steps"]
    model, magnitude = _svdf_float_model(call)
    golden = _emitted_golden(header, name, np.float16 if f16 else np.float32).astype(np.float64)
    eps = float(np.finfo(np.float16 if f16 else np.float32).eps)
    # The state projection, every filter's time reduction and the output each round once.
    bound = 4 * eps * (magnitude[-1].reshape(-1) if magnitude.ndim > 1 else magnitude) + eps * np.abs(golden)
    assert np.all(np.abs(golden - model.reshape(-1)) <= bound), (golden, model.reshape(-1), bound)


@pytest.mark.parametrize("name", FAULT_CASES)
def test_fault_cases_are_status_only_and_keep_their_harness(name: str, tmp_path: Path) -> None:
    op, header = _generate(name, tmp_path)
    assert op.status_only() and not op.uses_reference()
    assert op.reference is None
    assert not list((tmp_path / name).rglob("*.reference.json"))
    assert re.search(rf"{name}_(?:output_ref|expected_output)\[\]", header)


def test_the_op_file_carries_no_converter_path() -> None:
    text = _OP_FILE.read_text().lower()
    for word in ("litert", "tflite_path", "tensorflow", "keras", "interpreter"):
        assert word not in text


def test_float_input_activation_hint_is_rejected(tmp_path: Path) -> None:
    desc = dict(DESCRIPTORS["svdf_float_default_f32"])
    desc["hint"] = {**desc["hint"], "input_activation_min": -1.0}
    with pytest.raises(ValueError, match="input_activation_min/max are not supported"):
        OpSVDF(desc, 1, target_cpu="cortex-m55").generate_c_files(tmp_path)


# ------------------------------------------------------------- entry negatives

_INT_PARAMS = dict(rank=1, input_offset=0, output_offset=0, input_multiplier=1 << 30, input_shift=0,
                   output_multiplier=1 << 30, output_shift=0, input_activation_min=-128, input_activation_max=127,
                   output_activation_min=-128, output_activation_max=127)


def _units(features: int, rank: int) -> int:
    # An invalid rank keeps every other tensor well formed, so only the rank is wrong.
    return features // rank if rank > 0 and features % rank == 0 else features


def _int_case(S=1, B=2, I=3, F=4, M=3, U=None, state_dtype=np.int8, bias=True, **override):
    rank = override.pop("rank", 1)
    U = _units(F, rank) if U is None else U
    params = {**_INT_PARAMS, "rank": rank, **override}
    inputs = {"input": np.ones((S, B, I), np.int8), "weights_feature": np.ones((F, I), np.int8),
              "weights_time": np.ones((F, M), state_dtype),
              "bias": np.ones((U,), np.int32) if bias else np.zeros((0,), np.int32),
              "state": np.zeros((B, F, M), state_dtype)}
    outputs = {"output": (S, B, U), "state_out": (B, F, M)}
    return params, inputs, outputs


def _float_case(dtype=np.float32, S=2, B=1, I=4, F=4, M=3, rank=1, U=None):
    U = _units(F, rank) if U is None else U
    params = {"rank": rank, "activation_min": -1.0e30, "activation_max": 1.0e30}
    inputs = {"input": np.ones((S, B, I), dtype), "weights_feature": np.ones((F, I), dtype),
              "weights_time": np.ones((F, M), dtype), "bias": np.ones((U,), dtype), "state": np.zeros((B, F, M), dtype)}
    return params, inputs, {"output": (S, B, U), "state_out": (B, F, M)}


def _status(entry, params, inputs, outputs) -> str:
    with pytest.raises(ReferenceKernelError) as info:
        get_bindings().run(entry, params, inputs, outputs)
    return info.value.status


INT_ENTRIES = [("svdf_s8", np.int8), ("svdf_state_s16_s8", np.int16)]
FLOAT_ENTRIES = [("svdf_f32", np.float32), ("svdf_f16", np.float16)]


@pytest.mark.parametrize("entry,state_dtype", INT_ENTRIES)
def test_integer_entries_run_the_minimal_case(entry, state_dtype) -> None:
    params, inputs, outputs = _int_case(state_dtype=state_dtype, B=1, I=1, F=1, M=1)
    got = get_bindings().run(entry, params, inputs, outputs)
    # MBQM(1, 0.5) rounds the half away from zero to 1 (the state); MBQM(1 bias + 1, 0.5) = 1.
    assert got["state_out"].tolist() == [[[1]]] and got["output"].tolist() == [[[1]]]


@pytest.mark.parametrize("entry,state_dtype", INT_ENTRIES)
@pytest.mark.parametrize("kwargs,status", [
    ({"rank": 0}, "E_PARAM"),
    ({"rank": -1}, "E_PARAM"),
    ({"rank": 3}, "E_PARAM"),                 # F = 4 is not a multiple of rank 3
    ({"input_offset": 129}, "E_PARAM"),
    ({"input_offset": -128}, "E_PARAM"),
    ({"output_offset": 128}, "E_PARAM"),
    ({"input_multiplier": -1}, "E_PARAM"),
    ({"output_shift": 31}, "E_PARAM"),
    ({"input_shift": -32}, "E_PARAM"),
    ({"output_activation_min": 5, "output_activation_max": 4}, "E_PARAM"),
    ({"output_activation_max": 128}, "E_PARAM"),
    ({"input_activation_min": 3, "input_activation_max": 2}, "E_PARAM"),
])
def test_integer_entries_reject_bad_params(entry, state_dtype, kwargs, status) -> None:
    assert _status(entry, *_int_case(state_dtype=state_dtype, **kwargs)) == status


@pytest.mark.parametrize("entry,state_dtype", INT_ENTRIES)
def test_integer_entries_reject_state_range_beyond_the_state_dtype(entry, state_dtype) -> None:
    lo = int(np.iinfo(state_dtype).min) - 1
    assert _status(entry, *_int_case(state_dtype=state_dtype, input_activation_min=lo)) == "E_PARAM"


@pytest.mark.parametrize("entry,state_dtype", INT_ENTRIES)
def test_integer_entries_reject_an_int32_overflowing_accumulator(entry, state_dtype) -> None:
    params, inputs, outputs = _int_case(state_dtype=state_dtype, I=200000, B=1, F=1, M=1)
    inputs["input"][:] = -128
    inputs["weights_feature"][:] = -128
    params["input_offset"] = -127
    assert _status(entry, params, inputs, outputs) == "E_PARAM"


def _mutations(case):
    """Shape mismatches: each yields (inputs, outputs) with one tensor disagreeing."""
    params, inputs, outputs = case
    B, F, M = inputs["state"].shape
    S, _, I = inputs["input"].shape
    U = outputs["output"][2]
    yield "weights_feature cols", {**inputs, "weights_feature": np.ones((F, I + 1), inputs["weights_feature"].dtype)}, outputs
    yield "weights_time rows", {**inputs, "weights_time": np.ones((F + 1, M), inputs["weights_time"].dtype)}, outputs
    yield "state batches", {**inputs, "state": np.zeros((B + 1, F, M), inputs["state"].dtype)}, outputs
    yield "state memory", {**inputs, "state": np.zeros((B, F, M + 1), inputs["state"].dtype)}, outputs
    yield "bias length", {**inputs, "bias": np.ones((U + 1,), inputs["bias"].dtype)}, outputs
    yield "bias rank", {**inputs, "bias": np.ones((1, U), inputs["bias"].dtype)}, outputs
    yield "input rank", {**inputs, "input": inputs["input"].reshape(S * B, I)}, outputs
    yield "output units", inputs, {**outputs, "output": (S, B, U + 1)}
    yield "output steps", inputs, {**outputs, "output": (S + 1, B, U)}
    yield "state_out", inputs, {**outputs, "state_out": (B, F, M + 1)}


@pytest.mark.parametrize("entry,state_dtype", INT_ENTRIES)
def test_integer_entries_reject_shape_mismatches(entry, state_dtype) -> None:
    case = _int_case(state_dtype=state_dtype, S=2)
    for what, inputs, outputs in _mutations(case):
        assert _status(entry, case[0], inputs, outputs) == "E_SHAPE", what


@pytest.mark.parametrize("entry,dtype", FLOAT_ENTRIES)
def test_float_entries_reject_shape_mismatches(entry, dtype) -> None:
    case = _float_case(dtype)
    for what, inputs, outputs in _mutations(case):
        assert _status(entry, case[0], inputs, outputs) == "E_SHAPE", what


@pytest.mark.parametrize("entry,dtype", FLOAT_ENTRIES)
@pytest.mark.parametrize("rank", [0, -2, 3])
def test_float_entries_reject_bad_rank(entry, dtype, rank) -> None:
    params, inputs, outputs = _float_case(dtype)
    params["rank"] = rank
    assert _status(entry, params, inputs, outputs) == "E_PARAM"


@pytest.mark.parametrize("entry,dtype", FLOAT_ENTRIES)
def test_float_entries_reject_an_inverted_activation_range(entry, dtype) -> None:
    params, inputs, outputs = _float_case(dtype)
    params.update(activation_min=1.0, activation_max=-1.0)
    assert _status(entry, params, inputs, outputs) == "E_PARAM"


@pytest.mark.parametrize("entry,dtype", FLOAT_ENTRIES)
def test_float_entries_take_an_absent_bias_and_clamp_the_output(entry, dtype) -> None:
    params, inputs, outputs = _float_case(dtype, S=1, I=2, F=2, M=2, rank=2)
    inputs["bias"] = np.zeros((0,), dtype)
    params.update(activation_min=-1.0, activation_max=3.0)
    got = get_bindings().run(entry, params, inputs, outputs)
    # projection = 2 per filter into the newest slot; per unit 2 filters * 2 = 4, clamped to 3.
    assert got["output"].tolist() == [[[3.0]]]
    np.testing.assert_array_equal(got["state_out"], np.array([[[0, 2], [0, 2]]], dtype))


@pytest.mark.parametrize("entry,dtype", FLOAT_ENTRIES)
def test_float_entries_reject_wrong_element_types(entry, dtype) -> None:
    params, inputs, outputs = _float_case(dtype)
    other = np.float16 if dtype == np.float32 else np.float32
    with pytest.raises((TypeError, ReferenceKernelError)):
        get_bindings().run(entry, params, {**inputs, "state": inputs["state"].astype(other)}, outputs)
