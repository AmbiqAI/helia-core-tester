"""S8 Add/Mul `quant_preset: distinct` gives each operand its own quantization (#349)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared.binary_basic_math_base import resolve_quant_preset
from helia_core_tester.generation.ops.BasicMathFunctions.add import ADD_QUANT_PRESETS, OpAdd
from helia_core_tester.generation.ops.BasicMathFunctions.mul import MUL_QUANT_PRESETS, OpMul
from helia_core_tester.generation.reference.bindings import get_bindings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
def _descriptors() -> list[dict]:
    return load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors"))


def _descriptor(name: str) -> dict:
    return next(d for d in _descriptors() if d["name"] == name)


_OPS = {"add": OpAdd, "mul": OpMul}


def _generate(desc: dict, tmp_path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, str]:
    """The recorded reference call of one generated case: its prepared parameters, inputs, golden and entry."""
    op = _OPS[desc["operator"].lower()](dict(desc), seed=500, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    call = op.reference
    assert call is not None and not list(tmp_path.rglob("*.tflite"))
    return dict(call.params), call.inputs["input1"], call.inputs["input2"], call.output().reshape(-1), call.entry


def _simulate(entry: str, p: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = get_bindings().run(entry, p, {"input1": a, "input2": b},
                             {"output": np.broadcast_shapes(a.shape, b.shape)})["output"]
    return out.reshape(-1)


def _swapped(p: dict, *keys: str) -> dict:
    q = dict(p)
    for k in keys:
        q[f"input1_{k}"], q[f"input2_{k}"] = p[f"input2_{k}"], p[f"input1_{k}"]
    return q


def _swaps(op: str) -> list[tuple[str, ...]]:
    return [("offset",), ("multiplier", "shift"), ("offset", "multiplier", "shift")] if op == "add" else [("offset",)]


@pytest.mark.parametrize(
    "name",
    [
        "add_default_qdistinct_s8",
        "add_scalar_left_qdistinct_s8",
        "add_entry_row_broadcast_w1_qdistinct_s8",
        "mul_default_qdistinct_s8",
        "mul_scalar_right_qdistinct_s8",
        "mul_entry_row_broadcast_w2_qdistinct_s8",
    ],
)
def test_distinct_case_golden_changes_when_operand_quant_is_swapped(name: str, tmp_path: Path) -> None:
    op = name[:3]
    params, a, b, golden, entry = _generate(_descriptor(name), tmp_path)

    assert params["input1_offset"] != params["input2_offset"]
    if op == "add":
        assert (params["input1_multiplier"], params["input1_shift"]) != (params["input2_multiplier"],
                                                                          params["input2_shift"])
    assert np.array_equal(_simulate(entry, params, a, b), golden)
    # Count only moves the FVP's 1-LSB tolerance cannot absorb. mul_scalar_right has
    # 4 elements, two of which saturate either way.
    for keys in _swaps(op):
        moved = np.abs(_simulate(entry, _swapped(params, *keys), a, b).astype(np.int32) - golden) > 1
        assert np.mean(moved) >= 0.5, keys


@pytest.mark.parametrize("name", ["add_default_s8", "mul_default_s8"])
def test_shared_default_is_unchanged_and_blind_to_a_swap(name: str, tmp_path: Path) -> None:
    op = name[:3]
    params, a, b, golden, entry = _generate(_descriptor(name), tmp_path)

    assert params["input1_offset"] == params["input2_offset"]
    for keys in _swaps(op):
        assert np.array_equal(_simulate(entry, _swapped(params, *keys), a, b), golden)


@pytest.mark.parametrize("op", ["add", "mul"])
def test_every_s8_shape_has_a_distinct_twin(op: str) -> None:
    descs = [d for d in _descriptors() if d["operator"] == op.capitalize() and d["activation_dtype"] == "S8"]
    shared = {
        d["name"]: d for d in descs
        if d.get("quant_preset") is None and not d.get("expected_status")
    }
    distinct = {d["name"]: d for d in descs if d.get("quant_preset") == "distinct"}

    assert shared and len(distinct) == len(shared)
    for name, d in shared.items():
        twin = distinct[f"{name[:-len('_s8')]}_qdistinct_s8"]
        for key in ("entry", "input_1_shape", "input_2_shape", "activation", "hint"):
            assert twin.get(key) == d.get(key), (name, key)


@pytest.mark.parametrize(("op_cls", "base"), [(OpAdd, "add_default_s8"), (OpMul, "mul_default_s8")])
def test_explicit_shared_preset_matches_the_default(op_cls, base: str) -> None:
    assert op_cls({**_descriptor(base), "quant_preset": "shared"})._preset_quant == {}
    assert op_cls(_descriptor(base))._preset_quant == {}


@pytest.mark.parametrize(
    ("op_cls", "base", "overrides", "match"),
    [
        (OpAdd, "add_default_s8", {"quant_preset": "relu_range"}, "no quant_preset 'relu_range'"),
        (OpMul, "mul_default_s8", {"quant_preset": "Distinct"}, "no quant_preset 'Distinct'"),
        (OpAdd, "add_default_s8", {"quant_preset": ""}, "no quant_preset ''"),
        (OpMul, "mul_default_s8", {"quant_preset": ["distinct"]}, "no quant_preset"),
        (OpAdd, "add_default_s16", {"quant_preset": "distinct"}, "no S16 quantization"),
        (OpMul, "mul_default_s16", {"quant_preset": "distinct"}, "no S16 quantization"),
        (OpAdd, "add_float_default_f32", {"quant_preset": "distinct"}, "no FP32 quantization"),
        (OpMul, "mul_float_default_f16", {"quant_preset": "distinct"}, "no FP16 quantization"),
    ],
)
def test_bad_preset_is_rejected_at_load(op_cls, base: str, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        op_cls({**_descriptor(base), **overrides})


def test_preset_missing_a_tensor_is_rejected() -> None:
    broken = {"distinct": {"S8": {"input_1_quant": ([0.5], [0]), "output_quant": ([0.5], [0])}}}
    with pytest.raises(ValueError, match="must define"):
        resolve_quant_preset("Add", broken, "distinct", "S8")


@pytest.mark.parametrize("presets", [ADD_QUANT_PRESETS, MUL_QUANT_PRESETS])
def test_distinct_presets_differ_per_operand(presets: dict) -> None:
    q = presets["distinct"]["S8"]
    assert q["input_1_quant"][0] != q["input_2_quant"][0]
    assert q["input_1_quant"][1] != q["input_2_quant"][1]
    assert all(-128 <= zp[0] <= 127 for _, zp in q.values())
