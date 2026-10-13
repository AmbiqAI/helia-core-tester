"""Issue #123: HardSwishPrecise takes an explicit hint.extras.prescale.

The prescale derived from the scales is 0 for every realistic scale, so without
an override the kernel's prescale shift is never exercised.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.golden_check import case_problems
from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared.hard_swish_base import PRESCALE_MAX, HardSwishFamilyBase
from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_compat import OpHardSwishCompat
from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_precise import OpHardSwishPrecise

UNITY_S8 = {
    "input_offset": 0,
    "output_offset": 0,
    "output_mult": 1431655765,
    "output_shift": -5,
    "relu_q3": 384,
    "relu_q6": 768,
    "prescale": 4,
}


def _desc(dtype: str = "S8", length: int = 37, **extras) -> dict:
    base = {
        "input_scale": 0.0078125,
        "output_scale": 0.0078125,
        "input_zero_point": 0,
        "output_zero_point": 0,
    }
    base.update(extras)
    return {
        "operator": "HardSwishPrecise",
        "name": f"hard_swish_precise_case_{dtype.lower()}",
        "activation_dtype": dtype,
        "weight_dtype": "S8",
        "hint": {"call_style": "per_tensor", "extras": {k: v for k, v in base.items() if v is not None}},
        "input_shape": [1, length],
    }


def _context(monkeypatch, desc: dict, tmp_path: Path, op_cls=OpHardSwishPrecise) -> dict:
    captured = {}

    def capture(self, output_dir, stem, h_template, c_template, context, *args, **kwargs):
        captured.update(context)

    monkeypatch.setattr(op_cls, "_write_op_outputs", capture)
    op_cls(desc, seed=7, target_cpu="cortex-m55").generate_c_files(tmp_path)
    return captured


def _ints(literal: str) -> np.ndarray:
    return np.array([int(v) for v in re.findall(r"-?\d+", literal)], dtype=np.int64)


def test_override_reproduces_the_unity_parameter_set(monkeypatch, tmp_path: Path) -> None:
    ctx = _context(monkeypatch, _desc(prescale=4), tmp_path)
    assert {k: ctx[k] for k in UNITY_S8} == UNITY_S8


def test_without_override_prescale_is_still_derived(monkeypatch, tmp_path: Path) -> None:
    ctx = _context(monkeypatch, _desc(), tmp_path)
    assert ctx["prescale"] == 0
    assert (ctx["output_mult"], ctx["output_shift"]) == (1431655765, -9)


@pytest.mark.parametrize("dtype", ["S8", "S16"])
def test_golden_applies_the_prescale_shift(monkeypatch, tmp_path: Path, dtype: str) -> None:
    ctx = _context(monkeypatch, _desc(dtype, prescale=4), tmp_path)
    x = _ints(ctx["input_data_array"])
    golden = _ints(ctx["expected_output_array"])
    xr = (np.clip(x + 384, 0, 768) + 8) >> 4
    expected = np.array(
        [HardSwishFamilyBase._requantize(int(v) * int(r), 1431655765, -5) for v, r in zip(x, xr)]
    )
    lo, hi = (-128, 127) if dtype == "S8" else (-32768, 32767)
    np.testing.assert_array_equal(golden, np.clip(expected, lo, hi))
    # The shift loses precision, so prescale 4 and 0 must give different goldens.
    plain = _context(monkeypatch, _desc(dtype), tmp_path)
    assert plain["input_data_array"] == ctx["input_data_array"]
    assert plain["expected_output_array"] != ctx["expected_output_array"]


@pytest.mark.parametrize("prescale", [0, PRESCALE_MAX])
def test_boundary_prescales_are_accepted(monkeypatch, tmp_path: Path, prescale: int) -> None:
    assert _context(monkeypatch, _desc(prescale=prescale), tmp_path)["prescale"] == prescale


@pytest.mark.parametrize("prescale", [-1, PRESCALE_MAX + 1, 4.0, "4", True, None])
def test_invalid_prescales_are_rejected(monkeypatch, tmp_path: Path, prescale) -> None:
    desc = _desc()
    desc["hint"]["extras"]["prescale"] = prescale
    with pytest.raises(ValueError, match=r"hint\.extras\.prescale must be"):
        _context(monkeypatch, desc, tmp_path)


def test_prescale_that_overflows_int32_is_rejected(monkeypatch, tmp_path: Path) -> None:
    # s16 with a tiny scale: relu6 = 6 / 2^-16 = 393216 and |x| <= 32768.
    desc = _desc("S16", input_scale=2.0**-16, output_scale=2.0**-16, prescale=0)
    with pytest.raises(ValueError, match="overflows int32"):
        _context(monkeypatch, desc, tmp_path)
    # The derived prescale for the same scales is safe, and so is the boundary override.
    assert _context(monkeypatch, _desc("S16", input_scale=2.0**-16, output_scale=2.0**-16), tmp_path)["prescale"] == 3
    assert _context(
        monkeypatch, _desc("S16", input_scale=2.0**-16, output_scale=2.0**-16, prescale=3), tmp_path
    )["prescale"] == 3


def test_overflow_check_boundary() -> None:
    int32_max = np.iinfo(np.int32).max
    # x_max = 32768 for s16 with zero point 0; 65535 * 32768 fits, 65536 * 32768 does not.
    HardSwishFamilyBase._check_prescale_fits(0, 65535, 0, -32768, 32767)
    assert 65536 * 32768 > int32_max
    with pytest.raises(ValueError, match="overflows int32"):
        HardSwishFamilyBase._check_prescale_fits(0, 65536, 0, -32768, 32767)


def test_compat_rejects_prescale(monkeypatch, tmp_path: Path) -> None:
    desc = _desc(prescale=4)
    desc["operator"] = "HardSwishCompat"
    with pytest.raises(ValueError, match="applies only to HardSwishPrecise"):
        _context(monkeypatch, desc, tmp_path, op_cls=OpHardSwishCompat)


PRESCALE_CASES = [
    "hard_swish_precise_vector_prescale4_s8",
    "hard_swish_precise_vector_1760_lut_prescale4_s8",
    "hard_swish_precise_vector_prescale4_s16",
]


@pytest.mark.parametrize("name", PRESCALE_CASES)
def test_shipped_prescale_cases_have_live_goldens(tmp_path: Path, name: str) -> None:
    pytest.importorskip("tensorflow")
    from helia_core_tester.core.config import Config
    from helia_core_tester.generation.test_ops import generate_test

    descs = {d["name"]: d for d in load_all_descriptors(str(find_descriptors_dir()))}
    assert descs[name]["hint"]["extras"]["prescale"] == 4
    generate_test(descs[name], str(tmp_path), seed=Config.seed)
    (case_dir,) = tmp_path.glob(f"*/{name}")
    assert case_problems(case_dir) == []
    c_text = next(case_dir.rglob("*_hard_swish.c")).read_text()
    assert re.search(r"\b4,\s*// prescale", c_text)
