"""Issue #123: HardSwishPrecise takes an explicit hint.extras.prescale.

The prescale derived from the scales is 0 for every realistic scale, so without
an override the kernel's prescale shift is never exercised.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.golden_check import case_problems
from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import (
    _validate_and_normalize_descriptor,
    expand_descriptor_variations,
    load_descriptor,
)
from helia_core_tester.generation.ops._shared.hard_swish_base import PRESCALE_MAX, HardSwishFamilyBase
from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_compat import OpHardSwishCompat
from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_precise import OpHardSwishPrecise

# The Unity suite's s = 1/128 parameter set, keyed as in the generated kernel call.
UNITY_PARAMS = {
    "input_offset": 0,
    "output_offset": 0,
    "output_multiplier": 1431655765,
    "output_shift": -5,
    "relu_q3": 384,
    "relu_q6": 768,
    "prescale": 4,
}
# The template context names the multiplier output_mult.
UNITY_CONTEXT = {("output_mult" if k == "output_multiplier" else k): v for k, v in UNITY_PARAMS.items()}


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
    assert {k: ctx[k] for k in UNITY_CONTEXT} == UNITY_CONTEXT


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


# At s = 1/128, relu_q6 = 768: prescale 10 leaves xr in {0, 1}; 11 rounds it to 0 for every input.
@pytest.mark.parametrize("prescale", [0, 10])
def test_boundary_prescales_are_accepted(monkeypatch, tmp_path: Path, prescale: int) -> None:
    assert _context(monkeypatch, _desc(prescale=prescale), tmp_path)["prescale"] == prescale


@pytest.mark.parametrize("prescale", [11, PRESCALE_MAX])
def test_prescale_that_zeroes_the_relu6_term_is_rejected(monkeypatch, tmp_path: Path, prescale: int) -> None:
    with pytest.raises(ValueError, match="rounds the relu6 term .* to 0 for every input"):
        _context(monkeypatch, _desc(prescale=prescale), tmp_path)


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


S16 = (-32768, 32767)


@pytest.mark.parametrize(
    ("prescale", "relu_q6", "input_zp", "fits"),
    [
        # |x| <= 32768 at zero point 0: 65535 * 32768 fits, 65536 * 32768 does not.
        (0, 65535, 0, True),
        (0, 65536, 0, False),
        # The zero point moves the |x| bound: 32769 at zp 1 or -2 overflows, 32768 at zp -1 fits.
        (0, 65535, 1, False),
        (0, 65535, -2, False),
        (0, 65535, -1, True),
        # xr rounds half up: (131069 + 1) >> 1 = 65535 fits, (131071 + 1) >> 1 = 65536 does not.
        (1, 131069, 0, True),
        (1, 131071, 0, False),
    ],
)
def test_overflow_check_boundary(prescale: int, relu_q6: int, input_zp: int, fits: bool) -> None:
    args = (prescale, relu_q6, input_zp, *S16)
    assert HardSwishFamilyBase._prescale_overflows(*args) is (not fits)
    if fits:
        HardSwishFamilyBase._check_prescale_fits(*args)
    else:
        with pytest.raises(ValueError, match="overflows int32"):
            HardSwishFamilyBase._check_prescale_fits(*args)


def test_derived_prescale_uses_the_same_bound_as_the_override(monkeypatch, tmp_path: Path) -> None:
    # relu_q6 = 65538: 32767 * 65538 fits int32 but 32768 * 65538 does not, so the
    # smallest safe prescale is 1, and an explicit 0 is rejected for the same reason.
    scale = 6.0 / 65538
    assert HardSwishFamilyBase._derive_prescale(65538, 0, *S16) == 1
    assert _context(monkeypatch, _desc("S16", input_scale=scale, output_scale=scale), tmp_path)["prescale"] == 1
    with pytest.raises(ValueError, match="overflows int32"):
        _context(monkeypatch, _desc("S16", input_scale=scale, output_scale=scale, prescale=0), tmp_path)
    assert _context(monkeypatch, _desc("S16", input_scale=scale, output_scale=scale, prescale=1), tmp_path)["prescale"] == 1


@pytest.mark.parametrize("relu_q6", [0, 1, 6, 768, 65535, 393216, 2**31 - 1])
@pytest.mark.parametrize("input_zp", [0, 127, -128])
def test_derived_prescale_is_the_smallest_safe_one(relu_q6: int, input_zp: int) -> None:
    prescale = HardSwishFamilyBase._derive_prescale(relu_q6, input_zp, *S16)
    assert not HardSwishFamilyBase._prescale_overflows(prescale, relu_q6, input_zp, *S16)
    if prescale:
        assert HardSwishFamilyBase._prescale_overflows(prescale - 1, relu_q6, input_zp, *S16)


def test_compat_rejects_prescale(monkeypatch, tmp_path: Path) -> None:
    desc = _desc(prescale=4)
    desc["operator"] = "HardSwishCompat"
    with pytest.raises(ValueError, match="applies only to the integer HardSwishPrecise kernels"):
        _context(monkeypatch, desc, tmp_path, op_cls=OpHardSwishCompat)


@pytest.mark.parametrize("dtype", ["FP32", "FP16"])
def test_float_route_rejects_prescale(monkeypatch, tmp_path: Path, dtype: str) -> None:
    desc = _validate_and_normalize_descriptor(
        {
            "operator": "HardSwishPrecise",
            "name": f"hard_swish_float_case_{dtype.lower()}",
            "suite": "float",
            "tensor_dtypes": {"input": dtype, "output": dtype},
            "input_shape": [1, 7],
            "hint": {"extras": {"prescale": 4}},
        }
    )
    with pytest.raises(ValueError, match="applies only to the integer HardSwishPrecise kernels"):
        _context(monkeypatch, desc, tmp_path)


@pytest.mark.parametrize(
    ("extras", "dtype", "descriptor_quantized"),
    [
        ({}, "S8", False),
        ({"prescale": 4}, "S8", False),
        ({"input_scale": 0.0078125}, "S8", True),
        ({"output_zero_point": 0}, "S8", True),
        ({}, "S16", True),
    ],
)
def test_pinned_quantization_skips_the_calibrated_tflite(tmp_path: Path, extras, dtype, descriptor_quantized) -> None:
    desc = _desc(dtype)
    desc["hint"]["extras"] = dict(extras)
    op = OpHardSwishPrecise(desc, seed=7, target_cpu="cortex-m55")
    assert op.needs_keras_model() is (not descriptor_quantized)
    if descriptor_quantized:
        out = tmp_path / "case.tflite"
        op.convert_to_tflite(None, str(out), 7)
        assert not out.exists()


def _schema() -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / "generation" / "descriptors" / "schema.json").read_text())


def test_prescale_max_matches_the_schema() -> None:
    schema = _schema()
    prescale = schema["properties"]["hint"]["properties"]["extras"]["properties"]["prescale"]
    assert (prescale["minimum"], prescale["maximum"]) == (0, PRESCALE_MAX)


@pytest.mark.parametrize(
    ("operator", "ok"),
    [("HardSwishPrecise", True), ("HardSwishCompat", False), ("Relu", False)],
)
def test_schema_ties_prescale_to_hard_swish_precise(operator: str, ok: bool) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = _schema()
    (rule,) = [r for r in schema["allOf"] if "hint.extras.prescale" in r.get("$comment", "")]
    doc = {"operator": operator, "hint": {"extras": {"prescale": 4}}}
    assert jsonschema.Draft202012Validator(rule).is_valid(doc) is ok
    assert jsonschema.Draft202012Validator(rule).is_valid({"operator": operator, "hint": {"extras": {}}})

PRESCALE_CASES = [
    "hard_swish_precise_vector_prescale4_s8",
    "hard_swish_precise_vector_1760_lut_prescale4_s8",
    "hard_swish_precise_vector_prescale4_s16",
]


_SHIPPED = {
    expanded["name"]: expanded
    for desc in load_descriptor(str(find_descriptors_dir() / "ActivationFunctions" / "hard_swish_precise.yaml"))
    for expanded in expand_descriptor_variations(desc)
}


@pytest.mark.parametrize("name", PRESCALE_CASES)
def test_shipped_prescale_cases_have_live_goldens(tmp_path: Path, name: str) -> None:
    pytest.importorskip("tensorflow")
    from helia_core_tester.core.config import Config
    from helia_core_tester.generation.test_ops import generate_test

    assert _SHIPPED[name]["hint"]["extras"]["prescale"] == 4
    generate_test(_SHIPPED[name], str(tmp_path), seed=Config.seed)
    (case_dir,) = tmp_path.glob(f"*/{name}")
    assert case_problems(case_dir) == []
    # Every quantization parameter is pinned, so no calibrated model is shipped beside the case.
    assert list(case_dir.rglob("*.tflite")) == []
    c_text = next(case_dir.rglob("*_hard_swish.c")).read_text()
    # Through the real pipeline (int8 builds a calibrated .tflite), the
    # descriptor's scales and zero points must win, giving the Unity set.
    params = {
        key: int(value)
        for value, key in re.findall(r"^\s*(-?\d+),\s*// (\w+)", c_text, flags=re.M)
    }
    assert {k: params.get(k) for k in UNITY_PARAMS} == UNITY_PARAMS
