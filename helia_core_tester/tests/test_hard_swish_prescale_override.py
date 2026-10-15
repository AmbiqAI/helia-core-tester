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


# At s = 1/128 (relu_q3 384, relu_q6 768) xr follows x: s8 reaches x + 384 <= 511, so
# prescale 9 still leaves xr = 1 and 10 rounds it to 0 for every code; s16 reaches the
# full 768, so 10 leaves xr in {0, 1} and 11 zeroes it.
@pytest.mark.parametrize(("dtype", "prescale"), [("S8", 0), ("S8", 9), ("S16", 0), ("S16", 10)])
def test_boundary_prescales_are_accepted(monkeypatch, tmp_path: Path, dtype: str, prescale: int) -> None:
    assert _context(monkeypatch, _desc(dtype, prescale=prescale), tmp_path)["prescale"] == prescale


@pytest.mark.parametrize(("dtype", "prescale"), [("S8", 10), ("S8", PRESCALE_MAX), ("S16", 11), ("S16", PRESCALE_MAX)])
def test_prescale_that_zeroes_the_relu6_term_is_rejected(
    monkeypatch, tmp_path: Path, dtype: str, prescale: int
) -> None:
    with pytest.raises(ValueError, match="rounds the relu6 term .* to 0 for every input"):
        _context(monkeypatch, _desc(dtype, prescale=prescale), tmp_path)


@pytest.mark.parametrize("prescale", [-1, PRESCALE_MAX + 1, 4.0, "4", True, None])
def test_invalid_prescales_are_rejected(monkeypatch, tmp_path: Path, prescale) -> None:
    desc = _desc()
    desc["hint"]["extras"]["prescale"] = prescale
    with pytest.raises(ValueError, match=r"hint\.extras\.prescale must be"):
        _context(monkeypatch, desc, tmp_path)


def test_prescale_that_overflows_int32_is_rejected(monkeypatch, tmp_path: Path) -> None:
    # s16 at 2^-16: relu_q3 = 196608, relu_q6 = 393216. x = 32767 reaches x + relu_q3 = 229375,
    # so prescale 1 gives 32767 * 114688 (overflow) and 2 gives 32767 * 57344 (fits).
    tiny = {"input_scale": 2.0**-16, "output_scale": 2.0**-16}
    for prescale in (0, 1):
        with pytest.raises(ValueError, match="overflows int32"):
            _context(monkeypatch, _desc("S16", prescale=prescale, **tiny), tmp_path)
    assert _context(monkeypatch, _desc("S16", **tiny), tmp_path)["prescale"] == 2
    assert _context(monkeypatch, _desc("S16", prescale=2, **tiny), tmp_path)["prescale"] == 2


S16 = (-32768, 32767)


@pytest.mark.parametrize(
    ("prescale", "relu_q3", "relu_q6", "input_zp", "fits"),
    [
        # xr clamps to relu_q6 at x = 32767: 32767 * 65538 fits int32, 32767 * 65539 does not.
        # (An independent |x| <= 32768 bound would reject the first: x = -32768 only
        # reaches x + relu_q3 = 7232.)
        (0, 40000, 65538, 0, True),
        (0, 40000, 65539, 0, False),
        # Negative side at zp 32767 (x in [-65535, 0]): min of x * (x + relu_q3) is at
        # x = -relu_q3 / 2; 92681 gives -46340 * 46341 (fits), 92682 gives -46341^2 (overflow).
        (0, 92681, 2 * 92681, 32767, True),
        (0, 92682, 2 * 92682, 32767, False),
        # xr rounds half up: (131075 + 1) >> 1 = 65538 fits, (131077 + 1) >> 1 = 65539 does not;
        # flooring would give 65538 for both.
        (1, 100000, 131075, 0, True),
        (1, 100000, 131077, 0, False),
        # x + relu_q3 itself must stay in int32 before the clamp.
        (PRESCALE_MAX, 2**31 - 32768, 2**31 - 1, 0, True),
        (PRESCALE_MAX, 2**31 - 32767, 2**31 - 1, 0, False),
    ],
)
def test_overflow_check_boundary(prescale: int, relu_q3: int, relu_q6: int, input_zp: int, fits: bool) -> None:
    args = (prescale, relu_q3, relu_q6, input_zp, *S16)
    assert HardSwishFamilyBase._prescale_overflows(*args) is (not fits)
    if fits:
        HardSwishFamilyBase._check_prescale_fits(*args)
    else:
        with pytest.raises(ValueError, match="overflows int32"):
            HardSwishFamilyBase._check_prescale_fits(*args)


def _scalar_overflows(prescale: int, relu_q3: int, relu_q6: int, input_zp: int, qmin: int, qmax: int) -> bool:
    """Reference: the golden simulation's own scalar steps over every input code."""
    lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    for q in range(qmin, qmax + 1):
        x = q - input_zp
        xr = HardSwishFamilyBase._clamp(x + relu_q3, 0, relu_q6)
        xr = HardSwishFamilyBase._nonneg_divide_by_pot_s32(xr, prescale)
        if not (lo <= x + relu_q3 <= hi and lo <= x * xr <= hi):
            return True
    return False


@pytest.mark.parametrize("relu_q6", [0, 6, 768, 65538, 131072, 393216])
@pytest.mark.parametrize(("input_zp", "bounds"), [(0, S16), (32767, S16), (-128, (-128, 127)), (127, (-128, 127))])
def test_derived_prescale_is_the_smallest_safe_one(relu_q6: int, input_zp: int, bounds) -> None:
    relu_q3 = (relu_q6 + 1) // 2
    prescale = HardSwishFamilyBase._derive_prescale(relu_q3, relu_q6, input_zp, *bounds)
    assert not _scalar_overflows(prescale, relu_q3, relu_q6, input_zp, *bounds)
    if prescale:
        assert _scalar_overflows(prescale - 1, relu_q3, relu_q6, input_zp, *bounds)


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
    # Through the real pipeline the parameters come from the descriptor alone, giving the Unity set.
    params = {
        key: int(value)
        for value, key in re.findall(r"^\s*(-?\d+),\s*// (\w+)", c_text, flags=re.M)
    }
    assert {k: params.get(k) for k in UNITY_PARAMS} == UNITY_PARAMS
