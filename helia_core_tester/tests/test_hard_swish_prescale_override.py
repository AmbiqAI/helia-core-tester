"""Issue #123: HardSwishPrecise takes an explicit hint.extras.prescale.

The prescale derived from the scales is 0 for every realistic scale, so without
an override the kernel's prescale shift is never exercised. The C reference's
hard_swish_precise_prepare derives or pins it; these tests hold that against an
independent scalar model of the kernel's int32 steps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.golden_check import case_problems
from helia_core_tester.generation.io.descriptors import (
    _validate_and_normalize_descriptor,
    expand_descriptor_variations,
    load_descriptor,
)
from helia_core_tester.generation.ops._shared.hard_swish_base import PRESCALE_MAX
from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_compat import OpHardSwishCompat
from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_precise import OpHardSwishPrecise
from helia_core_tester.generation.reference.abi import dtype_code
from helia_core_tester.generation.reference.bindings import ReferenceKernelError, get_bindings
from helia_core_tester.tests.test_reference_activation import _model_hard_swish_precise

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
REJECTED = r"hint\.extras\.prescale \d+ either lets x \+ relu_q3 or x \* relu6 leave int32"


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

    def capture(self, output_dir, **kwargs):
        captured.update(kwargs["context"])
        captured["_reference"] = self.reference

    monkeypatch.setattr(op_cls, "render_harness_case", capture)
    monkeypatch.setattr(op_cls, "_generate_float_c_files", lambda self, *a, **k: None, raising=False)
    op_cls(desc, seed=7, target_cpu="cortex-m55").generate_c_files(tmp_path)
    return captured


def _ints(literal: str) -> np.ndarray:
    return np.array([int(v) for v in re.findall(r"-?\d+", literal)], dtype=np.int64)


def _prepare(dtype: str, scale: float, zero_point: int = 0, prescale: int = -1) -> dict:
    return get_bindings().prepare("hard_swish_precise_prepare", {
        "dtype": dtype_code("int8" if dtype == "S8" else "int16"), "input_scale": scale,
        "input_zero_point": zero_point, "output_scale": scale, "output_zero_point": zero_point,
        "prescale": prescale})


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
    kind = dtype.lower()
    np.testing.assert_array_equal(golden, [_model_hard_swish_precise(int(v), UNITY_PARAMS, kind) for v in x])
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
    with pytest.raises(ValueError, match=REJECTED):
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
        with pytest.raises(ValueError, match=REJECTED):
            _context(monkeypatch, _desc("S16", prescale=prescale, **tiny), tmp_path)
    assert _context(monkeypatch, _desc("S16", **tiny), tmp_path)["prescale"] == 2
    assert _context(monkeypatch, _desc("S16", prescale=2, **tiny), tmp_path)["prescale"] == 2


def _scalar_overflows(prescale: int, relu_q3: int, relu_q6: int, input_zp: int, qmin: int, qmax: int) -> bool:
    """Reference: the kernel's own scalar int32 steps over every input code."""
    lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    for q in range(qmin, qmax + 1):
        x = q - input_zp
        xr = min(max(x + relu_q3, 0), relu_q6)
        if prescale:
            xr = (xr + (1 << (prescale - 1))) >> prescale
        if not (lo <= x + relu_q3 <= hi and lo <= x * xr <= hi):
            return True
    return False


def _q(scale: float) -> tuple[int, int]:
    return int(np.floor(3.0 / scale + 0.5)), int(np.floor(6.0 / scale + 0.5))


@pytest.mark.parametrize("scale", [2.0**-7, 1e-4, 9.2e-5, 2.0**-14, 5e-5, 2.0**-16, 1e-5])
@pytest.mark.parametrize(("dtype", "zero_point", "bounds"), [("S16", 0, (-32768, 32767)), ("S8", -128, (-128, 127)),
                                                             ("S8", 127, (-128, 127))])
def test_derived_prescale_is_the_smallest_safe_one(scale: float, dtype: str, zero_point: int, bounds) -> None:
    try:
        p = _prepare(dtype, scale, zero_point)
    except ReferenceKernelError:
        pytest.skip("relu_q6 outside the kernel's range at this scale")
    relu_q3, relu_q6 = _q(float(np.float32(scale)))
    assert (p["relu_q3"], p["relu_q6"]) == (relu_q3, relu_q6)
    assert not _scalar_overflows(p["prescale"], relu_q3, relu_q6, zero_point, *bounds)
    if p["prescale"]:
        assert _scalar_overflows(p["prescale"] - 1, relu_q3, relu_q6, zero_point, *bounds)


def test_prepare_pins_a_prescale_only_where_the_kernel_stays_in_int32() -> None:
    # 2^-16: prescale 2 is the smallest safe one for s16 (see the overflow case above).
    assert _prepare("S16", 2.0**-16)["prescale"] == 2
    assert _prepare("S16", 2.0**-16, prescale=3)["prescale"] == 3
    for prescale in (0, 1, -2, PRESCALE_MAX + 1):
        with pytest.raises(ReferenceKernelError, match="E_PARAM"):
            _prepare("S16", 2.0**-16, prescale=prescale)


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
    ("extras", "dtype", "pinned"),
    [
        ({}, "S8", False),
        ({"prescale": 4}, "S8", False),
        ({"input_scale": 0.0078125}, "S8", True),
        ({"output_zero_point": 0}, "S8", True),
    ],
)
def test_pinned_quantization_comes_from_the_descriptor(monkeypatch, tmp_path: Path, extras, dtype, pinned) -> None:
    desc = _desc(dtype)
    desc["hint"]["extras"] = dict(extras)
    call = _context(monkeypatch, desc, tmp_path)["_reference"]
    if pinned:
        assert call.quant["input"]["scale"] == float(np.float32(extras.get("input_scale", 1.0)))
    else:
        assert call.quant["input"]["scale"] != 1.0


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
    from helia_core_tester.generation.test_ops import generate_test

    assert _SHIPPED[name]["hint"]["extras"]["prescale"] == 4
    generate_test(_SHIPPED[name], str(tmp_path))
    (case_dir,) = tmp_path.glob(f"*/{name}")
    assert case_problems(case_dir) == []
    assert list(case_dir.rglob("*.tflite")) == []
    reference = json.loads(next(case_dir.rglob("*.reference.json")).read_text())
    # Through the real pipeline the parameters come from the descriptor alone, giving the Unity set.
    assert {k: reference["params"][k] for k in UNITY_PARAMS} == UNITY_PARAMS
