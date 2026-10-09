"""Float weighted cases on the TFLM f32 kernels, and strict-JSON records of unbounded activations."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from helia_core_tester.generation.reference import weighted
from helia_core_tester.generation.reference.case import ReferenceCall
from helia_core_tester.generation.reference.weighted import ReferenceCaseError


def _fc_desc(**extra):
    return {"name": "fc_case", "input_shape": [3, 7], "filter_shape": [5, 7], **extra}


def _case(desc, dtype=np.float32, seed=0):
    spec = weighted.fc_spec(desc)
    draw = np.random.default_rng(seed + 1)
    return weighted.build_float_case(desc, spec, np.random.default_rng(seed),
                                     lambda: draw.uniform(-1, 1, spec.input_shape), dtype)


def test_float_fc_case_matches_numpy() -> None:
    case = _case(_fc_desc())
    expected = case.input.astype(np.float64) @ case.weights.T.astype(np.float64) + case.bias
    np.testing.assert_allclose(case.output, expected, rtol=1e-5, atol=1e-6)
    assert case.call.kernel == "fc_f32"
    np.testing.assert_array_equal(case.reference([case.input]), case.output)


def test_float_case_without_bias_and_with_clamp() -> None:
    case = _case(_fc_desc(use_bias=False, activation="RELU6", activation_max=0.1))
    assert case.bias is None
    expected = np.clip(case.input.astype(np.float64) @ case.weights.T.astype(np.float64), 0.0, 0.1)
    np.testing.assert_allclose(case.output, expected, rtol=1e-5, atol=1e-6)


def test_float16_case_rounds_operands_and_casts_the_golden_once() -> None:
    case = _case(_fc_desc(), dtype=np.float16)
    assert case.input.dtype == case.weights.dtype == case.bias.dtype == case.output.dtype == np.float16
    wide = (case.input.astype(np.float32) @ case.weights.T.astype(np.float32) + case.bias.astype(np.float32))
    np.testing.assert_array_equal(case.output, wide.astype(np.float16))


def test_float_case_draws_are_seeded() -> None:
    a, b = _case(_fc_desc(), seed=3), _case(_fc_desc(), seed=3)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.output, b.output)
    assert not np.array_equal(a.weights, _case(_fc_desc(), seed=4).weights)


def test_float_case_reference_propagates_nonfinite_inputs() -> None:
    case = _case(_fc_desc())
    poisoned = case.input.copy()
    poisoned[1, 0] = np.inf
    out = case.reference([poisoned])
    assert np.isfinite(out[[0, 2]]).all()
    assert not np.isfinite(out[1]).all()


def test_float_bounds_and_dtype_rejections() -> None:
    assert weighted.float_bounds({"activation": "NONE"}) == (-math.inf, math.inf)
    assert weighted.float_bounds({"activation": "RELU", "activation_max": 2.0}) == (0.0, 2.0)
    with pytest.raises(ReferenceCaseError, match="empty"):
        weighted.float_bounds({"name": "x", "activation": "RELU6", "activation_min": 7.0})
    with pytest.raises(ValueError, match="unsupported fused activation"):
        weighted.float_bounds({"activation": "TANH"})
    with pytest.raises(ReferenceCaseError, match="f32 or f16"):
        _case(_fc_desc(), dtype=np.float64)


def test_unbounded_activation_records_as_strict_json(tmp_path) -> None:
    call = ReferenceCall("fc_f32", {"act": {"min": 0, "max": 0, "fmin": -math.inf, "fmax": math.inf}},
                         {"input": np.ones((1, 2), np.float32), "filter": np.ones((1, 2), np.float32), "bias": None},
                         (1, 1), "float32")
    record = json.loads((call.to_json(tmp_path / "r.json")).read_text(), parse_constant=pytest.fail)
    assert record["params"]["act"] == {"min": 0, "max": 0, "fmin": "-inf", "fmax": "inf"}
    assert float(record["params"]["act"]["fmin"]) == -math.inf
