"""Check the exact-GELU reference and the tolerance the Gelu cases assert."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.ops.ActivationFunctions.gelu import OpGelu, gelu_exact_reference

ROOT = Path(__file__).resolve().parents[2]
CASES = list(yaml.safe_load_all((ROOT / "assets/descriptors/ActivationFunctions/gelu_float.yaml").read_text()))
FINITE = np.linspace(-9.0, 9.0, 721, dtype=np.float32)


def _bound(desc):
    return desc["comparison"]["atol"], desc["comparison"]["rtol"]


def _within(actual, expected, desc):
    atol, rtol = _bound(desc)
    return np.abs(actual.astype(np.float64) - expected) <= atol + rtol * np.abs(expected)


def test_every_case_asserts_the_same_bound():
    # 2^-24 floor and 2^-21 relative from ns-cmsis-nn #743, plus 2^-24 (float32 unit roundoff) for the golden.
    golden = [desc for desc in CASES if "contract_interval" not in desc]
    assert {_bound(desc) for desc in golden} == {(2.0**-24, 2.0**-21 + 2.0**-24)}
    intervals = [desc["contract_interval"] for desc in CASES if "contract_interval" in desc]
    assert intervals == [{"rtol": 2.0**-21, "atol": 2.0**-24}]


def test_reference_nonfinite_lanes():
    out = gelu_exact_reference(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    assert np.isnan(out[0]) and out[1] == np.inf and np.isnan(out[2])


@pytest.mark.parametrize("entry,dtype", [("gelu_f32", np.float32), ("gelu_f16", np.float16)])
def test_c_reference_is_the_exact_gelu_rounded_once(entry, dtype):
    from helia_core_tester.generation.reference.bindings import get_bindings

    values = np.concatenate([FINITE, np.array([0.0, -0.0, np.nan, np.inf, -np.inf], dtype=np.float32)]).astype(dtype)
    got = get_bindings().run(entry, {"unused": 0}, {"input": values}, {"output": values.shape})["output"]
    with np.errstate(over="ignore", invalid="ignore"):
        want = gelu_exact_reference(values).astype(dtype)
    assert got.dtype == dtype
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    finite = ~np.isnan(want)
    np.testing.assert_array_equal(got[finite], want[finite])
    # An input zero keeps its sign.
    zeros = values == 0
    np.testing.assert_array_equal(np.signbit(got[zeros]), np.signbit(values[zeros]))


def test_bound_rejects_the_tanh_approximation():
    x = FINITE.astype(np.float64)
    tanh_form = 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    curve = next(desc for desc in CASES if desc["name"] == "gelu_float_curve_f32")
    # The two forms agree to within the bound only near zero.
    sampled = (np.abs(x) >= 0.25) & (x >= curve["input_min"]) & (x <= curve["input_max"])
    assert not _within(tanh_form[sampled], gelu_exact_reference(FINITE)[sampled], curve).any()


@pytest.mark.parametrize("desc", CASES, ids=lambda desc: desc["name"])
def test_generated_case_calls_the_kernel(tmp_path, desc):
    op = OpGelu(desc, seed=500, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    assert op.reference is not None and op.reference.entry == "gelu_f32"
    assert not list(tmp_path.rglob("*.tflite"))
    source = (tmp_path / f"{desc['name']}_gelu.c").read_text()
    assert "arm_nn_gelu_f32(" in source
    assert "HELIA_VALIDATE_RETURN_FAILURES" in source
