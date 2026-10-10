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
    assert {_bound(desc) for desc in CASES} == {(2.0**-24, 2.0**-21 + 2.0**-24)}


def test_reference_nonfinite_lanes():
    out = gelu_exact_reference(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    assert np.isnan(out[0]) and out[1] == np.inf and np.isnan(out[2])


def test_reference_matches_tflite_exact_gelu(tmp_path):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    values = np.concatenate([FINITE, np.array([np.nan, np.inf, -np.inf], dtype=np.float32)])
    desc = {**CASES[0], "name": "gelu_reference_check", "input_shape": [1, 1, 1, values.size]}
    op = OpGelu(desc, seed=500, target_cpu="cortex-m55")
    path = tmp_path / "gelu.tflite"
    op.convert_to_tflite(op.build_keras_model(), str(path), 500)
    # The reference kernels, not XNNPACK, which flushes the far negative tail to -0.
    interpreter = Interpreter(model_path=str(path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    assert [o["op_name"] for o in interpreter._get_ops_details()] == ["GELU"]
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], values.reshape(1, 1, 1, -1))
    interpreter.invoke()
    actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).ravel()
    expected = gelu_exact_reference(values)

    finite = np.isfinite(expected)
    assert _within(actual[finite], expected[finite], CASES[0]).all()
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(actual[np.isinf(expected)], expected[np.isinf(expected)])


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
    op.convert_to_tflite(op.build_keras_model(), str(tmp_path / f"{desc['name']}.tflite"), 500)
    op.generate_c_files(tmp_path)
    source = (tmp_path / f"{desc['name']}_gelu.c").read_text()
    assert "arm_nn_gelu_f32(" in source
    assert "HELIA_VALIDATE_RETURN_FAILURES" in source
