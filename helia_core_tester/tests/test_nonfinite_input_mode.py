"""Cover the nonfinite_sweep input mode: token placement, expectations, C literals.

The C-literal half matters more than it looks: the generated arrays are compiled at
-Ofast, which implies -ffinite-math-only, and the whole point of these cases is that a
NaN actually reaches the kernel (ns-cmsis-nn#314).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _StubOp(OperationBase):
    """Minimal concrete OperationBase so the sampling helpers can be exercised alone."""

    def build_keras_model(self):  # pragma: no cover - never called by these tests
        raise NotImplementedError

    def generate_c_files(self, output_dir):  # pragma: no cover - never called
        raise NotImplementedError


def _op(**desc) -> _StubOp:
    return _StubOp(dict(desc), seed=500)


# --- schema registration ----------------------------------------------------


def _schema() -> dict:
    path = _repo_root() / "helia_core_tester" / "generation" / "descriptors" / "schema.json"
    return json.loads(path.read_text())


def test_schema_declares_input_mode_enum() -> None:
    prop = _schema()["properties"]["input_mode"]
    assert prop["type"] == "string"
    assert set(prop["enum"]) == {"uniform", "nonfinite_sweep"}


def test_schema_declares_the_nonfinite_token_enum() -> None:
    prop = _schema()["properties"]["nonfinite_tokens"]
    assert prop["type"] == "array"
    assert set(prop["items"]["enum"]) == {"nan", "inf", "-inf"}
    assert prop["uniqueItems"] is True


def test_every_descriptor_input_mode_value_is_in_the_schema_enum() -> None:
    allowed = set(_schema()["properties"]["input_mode"]["enum"])
    seen: list[tuple[str, str]] = []

    token_enum = set(_schema()["properties"]["nonfinite_tokens"]["items"]["enum"])

    for path in (_repo_root() / "assets" / "descriptors").rglob("*.yaml"):
        for doc in yaml.safe_load_all(path.read_text()):
            if not isinstance(doc, dict) or "input_mode" not in doc:
                continue
            seen.append((doc["name"], doc["input_mode"]))
            assert doc["input_mode"] in allowed, (path.name, doc["name"], doc["input_mode"])
            assert doc.get("suite") == "float", (path.name, doc["name"])
            assert set(doc.get("nonfinite_tokens", [])) <= token_enum, doc["name"]

    # Guards against the mode being silently dropped from every descriptor, which would
    # leave the machinery below green while testing nothing on real hardware.
    assert seen, "no descriptor requests a non-uniform input_mode"
    assert any(mode == "nonfinite_sweep" for _, mode in seen)


# --- token placement --------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_nonfinite_sweep_writes_the_expected_tokens_at_the_expected_positions(dtype) -> None:
    op = _op(input_mode="nonfinite_sweep", suite="float")
    swept = op._sample_uniform((1, 16), dtype=dtype)
    flat = swept.reshape(-1)

    assert swept.dtype == dtype
    assert op.nonfinite_tokens() == ("nan", "inf", "-inf")
    assert op.nonfinite_sweep_positions() == (0, 1, 2)
    assert np.isnan(flat[0])
    assert np.isposinf(flat[1])
    assert np.isneginf(flat[2])
    assert np.all(np.isfinite(flat[3:]))


def test_a_descriptor_can_restrict_the_token_set() -> None:
    op = _op(input_mode="nonfinite_sweep", suite="float", nonfinite_tokens=["inf", "-inf"])
    flat = op._sample_uniform((1, 16)).reshape(-1)

    assert op.nonfinite_sweep_positions() == (0, 1)
    assert np.isposinf(flat[0]) and np.isneginf(flat[1])
    assert not np.isnan(flat).any()


def test_unknown_and_duplicate_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported nonfinite_tokens"):
        _op(
            input_mode="nonfinite_sweep", suite="float", nonfinite_tokens=["-nan"]
        )._sample_uniform((1, 8))
    with pytest.raises(ValueError, match="must be unique"):
        _op(
            input_mode="nonfinite_sweep", suite="float", nonfinite_tokens=["nan", "nan"]
        )._sample_uniform((1, 8))


def test_nonfinite_sweep_leaves_the_finite_neighbours_exactly_as_drawn() -> None:
    baseline = _op()._sample_uniform((1, 16))
    swept = _op(input_mode="nonfinite_sweep", suite="float")._sample_uniform((1, 16))

    # Same seed, same draw: only the leading token elements may differ.
    np.testing.assert_array_equal(swept.reshape(-1)[3:], baseline.reshape(-1)[3:])


def test_dual_input_sweep_touches_only_the_left_operand() -> None:
    left, right = _op(
        input_mode="nonfinite_sweep", suite="float"
    )._sample_dual_uniform_inputs((1, 8), (1, 8))

    assert np.isnan(left.reshape(-1)[0])
    assert np.all(np.isfinite(right))


def test_uniform_mode_and_absent_mode_are_the_same_draw() -> None:
    np.testing.assert_array_equal(
        _op()._sample_uniform((1, 8)),
        _op(input_mode="uniform")._sample_uniform((1, 8)),
    )


# --- rejections -------------------------------------------------------------


def test_sweep_rejects_a_tensor_too_small_to_hold_every_token() -> None:
    with pytest.raises(ValueError, match="at least 3 elements"):
        _op(input_mode="nonfinite_sweep", suite="float")._sample_uniform((2,))


def test_sweep_rejects_integer_tensors() -> None:
    with pytest.raises(ValueError, match="requires a float tensor"):
        _op(input_mode="nonfinite_sweep", suite="float")._apply_nonfinite_sweep(
            np.zeros(8, dtype=np.int8)
        )


def test_unknown_input_mode_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="Unsupported input_mode"):
        _op(input_mode="wobbly", suite="float")._sample_uniform((1, 8))


def test_nonuniform_input_mode_is_rejected_outside_the_float_suite() -> None:
    """The sweep writes non-finite floats, which an int8 descriptor cannot carry."""
    with pytest.raises(ValueError, match="float-suite only"):
        _op(name="int_case", input_mode="nonfinite_sweep", suite="default")._sample_uniform((1, 8))


def test_a_requested_sweep_that_no_helper_consumed_is_a_generation_error() -> None:
    """Ops that sample outside the shared helpers must fail loudly, not silently finite.

    LSTM, GRU, softmax, batch_matmul and svdf all draw their own inputs, so without this
    guard a nonfinite_sweep descriptor on one of them would generate an ordinary finite
    case and pass on hardware while asserting nothing about non-finite behaviour.
    """
    unconsumed = _op(name="never_swept", input_mode="nonfinite_sweep", suite="float")
    with pytest.raises(ValueError, match="never applied it"):
        unconsumed.assert_input_mode_consumed()

    consumed = _op(name="swept", input_mode="nonfinite_sweep", suite="float")
    consumed._sample_uniform((1, 8))
    consumed.assert_input_mode_consumed()

    _op(name="plain").assert_input_mode_consumed()


# --- C serialization --------------------------------------------------------


def test_float_literals_use_c99_nonfinite_macros() -> None:
    fmt = TemplateContextBuilder.format_float_literal
    assert fmt(float("nan")) == "NAN"
    assert fmt(float("-nan")) == "-NAN"
    assert fmt(float("inf")) == "INFINITY"
    assert fmt(float("-inf")) == "-INFINITY"


def test_nonfinite_array_literals_render_for_both_float_widths() -> None:
    values = [float("nan"), float("inf"), float("-inf"), 0.5]

    f32 = TemplateContextBuilder.format_array_as_c_literal(np.array(values, dtype=np.float32))
    assert f32.split(", ")[:3] == ["    NAN", "INFINITY", "-INFINITY"]

    f16 = TemplateContextBuilder.format_array_as_c_literal(np.array(values, dtype=np.float16))
    assert "(float16_t)NAN" in f16
    assert "(float16_t)-INFINITY" in f16


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_generated_nonfinite_literals_survive_ofast(tmp_path: Path) -> None:
    """The tokens must still be non-finite after -Ofast, which asserts they are not.

    -ffinite-math-only licenses the optimizer to assume no NaN/Inf operands, so this
    checks the emitted bit patterns rather than isnan()/isinf(), which that same flag is
    entitled to fold to false.
    """
    literals = TemplateContextBuilder.format_array_as_c_literal(
        np.array([float("nan"), float("inf"), float("-inf")], dtype=np.float32)
    )
    source = tmp_path / "nonfinite.c"
    source.write_text(
        "#include <inttypes.h>\n"
        "#include <math.h>\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <string.h>\n"
        "_Static_assert(sizeof(float) == sizeof(uint32_t), \"binary32 float required\");\n"
        "static const float values[] = {\n" + literals + "\n};\n"
        "int main(void) {\n"
        "    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {\n"
        "        uint32_t bits;\n"
        "        memcpy(&bits, &values[i], sizeof(bits));\n"
        "        printf(\"%08\" PRIx32 \"\\n\", bits);\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    binary = tmp_path / "nonfinite"
    subprocess.run(
        ["cc", "-Ofast", "-std=c11", str(source), "-o", str(binary), "-lm"],
        check=True,
        capture_output=True,
    )
    bits = [int(line, 16) for line in subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    ).stdout.split()]

    exponent_all_ones = 0x7F800000
    assert all(value & exponent_all_ones == exponent_all_ones for value in bits)
    assert bits[0] & 0x007FFFFF != 0  # NaN
    assert bits[1] == 0x7F800000  # +Inf
    assert bits[2] == 0xFF800000  # -Inf


# --- reference expectations -------------------------------------------------


def test_activation_reference_encodes_the_propagation_and_clamping_contract() -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import (
        _activation_reference,
    )

    sweep = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    old = np.seterr(all="ignore")
    try:
        relu = _activation_reference(sweep, "ARM_NN_FLT_ACT_RELU", 0.0, "FP32")
        relu6 = _activation_reference(sweep, "ARM_NN_FLT_ACT_RELU6", 0.0, "FP32")
        leaky = _activation_reference(sweep, "ARM_NN_FLT_ACT_LEAKY_RELU", 0.125, "FP32")
        # tanh and sigmoid sweep +-Inf only; NaN is not a lane they assert.
        finite_sweep = sweep[1:]
        tanh = _activation_reference(finite_sweep, "ARM_NN_FLT_ACT_TANH", 0.0, "FP32")
        sigmoid = _activation_reference(finite_sweep, "ARM_NN_FLT_ACT_SIGMOID", 0.0, "FP32")
        tanh_f16 = _activation_reference(
            finite_sweep.astype(np.float16),
            "ARM_NN_FLT_ACT_TANH",
            0.0,
            "FP16",
            use_mve_tanh=True,
        )
    finally:
        np.seterr(**old)

    # Propagation, per arm_nnfunctions_flt.h:440-444: RELU/RELU6/LEAKY_RELU only.
    for values in (relu, relu6, leaky):
        assert np.isnan(values[0])

    np.testing.assert_array_equal(relu6[1:], np.array([6.0, 0.0], dtype=np.float32))
    assert np.isposinf(relu[1]) and relu[2] == 0.0
    assert np.isposinf(leaky[1]) and np.isneginf(leaky[2])

    # Clamping: the domain reduction bounds a non-finite input to a finite result.
    np.testing.assert_array_equal(tanh, np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(sigmoid, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(tanh_f16, np.array([1.0, -1.0], dtype=np.float16))


def test_tanh_and_sigmoid_descriptors_do_not_assert_a_nan_lane() -> None:
    """ns-cmsis-nn disclaims NaN on both: sigmoid calls it unsupported
    (arm_nnsupportfunctions_flt.h:184-186) and the MVE tanh legs destroy it by design
    (arm_nn_activation_flt.h:89-93 and :537-554). Neither IEEE propagation nor the
    observed divergence may be locked in as a golden.
    """
    path = (
        _repo_root() / "assets" / "descriptors" / "ActivationFunctions" / "nn_activation_float.yaml"
    )
    checked = 0
    for doc in yaml.safe_load_all(path.read_text()):
        if not isinstance(doc, dict) or doc.get("input_mode") != "nonfinite_sweep":
            continue
        activation = doc["activation_type"]
        if activation in ("ARM_NN_FLT_ACT_TANH", "ARM_NN_FLT_ACT_SIGMOID"):
            assert doc["nonfinite_tokens"] == ["inf", "-inf"], doc["name"]
            checked += 1
        else:
            assert "nan" in doc.get("nonfinite_tokens", ["nan"]), doc["name"]
        # The mux's hard swish is outside the NaN/Inf contract entirely
        # (arm_nnfunctions_flt.h:445-451); hard_swish_float.yaml covers that kernel.
        assert activation != "ARM_NN_FLT_ACT_HARDSWISH", doc["name"]

    assert checked == 4
