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


def test_every_descriptor_input_mode_value_is_in_the_schema_enum() -> None:
    allowed = set(_schema()["properties"]["input_mode"]["enum"])
    seen: list[tuple[str, str]] = []

    for path in (_repo_root() / "assets" / "descriptors").rglob("*.yaml"):
        for doc in yaml.safe_load_all(path.read_text()):
            if not isinstance(doc, dict) or "input_mode" not in doc:
                continue
            seen.append((doc["name"], doc["input_mode"]))
            assert doc["input_mode"] in allowed, (path.name, doc["name"], doc["input_mode"])

    # Guards against the mode being silently dropped from every descriptor, which would
    # leave the machinery below green while testing nothing on real hardware.
    assert seen, "no descriptor requests a non-uniform input_mode"
    assert any(mode == "nonfinite_sweep" for _, mode in seen)


# --- token placement --------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_nonfinite_sweep_writes_the_expected_tokens_at_the_expected_positions(dtype) -> None:
    swept = _op(input_mode="nonfinite_sweep")._sample_uniform((1, 16), dtype=dtype)
    flat = swept.reshape(-1)

    assert swept.dtype == dtype
    assert OperationBase.nonfinite_sweep_positions() == (0, 1, 2, 3)
    assert np.isnan(flat[0]) and np.isnan(flat[1])
    # -NaN must keep its sign bit through the astype, or the "-NAN" literal below is a lie.
    assert np.signbit(flat[0]) == False  # noqa: E712 - numpy bool_, not a Python bool
    assert np.signbit(flat[1]) == True  # noqa: E712
    assert np.isposinf(flat[2])
    assert np.isneginf(flat[3])
    assert np.all(np.isfinite(flat[4:]))


def test_nonfinite_sweep_leaves_the_finite_neighbours_exactly_as_drawn() -> None:
    baseline = _op()._sample_uniform((1, 16))
    swept = _op(input_mode="nonfinite_sweep")._sample_uniform((1, 16))

    # Same seed, same draw: only the leading four elements may differ.
    np.testing.assert_array_equal(swept.reshape(-1)[4:], baseline.reshape(-1)[4:])


def test_dual_input_sweep_touches_only_the_left_operand() -> None:
    left, right = _op(input_mode="nonfinite_sweep")._sample_dual_uniform_inputs((1, 8), (1, 8))

    assert np.isnan(left.reshape(-1)[0])
    assert np.all(np.isfinite(right))


def test_uniform_mode_and_absent_mode_are_the_same_draw() -> None:
    np.testing.assert_array_equal(
        _op()._sample_uniform((1, 8)),
        _op(input_mode="uniform")._sample_uniform((1, 8)),
    )


# --- rejections -------------------------------------------------------------


def test_sweep_rejects_a_tensor_too_small_to_hold_every_token() -> None:
    with pytest.raises(ValueError, match="at least 4 elements"):
        _op(input_mode="nonfinite_sweep")._sample_uniform((3,))


def test_sweep_rejects_integer_tensors() -> None:
    with pytest.raises(ValueError, match="requires a float tensor"):
        OperationBase._apply_nonfinite_sweep(np.zeros(8, dtype=np.int8))


def test_unknown_input_mode_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="Unsupported input_mode"):
        _op(input_mode="wobbly")._sample_uniform((1, 8))


# --- C serialization --------------------------------------------------------


def test_float_literals_use_c99_nonfinite_macros() -> None:
    fmt = TemplateContextBuilder.format_float_literal
    assert fmt(float("nan")) == "NAN"
    assert fmt(float("-nan")) == "-NAN"
    assert fmt(float("inf")) == "INFINITY"
    assert fmt(float("-inf")) == "-INFINITY"


def test_nonfinite_array_literals_render_for_both_float_widths() -> None:
    values = [float("nan"), float("-nan"), float("inf"), float("-inf"), 0.5]

    f32 = TemplateContextBuilder.format_array_as_c_literal(np.array(values, dtype=np.float32))
    assert f32.split(", ")[:4] == ["    NAN", "-NAN", "INFINITY", "-INFINITY"]

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
        np.array([float("nan"), float("-nan"), float("inf"), float("-inf")], dtype=np.float32)
    )
    source = tmp_path / "nonfinite.c"
    source.write_text(
        "#include <math.h>\n"
        "#include <stdio.h>\n"
        "#include <string.h>\n"
        "static const float values[] = {\n" + literals + "\n};\n"
        "int main(void) {\n"
        "    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); i++) {\n"
        "        unsigned int bits;\n"
        "        memcpy(&bits, &values[i], sizeof(bits));\n"
        "        printf(\"%08x\\n\", bits);\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    binary = tmp_path / "nonfinite"
    subprocess.run(
        ["cc", "-Ofast", "-std=c99", str(source), "-o", str(binary), "-lm"],
        check=True,
        capture_output=True,
    )
    bits = [int(line, 16) for line in subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    ).stdout.split()]

    exponent_all_ones = 0x7F800000
    assert all(value & exponent_all_ones == exponent_all_ones for value in bits)
    assert bits[0] & 0x007FFFFF != 0 and not bits[0] >> 31  # +NaN
    assert bits[1] & 0x007FFFFF != 0 and bits[1] >> 31  # -NaN
    assert bits[2] == 0x7F800000  # +Inf
    assert bits[3] == 0xFF800000  # -Inf


# --- reference expectations -------------------------------------------------


def test_activation_reference_encodes_the_propagation_and_clamping_contract() -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import (
        _activation_reference,
    )

    sweep = np.array([np.nan, -np.nan, np.inf, -np.inf], dtype=np.float32)
    old = np.seterr(all="ignore")
    try:
        tanh = _activation_reference(sweep, "ARM_NN_FLT_ACT_TANH", 0.0, "FP32")
        sigmoid = _activation_reference(sweep, "ARM_NN_FLT_ACT_SIGMOID", 0.0, "FP32")
        relu = _activation_reference(sweep, "ARM_NN_FLT_ACT_RELU", 0.0, "FP32")
        relu6 = _activation_reference(sweep, "ARM_NN_FLT_ACT_RELU6", 0.0, "FP32")
        tanh_f16 = _activation_reference(
            sweep.astype(np.float16), "ARM_NN_FLT_ACT_TANH", 0.0, "FP16", use_mve_tanh=True
        )
    finally:
        np.seterr(**old)

    # NaN propagates on every activation; +-Inf is clamped to the saturating bound.
    for values in (tanh, sigmoid, relu, relu6, tanh_f16):
        assert np.isnan(values[0]) and np.isnan(values[1])

    np.testing.assert_array_equal(tanh[2:], np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(sigmoid[2:], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(relu6[2:], np.array([6.0, 0.0], dtype=np.float32))
    assert np.isposinf(relu[2]) and relu[3] == 0.0
    np.testing.assert_array_equal(tanh_f16[2:], np.array([1.0, -1.0], dtype=np.float16))


def test_f16_mve_tanh_reference_does_not_cast_a_nan_to_an_index() -> None:
    """A NaN index cast is the numpy analogue of the ns-cmsis-nn#314 out-of-bounds read."""
    from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import (
        _tanh_reference_f16_mve,
    )

    old = np.seterr(all="raise")
    try:
        result = _tanh_reference_f16_mve(np.array([np.nan, 0.5], dtype=np.float16))
    finally:
        np.seterr(**old)

    assert np.isnan(result[0])
