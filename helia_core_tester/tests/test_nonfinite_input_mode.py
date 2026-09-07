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
from helia_core_tester.tests.test_float_nonfinite_compare import FLAG_SETS


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


def test_nonfinite_positions_place_the_tokens_off_the_leading_run() -> None:
    """The AmbiqAI/ns-cmsis-nn#429 placement: one token per reduction group, which the
    default leading run cannot express."""
    op = _op(
        input_mode="nonfinite_sweep",
        suite="float",
        nonfinite_tokens=["inf", "nan"],
        nonfinite_positions=[1, 3],
    )
    flat = op._sample_uniform((1, 12)).reshape(-1)

    assert op.nonfinite_sweep_positions() == (1, 3)
    assert np.isposinf(flat[1]) and np.isnan(flat[3])
    assert np.isfinite(flat[[0, 2, 4, 5, 6, 7, 8, 9, 10, 11]]).all()


def test_nonfinite_positions_must_pair_with_the_tokens() -> None:
    with pytest.raises(ValueError, match="nonfinite_positions for"):
        _op(
            input_mode="nonfinite_sweep",
            suite="float",
            nonfinite_tokens=["inf", "nan"],
            nonfinite_positions=[1],
        )._sample_uniform((1, 8))
    with pytest.raises(ValueError, match="nonfinite_positions must be unique"):
        _op(
            input_mode="nonfinite_sweep",
            suite="float",
            nonfinite_tokens=["inf", "nan"],
            nonfinite_positions=[2, 2],
        )._sample_uniform((1, 8))
    with pytest.raises(ValueError, match="without input_mode 'nonfinite_sweep'"):
        _op(suite="float", nonfinite_positions=[2]).nonfinite_sweep_positions()


def test_a_tensor_too_short_for_the_furthest_position_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs at least 73 elements"):
        _op(
            input_mode="nonfinite_sweep",
            suite="float",
            nonfinite_tokens=["nan"],
            nonfinite_positions=[72],
        )._sample_uniform((1, 8))


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
    unconsumed = _op(
        name="never_swept",
        input_mode="nonfinite_sweep",
        nonfinite_policy="strict",
        suite="float",
    )
    with pytest.raises(ValueError, match="never applied it"):
        unconsumed.assert_input_mode_consumed()

    consumed = _op(
        name="swept",
        input_mode="nonfinite_sweep",
        nonfinite_policy="strict",
        suite="float",
    )
    consumed._sample_uniform((1, 8))
    consumed.assert_input_mode_consumed()

    _op(name="plain").assert_input_mode_consumed()


# --- comparison policy ------------------------------------------------------


def test_schema_declares_the_nonfinite_policy_enum() -> None:
    prop = _schema()["properties"]["nonfinite_policy"]
    assert prop["type"] == "string"
    assert set(prop["enum"]) == {"strict", "mask"}


def test_every_descriptor_nonfinite_policy_value_is_in_the_schema_enum() -> None:
    allowed = set(_schema()["properties"]["nonfinite_policy"]["enum"])
    for path in sorted((_repo_root() / "assets" / "descriptors").rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if not isinstance(doc, dict) or "nonfinite_policy" not in doc:
                continue
            assert doc["nonfinite_policy"] in allowed, (path.name, doc.get("name"))
            assert doc.get("input_mode") == "nonfinite_sweep", (path.name, doc.get("name"))


def test_every_sweep_descriptor_declares_a_policy() -> None:
    """A sweep case with no policy would be answered by a default, and which policy
    applies is a per-kernel contract question no default can answer."""
    seen = 0
    for path in sorted((_repo_root() / "assets" / "descriptors").rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if not isinstance(doc, dict) or doc.get("input_mode") != "nonfinite_sweep":
                continue
            seen += 1
            assert "nonfinite_policy" in doc, (path.name, doc.get("name"))
    assert seen


def test_schema_requires_a_policy_alongside_a_sweep() -> None:
    gates = [
        block
        for block in _schema()["allOf"]
        if block.get("then", {}).get("required") == ["nonfinite_policy"]
    ]
    assert len(gates) == 1
    assert gates[0]["if"] == {
        "properties": {"input_mode": {"const": "nonfinite_sweep"}},
        "required": ["input_mode"],
    }


def test_a_sweep_without_a_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="without nonfinite_policy"):
        _op(name="plain", input_mode="nonfinite_sweep", suite="float").nonfinite_policy()


def test_strict_policy_emits_no_mask() -> None:
    op = _op(
        name="plain",
        input_mode="nonfinite_sweep",
        nonfinite_policy="strict",
        suite="float",
    )
    assert op.nonfinite_policy() == "strict"
    reference = np.array([float("nan"), 1.0, 2.0], dtype=np.float32)
    emitted, context = op.apply_nonfinite_policy(
        reference, reference=lambda operands: reference, inputs=[reference]
    )
    assert context == {}
    assert emitted is reference


def _masking_op():
    return _op(
        name="masked",
        input_mode="nonfinite_sweep",
        nonfinite_policy="mask",
        suite="float",
    )


def test_mask_covers_a_finite_lane_the_token_reaches() -> None:
    """The max_pool -Inf shape: every window output is finite, so a mask built from
    the reference alone would be empty and the case would pin the kernel's own fold
    order for the window that saw the token."""
    op = _masking_op()
    swept = np.array([float("-inf"), 1.0, 2.0, 3.0], dtype=np.float32)

    def windowed_max(operands):
        values = operands[0]
        return np.array(
            [np.max(values[index : index + 2]) for index in range(3)], dtype=np.float32
        )

    golden = windowed_max([swept])
    assert np.isfinite(golden).all()

    emitted, context = op.apply_nonfinite_policy(
        golden, reference=windowed_max, inputs=[swept]
    )

    assert context["nonfinite_masked_lanes"] == 1
    assert _mask_bytes(context) == [1, 0, 0]
    assert emitted.tolist() == [0.0, 2.0, 3.0]


def test_mask_covers_the_nonfinite_reference_lanes_and_zeroes_them() -> None:
    op = _masking_op()
    swept = np.array([float("nan"), float("inf"), float("-inf"), 1.5], dtype=np.float32)

    def passthrough(operands):
        return operands[0].astype(np.float32)

    emitted, context = op.apply_nonfinite_policy(
        passthrough([swept]), reference=passthrough, inputs=[swept]
    )

    assert context["nonfinite_masked_lanes"] == 3
    assert _mask_bytes(context) == [1, 1, 1, 0]
    # The golden must stay finite: the masked entries are written as zero.
    assert emitted.tolist() == [0.0, 0.0, 0.0, 1.5]
    assert emitted.dtype == np.float32


def test_reachability_survives_a_symmetric_reference() -> None:
    """Two probes of opposite sign coincide under abs, which would read as
    unreachable; the third probe is what breaks the tie."""
    op = _masking_op()
    swept = np.array([float("nan"), 1.0, 2.0, 3.0], dtype=np.float32)

    reachable = op._nonfinite_reachable_lanes(
        lambda operands: np.abs(operands[0]).astype(np.float32), [swept], (4,)
    )
    assert reachable.tolist() == [True, False, False, False]


def test_mask_policy_needs_an_input_carrying_a_token() -> None:
    op = _masking_op()
    finite = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="no input handed to apply_nonfinite_policy"):
        op.apply_nonfinite_policy(
            finite, reference=lambda operands: operands[0], inputs=[finite]
        )


def test_a_reference_that_swallows_every_probe_fails_generation() -> None:
    """A two-sided output clamp can saturate all three probes to the same bound, so
    every lane reads as unreachable and the measurement is not evidence that the token
    is confined."""
    op = _masking_op()
    swept = np.array([float("inf"), 12.0, 11.0, 13.0], dtype=np.float32)

    def clamped_window_max(operands):
        values = operands[0].astype(np.float32)
        return np.clip(
            np.array([np.max(values[index : index + 2]) for index in range(3)]), -5.0, 5.0
        ).astype(np.float32)

    with pytest.raises(ValueError, match="no output lane moved between the finite probes"):
        op.apply_nonfinite_policy(
            clamped_window_max([swept]), reference=clamped_window_max, inputs=[swept]
        )


def test_a_fully_masked_case_fails_generation() -> None:
    """Masking every lane leaves nothing but SUCCESS asserted, which is a case that
    passes whatever the kernel writes."""
    op = _masking_op()
    swept = np.array([float("nan"), 1.0], dtype=np.float32)

    def total(operands):
        return np.array([np.sum(operands[0])] * 2, dtype=np.float32)

    with pytest.raises(ValueError, match="masks all 2 output lanes"):
        op.apply_nonfinite_policy(total([swept]), reference=total, inputs=[swept])


def _mask_bytes(context) -> list[int]:
    return [int(v) for v in context["nonfinite_mask_array_str"].replace(",", " ").split()]


def test_mask_policy_without_a_sweep_is_rejected() -> None:
    with pytest.raises(ValueError, match="without input_mode"):
        _op(name="no_sweep", nonfinite_policy="mask", suite="float").nonfinite_policy()


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported nonfinite_policy"):
        _op(
            name="bad",
            input_mode="nonfinite_sweep",
            nonfinite_policy="ignore",
            suite="float",
        ).nonfinite_policy()


def test_a_mask_policy_the_op_never_applied_is_a_generation_error() -> None:
    """Same failure shape as an unconsumed input_mode, one step later.

    An op that sweeps its input but never asks for the mask would fall back to a
    strict comparison and pin a value the kernel never promised.
    """
    op = _op(
        name="masked",
        input_mode="nonfinite_sweep",
        nonfinite_policy="mask",
        suite="float",
    )
    op._sample_uniform((1, 8))
    with pytest.raises(ValueError, match="never called apply_nonfinite_policy"):
        op.assert_input_mode_consumed()

    swept = np.array([float("nan"), 1.0], dtype=np.float32)
    op.apply_nonfinite_policy(
        swept, reference=lambda operands: operands[0].astype(np.float32), inputs=[swept]
    )
    op.assert_input_mode_consumed()


def test_strict_spreading_descriptors_sweep_a_single_token() -> None:
    """A strict reduction group holding both infinities would make the golden say
    (+Inf) + (-Inf) instead of anything about the kernel. Under mask the group is
    don't-care, so several tokens are allowed -- either in one group (the two_token
    pairs) or one per group (the issue429 pairs)."""
    spreading_prefixes = ("mean_", "reduce_sum_", "softmax_", "avg_pool_", "max_pool_")
    seen = 0
    multi_token_masked = 0
    for path in sorted((_repo_root() / "assets" / "descriptors").rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if not isinstance(doc, dict) or doc.get("input_mode") != "nonfinite_sweep":
                continue
            name = doc.get("name", "")
            if not name.startswith(spreading_prefixes):
                continue
            seen += 1
            tokens = doc.get("nonfinite_tokens", [])
            if doc["nonfinite_policy"] == "mask" and len(tokens) > 1:
                multi_token_masked += 1
                continue
            assert len(tokens) == 1, name
    assert seen
    # Four two_token pairs plus eight issue429 cases, f32 and f16 for mean and
    # reduce_sum on both the flatten and the generic reduced axis.
    assert multi_token_masked == 12, "the multi-token mean/reduce_sum cases went missing"


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
@pytest.mark.parametrize("flags", FLAG_SETS, ids=lambda flags: " ".join(flags))
def test_generated_nonfinite_literals_survive_fast_math(tmp_path: Path, flags: list[str]) -> None:
    """The tokens must still be non-finite under the flag sets the float legs build with.

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
        ["cc", *flags, "-std=c11", str(source), "-o", str(binary), "-lm"],
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


_SOFT_FLOAT_TANH_NAN_CASE = "nn_activation_float_tanh_nan_softfloat_f32"


def test_tanh_and_sigmoid_descriptors_do_not_assert_a_nan_lane() -> None:
    """ns-cmsis-nn disclaims NaN on both: sigmoid calls it unsupported
    (arm_nnsupportfunctions_flt.h:184-186) and the MVE tanh legs destroy it by design
    (arm_nn_activation_flt.h:89-93 and :537-554). Neither IEEE propagation nor the
    observed divergence may be locked in as a golden. The single exception is the
    soft-float tanh case, which is contracted at arm_nn_activation_flt.h:103-118 and
    is gated on the soft_float capability so no hard-float leg ever runs it.
    """
    path = (
        _repo_root() / "assets" / "descriptors" / "ActivationFunctions" / "nn_activation_float.yaml"
    )
    checked = 0
    soft_float_cases = 0
    for doc in yaml.safe_load_all(path.read_text()):
        if not isinstance(doc, dict) or doc.get("input_mode") != "nonfinite_sweep":
            continue
        activation = doc["activation_type"]
        if doc["name"] == _SOFT_FLOAT_TANH_NAN_CASE:
            assert activation == "ARM_NN_FLT_ACT_TANH"
            assert doc["nonfinite_tokens"] == ["nan"]
            assert doc["required_capabilities"] == ["soft_float"]
            soft_float_cases += 1
        elif activation in ("ARM_NN_FLT_ACT_TANH", "ARM_NN_FLT_ACT_SIGMOID"):
            assert doc["nonfinite_tokens"] == ["inf", "-inf"], doc["name"]
            assert "required_capabilities" not in doc, doc["name"]
            checked += 1
        else:
            assert "nan" in doc.get("nonfinite_tokens", ["nan"]), doc["name"]
        # The mux's hard swish is outside the NaN/Inf contract entirely
        # (arm_nnfunctions_flt.h:445-451); hard_swish_float.yaml covers that kernel.
        assert activation != "ARM_NN_FLT_ACT_HARDSWISH", doc["name"]

    assert checked == 4
    assert soft_float_cases == 1


def test_soft_float_tanh_nan_case_expects_a_nan_back() -> None:
    """The #314 guard is only a guard if the golden is a NaN: the reference must
    propagate, not clamp the lane to tanh(xmax) the way the MVE leg does.
    """
    from helia_core_tester.generation.ops.ActivationFunctions.nn_activation_float import (
        _activation_reference,
    )

    old = np.seterr(all="ignore")
    try:
        result = _activation_reference(
            np.array([np.nan, 0.5], dtype=np.float32), "ARM_NN_FLT_ACT_TANH", 0.0, "FP32"
        )
    finally:
        np.seterr(**old)

    assert np.isnan(result[0])
    assert not np.isnan(result[1])


# --- activation bound typing ------------------------------------------------


def test_quantized_clamp_rejects_a_non_integral_activation_bound() -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.clamp import (
        _integral_activation_bound,
    )

    assert _integral_activation_bound({"act_min": -64}, "act_min", -128) == -64
    assert _integral_activation_bound({"act_min": -64.0}, "act_min", -128) == -64
    assert _integral_activation_bound({}, "act_min", -128) == -128

    for bad in (-64.5, float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="integral quantized code"):
            _integral_activation_bound({"act_min": bad}, "act_min", -128)
