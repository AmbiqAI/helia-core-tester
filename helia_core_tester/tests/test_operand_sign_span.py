"""
Generation-time operand sign-span rule (issue #81 property 2).

Int elementwise operands have to span negative, near-zero and positive AFTER
the input offset is applied: a one-signed operand cannot discriminate the
sign-dependent kernel paths (the packed DSP loop of ns-cmsis-nn#343 dropped
the sign of value + input_offset; PReLU and min/max branch on it directly).

These tests pin the three outcomes the rule can have -- pass, steer, refuse --
plus the descriptor opt-out and the shape of its required reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared.base import OperationBase


class _Op(OperationBase):
    """Minimal concrete OperationBase: the rule needs only self.desc."""

    SIGN_SPAN_OPERANDS = ("input", "input_1", "input_2", "alpha")

    def build_keras_model(self):  # pragma: no cover - never called
        raise NotImplementedError


def _op(**desc) -> _Op:
    desc.setdefault("name", "case")
    return _Op(desc, seed=1)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_spanning_operand_passes_unchanged() -> None:
    data = np.array([-40, 0, 37, -5, 61, 2, -3, 9], dtype=np.int8)
    (result,) = _op()._enforce_int_operand_sign_span((("input_1", data, 0),), steerable=("input_1",))
    np.testing.assert_array_equal(result, data)


def test_one_signed_runtime_operand_is_steered_not_rejected() -> None:
    # Uniform [-1, 1] float data that lands entirely on one side after
    # quantization is an accident, not a design choice: steer it.
    data = np.full(12, 40, dtype=np.int8)
    (result,) = _op()._enforce_int_operand_sign_span((("input_1", data, 0),), steerable=("input_1",))
    assert not OperationBase._sign_span_gaps(result.reshape(-1), 0)
    # Only the missing regions are planted: the data was already positive, so
    # the negative and near-zero plants move two elements and no more.
    assert OperationBase._sign_span_gaps(data, 0) == ["negative", "near-zero"]
    assert int((result != data).sum()) == 2


def test_steering_overwrites_the_least_extreme_elements() -> None:
    # Planting into the head would discard a full-scale element and the
    # saturation coverage that comes with it. The elements closest to the zero
    # point go instead, and only as many as there are missing regions.
    data = np.array([-128, 127, 20, -9, 3, 100], dtype=np.int8)
    assert OperationBase._sign_span_gaps(data, 0) == ["near-zero"]
    (result,) = _op()._enforce_int_operand_sign_span((("input_1", data, 0),), steerable=("input_1",))
    assert result[0] == -128 and result[1] == 127
    assert {int(i) for i in np.flatnonzero(result != data)} == {4}


def test_near_zero_is_absolute_not_a_fraction_of_the_range() -> None:
    # A large-magnitude s16 operand spans sign but never comes near the
    # boundary the kernels split on; the old relative rule called it covered.
    data = np.array([-20000, -9000, 8000, 17000, -3000, 12000], dtype=np.int16)
    assert OperationBase._sign_span_gaps(data, 0) == ["near-zero"]
    (result,) = _op()._enforce_int_operand_sign_span((("input_1", data, 0),), steerable=("input_1",))
    assert (np.abs(result.astype(np.int64)) <= OperationBase.NEAR_ZERO_MAX_ABS).any()
    assert result[0] == -20000 and result[3] == 17000
    # The operand already spanned sign, so only the near-zero plant moves.
    assert int((result != data).sum()) == 1


def test_steering_falls_back_to_the_full_triple_when_a_targeted_plant_reopens_a_gap() -> None:
    # The near-zero plant lands on the only negative element, so planting the
    # gap alone would trade one missing region for another. The fallback plants
    # all three and the operand ends up spanning.
    data = np.array([-2, 40, 60], dtype=np.int8)
    assert OperationBase._sign_span_gaps(data, 0) == ["near-zero"]
    (result,) = _op()._enforce_int_operand_sign_span((("input_1", data, 0),), steerable=("input_1",))
    assert not OperationBase._sign_span_gaps(result.reshape(-1), 0)


def test_steering_is_deterministic() -> None:
    data = np.full(12, 40, dtype=np.int8)
    first = _op()._enforce_int_operand_sign_span((("input_1", data, 0),), steerable=("input_1",))[0]
    second = _op()._enforce_int_operand_sign_span((("input_1", data, 0),), steerable=("input_1",))[0]
    np.testing.assert_array_equal(first, second)


def test_unsteerable_one_signed_operand_fails_generation() -> None:
    # A model constant (steerable=()) cannot be fixed by the generator, so the
    # descriptor has to declare the reason instead.
    data = np.full(12, 26, dtype=np.int8)
    with pytest.raises(ValueError, match="operand_sign_span_exempt"):
        _op()._enforce_int_operand_sign_span((("alpha", data, 0),), steerable=())


def test_zero_point_that_pins_the_whole_domain_cannot_be_steered() -> None:
    # zero_point -128 maps the entire int8 domain to [0, 255] post-offset:
    # no planted value can make the operand negative. The refusal has to say
    # steering was tried and name the zero point, not blame a pinned operand.
    data = np.array([-128, -100, -60, -30], dtype=np.int8)
    with pytest.raises(ValueError, match="do not span negative") as excinfo:
        _op()._enforce_int_operand_sign_span((("input_1", data, -128),), steerable=("input_1",))
    assert "steering cannot reach them" in str(excinfo.value)
    assert "zero_point -128" in str(excinfo.value)


def test_refusal_of_an_unsteerable_operand_names_the_pinning_not_the_clipping() -> None:
    data = np.full(12, 26, dtype=np.int8)
    with pytest.raises(ValueError) as excinfo:
        _op()._enforce_int_operand_sign_span((("alpha", data, 0),), steerable=())
    assert "model-baked or descriptor-pinned" in str(excinfo.value)
    assert "steering cannot reach" not in str(excinfo.value)


def test_short_operands_are_out_of_scope() -> None:
    # A broadcast scalar cannot hold three regions at once; the operand it
    # broadcasts against still has to.
    data = np.array([40, 41], dtype=np.int8)
    (result,) = _op()._enforce_int_operand_sign_span((("input_2", data, 0),), steerable=())
    np.testing.assert_array_equal(result, data)


def test_declared_exemption_waives_only_the_named_operand() -> None:
    alpha = np.full(12, 26, dtype=np.int8)
    bad_input = np.full(12, 26, dtype=np.int8)
    op = _op(operand_sign_span_exempt={"alpha": "model constant, positive PReLU slope"})
    op._enforce_int_operand_sign_span((("alpha", alpha, 0),), steerable=())
    with pytest.raises(ValueError, match="operand_sign_span_exempt\\[input\\]"):
        op._enforce_int_operand_sign_span((("input", bad_input, 0),), steerable=())


@pytest.mark.parametrize("value", [{}, "just a string", {"alpha": ""}, {"alpha": None}])
def test_exemption_requires_an_operand_keyed_reason(value) -> None:
    op = _op(operand_sign_span_exempt=value)
    with pytest.raises(ValueError, match="operand_sign_span_exempt"):
        op._enforce_int_operand_sign_span((("alpha", np.full(12, 26, dtype=np.int8), 0),), steerable=())


def test_a_waiver_on_an_operand_the_operator_never_checks_is_rejected() -> None:
    # A key the rule never looks up waives nothing while reading as if it
    # covered a gap; the operator's declared labels are the check.
    op = _op(operand_sign_span_exempt={"inpt_1": "typo for input_1"})
    with pytest.raises(ValueError, match="waives nothing"):
        op._enforce_int_operand_sign_span(
            (("input_1", np.full(12, 26, dtype=np.int8), 0),), steerable=()
        )


def test_the_rule_rejects_an_operand_the_operator_did_not_declare() -> None:
    # SIGN_SPAN_OPERANDS is what the waiver keys are checked against, so a call
    # site passing a label outside it would create an unwaivable operand.
    class _Undeclared(_Op):
        SIGN_SPAN_OPERANDS = ("input_1",)

    op = _Undeclared({"name": "case"}, seed=1)
    with pytest.raises(ValueError, match="SIGN_SPAN_OPERANDS"):
        op._enforce_int_operand_sign_span(
            (("input_2", np.full(12, 26, dtype=np.int8), 0),), steerable=("input_2",)
        )


def test_shipped_exemptions_are_well_formed() -> None:
    # Scoping is enforced at generation time against the operator's declared
    # labels, so this only has to pin the reason contract.
    descriptors = load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    exempt = [d for d in descriptors if d.get("operand_sign_span_exempt") is not None]
    assert exempt, "the audit left at least the PReLU alpha waivers in place"
    for desc in exempt:
        mapping = desc["operand_sign_span_exempt"]
        assert isinstance(mapping, dict) and mapping, desc["name"]
        for operand, reason in mapping.items():
            assert isinstance(reason, str) and reason.strip(), (desc["name"], operand)
