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
    # Only the planted head moves; the rest of the case's data is untouched.
    np.testing.assert_array_equal(result[3:], data[3:])


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
    # no planted value can make the operand negative.
    data = np.array([-128, -100, -60, -30], dtype=np.int8)
    with pytest.raises(ValueError, match="do not span negative"):
        _op()._enforce_int_operand_sign_span((("input_1", data, -128),), steerable=("input_1",))


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


def test_shipped_exemptions_are_well_formed_and_scoped() -> None:
    descriptors = load_all_descriptors(str(_repo_root() / "assets" / "descriptors"))
    exempt = [d for d in descriptors if d.get("operand_sign_span_exempt") is not None]
    assert exempt, "the audit left at least the PReLU alpha waivers in place"
    for desc in exempt:
        mapping = desc["operand_sign_span_exempt"]
        assert isinstance(mapping, dict) and mapping, desc["name"]
        for operand, reason in mapping.items():
            assert isinstance(reason, str) and reason.strip(), (desc["name"], operand)
        # Nothing waives a runtime input: those are steered instead.
        assert set(mapping) <= {"alpha"}, desc["name"]
