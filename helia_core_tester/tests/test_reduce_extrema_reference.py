"""The float reduce-min/max reference, clause by clause against ns-cmsis-nn#498.

This reference is the oracle for every float extrema case, so a mistake here is invisible
in the worst way: the generated golden and the kernel would have to disagree for anything
to show, and a wrong golden that happens to match a wrong kernel shows nothing at all.
Each contract clause therefore gets a test that fails if the rule is dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from helia_core_tester.generation.ops._shared.reduce_extrema_reference import (
    canonical_qnan,
    reduce_extrema_reference,
)

F32 = np.float32
F16 = np.float16


def _bits(x):
    x = np.asarray(x)
    width = np.uint32 if x.dtype == np.dtype(F32) else np.uint16
    return int(np.frombuffer(np.asarray(x, dtype=x.dtype).tobytes(), dtype=width)[0])


def _from_bits(value: int, dtype) -> np.floating:
    dtype = np.dtype(dtype)
    width = np.uint32 if dtype == np.dtype(F32) else np.uint16
    return np.frombuffer(width(value).tobytes(), dtype=dtype)[0]


@pytest.mark.parametrize("dtype,qnan", [(F32, 0x7FC00000), (F16, 0x7E00)])
def test_canonical_qnan_is_the_bit_pattern_the_contract_names(dtype, qnan):
    assert _bits(canonical_qnan(dtype)) == qnan


# --- Equal values retain the first input in row-major order, including zero signs ------

@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("dtype", [F32, F16])
def test_a_tie_retains_the_first_input_and_its_zero_sign(kind, dtype):
    """The clause numpy gets backwards, which is why this reference exists.

    np.max([-0.0, 0.0]) is +0.0 and np.max([0.0, -0.0]) is -0.0: numpy keeps the last
    equal element. The contract keeps the first. A golden built on numpy would disagree
    with a correct kernel on every signed-zero tie and agree with an inverted one.
    """
    first_negative = np.array([[dtype(-0.0), dtype(0.0)]], dtype=dtype)
    first_positive = np.array([[dtype(0.0), dtype(-0.0)]], dtype=dtype)
    neg_zero = _bits(dtype(-0.0))
    pos_zero = _bits(dtype(0.0))

    assert _bits(reduce_extrema_reference(first_negative, [1], kind)[0, 0]) == neg_zero
    assert _bits(reduce_extrema_reference(first_positive, [1], kind)[0, 0]) == pos_zero


@pytest.mark.parametrize("kind", ["max", "min"])
def test_a_tie_among_equal_nonzero_values_keeps_the_first(kind):
    # Same rule away from zero, where the bits are indistinguishable; this pins the
    # ordering logic rather than the sign, so a reference that scanned in reverse and
    # happened to pass the zero test still fails here.
    values = np.array([[F32(2.5), F32(2.5), F32(1.0)]], dtype=F32)
    got = reduce_extrema_reference(values, [1], kind)[0, 0]
    assert got == (F32(2.5) if kind == "max" else F32(1.0))


# --- Any NaN in a reduction yields canonical quiet NaN ---------------------------------

@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("payload", [0x7FA00001, 0x7F800001, 0xFFC00000])
def test_any_nan_in_the_domain_yields_the_canonical_quiet_nan(kind, payload):
    """Whatever NaN goes in, the canonical one comes out. Payload and sign are discarded."""
    values = np.array([[_from_bits(payload, F32), F32(1.0), F32(-3.0)]], dtype=F32)
    got = reduce_extrema_reference(values, [1], kind)[0, 0]
    assert _bits(got) == 0x7FC00000


@pytest.mark.parametrize("kind", ["max", "min"])
def test_a_nan_decides_rather_than_competes(kind):
    """A NaN anywhere wins, including last and including against an infinity."""
    values = np.array([[F32(np.inf), F32(-np.inf), _from_bits(0x7FA00001, F32)]], dtype=F32)
    assert _bits(reduce_extrema_reference(values, [1], kind)[0, 0]) == 0x7FC00000


# --- A zero mask copies bits unchanged; a singleton axis canonicalises -----------------

def test_a_zero_mask_copies_bits_unchanged_including_nan_payloads():
    payload = _from_bits(0x7FA00001, F32)
    values = np.array([[payload, F32(-0.0), F32(np.inf)]], dtype=F32)
    got = reduce_extrema_reference(values, [], "max")
    assert [_bits(v) for v in got[0]] == [0x7FA00001, _bits(F32(-0.0)), _bits(F32(np.inf))]


def test_reducing_a_singleton_axis_canonicalises_where_a_zero_mask_would_not():
    """The contract draws this distinction explicitly, so the reference must too.

    Reducing nothing preserves a NaN payload; reducing an axis of extent one is still a
    reduction and yields the canonical NaN. A reference that treated a singleton axis as
    a no-op copy would pass every other test here.
    """
    payload = _from_bits(0x7FA00001, F32)
    values = np.array([[payload]], dtype=F32)
    assert _bits(reduce_extrema_reference(values, [], "max")[0, 0]) == 0x7FA00001
    assert _bits(reduce_extrema_reference(values, [1], "max")[0, 0]) == 0x7FC00000


# --- Infinities and subnormals retain their bits ---------------------------------------

@pytest.mark.parametrize("kind,expected_bits", [("max", 0x7F800000), ("min", 0x00000001)])
def test_infinities_and_subnormals_retain_their_bits(kind, expected_bits):
    smallest_subnormal = _from_bits(0x00000001, F32)
    values = np.array([[F32(np.inf), smallest_subnormal, F32(1.0)]], dtype=F32)
    assert _bits(reduce_extrema_reference(values, [1], kind)[0, 0]) == expected_bits


# --- An empty reduced domain produces the identity --------------------------------------

@pytest.mark.parametrize("kind,expected", [("max", -np.inf), ("min", np.inf)])
def test_an_empty_reduced_domain_produces_the_identity(kind, expected):
    values = np.zeros((2, 0), dtype=F32)
    got = reduce_extrema_reference(values, [1], kind)
    assert got.shape == (2, 1)
    assert np.all(got == F32(expected))


# --- Ordinary selection, cross-checked against numpy where they must agree --------------

@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize(
    "shape,axes",
    [
        ((4,), [0]),
        ((2, 3), [1]),
        ((2, 3), [0]),
        ((2, 3, 4), [1]),
        ((2, 3, 4), [0, 2]),
        ((2, 3, 4, 5), [1, 2]),
        ((2, 3, 4, 5), [0, 1, 2, 3]),
    ],
)
def test_ordinary_selection_agrees_with_numpy_where_the_rules_coincide(kind, shape, axes):
    """Guards the traversal itself.

    On data with no ties and no NaN the contract and numpy must agree exactly, so numpy
    is a fair cross-check of the axis handling even though it cannot be the oracle. Values
    are distinct by construction so no tie rule is in play.
    """
    rng = np.random.default_rng(20260911)
    values = rng.permutation(int(np.prod(shape))).astype(F32).reshape(shape)
    got = reduce_extrema_reference(values, axes, kind, keepdims=True)
    want = (np.max if kind == "max" else np.min)(values, axis=tuple(axes), keepdims=True)
    assert got.shape == want.shape
    assert np.array_equal(got, want)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_keepdims_false_drops_the_reduced_axes(kind):
    rng = np.random.default_rng(7)
    values = rng.permutation(24).astype(F32).reshape((2, 3, 4))
    got = reduce_extrema_reference(values, [1], kind, keepdims=False)
    want = (np.max if kind == "max" else np.min)(values, axis=1)
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_subnormals_survive_a_process_that_has_loaded_tensorflow():
    """Guards against flush-to-zero contamination of the oracle (ns-cmsis-nn#498 review).

    The kernel lane's reviewer found that loading a GCC -Ofast host shared library
    installed FTZ/DAZ in the process, after which a float32-to-float64 conversion in their
    NumPy reference silently flushed a subnormal to zero. Their kernel was right and the
    reference was wrong, which is the dangerous direction.

    Generation always imports TensorFlow, so the same exposure exists here. This drives the
    reference in a process that has loaded it and asserts the subnormal comes back bit for
    bit. The reference stays in its own float width and never widens, which is why it
    survives; this test is what keeps that true if someone adds a float64 step later.
    """
    import tensorflow  # noqa: F401  -- imported for its side effects on FP control state

    smallest_subnormal = _from_bits(0x00000001, F32)
    values = np.array([[F32(np.inf), smallest_subnormal, F32(1.0)]], dtype=F32)
    assert _bits(reduce_extrema_reference(values, [1], "min")[0, 0]) == 0x00000001


def test_float16_selection_stays_in_float16():
    values = np.array([[F16(1.5), F16(-2.25), F16(3.75)]], dtype=F16)
    got = reduce_extrema_reference(values, [1], "max")
    assert got.dtype == np.dtype(F16)
    assert got[0, 0] == F16(3.75)
