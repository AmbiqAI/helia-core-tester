"""Independent reference for the float reduce-min and reduce-max bit contract.

The kernel contract (ns-cmsis-nn#498) is a bit contract, not a numeric one:

    Values are selected without floating-point arithmetic, accumulation or conversion.
    Any NaN in a reduction yields canonical quiet NaN (0x7fc00000); infinities and
    subnormals retain their bits. Equal numeric values retain the first input in
    row-major order, including zero signs. A zero mask copies bits unchanged,
    including NaN payloads. Reducing a singleton axis instead canonicalizes NaNs.
    An empty reduced domain produces -Inf (max) or +Inf (min).

``numpy`` cannot stand in for this. ``np.max`` keeps the *last* equal element's sign
where the contract keeps the first::

    np.max([-0.0, 0.0]) -> 0.0     # positive
    np.max([0.0, -0.0]) -> -0.0    # negative

so a numpy golden would disagree with a correct kernel on every signed-zero tie, and
would agree with one that had the rule backwards. numpy does canonicalise NaN, which
happens to match, but that is an implementation detail rather than a promise.

Writing the rules out is therefore not busywork: it is the only way to get the bits
right, and it keeps the reference an independent formulation rather than a borrowing of
numpy's semantics -- which is the shared-misunderstanding failure #127 turned out to be.

Ordering is decided on integers decoded from the stored bits, never by a floating-point
comparison. That is not a stylistic choice. An earlier version of this reference compared
with ``>`` and ``<`` on float32 values and was wrong whenever the process had DAZ set:
the hardware reads a subnormal operand as zero, so distinct subnormals compare equal to
each other and to zero, and the selection picks the wrong element. Demonstrated by the
kernel lane at MXCSR 0x9fe2 (ns-cmsis-nn#498 review), four cases including
``max(0x00000000, 0x00000001)`` returning ``0x00000000``.

Note what that corrects: an earlier comment here claimed the reference was safe because
it never widened to float64. Widening was never the mechanism. Staying in float32 does
nothing, because the comparison instruction itself honours the control state. Only
removing the floating-point comparison removes the exposure.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Canonical quiet NaN per the contract, per width.
_CANONICAL_QNAN_BITS = {np.dtype(np.float32): 0x7FC00000, np.dtype(np.float16): 0x7E00}
_BITS_DTYPE = {np.dtype(np.float32): np.uint32, np.dtype(np.float16): np.uint16}


def canonical_qnan(dtype: np.dtype) -> np.floating:
    """The canonical quiet NaN the contract requires a reduction over a NaN to yield."""
    dtype = np.dtype(dtype)
    bits = _BITS_DTYPE[dtype](_CANONICAL_QNAN_BITS[dtype])
    return np.frombuffer(bits.tobytes(), dtype=dtype)[0]


def _empty_domain_value(kind: str, dtype: np.dtype) -> np.floating:
    """An empty reduced domain produces the identity at the far end of the range."""
    dtype = np.dtype(dtype)
    return dtype.type(-np.inf if kind == "max" else np.inf)


def reduce_extrema_reference(
    values: np.ndarray,
    axes: Iterable[int],
    kind: str,
    *,
    keepdims: bool = True,
) -> np.ndarray:
    """Reduce ``values`` along ``axes`` under the kernel's selection rules.

    ``kind`` is "max" or "min". Returns an array of the same floating dtype.

    Selection walks each reduction domain in row-major order and keeps the first element
    that is strictly better than the incumbent, so an equal value never displaces the one
    before it. That is what preserves the first input's zero sign on a tie, and it is the
    single place this differs from ``np.max``/``np.min``.
    """
    if kind not in ("max", "min"):
        raise ValueError(f"kind must be 'max' or 'min', not {kind!r}")
    dtype = np.dtype(values.dtype)
    if dtype not in _CANONICAL_QNAN_BITS:
        raise ValueError(f"reduce extrema reference handles float32 and float16, not {dtype}")

    axes = sorted({int(a) % values.ndim for a in axes})

    # A zero mask copies bits unchanged, NaN payloads included. Reducing nothing is not
    # the same as reducing a singleton axis, which canonicalises; keep them distinct.
    if not axes:
        return values.copy()

    kept = [d for d in range(values.ndim) if d not in axes]
    out_shape = tuple(values.shape[d] for d in kept)

    # Move the reduced axes to the end so each output position owns one contiguous,
    # row-major reduction domain -- the order the contract's tie rule is defined against.
    moved = np.transpose(values, kept + axes)
    domain_size = int(np.prod([values.shape[a] for a in axes]))
    output_count = int(np.prod(out_shape)) if out_shape else 1
    # Both counts are computed rather than inferred with -1: a reduced axis of extent zero
    # makes domain_size 0, and -1 cannot be resolved against a zero dimension.
    flat = moved.reshape((output_count, domain_size))

    result = np.empty(flat.shape[0], dtype=dtype)
    for i, domain in enumerate(flat):
        result[i] = _reduce_one_domain(domain, kind, dtype)

    if keepdims:
        full = [values.shape[d] if d not in axes else 1 for d in range(values.ndim)]
        # The kept axes were moved to the front above, so restore their original order.
        return result.reshape(out_shape if out_shape else ()).reshape(tuple(full))
    return result.reshape(out_shape if out_shape else ())


def _reduce_one_domain(domain: Sequence[np.floating], kind: str, dtype: np.dtype) -> np.floating:
    """Select over one reduction domain, in row-major order, under the contract's rules."""
    if len(domain) == 0:
        return _empty_domain_value(kind, dtype)

    # Any NaN anywhere in the domain yields the canonical quiet NaN, whatever payload the
    # input carried. This is checked before selection: a NaN does not compete, it decides.
    if np.isnan(np.asarray(domain, dtype=dtype)).any():
        return canonical_qnan(dtype)

    raw = np.asarray(domain, dtype=dtype).view(_BITS_DTYPE[dtype])
    winner_index = 0
    winner_key = _numeric_order_key(int(raw[0]), dtype)
    for index in range(1, len(raw)):
        key = _numeric_order_key(int(raw[index]), dtype)
        better = key > winner_key if kind == "max" else key < winner_key
        # Strictly better only. An equal key leaves the incumbent in place, which is what
        # retains the first input's bits -- including its zero sign -- on a tie.
        if better:
            winner_key = key
            winner_index = index
    return dtype.type(domain[winner_index])


def _numeric_order_key(bits: int, dtype: np.dtype) -> int:
    """An exact integer whose ordering matches the float's numeric ordering.

    Decoded from the stored bits so no floating-point comparison is involved and the
    process's FP control state cannot influence the result. Subnormals order correctly
    because their magnitude field is an ordinary small integer here.

    Both zeros map to 0, because +0.0 and -0.0 are numerically equal and the contract
    therefore treats them as a tie, which the caller resolves in favour of the first
    element. Ordering them by their bits would make -0.0 strictly smaller and quietly
    break the clause this reference exists to enforce.
    """
    width = 32 if dtype == np.dtype(np.float32) else 16
    sign = bits >> (width - 1)
    magnitude = bits & ((1 << (width - 1)) - 1)
    if magnitude == 0:
        return 0
    return -magnitude if sign else magnitude
