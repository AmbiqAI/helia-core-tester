"""Per-element contract intervals for float16/float32 outputs.

A contract of the form |out - ref| <= rtol * |ref| + atol, with ref a float64
reference, allows a contiguous run of output values. This module finds the
lowest and highest bit patterns in that run, in the sign-magnitude order that
helia_test_float_interval compares in (so -0 sits just below +0). The
comparison then asserts the contract itself rather than a tolerance around a
rounded golden.
"""

from fractions import Fraction
from typing import Tuple

import numpy as np

_BITS = {np.dtype(np.float16): np.uint16, np.dtype(np.float32): np.uint32}
_CANONICAL_NAN = {np.dtype(np.float16): 0x7E00, np.dtype(np.float32): 0x7FC00000}
# Near the contract edge a candidate is re-tested in exact rational arithmetic
# against the float64 reference. That reference is itself approximate: against
# an 80-digit evaluation over float16 and float32 inputs, math.erfc-based GELU
# stays within 2^-42.27 of its magnitude where the result is a normal float64,
# and a subnormal result is off by up to 813 * 2^-1074. A candidate within
# that error of the edge, an exact tie included, cannot be classified and is
# reported instead of guessed.
_REF_ERROR = 2.0**-40
_REF_ERROR_ABS = 2.0**-1060
# Float64 rounding in forming |value - ref| and the bound, relative to the bound.
_ROUNDING_MARGIN = 2.0**-50


def _key(bits: np.ndarray, sign: int) -> np.ndarray:
    magnitude = (bits & (sign - 1)).astype(np.int64)
    return np.where(bits & sign, -magnitude - 1, magnitude)


def _bits(key: np.ndarray, sign: int, bits_dtype) -> np.ndarray:
    return np.where(key < 0, (-key - 1) | sign, key).astype(bits_dtype)


def contract_interval(ref: np.ndarray, dtype, rtol: float, atol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) bit arrays of the outputs within rtol * |ref| + atol of ref.

    NaN references give NaN for both ends and infinite references the same
    infinity. An exactly zero reference allows only the zero of its own sign.
    """
    dtype = np.dtype(dtype)
    bits_dtype = _BITS[dtype]
    sign = 1 << (8 * dtype.itemsize - 1)
    ref = np.asarray(ref, dtype=np.float64).ravel()
    finite = np.isfinite(ref) & (ref != 0)

    with np.errstate(over="ignore", invalid="ignore"):
        golden = ref.astype(dtype)
    lo = golden.view(bits_dtype).copy()
    hi = lo.copy()
    lo[np.isnan(ref)] = hi[np.isnan(ref)] = _CANONICAL_NAN[dtype]

    bound = rtol * np.abs(ref) + atol
    ref_error = _REF_ERROR * np.abs(ref) + _REF_ERROR_ABS
    margin = ref_error + _ROUNDING_MARGIN * bound

    def inside(bits: np.ndarray, lanes: np.ndarray) -> np.ndarray:
        value = bits.view(dtype).astype(np.float64)
        with np.errstate(invalid="ignore"):
            diff = np.abs(value - ref[lanes])
        ok = np.isfinite(value) & (diff <= bound[lanes])
        for i in np.flatnonzero(np.abs(diff - bound[lanes]) <= margin[lanes]):
            r = Fraction(float(ref[lanes[i]]))
            excess = abs(Fraction(float(value[i])) - r) - (Fraction(rtol) * abs(r) + Fraction(atol))
            if abs(excess) <= Fraction(float(ref_error[lanes[i]])):
                raise ValueError(
                    f"output {float(value[i])!r} lies within float64 error of the contract edge for "
                    f"reference {float(r)!r}; the interval cannot be decided"
                )
            ok[i] = excess <= 0
        return ok

    lanes = np.flatnonzero(finite)
    if not inside(lo[lanes], lanes).all():
        raise ValueError("a reference rounded to the output type lies outside its own contract")
    largest = float(np.finfo(dtype).max)
    for end, step in ((lo, -1), (hi, 1)):
        # Start at the edge rounded to the output type, which lies on the golden's side of it
        # or one step beyond; step inward until inside, then outward while the next one is.
        edge = ref[lanes] + step * bound[lanes]
        with np.errstate(over="ignore"):
            candidate = np.clip(edge, -largest, largest).astype(dtype).view(bits_dtype).copy()
        active = np.arange(lanes.size)
        while active.size:
            active = active[~inside(candidate[active], lanes[active])]
            candidate[active] = _bits(_key(candidate[active], sign) - step, sign, bits_dtype)
        active = np.arange(lanes.size)
        while active.size:
            outward = _bits(_key(candidate[active], sign) + step, sign, bits_dtype)
            ok = inside(outward, lanes[active])
            candidate[active[ok]] = outward[ok]
            active = active[ok]
        end[lanes] = candidate
    return lo, hi
