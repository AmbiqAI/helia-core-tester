"""Independent CORE index contract: first NaN, else first numeric extremum.

NaN selection deliberately differs from observed LiteRT sequences. Operands
are decoded as integers so subnormal ordering cannot depend on host FTZ/DAZ.
"""

import numpy as np


def arg_extrema_reference(bits, axis, kind):
    """Return exact S32 axis-relative indices for a canonical 4D bit tensor."""
    bits = np.asarray(bits)
    if bits.dtype not in (np.dtype("uint16"), np.dtype("uint32")):
        raise ValueError("Expected uint16 or uint32 operand bits")
    if bits.ndim != 4 or not isinstance(axis, (int, np.integer)) or not 0 <= axis < 4:
        raise ValueError("Expected canonical 4D input and axis 0..3")
    if kind not in ("min", "max"):
        raise ValueError("Expected min or max")
    if bits.shape[axis] == 0:
        raise ValueError("Empty reduction axis has no index")
    shape = tuple(n for i, n in enumerate(bits.shape) if i != axis)
    count = int(np.prod(shape))
    domains = np.moveaxis(bits, axis, -1).reshape(count, bits.shape[axis])
    result = np.empty(count, dtype=np.int32)
    sign = 1 << (bits.dtype.itemsize * 8 - 1)
    infinity = 0x7C00 if bits.dtype.itemsize == 2 else 0x7F800000
    for out, domain in enumerate(domains):
        winner, best = 0, None
        for index, word in enumerate(domain):
            word = int(word)
            magnitude = word & (sign - 1)
            if magnitude > infinity:
                winner = index
                break
            key = -magnitude if word & sign else magnitude
            if best is None or (key < best if kind == "min" else key > best):
                winner, best = index, key
        result[out] = winner
    return result.reshape(shape)
