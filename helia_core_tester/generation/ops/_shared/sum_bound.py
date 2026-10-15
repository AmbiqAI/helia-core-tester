"""Data-scaled float tolerance for long sums.

bound = SUM_BOUND * sqrt(K) * eps32 * sqrt(sum t_j^2)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Measured: reorders 0.11, f16 loss 6.6.
SUM_BOUND = 3.0
# Host sums may spend this much.
DRAW_SHARE = 0.5


def tconv_terms(x: np.ndarray, w: np.ndarray, params: Mapping[str, Any], out_hw: tuple[int, int]) -> tuple:
    """Summands and counts per output."""
    # Columns: input pixel, tap, channel.
    n, h, w_in, cin = x.shape
    cout, kh, kw, _ = w.shape
    oh, ow = out_hw
    iy, ix, ky, kx = (a.ravel() for a in np.meshgrid(
        np.arange(h), np.arange(w_in), np.arange(kh), np.arange(kw), indexing="ij"))
    oy, ox = iy * params["stride_h"] + ky - params["pad_h"], ix * params["stride_w"] + kx - params["pad_w"]
    keep = (oy >= 0) & (oy < oh) & (ox >= 0) & (ox < ow)
    pixel, source, tap = (oy * ow + ox)[keep], (iy * w_in + ix)[keep], (ky * kw + kx)[keep]
    # Stable sort keeps input order per output.
    order = np.argsort(pixel, kind="stable")
    pixel, source, tap = pixel[order], source[order], tap[order]
    counts = np.bincount(pixel, minlength=oh * ow)
    slot = np.arange(pixel.size) - np.repeat(np.cumsum(counts) - counts, counts)
    width = max(int(counts.max()), 1)
    # Padding slots read a zero pixel.
    sources = np.full((oh * ow, width), h * w_in)
    taps = np.zeros((oh * ow, width), dtype=np.int64)
    sources[pixel, slot], taps[pixel, slot] = source, tap
    weights = w.astype(np.float64).transpose(1, 2, 0, 3).reshape(kh * kw, cout, cin)
    rows = []
    for batch in x.astype(np.float64).reshape(n, h * w_in, cin):
        pixels = np.concatenate([batch, np.zeros((1, cin))])[sources]
        terms = pixels[:, :, None, :] * weights[taps]
        rows.append(terms.transpose(0, 2, 1, 3).reshape(oh * ow * cout, width * cin))
    count = np.repeat(counts, cout) * cin
    return np.concatenate(rows), np.tile(count, n)


def _running_sum(terms: np.ndarray, start: np.ndarray, acc_dtype) -> np.ndarray:
    """Left-to-right sum in acc_dtype."""
    acc = start.astype(acc_dtype)
    for column in np.asfortranarray(terms.astype(acc_dtype)).T:
        acc = (acc + column).astype(acc_dtype)
    return acc


def _pairwise_sum(terms: np.ndarray) -> np.ndarray:
    """Tree sum in float32."""
    level = terms.astype(np.float32)
    while level.shape[1] > 1:
        if level.shape[1] % 2:
            level = np.pad(level, ((0, 0), (0, 1)))
        level = level[:, 0::2] + level[:, 1::2]
    return level[:, 0]


def reordered_sums(terms: np.ndarray, bias: np.ndarray) -> dict[str, np.ndarray]:
    """f32 sums in other orders."""
    zero = np.zeros(terms.shape[0])
    return {
        "reversed": _running_sum(terms[:, ::-1], zero, np.float32) + bias.astype(np.float32),
        "pairwise": _pairwise_sum(np.column_stack([bias, terms])),
    }


def tconv_tolerance(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray | None,
    params: Mapping[str, Any],
    ref: np.ndarray,
    rtol: float,
    atol: float,
) -> float | None:
    """Scaled atol; None when reorders drift."""
    dtype = ref.dtype
    with np.errstate(invalid="ignore"):
        terms, count = tconv_terms(x, w, params, ref.shape[1:3])
    # Lanes a NaN or Inf reaches.
    finite = np.isfinite(terms)
    tainted = ~finite.all(axis=1)
    terms = np.where(finite, terms, 0)
    rows = np.tile(np.zeros(w.shape[0]) if bias is None else bias.astype(np.float64), terms.shape[0] // w.shape[0])
    scale = np.sqrt(np.square(terms).sum(axis=1) + np.square(rows))
    # Both dtypes accumulate in f32.
    bound = SUM_BOUND * np.sqrt(count + 1) * float(np.finfo(np.float32).eps) * scale
    atol = max(atol, float(bound.max()))
    golden = ref.astype(np.float64).ravel()
    tol = atol + rtol * np.abs(golden)
    lanes = np.isfinite(golden) & ~tainted
    low, high = float(params["activation_min"]), float(params["activation_max"])
    exact = np.clip(terms.sum(axis=1) + rows, low, high)
    if np.any(np.abs(golden - exact)[lanes] > DRAW_SHARE * tol[lanes]):
        return None
    for total in reordered_sums(terms, rows).values():
        out = np.clip(total.astype(dtype).astype(np.float64), low, high)
        with np.errstate(invalid="ignore"):
            drift = np.abs(out - golden)[lanes]
        # A NaN drift fails too.
        if not np.all(drift <= DRAW_SHARE * tol[lanes]):
            return None
    return atol
