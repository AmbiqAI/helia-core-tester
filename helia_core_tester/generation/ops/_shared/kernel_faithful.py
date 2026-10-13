"""Kernel-faithful float goldens: the arithmetic of the ns-cmsis-nn route a case runs on its target,
reproduced by a named C reference variant, compared at torch.testing.assert_close's default
tolerances. The exact reference stays the oracle: an emulated golden must lie within the rigorous
rounding-error bound of the exact result, or generation fails instead of encoding the deviation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from helia_core_tester.core.cpu_targets import get_cpu_profile
from helia_core_tester.generation.io.dtypes import _DEFAULT_FLOAT_COMPARISON as _DTYPE_DEFAULTS
from helia_core_tester.generation.reference.call import ReferenceCall

# torch.testing.assert_close defaults: (atol, rtol), the same values as dtypes.default_comparison_for_dtype.
PYTORCH_TOLERANCE = {
    np.dtype(np.float32): (_DTYPE_DEFAULTS["FP32"]["atol"], _DTYPE_DEFAULTS["FP32"]["rtol"]),
    np.dtype(np.float16): (_DTYPE_DEFAULTS["FP16"]["atol"], _DTYPE_DEFAULTS["FP16"]["rtol"]),
}

# ARM_NN_MVE_F16_MAX_GATHER_STRIDE_8: the largest row count an eight-lane gather can stride.
MVE_F16_GATHER_ROWS = 65535 // 7

_U = {np.dtype(np.float16): 2.0 ** -11, np.dtype(np.float32): 2.0 ** -24}


class KernelDeviationError(ValueError):
    """An emulated route's result leaves the rounding-error bound of the exact result."""


def mve_float16_target(cpu: str) -> bool:
    profile = get_cpu_profile(cpu)
    return profile.has_mve and profile.supports_capability("fp16_execution")


def float_target(cpu: str) -> int:
    """The float32 leg a target runs: 0 Helium, 1 scalar with an FPU, 2 soft float."""
    profile = get_cpu_profile(cpu)
    if profile.has_mve:
        return 0
    return 2 if profile.supports_capability("soft_float") else 1


def shipped(op) -> bool:
    """Whether the case's kernels build as shipped (-Ofast, no coverage): only then do emulated goldens apply."""
    numerics = getattr(op, "kernel_numerics", "shipped")
    if numerics not in ("shipped", "bounded"):
        raise ValueError(f"kernel_numerics must be 'shipped' or 'bounded', got {numerics!r}")
    return numerics == "shipped"


def mve_float16_route(op) -> bool:
    """The case runs a float16 Helium route that an emulated golden reproduces."""
    return shipped(op) and mve_float16_target(op.target_cpu)


def float_route(op) -> Optional[int]:
    """The float32 leg an emulated golden reproduces (see float_target), or None off the shipped build."""
    return float_target(op.target_cpu) if shipped(op) else None


def bounded_context(dtype, atol_extra: float, rtol_extra: float = 0.0) -> dict:
    """The comparison for an exact golden: assert_close's defaults widened by the case's derived bound."""
    if not (np.isfinite(atol_extra) and np.isfinite(rtol_extra)) or atol_extra < 0 or rtol_extra < 0:
        raise ValueError(f"bound must be finite and non-negative, got atol {atol_extra!r}, rtol {rtol_extra!r}")
    atol, rtol = PYTORCH_TOLERANCE[np.dtype(dtype)]
    return {"validation_atol": atol + float(atol_extra), "validation_rtol": rtol + float(rtol_extra),
            "kernel_faithful": False}


def descriptor_tightening(desc) -> Optional[Tuple[float, float]]:
    """The (atol, rtol) a descriptor states, which may only tighten the defaults (test_float_tolerance_policy)."""
    stated = desc.get("comparison") if isinstance(desc, dict) else None
    if not isinstance(stated, dict) or stated.get("atol") is None:
        return None
    return float(stated["atol"]), float(stated.get("rtol", 0.0))


def comparison_context(op, dtype, atol_extra: float = 0.0, rtol_extra: float = 0.0) -> dict:
    """The comparison a float case emits. A descriptor's tightened comparison wins; otherwise an emulated golden
    (no extra) is held to assert_close's defaults and an exact golden to those widened by its derived bound."""
    stated = descriptor_tightening(op.desc)
    if stated is not None:
        return {"validation_atol": stated[0], "validation_rtol": stated[1],
                "kernel_faithful": shipped(op) and not (atol_extra or rtol_extra)}
    if atol_extra or rtol_extra:
        return bounded_context(dtype, atol_extra, rtol_extra)
    return tolerance_context(dtype)


def linear_bound(magnitude: np.ndarray, k: int, dtype) -> Tuple[float, float]:
    """(atol, rtol) covering any evaluation order of a k-term sum in `dtype` against the exact golden:
    |computed - golden| <= (k + 2) u sum|terms| + 2 u |exact|. `magnitude` is sum|terms| per output, e.g. the
    exact reference re-run on the operands' absolute values with the activation clamp off."""
    if k < 0:
        raise ValueError(f"term count must be non-negative, got {k}")
    mag = np.asarray(magnitude, dtype=np.float64)
    finite = mag[np.isfinite(mag)]
    u = _U[np.dtype(dtype)]
    return (k + 2) * u * float(np.max(finite, initial=0.0)), 2.0 * u


def check_close_to_exact(name: str, call: ReferenceCall, exact: ReferenceCall, atol: float) -> None:
    """An approximating variant (kernel activation tables) must stay within `atol` of the exact
    reference on every finite lane, or the emulation itself is wrong."""
    a = call.output().astype(np.float64)
    b = exact.output().astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    worst = float(np.max(np.abs(a - b)[finite], initial=0.0))
    if worst > atol:
        raise KernelDeviationError(f"{name}: {call.entry} is {worst:.3e} from {exact.entry}, beyond {atol:.1e}")


def tolerance_context(dtype) -> dict:
    atol, rtol = PYTORCH_TOLERANCE[np.dtype(dtype)]
    return {"validation_atol": atol, "validation_rtol": rtol, "kernel_faithful": True}


def check_within_bound(name: str, emulated: np.ndarray, lhs: np.ndarray, rhs: np.ndarray,
                       bias: Optional[np.ndarray], dtype) -> None:
    """|emulated - exact| <= (K + 2) u sum|a b| + u |exact| on every finite lane (lhs [B,M,K], rhs [B,N,K])."""
    a = lhs.astype(np.float64)
    b = rhs.astype(np.float64)
    exact = np.einsum("bmk,bnk->bmn", a, b)
    mag = np.einsum("bmk,bnk->bmn", np.abs(a), np.abs(b))
    if bias is not None and bias.size:
        exact = exact + bias.astype(np.float64)
        mag = mag + np.abs(bias.astype(np.float64))
    u = _U[np.dtype(dtype)]
    k = lhs.shape[-1]
    bound = (k + 2) * u * mag + u * np.abs(exact)
    emu = emulated.astype(np.float64).reshape(exact.shape)
    finite = np.isfinite(emu) & np.isfinite(exact)
    with np.errstate(invalid="ignore"):  # nonfinite lanes are excluded by `finite`
        excess = np.where(finite, np.abs(emu - exact) - bound, -np.inf)
    if np.any(excess > 0):
        i = np.unravel_index(int(np.argmax(excess)), excess.shape)
        raise KernelDeviationError(
            f"{name}: emulated route result {emu[i]!r} at {i} is {abs(emu[i] - exact[i]):.3e} from the exact "
            f"{exact[i]!r}, beyond the rounding bound {bound[i]:.3e}")


def matmul_f16_mve_call(lhs: np.ndarray, rhs: np.ndarray, bias: Optional[np.ndarray], *, route: int, acc16: bool,
                        activation: Tuple[float, float]) -> ReferenceCall:
    """lhs [B,M,K], rhs [B,N,K] (row n is output column n), bias [N] or None. route 0: nt_t matmul;
    1: arm_batch_matmul_f16 adjoint lanes; 2: its fused scalar loop."""
    lhs = np.ascontiguousarray(lhs, dtype=np.float16)
    rhs = np.ascontiguousarray(rhs, dtype=np.float16)
    if lhs.ndim != 3 or rhs.ndim != 3 or lhs.shape[0] != rhs.shape[0] or lhs.shape[2] != rhs.shape[2]:
        raise ValueError(f"matmul operands must be [B,M,K] and [B,N,K], got {lhs.shape} and {rhs.shape}")
    b = np.zeros((0,), np.float16) if bias is None else np.ascontiguousarray(bias, dtype=np.float16)
    return ReferenceCall("matmul_nt_t_f16_mve",
                         {"route": int(route), "acc16": int(bool(acc16)),
                          "activation_min": float(activation[0]), "activation_max": float(activation[1])},
                         {"lhs": lhs, "rhs": rhs, "bias": b},
                         {"output": (lhs.shape[0], lhs.shape[1], rhs.shape[1])})


def conv_1xn_f16_mve_call(x: np.ndarray, weights: np.ndarray, bias: Optional[np.ndarray], *, stride_w: int,
                          pad_w: int, out_w: int, acc16: bool, packed: bool,
                          activation: Tuple[float, float]) -> ReferenceCall:
    """arm_convolve_1_x_n_f16 (or _acc16) on Helium: x [N,1,W,C], weights OHWI [O,1,KW,C] (unpacked even for an
    NT_N_PACKED case), bias [O] or None; output [N,1,out_w,O] (out_w as the case's output dims give it: SAME
    padding pads the trailing edge by up to one more than pad_w)."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    weights = np.ascontiguousarray(weights, dtype=np.float16)
    if x.ndim != 4 or weights.ndim != 4 or x.shape[1] != 1 or weights.shape[1] != 1 or x.shape[3] != weights.shape[3]:
        raise ValueError(f"1xN conv operands must be [N,1,W,C] and [O,1,KW,C], got {x.shape} and {weights.shape}")
    if stride_w < 1 or pad_w < 0 or out_w < 1:
        raise ValueError(f"1xN conv needs stride_w >= 1, pad_w >= 0 and out_w >= 1, got {stride_w}, {pad_w}, {out_w}")
    b = np.zeros((0,), np.float16) if bias is None else np.ascontiguousarray(bias, dtype=np.float16)
    return ReferenceCall("conv_1xn_f16_mve",
                         {"stride_w": int(stride_w), "pad_w": int(pad_w), "acc16": int(bool(acc16)),
                          "packed": int(bool(packed)), "activation_min": float(activation[0]),
                          "activation_max": float(activation[1])},
                         {"input": x, "filter": weights, "bias": b},
                         {"output": (x.shape[0], 1, out_w, weights.shape[0])})


# arm_nn_mat_mult_nt_n_packed_f16 numerical policies (matmul_nt_n_packed_f16_mve's `policy`).
NT_N_PACKED_FOLD, NT_N_PACKED_ACC16, NT_N_PACKED_PRECISE = 0, 1, 2


def matmul_nt_n_packed_f16_mve_call(lhs: np.ndarray, rhs: np.ndarray, bias: Optional[np.ndarray], *, policy: int,
                                    activation: Tuple[float, float]) -> ReferenceCall:
    """arm_nn_mat_mult_nt_n_packed_f16 on Helium: lhs [B,M,K], rhs [B,N,K] (the weights before NT_N packing; row n
    is output column n), bias [N] or None. policy: NT_N_PACKED_FOLD (the plain entry), NT_N_PACKED_ACC16 (_acc16)
    or NT_N_PACKED_PRECISE (_precise)."""
    lhs = np.ascontiguousarray(lhs, dtype=np.float16)
    rhs = np.ascontiguousarray(rhs, dtype=np.float16)
    if lhs.ndim != 3 or rhs.ndim != 3 or lhs.shape[0] != rhs.shape[0] or lhs.shape[2] != rhs.shape[2]:
        raise ValueError(f"matmul operands must be [B,M,K] and [B,N,K], got {lhs.shape} and {rhs.shape}")
    if policy not in (NT_N_PACKED_FOLD, NT_N_PACKED_ACC16, NT_N_PACKED_PRECISE):
        raise ValueError(f"unknown nt_n_packed policy {policy!r}")
    b = np.zeros((0,), np.float16) if bias is None else np.ascontiguousarray(bias, dtype=np.float16)
    return ReferenceCall("matmul_nt_n_packed_f16_mve",
                         {"policy": int(policy), "activation_min": float(activation[0]),
                          "activation_max": float(activation[1])},
                         {"lhs": lhs, "rhs": rhs, "bias": b},
                         {"output": (lhs.shape[0], lhs.shape[1], rhs.shape[1])})


# conv_f16_mve routes (see reference/src/conv/conv_f16_mve.c).
CONV_F16_PATCH_NT_T, CONV_F16_PATCH_PACKED, CONV_F16_DIRECT_OHWI, CONV_F16_DIRECT_PACKED = 0, 1, 2, 3
CONV_F16_SMALL_C, CONV_F16_CONV1D, CONV_F16_CONV1D_PACKED = 4, 5, 6
CONV_F16_ROUTE_NAMES = {CONV_F16_PATCH_NT_T: "patch_nt_t", CONV_F16_PATCH_PACKED: "patch_nt_n_packed",
                        CONV_F16_DIRECT_OHWI: "direct_ohwi", CONV_F16_DIRECT_PACKED: "direct_packed",
                        CONV_F16_SMALL_C: "small_c", CONV_F16_CONV1D: "conv1d", CONV_F16_CONV1D_PACKED: "conv1d_packed"}


def conv_f16_mve_call(x: np.ndarray, weights: np.ndarray, bias: Optional[np.ndarray], *, route: int,
                      window: dict, output_shape, acc16: bool, activation: Tuple[float, float]) -> ReferenceCall:
    """A float16 convolution route of ns-cmsis-nn on Helium (conv_f16_mve): x [N,H,W,C], weights OHWI [O,KH,KW,C]
    (unpacked even for an NT_N_PACKED case), bias [O] or None, window {stride,pad,dilation}_{h,w} (leading pads),
    output [N,OH,OW,O] as the case's output dims give it."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    weights = np.ascontiguousarray(weights, dtype=np.float16)
    if x.ndim != 4 or weights.ndim != 4 or x.shape[3] != weights.shape[3]:
        raise ValueError(f"conv operands must be [N,H,W,C] and [O,KH,KW,C], got {x.shape} and {weights.shape}")
    if route not in CONV_F16_ROUTE_NAMES:
        raise ValueError(f"unknown conv_f16_mve route {route!r}")
    out = tuple(int(d) for d in output_shape)
    if len(out) != 4 or out[0] != x.shape[0] or out[3] != weights.shape[0] or min(out) < 1:
        raise ValueError(f"conv output must be [N,OH,OW,O] for {x.shape} and {weights.shape}, got {out}")
    b = np.zeros((0,), np.float16) if bias is None else np.ascontiguousarray(bias, dtype=np.float16)
    params = {k: int(window[k]) for k in ("stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w")}
    return ReferenceCall("conv_f16_mve",
                         {"route": int(route), **params, "acc16": int(bool(acc16)),
                          "activation_min": float(activation[0]), "activation_max": float(activation[1])},
                         {"input": x, "filter": weights, "bias": b},
                         {"output": out})

# ------------------------------------------------------------------ misc family (float16 on Helium)

# arm_transpose_conv_f16 gathers the eight output channels' weights at stride kh * kw * ic; past
# ARM_NN_MVE_F16_MAX_GATHER_STRIDE_8 it runs a scalar loop GCC vectorizes unfused (not modelled).
MVE_F16_GATHER_STRIDE_8 = 65535 // 7


def transpose_conv_f16_mve_route(kh: int, kw: int, in_c: int) -> bool:
    """True when arm_transpose_conv_f16 takes its gathered vfmaq_m path for this filter."""
    return kh * kw * in_c <= MVE_F16_GATHER_STRIDE_8


def transpose_conv_f16_mve_call(x: np.ndarray, weights: np.ndarray, bias: Optional[np.ndarray],
                                window: dict, output_shape, activation: Tuple[float, float]) -> ReferenceCall:
    """arm_transpose_conv_nhwc_f16 on Helium: x NHWC, weights OHWI, bias [O] or None, window with
    stride/dilation/pad h and w; output_shape NHWC."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    weights = np.ascontiguousarray(weights, dtype=np.float16)
    if x.ndim != 4 or weights.ndim != 4 or x.shape[3] != weights.shape[3]:
        raise ValueError(f"transpose conv operands must be NHWC and OHWI, got {x.shape} and {weights.shape}")
    b = np.zeros((0,), np.float16) if bias is None else np.ascontiguousarray(bias, dtype=np.float16)
    params = {k: int(window[k]) for k in ("stride_h", "stride_w", "dilation_h", "dilation_w", "pad_h", "pad_w")}
    return ReferenceCall("transpose_conv_f16_mve",
                         {**params, "activation_min": float(activation[0]), "activation_max": float(activation[1])},
                         {"input": x, "filter": weights, "bias": b}, {"output": tuple(int(d) for d in output_shape)})


def transpose_conv_patches(x: np.ndarray, weights: np.ndarray, window: dict, output_shape) -> Tuple[np.ndarray,
                                                                                                    np.ndarray]:
    """[1, N*OH*OW, KH*KW*C] input patches (zero where no input lands) and [1, O, KH*KW*C] filter rows, so a
    transpose convolution is patches @ rows.T (for check_within_bound)."""
    n, h, w, c = x.shape
    o, kh, kw, _ = weights.shape
    _, oh, ow, _ = output_shape
    patches = np.zeros((n, oh, ow, kh, kw, c), dtype=np.float64)
    for iy in range(h):
        for ix in range(w):
            for ky in range(kh):
                oy = iy * window["stride_h"] - window["pad_h"] + ky * window["dilation_h"]
                if not 0 <= oy < oh:
                    continue
                for kx in range(kw):
                    ox = ix * window["stride_w"] - window["pad_w"] + kx * window["dilation_w"]
                    if 0 <= ox < ow:
                        patches[:, oy, ox, ky, kx, :] = x[:, iy, ix, :]
    return patches.reshape(1, -1, kh * kw * c), weights.astype(np.float64).reshape(1, o, -1)


def avg_pool_f16_mve_call(x: np.ndarray, window: dict, output_shape,
                          activation: Tuple[float, float]) -> ReferenceCall:
    """arm_avg_pool_f16 on Helium: x NHWC, window with stride/filter/pad h and w; output_shape NHWC."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    if x.ndim != 4:
        raise ValueError(f"avg pool input must be NHWC, got {x.shape}")
    params = {k: int(window[k]) for k in ("stride_h", "stride_w", "filter_h", "filter_w", "pad_h", "pad_w")}
    return ReferenceCall("avg_pool_f16_mve",
                         {**params, "activation_min": float(activation[0]), "activation_max": float(activation[1])},
                         {"input": x}, {"output": tuple(int(d) for d in output_shape)})


def avg_pool_windows(x: np.ndarray, window: dict, output_shape) -> Tuple[np.ndarray, np.ndarray]:
    """[B, 1, KH*KW] window taps (zero outside the image) and [B, 1, KH*KW] weights 1 / count on the taps, one
    batch row per output element, so an average pool is a batch of dot products (for check_within_bound)."""
    n, h, w, c = x.shape
    _, oh, ow, _ = output_shape
    fh, fw = window["filter_h"], window["filter_w"]
    taps = np.zeros((n, oh, ow, c, fh * fw), dtype=np.float64)
    weights = np.zeros_like(taps)
    for oy in range(oh):
        for ox in range(ow):
            by, bx = oy * window["stride_h"] - window["pad_h"], ox * window["stride_w"] - window["pad_w"]
            cells = [(ky, kx) for ky in range(fh) for kx in range(fw) if 0 <= by + ky < h and 0 <= bx + kx < w]
            for ky, kx in cells:
                taps[:, oy, ox, :, ky * fw + kx] = x[:, by + ky, bx + kx, :]
                weights[:, oy, ox, :, ky * fw + kx] = 1.0 / len(cells)
    return taps.reshape(-1, 1, fh * fw), weights.reshape(-1, 1, fh * fw)


def softmax_f16_mve_call(x: np.ndarray) -> ReferenceCall:
    """arm_softmax_f16 on Helium over the last dim (the 1 x 2 specialisation included)."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    return ReferenceCall("softmax_f16_mve", {"unused": 0}, {"input": x}, {"output": x.shape})


def softmax_f16_atol(x: np.ndarray) -> float:
    """Distance allowed between the kernel's float16 softmax and the exact one: x - max is rounded to float16
    (relative exp error u |x - max|), the polynomial or LUT exp is within 1e-6 and is rounded once (u), the
    eight-lane float16 reduce and float32 sum (4u relative), 1 / sum rounded to float16 (2u) and the final
    product (u). Every output is <= 1, so (|x - max| + 8) u bounds the absolute error."""
    a = x.astype(np.float64).reshape(-1, x.shape[-1])
    a = np.where(np.isfinite(a), a, np.nan)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(a, axis=1) - np.nanmin(a, axis=1)
    worst = float(np.nanmax(spread)) if np.any(np.isfinite(spread)) else 0.0
    return (worst + 8.0) * _U[np.dtype(np.float16)]


def reduce_f16_route(kind: str, shape, axes) -> int:
    """The accumulation order arm_reduce_sum_f16 ("sum") or arm_nn_mean_f16 ("mean") takes on Helium for a 4-D
    input and reduced axes: 0 one running float32 sum, 1 the flatten path's eight lanes."""
    dims = [int(d) for d in shape]
    mask = [1 if d in set(int(a) for a in axes) else 0 for d in range(4)]
    if len(dims) != 4:
        raise ValueError(f"reduction routes need a 4-D input, got {shape}")
    if kind == "sum" and not mask[0] and not mask[3] and (mask[1] or mask[2]) and all(d > 0 for d in dims):
        return 0  # arm_reduce_sum_spatial_mve_f16: one float32 lane per kept element
    if kind not in ("sum", "mean"):
        raise ValueError(f"unknown reduction {kind!r}")
    union = [m or d == 1 for m, d in zip(mask, dims)]
    first = next((d for d in range(4) if mask[d]), None)
    if first is not None and all(union[first:]):
        return 1  # arm_reduce_get_flatten_suffix_start_from_arrays >= 0
    return 0  # middle block or generic: one running float32 sum


def reduce_f16_f32acc_call(x: np.ndarray, axes, *, mean: bool, route: int) -> ReferenceCall:
    """arm_reduce_sum_f16 / arm_nn_mean_f16 on Helium; the output keeps the rank with reduced dims 1."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    axes = sorted(set(int(a) for a in axes))
    out = tuple(1 if d in axes else int(s) for d, s in enumerate(x.shape))
    return ReferenceCall("mean_f16_f32acc" if mean else "reduce_sum_f16_f32acc",
                         {"axis_mask": sum(1 << a for a in axes), "route": int(route)}, {"input": x}, {"output": out})


def reduce_operands(x: np.ndarray, axes, *, mean: bool) -> Tuple[np.ndarray, np.ndarray]:
    """[B, 1, K] reduced elements per output and [B, 1, K] weights (1, or 1 / K for a mean)."""
    axes = sorted(set(int(a) for a in axes))
    kept = [d for d in range(x.ndim) if d not in axes]
    moved = np.transpose(x.astype(np.float64), kept + axes)
    k = int(np.prod([x.shape[d] for d in axes])) if axes else 1
    rows = moved.reshape(-1, 1, k)
    return rows, np.full_like(rows, 1.0 / k if mean else 1.0)


def unclamped_output(call: ReferenceCall) -> np.ndarray:
    """The emulated call's output without its activation clamp, for check_within_bound (which bounds the
    unclamped sum); a call without activation params is returned as is."""
    from dataclasses import replace

    if "activation_min" not in call.params:
        return call.output()
    return replace(call, params={**call.params, "activation_min": float("-inf"),
                                 "activation_max": float("inf")}).output()

# ------------------------------------------------- float16 recurrent kernels on Helium (rnn_f16_mve.c)

RNN_F16_MVE_ENTRIES = {"lstm_f16": "lstm_f16_mve", "gru_f16": "gru_f16_mve", "svdf_f16": "svdf_f16_mve"}

# lstm_f16_mve / gru_f16_mve against the exact lstm_f16 / gru_f16: both round every step's state to float16,
# but the kernel's sigmoid (float16 exp2 table, float16 exp) and tanh (float16 table over [0, 4]) each land
# within a few float16 ulps of the exact activation (ulp(1) = 9.8e-4), and the recurrence carries those
# differences into later steps through weights of magnitude < 1. 5e-3 is five ulps at 1.0; the descriptors'
# cases measure at most 7.4e-4 over seven seeds.
RNN_F16_ATOL = 5e-3


def rnn_f16_mve_call(exact: ReferenceCall) -> ReferenceCall:
    """The Helium emulation of an exact lstm_f16 / gru_f16 / svdf_f16 call: same params, operands and outputs."""
    entry = RNN_F16_MVE_ENTRIES.get(exact.entry)
    if entry is None:
        raise ValueError(f"{exact.entry} has no float16 Helium recurrent variant (one of {sorted(RNN_F16_MVE_ENTRIES)})")
    return ReferenceCall(entry, dict(exact.params), dict(exact.inputs), dict(exact.output_shapes))


def check_svdf_within_bound(name: str, emulated: np.ndarray, call: ReferenceCall) -> None:
    """The emulated SVDF output [S,B,U] against the exact float64 SVDF, every step's error bounded by
    propagating the summation bound through the state: a feature projection within (I+2)u sum|w x| + u|s|,
    each time reduction within sum|w_t| e_s + (M+2)u sum|w_t| (|s| + e_s) + u|t|, and the rank sum within
    sum e_t + (rank+2)u (|b| + sum(|t| + e_t)) + u|out|."""
    ins = {k: v.astype(np.float64) for k, v in call.inputs.items()}
    x, wf, wt, bias, state = ins["input"], ins["weights_feature"], ins["weights_time"], ins["bias"], ins["state"]
    rank = int(call.params["rank"])
    u = _U[np.dtype(call.inputs["input"].dtype)]
    steps, batches, _ = x.shape
    feats, mem = wt.shape
    units = feats // rank
    s = state.copy()
    e = np.zeros_like(s)
    emu = np.asarray(emulated, np.float64).reshape(steps, batches, units)
    lo, hi = float(call.params["activation_min"]), float(call.params["activation_max"])
    for step in range(steps):
        s = np.roll(s, -1, axis=2)
        e = np.roll(e, -1, axis=2)
        proj = np.einsum("fi,bi->bf", wf, x[step])
        s[:, :, -1] = proj
        e[:, :, -1] = (x.shape[2] + 2) * u * np.einsum("fi,bi->bf", np.abs(wf), np.abs(x[step])) + u * np.abs(proj)
        t = np.einsum("fm,bfm->bf", wt, s)
        et = (np.einsum("fm,bfm->bf", np.abs(wt), e) + (mem + 2) * u * np.einsum("fm,bfm->bf", np.abs(wt), np.abs(s) + e)
              + u * np.abs(t))
        out = t.reshape(batches, units, rank).sum(-1)
        mag = (np.abs(t) + et).reshape(batches, units, rank).sum(-1)
        if bias.size:
            out = out + bias
            mag = mag + np.abs(bias)
        bound = et.reshape(batches, units, rank).sum(-1) + (rank + 2) * u * mag + u * np.abs(out)
        exact = np.clip(out, lo, hi)
        finite = np.isfinite(emu[step]) & np.isfinite(exact) & np.isfinite(bound)
        with np.errstate(invalid="ignore"):
            excess = np.where(finite, np.abs(emu[step] - exact) - bound, -np.inf)
        if np.any(excess > 0):
            i = np.unravel_index(int(np.argmax(excess)), excess.shape)
            raise KernelDeviationError(
                f"{name}: emulated SVDF step {step} output {emu[step][i]!r} at {i} is "
                f"{abs(emu[step][i] - exact[i]):.3e} from the exact {exact[i]!r}, beyond the bound {bound[i]:.3e}")


def depthwise_f16_mve_call(x: np.ndarray, weights: np.ndarray, bias: Optional[np.ndarray], *,
                           window: dict, out_hw: Tuple[int, int], acc16: bool,
                           activation: Tuple[float, float]) -> ReferenceCall:
    """ns-cmsis-nn's float16 depthwise routes that chain over the in-range taps (direct ch_mult 1, generic,
    conv1d k3, 2x5) on Helium: x [N,H,W,C], weights [1,KH,KW,C*M], bias [C*M] or None; output [N,OH,OW,C*M]."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    weights = np.ascontiguousarray(weights, dtype=np.float16)
    if x.ndim != 4 or weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[3] % x.shape[3]:
        raise ValueError(f"depthwise operands must be [N,H,W,C] and [1,KH,KW,C*M], got {x.shape} and {weights.shape}")
    b = np.zeros((0,), np.float16) if bias is None else np.ascontiguousarray(bias, dtype=np.float16)
    params = {k: int(window[k]) for k in ("stride_h", "stride_w", "dilation_h", "dilation_w", "pad_h", "pad_w")}
    return ReferenceCall("depthwise_f16_mve",
                         {**params, "acc16": int(bool(acc16)), "activation_min": float(activation[0]),
                          "activation_max": float(activation[1])},
                         {"input": x, "filter": weights, "bias": b},
                         {"output": (x.shape[0], int(out_hw[0]), int(out_hw[1]), weights.shape[3])})

# ------------------------------------------------ activation arithmetic per target (act family, step 2)

# The activations whose float golden follows the kernel's approximation (the others are exact in the kernel:
# relu, relu6, leaky relu and none round at most one product, as the exact reference does).
APPROXIMATED_ACTIVATIONS = ("ARM_NN_FLT_ACT_SIGMOID", "ARM_NN_FLT_ACT_TANH", "ARM_NN_FLT_ACT_HARDSWISH")

# Distance allowed between a kernel activation and the exact one rounded once (both legs of the comparison
# round, so each budget carries the golden's half ulp too):
#  - float32 tanh: the 384-segment table's interpolation error, 2.35e-5 inside |x| < 6 per the kernel header,
#    and the 1 - tanh(6) = 1.2e-5 saturation step, plus two roundings at |y| <= 1 (2^-24).
#  - float32 sigmoid: linear interpolation of 2^f over 1/256 steps is within (ln 2)^2 2 / (8 * 256^2) = 1.8e-6
#    relative of exp, the table entries within 2^-24; sigmoid's sensitivity to e's relative error is
#    s (1 - s) <= 1/4, and three float32 roundings follow, so 1e-6.
#  - float16 sigmoid: the float16 exp2 table and e's float16 rounding put e within 2^-11 relative, a quarter
#    of that reaches the output, then the float32 result, the float16 output and the golden each round:
#    2^-10 (two float16 ulps below 1).
#  - tanh f16 (the [0, 4] float16 table): the table and interpolation error of tanh_lut_mve_f16, 2^-9.
ACTIVATION_ATOL = {
    (np.dtype(np.float32), "ARM_NN_FLT_ACT_TANH"): 2.5e-5,
    (np.dtype(np.float32), "ARM_NN_FLT_ACT_SIGMOID"): 1e-6,
    (np.dtype(np.float16), "ARM_NN_FLT_ACT_SIGMOID"): 2.0 ** -10,
    (np.dtype(np.float16), "ARM_NN_FLT_ACT_TANH"): 2.0 ** -9,
}

# |h(1/6) * 6 - 1| (2.44e-4): the float16 kernels scale by 1/6 rounded to float16.
_SIXTH_F16_REL = abs(float(np.float16(1.0 / 6.0)) * 6.0 - 1.0)


def hard_swish_atol(x: np.ndarray) -> float:
    """Distance allowed between a kernel hard swish and the exact one: the gate x / 6 + 1/2 (or (x + 3) / 6) is
    within 2u (two roundings) plus the scale's own error (float16: h(1/6)), the gate is in [0, 1], and the
    product and the golden each round, so |x| (4u + scale error) bounds it; |x| is taken at least 1."""
    dtype = np.dtype(x.dtype)
    a = np.abs(x.astype(np.float64))
    worst = max(1.0, float(np.max(a[np.isfinite(a)], initial=0.0)))
    scale = _SIXTH_F16_REL if dtype == np.dtype(np.float16) else 0.0
    return worst * (4.0 * _U[dtype] + scale)


def activation_atol(x: np.ndarray, activation_type: str) -> float:
    if activation_type == "ARM_NN_FLT_ACT_HARDSWISH":
        return hard_swish_atol(x)
    return ACTIVATION_ATOL[(np.dtype(x.dtype), activation_type)]


def nn_activation_f32_cmsis_call(x: np.ndarray, activation_code: int, act_param: float, target: int) -> ReferenceCall:
    """arm_nn_activation_f32 on float target `target` (float_target): the kernel's sigmoid/tanh tables and
    hardswish contraction as that target compiles them."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if target not in (0, 1, 2):
        raise ValueError(f"unknown float32 target {target!r}")
    return ReferenceCall("nn_activation_f32_cmsis",
                         {"activation_type": int(activation_code), "act_param": float(act_param),
                          "target": int(target)}, {"input": x}, {"output": x.shape})


def nn_activation_f16_mve_call(x: np.ndarray, activation_code: int, act_param: float) -> ReferenceCall:
    """arm_nn_activation_f16 on Helium."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    return ReferenceCall("nn_activation_f16_mve", {"activation_type": int(activation_code),
                                                   "act_param": float(act_param)}, {"input": x}, {"output": x.shape})


def hard_swish_cmsis_call(x: np.ndarray, cpu: str) -> Optional[ReferenceCall]:
    """arm_hard_swish_f32 (every target) or arm_hard_swish_f16 on Helium; None for float16 off Helium."""
    if np.dtype(x.dtype) == np.dtype(np.float32):
        x = np.ascontiguousarray(x)
        return ReferenceCall("hard_swish_f32_cmsis", {"target": float_target(cpu)}, {"input": x},
                             {"output": x.shape})
    if np.dtype(x.dtype) == np.dtype(np.float16) and mve_float16_target(cpu):
        x = np.ascontiguousarray(x)
        return ReferenceCall("hard_swish_f16_mve", {"unused": 0}, {"input": x}, {"output": x.shape})
    return None


def softmax_f32_cmsis_call(x: np.ndarray, target: int) -> ReferenceCall:
    """arm_softmax_f32 on float target `target` over the last dim (the 1 x 2 specialisation included)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if target not in (0, 1, 2):
        raise ValueError(f"unknown float32 target {target!r}")
    return ReferenceCall("softmax_f32_cmsis", {"target": int(target)}, {"input": x}, {"output": x.shape})


# exp's relative error in arm_softmax_f32: the LUT exp's interpolation (1.8e-6, see ACTIVATION_ATOL) and table
# (2^-24) on the scalar legs; arm_nn_vexpq_poly_mve_f32's degree-7 polynomial is tighter.
SOFTMAX_F32_EXP_REL = 2e-6


def softmax_f32_atol(x: np.ndarray) -> float:
    """Distance allowed between the kernel's float32 softmax and the exact one: x - max rounds (u |x - max|
    relative in exp), exp is within SOFTMAX_F32_EXP_REL, the sum (at most a few float32 roundings per element,
    relative per element of the row), 1 / sum and the product each round. Every output is <= 1, so
    (|x - max| + cols + 8) u plus twice the exp error (numerator and sum) bounds the absolute error."""
    a = x.astype(np.float64).reshape(-1, x.shape[-1])
    a = np.where(np.isfinite(a), a, np.nan)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(a, axis=1) - np.nanmin(a, axis=1)
    worst = float(np.nanmax(spread)) if np.any(np.isfinite(spread)) else 0.0
    return (worst + x.shape[-1] + 8.0) * _U[np.dtype(np.float32)] + 2.0 * SOFTMAX_F32_EXP_REL

# ------------------------------------------------------- float32 GRU per target (gru_f32_cmsis.c)

# gru_f32_cmsis against the exact gru_f32: the kernel's sigmoid (exp2 table, within ~1e-7 relative) and tanh
# (384-segment table, interpolation error <= 2.35e-5, a step of 1.2e-5 at the saturation point) are the only
# approximations; the recurrence carries them through weights of magnitude < 1. The FVP measured 7.8e-6..1.9e-5
# against the exact golden on every target; 1e-4 (the LSTM variant's bound) leaves room without hiding a wrong
# emulation, whose error would be of the order of the activations themselves.
GRU_F32_ATOL = 1e-4


def gru_f32_cmsis_call(exact: ReferenceCall, cpu: str) -> ReferenceCall:
    """The emulation of an exact gru_f32 call as arm_gru_unidirectional_f32 computes it on `cpu`."""
    if exact.entry != "gru_f32":
        raise ValueError(f"gru_f32_cmsis emulates gru_f32, got {exact.entry}")
    return ReferenceCall("gru_f32_cmsis", {**exact.params, "target": float_target(cpu)}, dict(exact.inputs),
                         dict(exact.output_shapes))

# ------------------------------------------- exact-golden bounds (rest family: pooling, reductions, recurrent)

# The guard budget of lstm_f32_cmsis against the exact lstm_f32, the counterpart of GRU_F32_ATOL.
LSTM_F32_ATOL = 1e-4


def abs_magnitude(call: ReferenceCall, output: Optional[str] = None) -> np.ndarray:
    """sum|terms| per output of an accumulating exact entry (linear_bound's `magnitude`): `call` re-run on the
    absolute values of its float operands with its activation clamp off, as float64. A float16 entry with a float32
    twin (`<op>_f32`) runs that twin on the widened operands, so the magnitude is not rounded to float16. It is
    evaluated here, never recorded as the case's golden."""
    from dataclasses import replace

    from helia_core_tester.generation.reference.abi import load_spec

    entry = call.entry
    widen = entry.endswith("_f16") and f"{entry[:-4]}_f32" in load_spec().kernels
    if widen:
        entry = f"{entry[:-4]}_f32"

    def _abs(v):
        if not np.issubdtype(v.dtype, np.floating):
            return v
        return np.abs(v).astype(np.float32) if widen and v.dtype == np.float16 else np.abs(v)

    inputs = {k: _abs(v) for k, v in call.inputs.items()}
    params = dict(call.params)
    if "activation_min" in params or "activation_max" in params:
        params.update(activation_min=float("-inf"), activation_max=float("inf"))
    probe = replace(call, entry=entry, params=params, inputs=inputs)
    out = probe.output() if output is None else probe.run()[output]
    return np.asarray(out, dtype=np.float64)


def float32_accumulation_bound(magnitude: np.ndarray, k: int, dtype) -> Tuple[float, float]:
    """linear_bound for a kernel that accumulates (and scales) in float32 on every route and rounds once to
    `dtype` when it stores (arm_reduce_sum_f16, arm_nn_mean_f16): the float32 summation bound as atol, and that
    store plus the golden's own rounding, 2 u(dtype), as rtol."""
    atol, _ = linear_bound(magnitude, k, np.float32)
    return atol, 2.0 * _U[np.dtype(dtype)]


def recurrent_dot_bound(x: np.ndarray, input_weights: np.ndarray, recurrent_weights: np.ndarray,
                        biases, dtype, state_bound: Optional[float] = None) -> float:
    """The atol an exact LSTM/GRU golden widens by for the gate dot products, per step.

    Each gate pre-activation is a sum of I input terms, H recurrent terms (|h| <= 1: h is o tanh(c), or a convex
    combination of tanh outputs) and the biases, so linear_bound bounds its error by d = (k + 2) u M + 2 u M over
    the largest such magnitude M. A step's output moves by at most d (1.5 + S / 4): the sigmoid gates pass a
    quarter of their error, tanh all of it, and the error of the gate multiplying a carried quantity of magnitude
    S scales by S: the cell state for an LSTM (`state_bound`), W_h h + b_h for a GRU's candidate (None: M bounds
    it). The carry of earlier steps' error is left to the guard budget this is added to."""
    w_in = np.abs(np.asarray(input_weights, np.float64))
    w_h = np.abs(np.asarray(recurrent_weights, np.float64))
    if w_in.ndim != 3 or w_h.ndim != 3 or w_in.shape[:2] != w_h.shape[:2] or w_h.shape[1] != w_h.shape[2]:
        raise ValueError(f"gate weights must be [G,H,I] and [G,H,H], got {w_in.shape} and {w_h.shape}")
    if state_bound is not None and (not np.isfinite(state_bound) or state_bound < 0):
        raise ValueError(f"state bound must be finite and non-negative, got {state_bound!r}")
    rows = np.abs(np.asarray(x, np.float64)).reshape(-1, w_in.shape[2])
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    mag = np.einsum("ghi,ri->grh", w_in, rows) + w_h.sum(axis=2)[:, None, :]
    terms = w_in.shape[2] + w_h.shape[2]
    for b in biases:
        b = np.asarray(b, np.float64)
        if b.size:
            mag = mag + np.abs(b).reshape(w_in.shape[0], 1, w_in.shape[1])
            terms += 1
    atol, rtol = linear_bound(mag, terms, dtype)
    worst = float(np.max(mag, initial=0.0))
    carried = worst if state_bound is None else float(state_bound)
    return (atol + rtol * worst) * (1.5 + carried / 4.0)
