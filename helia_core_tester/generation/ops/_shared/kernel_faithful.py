"""Kernel-faithful float goldens: the arithmetic of the ns-cmsis-nn route a case runs on its target,
reproduced by a named C reference variant, compared at torch.testing.assert_close's default
tolerances. The exact reference stays the oracle: an emulated golden must lie within the rigorous
rounding-error bound of the exact result, or generation fails instead of encoding the deviation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from helia_core_tester.core.cpu_targets import get_cpu_profile
from helia_core_tester.generation.reference.call import ReferenceCall

# torch.testing.assert_close defaults: (atol, rtol).
PYTORCH_TOLERANCE = {np.dtype(np.float32): (1e-5, 1.3e-6), np.dtype(np.float16): (1e-5, 1e-3)}

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
