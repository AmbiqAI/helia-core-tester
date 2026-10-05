"""The inner kernel an s8 wrapper calls.

Mirrors the dispatch of arm_convolve_wrapper_s8,
arm_depthwise_conv_wrapper_s8 and arm_fully_connected_wrapper_s8 in
ns-cmsis-nn main (GCC builds; MVE vs not). v7.38.0 lacks the 1xN
padding gate, so there a gated-out layer runs arm_convolve_1_x_n_s8.
Dims are cmsis_nn_dims tuples (n, h, w, c); stride, padding and
dilation are (h, w).

test_wrapper_route runs the real C wrappers against this mirror. To
check a modified kernel tree, point CMSIS_NN_ROOT at it and run that
test.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..core.cpu_targets import get_cpu_profile

Dims = tuple[int, int, int, int]
Pair = tuple[int, int]

CONV_WRAPPER = "arm_convolve_wrapper_s8"
DW_WRAPPER = "arm_depthwise_conv_wrapper_s8"
FC_WRAPPER = "arm_fully_connected_wrapper_s8"
# GCC value; armclang uses 8.
DW_TO_CONV_THRESHOLD = 1


def conv_route(i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, mve: bool) -> str:
    """arm_convolve_wrapper_s8's callee."""
    if pad == (0, 0) and f[1:3] == (1, 1) and dil == (1, 1) and i[3] == f[3]:
        return "arm_convolve_1x1_s8_fast" if stride == (1, 1) else "arm_convolve_1x1_s8"
    if (
        i[1] == 1 and dil[1] == 1 and f[1] == 1 and (stride[1] * i[3]) % 4 == 0 and i[3] == f[3]
        and o[1] == 1 and pad[0] == 0 and pad[1] >= 0 and f[2] >= 1
    ):
        return "arm_convolve_1_x_n_s8"
    if mve and o[1:3] == (1, 1) and (stride[1] * i[3]) % 4 == 0 and i[3] == f[3]:
        return "arm_convolve_1x1_out_s8"
    if mve and _small_cin(i, f, o, dil):
        return "arm_convolve_s8_small_cin"
    if mve and i[3] == 16 and f[3] == 16 and f[1:3] == (3, 3) and stride == (1, 1) and dil == (1, 1):
        return "arm_convolve_s8_3x3_c16_s1"
    return "arm_convolve_s8"


def _small_cin(i: Dims, f: Dims, o: Dims, dil: Pair) -> bool:
    """arm_nn_is_convolve_s8_small_cin."""
    kh, kw, cin = f[1], f[2], i[3]
    return (
        f[3] == cin and 1 <= cin <= 3 and dil == (1, 1) and kw >= 1 and kh >= 1
        and kw * cin <= 16 and kw * kh * cin <= 48 and o[3] > 0 and o[3] % 4 == 0
    )


def dw_route(
    i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, ch_mult: int, mve: bool
) -> str:
    """arm_depthwise_conv_wrapper_s8's callee; DW->conv names the conv callee."""
    if mve and i[3] == 1 and o[3] > DW_TO_CONV_THRESHOLD:
        # Transposed filter: {c, h, w, n}.
        return conv_route(i, (f[3], f[1], f[2], f[0]), o, stride, pad, dil, mve)
    dil_ok = dil == (1, 1) or (
        dil[0] == 1 and dil[1] >= 1 and f[1] == 1 and i[1] == 1 and o[1] == 1
        and stride == (1, 1) and pad[0] == 0
    )
    if ch_mult == 1 and i[0] == 1 and dil_ok:
        if not mve and f[1:3] == (3, 3) and pad[0] <= 1 and pad[1] <= 1 and dil == (1, 1):
            return "arm_depthwise_conv_3x3_s8"
        return "arm_depthwise_conv_s8_opt"
    return "arm_depthwise_conv_s8"


def inner_symbol(timed_symbol: str, manifest: Mapping) -> Optional[str]:
    """The kernel a wrapper case runs, else None."""
    if timed_symbol == FC_WRAPPER:
        # The adapter always passes per-channel quantization.
        return "arm_fully_connected_per_channel_s8"
    if timed_symbol not in (CONV_WRAPPER, DW_WRAPPER):
        return None
    p = manifest["serialized_scalar_parameters"]
    dims = {blob["role"]: tuple(blob["dimensions"]) for blob in manifest["blob_roles"]}
    i = dims["input_0"]
    o = (1, p["output_h"], p["output_w"], p["output_c"])
    stride, pad, dil = (p["stride_h"], p["stride_w"]), (p["pad_h"], p["pad_w"]), (p["dilation_h"], p["dilation_w"])
    mve = get_cpu_profile(manifest["target_cpu"]).has_mve
    if timed_symbol == DW_WRAPPER:
        return dw_route(i, dims["weights"], o, stride, pad, dil, p["ch_mult"], mve)
    # Conv weights blob is (h, w, c, n).
    kh, kw, fc, oc = dims["weights"]
    return conv_route(i, (oc, kh, kw, fc), o, stride, pad, dil, mve)
