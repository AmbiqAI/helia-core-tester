"""The inner kernel a conv, depthwise or FC wrapper calls.

Mirrors the dispatch of the s8 conv, depthwise and FC wrappers and
the s4 and s16 conv and depthwise wrappers in ns-cmsis-nn v7.38.0
and later (GCC builds; MVE vs not). The 1xN padding gate (v7.38.1,
s8 and s4) is read from the built tree; an unknown tree gives no
route. Names the wrapper's direct callee. Off MVE,
arm_convolve_1_x_n_s8 just forwards to arm_convolve_s8.

inner_variant goes one level down: arm_depthwise_conv_s8_opt runs a
planar or a channelwise algorithm. It is named by the matching
public entry (arm_depthwise_conv_s8_opt_planar or _channelwise).
Dims are cmsis_nn_dims tuples (n, h, w, c); stride, padding and
dilation are (h, w).

test_wrapper_route runs the real C wrappers against this mirror. To
check a modified kernel tree, point CMSIS_NN_ROOT at it and run that
test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Optional

from ..core.cpu_targets import get_cpu_profile

Dims = tuple[int, int, int, int]
Pair = tuple[int, int]

CONV_WRAPPER = "arm_convolve_wrapper_s8"
DW_WRAPPER = "arm_depthwise_conv_wrapper_s8"
FC_WRAPPER = "arm_fully_connected_wrapper_s8"
CONV_S4_WRAPPER = "arm_convolve_wrapper_s4"
CONV_S16_WRAPPER = "arm_convolve_wrapper_s16"
DW_S4_WRAPPER = "arm_depthwise_conv_wrapper_s4"
DW_S16_WRAPPER = "arm_depthwise_conv_wrapper_s16"
CONV_WRAPPERS = (CONV_WRAPPER, CONV_S4_WRAPPER, CONV_S16_WRAPPER)
DW_WRAPPERS = (DW_WRAPPER, DW_S4_WRAPPER, DW_S16_WRAPPER)
DW_OPT = "arm_depthwise_conv_s8_opt"
PLANAR = "arm_depthwise_conv_s8_opt_planar"
CHANNELWISE = "arm_depthwise_conv_s8_opt_channelwise"
# Planar plane slack; opt scratch channel block.
PLANAR_SLACK = 32
CH_IN_BLOCK_MVE = 124
# GCC value; armclang uses 8.
DW_TO_CONV_THRESHOLD = 1
CONV_SOURCE = Path("Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c")
# Known 1xN conditions: padding gate or not.
GATES_1XN = {
    "arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims)": False,
    "arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims) && "
    "arm_nn_convolve_1_x_n_s8_padding_supported(conv_params, filter_dims, output_dims)": True,
}
_GATE_1XN = re.compile(r"else if \((arm_nn_is_convolve_1_x_n\(.*?)\)\s*\{", re.S)


def tree_gate(kernel_root: Optional[Path]) -> Optional[bool]:
    """A tree's 1xN padding gate; None if unknown."""
    try:
        text = (kernel_root / CONV_SOURCE).read_text(encoding="utf-8")
    except (OSError, TypeError):
        return None
    match = _GATE_1XN.search(text)
    # Older trees lack the small_cin route.
    if match is None or "arm_nn_is_convolve_s8_small_cin(" not in text:
        return None
    return GATES_1XN.get(" ".join(match.group(1).split()))


def build_gate(build_dir: Optional[Path]) -> Optional[bool]:
    """The built tree's 1xN padding gate."""
    from .firmware_build import nsx_app_dir
    from .nsx_app import kernel_dir, saved_options

    if build_dir is None:
        return None
    app_dir = nsx_app_dir(build_dir)
    options = saved_options(app_dir)
    return None if options is None else tree_gate(kernel_dir(app_dir, options))


def conv_route(
    i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, mve: bool, gate_1xn: bool = True
) -> str:
    """arm_convolve_wrapper_s8's callee."""
    if pad == (0, 0) and f[1:3] == (1, 1) and dil == (1, 1) and i[3] == f[3]:
        return "arm_convolve_1x1_s8_fast" if stride == (1, 1) else "arm_convolve_1x1_s8"
    if _is_1xn(i, f, stride, dil) and (
        not gate_1xn or (o[1] == 1 and pad[0] == 0 and pad[1] >= 0 and f[2] >= 1)
    ):
        return "arm_convolve_1_x_n_s8"
    if mve and o[1:3] == (1, 1) and (stride[1] * i[3]) % 4 == 0 and i[3] == f[3]:
        return "arm_convolve_1x1_out_s8"
    if mve and _small_cin(i, f, o, dil):
        return "arm_convolve_s8_small_cin"
    if mve and i[3] == 16 and f[3] == 16 and f[1:3] == (3, 3) and stride == (1, 1) and dil == (1, 1):
        return "arm_convolve_s8_3x3_c16_s1"
    return "arm_convolve_s8"


def _is_1xn(i: Dims, f: Dims, stride: Pair, dil: Pair) -> bool:
    """arm_nn_is_convolve_1_x_n."""
    return i[1] == 1 and dil[1] == 1 and f[1] == 1 and (stride[1] * i[3]) % 4 == 0 and i[3] == f[3]


def _1xn_pad_ok(i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair) -> bool:
    """arm_nn_convolve_1_x_n_padding_supported."""
    if o[1] != 1 or pad[0] != 0:
        return False
    sx, px = stride[1], pad[1]
    if sx <= 0:
        return True
    total = (o[2] - 1) * sx + f[2] - i[2]
    if total < 0 or px * 2 + total % 2 != total:
        return False
    asym = total % 2
    right = max(1, (px + asym + sx - 1) // sx) if px + asym else 0
    left = max(1, (px + sx - 1) // sx) if px else 0
    return left + right <= o[2] and px + asym < f[2] and f[2] <= i[2]


def conv_s4_route(
    i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, mve: bool, gate_1xn: bool = True
) -> str:
    """arm_convolve_wrapper_s4's callee."""
    if pad == (0, 0) and f[1:3] == (1, 1) and dil == (1, 1) and i[3] == f[3]:
        return "arm_convolve_1x1_s4_fast" if stride == (1, 1) else "arm_convolve_1x1_s4"
    if _is_1xn(i, f, stride, dil) and (not gate_1xn or _1xn_pad_ok(i, f, o, stride, pad)):
        return "arm_convolve_1_x_n_s4"
    if mve and (f[1] * f[2] * i[3]) % 2 == 0:
        return "arm_convolve_even_s4"
    return "arm_convolve_s4"


def conv_s16_route(i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, mve: bool) -> str:
    """arm_convolve_wrapper_s16's callee."""
    if mve and i[3] == f[3] and stride == (1, 1) and pad == (0, 0) and f[1:3] == (1, 1) and dil == (1, 1):
        return "arm_convolve_1x1_s16_ns_np_nd"
    if mve and f[1] * f[2] * f[3] < 9 and pad == (0, 0):
        return "arm_convolve_s16_fast_small_kernel"
    if f[3] == 1 and i[3] == o[3]:
        return "arm_convolve_s16_group_ch_mult_1"
    return "arm_convolve_s16"


def _small_cin(i: Dims, f: Dims, o: Dims, dil: Pair) -> bool:
    """arm_nn_is_convolve_s8_small_cin."""
    kh, kw, cin = f[1], f[2], i[3]
    return (
        f[3] == cin and 1 <= cin <= 3 and dil == (1, 1) and kw >= 1 and kh >= 1
        and kw * cin <= 16 and kw * kh * cin <= 48 and o[3] > 0 and o[3] % 4 == 0
    )


def dw_route(
    i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, ch_mult: int, mve: bool,
    gate_1xn: bool = True,
) -> str:
    """arm_depthwise_conv_wrapper_s8's callee; DW->conv names the conv callee."""
    if mve and i[3] == 1 and o[3] > DW_TO_CONV_THRESHOLD:
        # Transposed filter: {c, h, w, n}.
        return conv_route(i, (f[3], f[1], f[2], f[0]), o, stride, pad, dil, mve, gate_1xn)
    if ch_mult == 1 and i[0] == 1 and _dil_ok(i, f, o, stride, pad, dil):
        if not mve and f[1:3] == (3, 3) and pad[0] <= 1 and pad[1] <= 1 and dil == (1, 1):
            return "arm_depthwise_conv_3x3_s8"
        return "arm_depthwise_conv_s8_opt"
    return "arm_depthwise_conv_s8"


def _dil_ok(i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair) -> bool:
    """arm_nn_dw_conv_opt_dilation_supported."""
    return dil == (1, 1) or (
        dil[0] == 1 and dil[1] >= 1 and f[1] == 1 and i[1] == 1 and o[1] == 1
        and stride == (1, 1) and pad[0] == 0
    )


def dw_s4_route(i: Dims, dil: Pair, ch_mult: int) -> str:
    """arm_depthwise_conv_wrapper_s4's callee."""
    if ch_mult == 1 and i[0] == 1 and dil == (1, 1):
        return "arm_depthwise_conv_s4_opt"
    return "arm_depthwise_conv_s4"


def dw_s16_route(i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, ch_mult: int) -> str:
    """arm_depthwise_conv_wrapper_s16's callee."""
    if ch_mult == 1 and _dil_ok(i, f, o, stride, pad, dil) and f[1] * f[2] < 512:
        return "arm_depthwise_conv_fast_s16"
    return "arm_depthwise_conv_s16"


def opt_scratch_size(i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, mve: bool) -> int:
    """The ctx size the adapter gives opt.

    The firmware sizes the s8 wrapper's ctx with
    arm_depthwise_conv_wrapper_s8_get_buffer_size; on the opt
    route that is the MVE opt scratch.
    """
    if not mve or i[3] != o[3] or i[0] != 1 or not _dil_ok(i, f, o, stride, pad, dil):
        return 0
    return 4 * CH_IN_BLOCK_MVE * f[1] * f[2]


def planar_bytes(i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, ch_mult: int) -> int:
    """arm_nn_depthwise_conv_s8_planar_bytes; -1 if declined."""
    ch, dx = i[3], dil[1]
    if (
        ch != o[3] or i[0] != 1 or ch_mult != 1 or stride != (1, 1) or dil[0] != 1 or dx < 1
        or ch < 1 or o[2] < 1 or o[1] < 1 or f[2] < 1 or f[1] < 1
    ):
        return -1
    if ch > 32 or dx > 128 // ch:
        return -1
    plane_w = o[2] + (f[2] - 1) * dx
    if o[1] == 1 and i[1] == 1 and f[1] == 1 and pad[0] == 0 and 5 <= f[2] <= 16:
        # The 1xk dot product path.
        if o[2] < 8 or (ch > 16 and o[2] < 24 * dx):
            return -1
        return -(-plane_w // dx) * dx + PLANAR_SLACK
    if o[1] == 1:
        profitable = ch <= 16 or (ch <= 32 and o[2] >= 32)
    else:
        profitable = ch <= 8 or (ch <= 16 and o[2] >= 16)
    if o[2] < 8 or not profitable:
        return -1
    return plane_w * (o[1] + f[1] - 1) + PLANAR_SLACK


def dw_opt_variant(
    i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, ch_mult: int, mve: bool, ctx_size: int
) -> str:
    """The algorithm arm_depthwise_conv_s8_opt runs.

    Planar declines when its plane exceeds ctx_size
    or the MVE opt scratch; channelwise then runs.
    """
    plane = planar_bytes(i, f, o, stride, pad, dil, ch_mult)
    if not mve or plane < 0 or ctx_size <= 0 or plane > ctx_size or plane > 4 * CH_IN_BLOCK_MVE * f[1] * f[2]:
        return CHANNELWISE
    return PLANAR


def _layer(manifest: Mapping) -> tuple:
    """Wrapper args from a case manifest."""
    p = manifest["serialized_scalar_parameters"]
    # Firmware defaults: omitted or 0 stride/dilation is 1.
    stride = (p.get("stride_h") or 1, p.get("stride_w") or 1)
    dil = (p.get("dilation_h") or 1, p.get("dilation_w") or 1)
    pad = (p.get("pad_h", 0), p.get("pad_w", 0))
    o = (1, p.get("output_h", 0), p.get("output_w", 0), p.get("output_c", 0))
    dims = {blob["role"]: tuple(blob["dimensions"]) for blob in manifest["blob_roles"]}
    mve = get_cpu_profile(manifest["target_cpu"]).has_mve
    return dims["input_0"], dims["weights"], o, stride, pad, dil, p.get("ch_mult", 0), mve


def inner_symbol(timed_symbol: str, manifest: Mapping, gate_1xn: Optional[bool]) -> Optional[str]:
    """The kernel a wrapper case runs, else None."""
    if timed_symbol not in (*CONV_WRAPPERS, *DW_WRAPPERS, FC_WRAPPER) or gate_1xn is None:
        return None
    if timed_symbol == FC_WRAPPER:
        # The adapter always passes per-channel quantization.
        return "arm_fully_connected_per_channel_s8"
    i, w, o, stride, pad, dil, ch_mult, mve = _layer(manifest)
    if timed_symbol == DW_WRAPPER:
        return dw_route(i, w, o, stride, pad, dil, ch_mult, mve, gate_1xn)
    if timed_symbol == DW_S4_WRAPPER:
        return dw_s4_route(i, dil, ch_mult)
    if timed_symbol == DW_S16_WRAPPER:
        return dw_s16_route(i, w, o, stride, pad, dil, ch_mult)
    # Conv weights blob is (h, w, c, n).
    kh, kw, fc, oc = w
    f = (oc, kh, kw, fc)
    if timed_symbol == CONV_S4_WRAPPER:
        return conv_s4_route(i, f, o, stride, pad, dil, mve, gate_1xn)
    if timed_symbol == CONV_S16_WRAPPER:
        return conv_s16_route(i, f, o, stride, pad, dil, mve)
    return conv_route(i, f, o, stride, pad, dil, mve, gate_1xn)


def inner_variant(timed_symbol: str, manifest: Mapping, gate_1xn: Optional[bool]) -> Optional[str]:
    """The algorithm opt runs, else None."""
    if timed_symbol != DW_WRAPPER or inner_symbol(timed_symbol, manifest, gate_1xn) != DW_OPT:
        return None
    i, f, o, stride, pad, dil, ch_mult, mve = _layer(manifest)
    ctx_size = opt_scratch_size(i, f, o, stride, pad, dil, mve)
    return dw_opt_variant(i, f, o, stride, pad, dil, ch_mult, mve, ctx_size)
