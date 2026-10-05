"""The inner kernel an s8 wrapper calls.

Mirrors the dispatch of arm_convolve_wrapper_s8,
arm_depthwise_conv_wrapper_s8 and arm_fully_connected_wrapper_s8 in
ns-cmsis-nn v7.38.0 and later (GCC builds; MVE vs not). The 1xN
padding gate (v7.38.1) is read from the built tree; an unknown
tree gives no route. Names the wrapper's direct callee. Off MVE,
arm_convolve_1_x_n_s8 just forwards to arm_convolve_s8.
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
    if (
        i[1] == 1 and dil[1] == 1 and f[1] == 1 and (stride[1] * i[3]) % 4 == 0 and i[3] == f[3]
        and (not gate_1xn or (o[1] == 1 and pad[0] == 0 and pad[1] >= 0 and f[2] >= 1))
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
    i: Dims, f: Dims, o: Dims, stride: Pair, pad: Pair, dil: Pair, ch_mult: int, mve: bool,
    gate_1xn: bool = True,
) -> str:
    """arm_depthwise_conv_wrapper_s8's callee; DW->conv names the conv callee."""
    if mve and i[3] == 1 and o[3] > DW_TO_CONV_THRESHOLD:
        # Transposed filter: {c, h, w, n}.
        return conv_route(i, (f[3], f[1], f[2], f[0]), o, stride, pad, dil, mve, gate_1xn)
    dil_ok = dil == (1, 1) or (
        dil[0] == 1 and dil[1] >= 1 and f[1] == 1 and i[1] == 1 and o[1] == 1
        and stride == (1, 1) and pad[0] == 0
    )
    if ch_mult == 1 and i[0] == 1 and dil_ok:
        if not mve and f[1:3] == (3, 3) and pad[0] <= 1 and pad[1] <= 1 and dil == (1, 1):
            return "arm_depthwise_conv_3x3_s8"
        return "arm_depthwise_conv_s8_opt"
    return "arm_depthwise_conv_s8"


def inner_symbol(timed_symbol: str, manifest: Mapping, gate_1xn: Optional[bool]) -> Optional[str]:
    """The kernel a wrapper case runs, else None."""
    if timed_symbol not in (CONV_WRAPPER, DW_WRAPPER, FC_WRAPPER) or gate_1xn is None:
        return None
    if timed_symbol == FC_WRAPPER:
        # The adapter always passes per-channel quantization.
        return "arm_fully_connected_per_channel_s8"
    p = manifest["serialized_scalar_parameters"]
    # Firmware defaults: omitted or 0 stride/dilation is 1.
    stride = (p.get("stride_h") or 1, p.get("stride_w") or 1)
    dil = (p.get("dilation_h") or 1, p.get("dilation_w") or 1)
    pad = (p.get("pad_h", 0), p.get("pad_w", 0))
    o = (1, p.get("output_h", 0), p.get("output_w", 0), p.get("output_c", 0))
    dims = {blob["role"]: tuple(blob["dimensions"]) for blob in manifest["blob_roles"]}
    i = dims["input_0"]
    mve = get_cpu_profile(manifest["target_cpu"]).has_mve
    if timed_symbol == DW_WRAPPER:
        return dw_route(i, dims["weights"], o, stride, pad, dil, p.get("ch_mult", 0), mve, gate_1xn)
    # Conv weights blob is (h, w, c, n).
    kh, kw, fc, oc = dims["weights"]
    return conv_route(i, (oc, kh, kw, fc), o, stride, pad, dil, mve, gate_1xn)
