"""The route mirror against the real C wrappers."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware.wrapper_route import (
    CHANNELWISE, CONV_SOURCE, DW_OPT, PLANAR, conv_route, conv_s4_route, conv_s16_route, dw_opt_variant, dw_route,
    dw_s4_route, dw_s16_route, inner_symbol, inner_variant, opt_scratch_size, planar_bytes, tree_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CMSIS_NN_ROOT = Path(os.environ.get("CMSIS_NN_ROOT") or PROJECT_ROOT.parent.parent)
HARNESS = Path(__file__).parent / "fixtures" / "wrapper_route" / "route_harness.c"


def _rows(seed: int = 7, count: int = 20000) -> list[tuple]:
    """Layers near every branch boundary.

    Kinds: c/d s8, p/q s4, r/s s16 (conv/dw).
    """
    rng = random.Random(seed)
    pick = rng.choice
    rows = []
    for _ in range(count):
        kind = pick("cdcdcdpqrs")
        ic = pick((1, 2, 3, 4, 5, 8, 16, 16, 16, 24, 33))
        kh, kw = pick((1, 1, 3, 4, 7)), pick((1, 1, 2, 3, 3, 4, 5, 7, 8, 9, 16, 17))
        oc = pick((1, 2, 3, 4, 8, 16))
        ch_mult = pick((1, 1, 2, 4))
        i = (pick((1, 1, 1, 2)), pick((1, 1, 4)), pick((1, 4, 9, 40)), ic)
        if kind in "cpr":
            f = (oc, kh, kw, pick((ic, ic, max(ic // 2, 1), 1)))
        else:
            oc = pick((oc, ic, ic, ic * ch_mult))
            f = (1, kh, kw, oc)
        o = (i[0], pick((1, 1, 3, 9)), pick((1, 2, 5, 8, 16, 33, 64)), oc)
        stride = (pick((1, 1, 2)), pick((1, 1, 2, 3)))
        pad = (pick((0, 0, 0, 1, 2)), pick((0, 0, 0, 1, 2)))
        dil = (pick((1, 1, 1, 2)), pick((1, 1, 1, 2, 4)))
        rows.append((kind, i, f, o, stride, pad, dil, ch_mult))
    return rows + _EDGE_ROWS


# Rare branches the random rows miss.
_EDGE_ROWS = [
    # Planar plane past ctx: 1x1 kernel, 32x32.
    ("d", (1, 32, 32, 8), (1, 1, 1, 8), (1, 32, 32, 8), (1, 1), (0, 0), (1, 1), 1),
    # Planar 1xk dot path, dilated.
    ("d", (1, 1, 64, 8), (1, 1, 7, 8), (1, 1, 52, 8), (1, 1), (0, 0), (1, 2), 1),
    ("d", (1, 1, 96, 24), (1, 1, 5, 24), (1, 1, 96, 24), (1, 1), (0, 2), (1, 1), 1),
    # s4 1xN: SAME pad, stride 2.
    ("p", (1, 1, 15, 4), (8, 1, 3, 4), (1, 1, 8, 8), (1, 2), (0, 1), (1, 1), 1),
    ("p", (1, 1, 16, 4), (8, 1, 3, 4), (1, 1, 8, 8), (1, 2), (0, 0), (1, 1), 1),
]


def _expected(row: tuple, mve: bool, gate_1xn: bool = True) -> str:
    kind, i, f, o, stride, pad, dil, ch_mult = row
    if kind == "c":
        return conv_route(i, f, o, stride, pad, dil, mve, gate_1xn)
    if kind == "p":
        return conv_s4_route(i, f, o, stride, pad, dil, mve, gate_1xn)
    if kind == "r":
        return conv_s16_route(i, f, o, stride, pad, dil, mve)
    if kind == "q":
        return dw_s4_route(i, dil, ch_mult)
    if kind == "s":
        return dw_s16_route(i, f, o, stride, pad, dil, ch_mult)
    route = dw_route(i, f, o, stride, pad, dil, ch_mult, mve, gate_1xn)
    if route != DW_OPT:
        return route
    ctx_size = opt_scratch_size(i, f, o, stride, pad, dil, mve)
    return f"{route}+{dw_opt_variant(i, f, o, stride, pad, dil, ch_mult, mve, ctx_size)}"


@pytest.mark.parametrize("mve", [False, True], ids=["plain", "mve"])
def test_mirror_matches_c_wrappers(tmp_path: Path, mve: bool) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")
    if not (CMSIS_NN_ROOT / "Include" / "arm_nnfunctions.h").is_file():
        pytest.skip(f"no real ns-cmsis-nn checkout found at {CMSIS_NN_ROOT}")
    gate_1xn = tree_gate(CMSIS_NN_ROOT)
    assert gate_1xn is not None, "unknown 1xN gate; update GATES_1XN"
    binary = tmp_path / "route_harness"
    subprocess.run(
        [cc, "-std=c99", *(["-DHCT_ROUTE_MVE"] if mve else []), "-I", str(CMSIS_NN_ROOT / "Include"),
         "-I", str(CMSIS_NN_ROOT / "Source"), str(HARNESS), "-o", str(binary)],
        check=True,
    )
    rows = _rows()
    lines = [" ".join([kind, *map(str, (*i, *f, *o, *stride, *pad, *dil, cm))]) for kind, i, f, o, stride, pad, dil, cm in rows]
    lines.append("f " + " ".join(["1"] * 19))
    out = subprocess.run([str(binary)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    routes = out.stdout.split()
    assert len(routes) == len(lines)
    assert routes[-1] == "arm_fully_connected_per_channel_s8"
    mismatches = [
        (row, route) for row, route in zip(rows, routes) if route.removeprefix("to_conv:") != _expected(row, mve, gate_1xn)
    ]
    assert not mismatches, f"{len(mismatches)} rows differ, first: {mismatches[:3]}"
    seen = set(routes)
    wanted = {
        "arm_convolve_1x1_s8_fast", "arm_convolve_1x1_s8", "arm_convolve_1_x_n_s8", "arm_convolve_s8",
        f"{DW_OPT}+{CHANNELWISE}", "arm_depthwise_conv_s8",
        "arm_convolve_1x1_s4_fast", "arm_convolve_1x1_s4", "arm_convolve_1_x_n_s4", "arm_convolve_s4",
        "arm_depthwise_conv_s4_opt", "arm_depthwise_conv_s4", "arm_convolve_s16_group_ch_mult_1", "arm_convolve_s16",
        "arm_depthwise_conv_fast_s16", "arm_depthwise_conv_s16",
    }
    if mve:
        wanted |= {
            "arm_convolve_1x1_out_s8", "arm_convolve_s8_small_cin", "arm_convolve_s8_3x3_c16_s1", f"{DW_OPT}+{PLANAR}",
            "arm_convolve_even_s4", "arm_convolve_1x1_s16_ns_np_nd", "arm_convolve_s16_fast_small_kernel",
        }
        assert any(route.startswith("to_conv:") for route in routes)
        # Planar declines a plane past ctx.
        assert any(_ctx_fallback(row) for row in rows)
    else:
        wanted.add("arm_depthwise_conv_3x3_s8")
    assert wanted <= seen


def _ctx_fallback(row: tuple) -> bool:
    """A planar layer whose plane exceeds ctx."""
    kind, i, f, o, stride, pad, dil, ch_mult = row
    if kind != "d" or dw_route(i, f, o, stride, pad, dil, ch_mult, True) != DW_OPT:
        return False
    plane = planar_bytes(i, f, o, stride, pad, dil, ch_mult)
    return plane > opt_scratch_size(i, f, o, stride, pad, dil, True) > 0


def _manifest(cpu: str, weights: list[int], **params) -> dict:
    base = {"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0, "dilation_h": 1, "dilation_w": 1,
            "output_h": 8, "output_w": 8, "output_c": 16, "ch_mult": 1}
    scalars = {key: value for key, value in {**base, **params}.items() if value is not None}
    return {
        "target_cpu": cpu,
        "serialized_scalar_parameters": scalars,
        "blob_roles": [{"role": "input_0", "dimensions": [1, 8, 8, 16]}, {"role": "weights", "dimensions": weights}],
    }


def test_inner_symbol_reads_the_manifest() -> None:
    conv = "arm_convolve_wrapper_s8"
    # Conv weights blob is (h, w, c, n).
    m55 = _manifest("cortex-m55", [3, 3, 16, 16])
    assert inner_symbol(conv, m55, True) == "arm_convolve_s8_3x3_c16_s1"
    assert inner_symbol(conv, {**m55, "target_cpu": "cortex-m4"}, True) == "arm_convolve_s8"
    assert inner_symbol(conv, _manifest("cortex-m55", [1, 1, 16, 16], stride_w=2), True) == "arm_convolve_1x1_s8"
    # Depthwise weights blob is (n, h, w, c).
    dw = _manifest("cortex-m55", [1, 3, 3, 16], pad_h=1, pad_w=1)
    assert inner_symbol("arm_depthwise_conv_wrapper_s8", dw, True) == "arm_depthwise_conv_s8_opt"
    assert inner_symbol("arm_depthwise_conv_wrapper_s8", {**dw, "target_cpu": "cortex-m4"}, True) == (
        "arm_depthwise_conv_3x3_s8"
    )
    assert inner_symbol("arm_fully_connected_wrapper_s8", {}, True) == "arm_fully_connected_per_channel_s8"
    assert inner_symbol("arm_abs_s8", {}, True) is None
    # Unknown kernel tree: no route.
    assert inner_symbol(conv, m55, None) is None


def test_inner_symbol_applies_firmware_defaults() -> None:
    conv = "arm_convolve_wrapper_s8"
    fast = "arm_convolve_1x1_s8_fast"
    # The firmware runs omitted or 0 stride/dilation as 1.
    for value in (None, 0):
        unset = dict.fromkeys(("stride_h", "stride_w", "dilation_h", "dilation_w"), value)
        assert inner_symbol(conv, _manifest("cortex-m55", [1, 1, 16, 16], **unset), True) == fast


def test_old_tree_skips_the_1xn_padding_gate() -> None:
    # M4: v7.38.0 calls 1xN; v7.38.1 gates it out.
    args = ((1, 1, 9, 4), (8, 1, 3, 4), (1, 3, 7, 8), (1, 1), (0, 1), (1, 1), False)
    assert conv_route(*args, gate_1xn=False) == "arm_convolve_1_x_n_s8"
    assert conv_route(*args, gate_1xn=True) == "arm_convolve_s8"


def test_tree_gate_reads_the_wrapper(tmp_path: Path) -> None:
    source = tmp_path / CONV_SOURCE
    source.parent.mkdir(parents=True)
    small_cin = "if (arm_nn_is_convolve_s8_small_cin(conv_params, input_dims, filter_dims, output_dims, NULL))\n"
    gate = "    else if (arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims){})\n    {{\n"
    padding = " &&\n             arm_nn_convolve_1_x_n_s8_padding_supported(conv_params, filter_dims, output_dims)"
    for extra, expected in (("", False), (padding, True), (" && other(conv_params)", None)):
        source.write_text(gate.format(extra) + small_cin, encoding="utf-8")
        assert tree_gate(tmp_path) is expected
    # Trees before the small_cin route are unknown.
    source.write_text(gate.format(""), encoding="utf-8")
    assert tree_gate(tmp_path) is None
    assert tree_gate(tmp_path / "missing") is None and tree_gate(None) is None


def test_inner_symbol_names_s4_and_s16_routes() -> None:
    m55 = _manifest("cortex-m55", [3, 3, 16, 16])
    assert inner_symbol("arm_convolve_wrapper_s4", m55, True) == "arm_convolve_even_s4"
    assert inner_symbol("arm_convolve_wrapper_s4", {**m55, "target_cpu": "cortex-m4"}, True) == "arm_convolve_s4"
    assert inner_symbol("arm_convolve_wrapper_s16", m55, True) == "arm_convolve_s16"
    one = _manifest("cortex-m55", [1, 1, 16, 16])
    assert inner_symbol("arm_convolve_wrapper_s16", one, True) == "arm_convolve_1x1_s16_ns_np_nd"
    dw = _manifest("cortex-m55", [1, 3, 3, 16], pad_h=1, pad_w=1)
    assert inner_symbol("arm_depthwise_conv_wrapper_s4", dw, True) == "arm_depthwise_conv_s4_opt"
    assert inner_symbol("arm_depthwise_conv_wrapper_s16", dw, True) == "arm_depthwise_conv_fast_s16"
    assert inner_symbol("arm_depthwise_conv_wrapper_s16", _manifest("cortex-m55", [1, 3, 3, 16], ch_mult=2), True) == (
        "arm_depthwise_conv_s16"
    )
    # Variants exist only under the s8 opt route.
    assert inner_variant("arm_depthwise_conv_wrapper_s16", dw, True) is None


def test_inner_variant_splits_the_opt_route() -> None:
    dw = "arm_depthwise_conv_wrapper_s8"
    # C = 16 above 8 on a 8x8 plane: channelwise.
    assert inner_variant(dw, _manifest("cortex-m55", [1, 3, 3, 16], pad_h=1, pad_w=1), True) == CHANNELWISE
    eight = _manifest("cortex-m55", [1, 3, 3, 8], pad_h=1, pad_w=1, output_c=8)
    eight["blob_roles"][0]["dimensions"] = [1, 8, 8, 8]
    assert inner_variant(dw, eight, True) == PLANAR
    # Off MVE, 3x3 skips opt; other opt layers are channelwise.
    assert inner_variant(dw, {**eight, "target_cpu": "cortex-m4"}, True) is None
    five = _manifest("cortex-m4", [1, 5, 5, 8], pad_h=2, pad_w=2, output_c=8)
    five["blob_roles"][0]["dimensions"] = [1, 8, 8, 8]
    assert inner_variant(dw, five, True) == CHANNELWISE
    assert inner_variant(dw, {**five, "target_cpu": "cortex-m55"}, True) == PLANAR
    assert inner_variant("arm_convolve_wrapper_s8", eight, True) is None


def test_planar_declines_past_ctx() -> None:
    # 1x1 kernel, 32x32 plane: 1056 bytes > 496.
    args = ((1, 32, 32, 8), (1, 1, 1, 8), (1, 32, 32, 8), (1, 1), (0, 0), (1, 1))
    ctx_size = opt_scratch_size(*args, True)
    assert ctx_size == 496
    assert planar_bytes(*args, 1) == 32 * 32 + 32
    assert dw_opt_variant(*args, 1, True, ctx_size) == CHANNELWISE
    assert dw_opt_variant(*args, 1, True, 1 << 20) == CHANNELWISE
    small = ((1, 16, 16, 8), (1, 1, 1, 8), (1, 16, 16, 8), (1, 1), (0, 0), (1, 1))
    assert dw_opt_variant(*small, 1, True, opt_scratch_size(*small, True)) == PLANAR
    assert dw_opt_variant(*small, 1, True, 0) == CHANNELWISE
