"""The route mirror against the real C wrappers."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware.wrapper_route import CONV_SOURCE, conv_route, dw_route, inner_symbol, tree_gate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CMSIS_NN_ROOT = Path(os.environ.get("CMSIS_NN_ROOT") or PROJECT_ROOT.parent.parent)
HARNESS = Path(__file__).parent / "fixtures" / "wrapper_route" / "route_harness.c"


def _rows(seed: int = 7, count: int = 20000) -> list[tuple]:
    """Layers near every branch boundary."""
    rng = random.Random(seed)
    pick = rng.choice
    rows = []
    for _ in range(count):
        kind = pick("cd")
        ic = pick((1, 2, 3, 4, 5, 8, 16, 16, 16))
        kh, kw = pick((1, 1, 3, 4, 7)), pick((1, 1, 2, 3, 3, 4, 5, 7, 8, 9, 16, 17))
        oc = pick((1, 2, 3, 4, 8, 16))
        i = (pick((1, 1, 1, 2)), pick((1, 1, 4)), pick((1, 4, 9)), ic)
        if kind == "c":
            f = (oc, kh, kw, pick((ic, ic, max(ic // 2, 1))))
        else:
            f = (1, kh, kw, oc)
        o = (i[0], pick((1, 1, 3)), pick((1, 2, 5)), oc)
        stride = (pick((1, 2)), pick((1, 1, 2, 3)))
        pad = (pick((0, 0, 0, 1, 2)), pick((0, 0, 0, 1, 2)))
        dil = (pick((1, 1, 1, 2)), pick((1, 1, 1, 2)))
        rows.append((kind, i, f, o, stride, pad, dil, pick((1, 1, 2, 4))))
    return rows


def _expected(row: tuple, mve: bool, gate_1xn: bool = True) -> str:
    kind, i, f, o, stride, pad, dil, ch_mult = row
    if kind == "c":
        return conv_route(i, f, o, stride, pad, dil, mve, gate_1xn)
    return dw_route(i, f, o, stride, pad, dil, ch_mult, mve, gate_1xn)


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
        "arm_depthwise_conv_s8_opt", "arm_depthwise_conv_s8",
    }
    if mve:
        wanted |= {"arm_convolve_1x1_out_s8", "arm_convolve_s8_small_cin", "arm_convolve_s8_3x3_c16_s1"}
        assert any(route.startswith("to_conv:") for route in routes)
    else:
        wanted.add("arm_depthwise_conv_3x3_s8")
    assert wanted <= seen


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


def test_pinned_tree_skips_the_1xn_padding_gate() -> None:
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
