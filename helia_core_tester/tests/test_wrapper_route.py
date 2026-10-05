"""The route mirror against the real C wrappers."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware.wrapper_route import conv_route, dw_route, inner_symbol

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


def _expected(row: tuple, mve: bool) -> str:
    kind, i, f, o, stride, pad, dil, ch_mult = row
    if kind == "c":
        return conv_route(i, f, o, stride, pad, dil, mve)
    return dw_route(i, f, o, stride, pad, dil, ch_mult, mve)


@pytest.mark.parametrize("mve", [False, True], ids=["plain", "mve"])
def test_mirror_matches_c_wrappers(tmp_path: Path, mve: bool) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")
    if not (CMSIS_NN_ROOT / "Include" / "arm_nnfunctions.h").is_file():
        pytest.skip(f"no real ns-cmsis-nn checkout found at {CMSIS_NN_ROOT}")
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
        (row, route) for row, route in zip(rows, routes) if route.removeprefix("to_conv:") != _expected(row, mve)
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
    return {
        "target_cpu": cpu,
        "serialized_scalar_parameters": {**base, **params},
        "blob_roles": [{"role": "input_0", "dimensions": [1, 8, 8, 16]}, {"role": "weights", "dimensions": weights}],
    }


def test_inner_symbol_reads_the_manifest() -> None:
    # Conv weights blob is (h, w, c, n).
    conv = _manifest("cortex-m55", [3, 3, 16, 16])
    assert inner_symbol("arm_convolve_wrapper_s8", conv) == "arm_convolve_s8_3x3_c16_s1"
    assert inner_symbol("arm_convolve_wrapper_s8", {**conv, "target_cpu": "cortex-m4"}) == "arm_convolve_s8"
    assert inner_symbol("arm_convolve_wrapper_s8", _manifest("cortex-m55", [1, 1, 16, 16], stride_w=2)) == (
        "arm_convolve_1x1_s8"
    )
    # Depthwise weights blob is (n, h, w, c).
    dw = _manifest("cortex-m55", [1, 3, 3, 16], pad_h=1, pad_w=1)
    assert inner_symbol("arm_depthwise_conv_wrapper_s8", dw) == "arm_depthwise_conv_s8_opt"
    assert inner_symbol("arm_depthwise_conv_wrapper_s8", {**dw, "target_cpu": "cortex-m4"}) == (
        "arm_depthwise_conv_3x3_s8"
    )
    assert inner_symbol("arm_fully_connected_wrapper_s8", {}) == "arm_fully_connected_per_channel_s8"
    assert inner_symbol("arm_abs_s8", {}) is None
