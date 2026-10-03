"""The s16 depthwise fast-path cases must stay inside ns-cmsis-nn's direct Helium gates.

arm_depthwise_conv_fast_s16 hands a 3x3 layer to arm_depthwise_conv_s16_opt_3x3 when batch is 1,
ch_mult is 1, C >= 8 with C % 4 == 0, stride and padding are at most 2 and 1, the output has at
least 3 rows and 16 pixels, and the last window starts inside the input; a 1xK layer at stride 1
without dilation goes to arm_depthwise_conv_s16_opt_planar. These cases exist only to reach those
paths, so a shape edited outside a gate would silently drop the coverage they were added for.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.io.dtypes import resolve_comparison
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CASES = [
    desc
    for desc in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors"))
    if desc["name"].startswith(("depthwise_conv_fast_3x3_", "depthwise_conv_fast_planar_"))
]


def _same_geometry(in_size: int, kernel: int, stride: int) -> tuple[int, int]:
    out = math.ceil(in_size / stride)
    pad_total = max((out - 1) * stride + kernel - in_size, 0)
    return out, pad_total // 2


def _geometry(desc: dict) -> tuple[int, int, int, int]:
    _, in_h, in_w, _ = desc["input_shape"]
    k_h, k_w, _, _ = desc["filter_shape"]
    s_h, s_w = desc["strides"]
    if desc["padding"] == "SAME":
        (out_h, pad_h), (out_w, pad_w) = _same_geometry(in_h, k_h, s_h), _same_geometry(in_w, k_w, s_w)
    else:
        out_h, out_w, pad_h, pad_w = (in_h - k_h) // s_h + 1, (in_w - k_w) // s_w + 1, 0, 0
    return out_h, out_w, pad_h, pad_w


@pytest.mark.parametrize("desc", CASES, ids=lambda desc: desc["name"])
def test_case_reaches_a_direct_helium_path(desc: dict) -> None:
    assert desc["operator"] == "DepthwiseConv"
    assert (desc["activation_dtype"], desc["weight_dtype"]) == ("S16", "S8")
    assert resolve_comparison(desc) == {"mode": "exact_int"}
    batch, in_h, in_w, ch = desc["input_shape"]
    k_h, k_w, filter_ch, ch_mult = desc["filter_shape"]
    assert batch == 1 and ch_mult == 1 and desc["depth_multiplier"] == 1 and filter_ch == ch
    assert desc["dilation"] == [1, 1]
    out_h, out_w, pad_h, pad_w = _geometry(desc)
    s_h, s_w = desc["strides"]
    if "_planar_" in desc["name"]:
        assert k_h == 1 and k_w > 1 and (s_h, s_w) == (1, 1)
    else:
        assert (k_h, k_w) == (3, 3)
        assert ch >= 8 and ch % 4 == 0
        assert 1 <= s_h <= 2 and 1 <= s_w <= 2 and pad_h <= 1 and pad_w <= 1
        assert out_h >= 3 and out_w * out_h >= 16
        assert (out_w - 1) * s_w - pad_w + 1 < in_w


def test_cases_cover_the_requested_variants() -> None:
    names = {desc["name"] for desc in CASES}
    assert len(CASES) == 14
    for desc in CASES:
        twin = desc["name"].replace("_no_bias_", "_bias_") if not desc["use_bias"] else desc["name"].replace("_bias_", "_no_bias_")
        assert twin in names
    three_by_three = [desc for desc in CASES if "_3x3_" in desc["name"]]
    assert {tuple(desc["strides"]) for desc in three_by_three} >= {(1, 1), (2, 2), (1, 2)}
    assert {desc["padding"] for desc in three_by_three} == {"SAME", "VALID"}
    assert any(desc["input_shape"][3] == 8 for desc in three_by_three)  # smallest gated channel count
    assert any(desc["input_shape"][3] > 64 for desc in three_by_three)  # more than one 64-channel pass
    # The second 64-channel pass also over padded edges, and a stride-2 window overhanging only the far side.
    assert any(desc["input_shape"][3] > 64 and desc["padding"] == "SAME" for desc in three_by_three)
    assert any(
        desc["strides"] == [2, 2] and _geometry(desc)[2:] == (0, 0) and desc["padding"] == "SAME"
        for desc in three_by_three
    )


def _header_array(header: str, name: str) -> list[int] | None:
    found = re.search(rf"\b{re.escape(name)}\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}", header)
    return None if found is None else [int(value, 0) for value in found.group(1).replace("LL", "").split(",") if value.strip()]


@pytest.mark.parametrize("desc", CASES, ids=lambda desc: desc["name"])
def test_generated_quantization_is_inside_the_direct_paths_range(tmp_path: Path, desc: dict) -> None:
    # Both paths also decline, falling back to im2col with identical output, when a shift is
    # outside [-31, 7] or a bias could overflow the int32 accumulator (arm_nn_dw_s16_quant_ok).
    generate_test(desc, str(tmp_path), seed=Config.seed)
    (header_path,) = sorted(tmp_path.rglob(f"{desc['name']}*.h"))
    header = header_path.read_text()
    shifts = _header_array(header, f"{desc['name']}_shift")
    assert shifts and all(-31 <= shift <= 7 for shift in shifts)
    biases = _header_array(header, f"{desc['name']}_biases")
    if desc["use_bias"]:
        k_h, k_w, _, _ = desc["filter_shape"]
        limit = 2**31 - 1 - k_h * k_w * (1 << 22)
        assert biases and all(abs(bias) <= limit for bias in biases)

