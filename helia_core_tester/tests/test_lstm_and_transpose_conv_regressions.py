from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def test_transpose_conv_s8_non_reverse_buffer_covers_mve_stride_h_width_term() -> None:
    buffer_size = TemplateContextBuilder.calculate_transpose_conv_buffer_size_max(
        input_dims={"n": 1, "h": 3, "w": 3, "c": 32},
        filter_dims={"n": 32, "h": 2, "w": 2, "c": 8},
        output_dims={"n": 1, "h": 8, "w": 4, "c": 32},
        output_dtype="S8",
        stride_h=3,
        stride_w=1,
    )

    assert buffer_size == 1920


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_transpose_conv_descriptor(name: str) -> dict:
    desc_path = _repo_root() / "assets" / "descriptors" / "ConvolutionFunctions" / "transpose_conv.yaml"
    for desc in load_descriptor(str(desc_path)):
        if desc.get("name") == name:
            return desc
    raise AssertionError(f"descriptor {name!r} not found in {desc_path}")


def test_transpose_conv_s8_reverse_buffer_covers_rolling_buffer_after_ns_cmsis_nn_262() -> None:
    """Regression for ns-cmsis-nn#261 / PR#262.

    arm_transpose_conv_s8_get_buffer_size (and the _mve variant) now returns
    MAX(reverse-conv size, rolling-buffer size) whenever the reverse-conv route
    is taken (stride_w<=2 && stride_h<=2 && input_c>16), because other direct
    callers of arm_transpose_conv_s8 still need the rolling-buffer sizing. The
    tester's static scratch buffer must stay an upper bound on that new max.

    Dims below are read from the transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8
    descriptor (input_c=17, kernel 1x1, stride 2x2, VALID, filter [1, 1, 5, 17]: 17 -> 5
    channels). Generation uses all four declared filter dims (issue #144). The parametrized
    test below still covers the original 17 -> 17 shape on which PR#262 was found.
    """
    desc = _load_transpose_conv_descriptor(
        "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8"
    )
    strides = desc["strides"]
    stride_h, stride_w = int(strides[0]), int(strides[1])
    in_n, in_h, in_w, in_c = (int(v) for v in desc["input_shape"])
    kh, kw, out_ch, in_ch = (int(v) for v in desc["filter_shape"])

    assert desc["padding"].upper() == "VALID"
    # Still the reverse-conv route: stride <= 2 and input_c > 16.
    assert (in_c, in_ch, out_ch, kh, kw, stride_h, stride_w) == (17, 17, 5, 1, 1, 2, 2)

    input_dims = {"n": in_n, "h": in_h, "w": in_w, "c": in_c}
    filter_dims = {"n": out_ch, "h": kh, "w": kw, "c": in_ch}
    output_dims = {"n": in_n, "h": 2, "w": 4, "c": out_ch}

    buffer_size = TemplateContextBuilder.calculate_transpose_conv_buffer_size_max(
        input_dims=input_dims,
        filter_dims=filter_dims,
        output_dims=output_dims,
        output_dtype="S8",
        stride_h=stride_h,
        stride_w=stride_w,
    )

    # Rolling buffer: ((w - 1) * stride_w + max(kw, stride)) * out_ch * max(kh, stride_h) * 4
    # = (1 * 2 + 2) * 5 * 2 * 4 = 160 bytes, the floor PR#262 added.
    assert buffer_size >= 160
    assert buffer_size == 160


@pytest.mark.parametrize(
    ("input_dims", "filter_dims", "output_dims", "stride_h", "stride_w"),
    (
        (
            {"n": 1, "h": 1, "w": 2, "c": 17},
            {"n": 17, "h": 1, "w": 1, "c": 17},
            {"n": 1, "h": 2, "w": 4, "c": 17},
            2,
            2,
        ),
        (
            {"n": 1, "h": 2, "w": 2, "c": 20},
            {"n": 20, "h": 1, "w": 1, "c": 20},
            {"n": 1, "h": 4, "w": 4, "c": 20},
            2,
            2,
        ),
        (
            {"n": 1, "h": 1, "w": 4, "c": 24},
            {"n": 24, "h": 1, "w": 1, "c": 24},
            {"n": 1, "h": 2, "w": 8, "c": 24},
            2,
            1,
        ),
    ),
)
def test_transpose_conv_s8_reverse_buffer_covers_rolling_formula(
    input_dims: dict,
    filter_dims: dict,
    output_dims: dict,
    stride_h: int,
    stride_w: int,
) -> None:
    """calculate_transpose_conv_buffer_size_max must bound the rolling-buffer
    formula whenever the reverse-conv route is possible+efficient (mirrors
    ns-cmsis-nn PR#262's MAX(reverse, rolling) change)."""
    output_c = output_dims["c"]
    filter_w = filter_dims["w"]
    filter_h = filter_dims["h"]

    buf_x = ((input_dims["w"] - 1) * stride_w + max(filter_w, stride_w)) * output_c
    buf_x_mve = ((input_dims["w"] - 1) * stride_w + max(filter_w, stride_h)) * output_c
    buf_y = max(filter_h, stride_h)
    expected_rolling = max(buf_x, buf_x_mve) * buf_y * 4

    buffer_size = TemplateContextBuilder.calculate_transpose_conv_buffer_size_max(
        input_dims=input_dims,
        filter_dims=filter_dims,
        output_dims=output_dims,
        output_dtype="S8",
        stride_h=stride_h,
        stride_w=stride_w,
    )

    assert buffer_size >= expected_rolling
