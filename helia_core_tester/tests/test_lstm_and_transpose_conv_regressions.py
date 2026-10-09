from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.utils import lstm_data
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
    descriptor (input_c=17, kernel 1x1, stride 2x2, VALID) and reflect the dims
    actually plugged into the harness at generation time: filter/output channel
    count is 17 because Keras' Conv2DTranspose infers the kernel's input-channel
    dimension from the real input tensor (17), not from the descriptor's nominal
    filter_shape InCh (5) -- confirmed by inspecting a generated harness for this
    descriptor.
    """
    desc = _load_transpose_conv_descriptor(
        "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8"
    )
    strides = desc["strides"]
    stride_h, stride_w = int(strides[0]), int(strides[1])
    in_n, in_h, in_w, in_c = (int(v) for v in desc["input_shape"])
    kh, kw, out_ch, _in_ch = (int(v) for v in desc["filter_shape"])

    assert desc["padding"].upper() == "VALID"
    assert (in_c, kh, kw, stride_h, stride_w) == (17, 1, 1, 2, 2)

    input_dims = {"n": in_n, "h": in_h, "w": in_w, "c": in_c}
    # Real output channel count generated for this descriptor (see docstring).
    filter_dims = {"n": out_ch, "h": kh, "w": kw, "c": in_c}
    output_dims = {"n": in_n, "h": 2, "w": 4, "c": in_c}

    buffer_size = TemplateContextBuilder.calculate_transpose_conv_buffer_size_max(
        input_dims=input_dims,
        filter_dims=filter_dims,
        output_dims=output_dims,
        output_dtype="S8",
        stride_h=stride_h,
        stride_w=stride_w,
    )

    # Old reverse-only sizing yields 128 (ctx) vs. an output_ctx of 289, so the
    # old function returns 289 for this shape -- comfortably short of the
    # rolling-buffer requirement of 544 introduced by PR#262.
    assert buffer_size >= 160
    assert buffer_size == 544


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


@pytest.mark.parametrize("dtype, time_major", [("S8", True), ("S16", False)])
def test_lstm_data_comes_from_the_tflm_reference(dtype: str, time_major: bool) -> None:
    data = lstm_data.generate_lstm_data(
        rng=np.random.default_rng(123), activation_dtype=dtype, batch_size=2, time_steps=5, input_size=6,
        hidden_size=4, time_major=time_major, input_zero_point_override=-9 if dtype == "S8" else None,
        output_zero_point_override=3 if dtype == "S8" else None,
    )
    call = data.reference_call
    assert call is not None and call.kernel == ("lstm_s8" if dtype == "S8" else "lstm_s16")
    lead = (5, 2) if time_major else (2, 5)
    assert data.tensors["input_tensor"].shape == lead + (6,)
    assert data.tensors["output"].shape == lead + (4,)
    assert np.unique(data.tensors["output"]).size > 4
    # The harness offset convention: INPUT_ZERO_POINT is -zp, OUTPUT_ZERO_POINT +zp.
    assert data.params["input_zero_point"] == (9 if dtype == "S8" else 0)
    assert data.params["output_zero_point"] == (3 if dtype == "S8" else 0)
    context = lstm_data.build_lstm_context(
        name="lstm_probe", dataset="lstm_probe", activation_dtype=dtype, batch_size=2, time_steps=5,
        input_size=6, hidden_size=4, time_major=time_major, data=data,
    )
    assert set(context["mult_shift"]) == {f"{k}_{s}" for k in lstm_data._MULTIPLIER_KEYS for s in ("multiplier", "shift")}
    # Distinct per-gate weight scales: a kernel that swapped two gates' multipliers would fail.
    gate_mults = {context["mult_shift"][f"{g}_gate_input_multiplier"] for g in ("input", "forget", "cell", "output")}
    assert len(gate_mults) == 4


def test_lstm_s16_refuses_zero_points() -> None:
    with pytest.raises(ValueError, match="s16"):
        lstm_data.generate_lstm_data(
            rng=np.random.default_rng(0), activation_dtype="S16", batch_size=1, time_steps=2, input_size=3,
            hidden_size=2, time_major=False, input_zero_point_override=-9,
        )
    with pytest.raises(ValueError, match="activation_dtype"):
        lstm_data.generate_lstm_data(
            rng=np.random.default_rng(0), activation_dtype="FP32", batch_size=1, time_steps=2, input_size=3,
            hidden_size=2, time_major=False,
        )
