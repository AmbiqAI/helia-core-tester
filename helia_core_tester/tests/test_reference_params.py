"""Parameter preparation through the shim's TFLM code, and the Python glue around it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference import params
from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift


@pytest.fixture(scope="module")
def lib() -> b.Bindings:
    return b.get_bindings()


def test_quantize_multiplier_matches_the_tester_port_on_random_scales(lib) -> None:
    # The tester's frexp + round-half-up port must agree with TFLM's
    # QuantizeMultiplier wherever TFLM does not flush the multiplier to zero.
    rng = np.random.default_rng(20261008)
    scales = np.exp(rng.uniform(np.log(1e-9), np.log(1e3), size=100_000))
    mismatches = []
    for scale in scales:
        ours = calculate_multiplier_shift(float(scale))
        ref = lib.quantize_multiplier(float(scale))
        if ours != ref:
            mismatches.append((float(scale), ours, ref))
    assert not mismatches[:5], f"{len(mismatches)} mismatches, first: {mismatches[:5]}"


@pytest.mark.parametrize(
    "scale, expected",
    [
        (0.0, (0, 0)),
        (0.5, (1 << 30, 0)),
        (1.0, (1 << 30, 1)),
        (0.25, (1 << 30, -1)),
        # Rounds up to 2^31: TFLM halves it and bumps the shift.
        (1.0 - 2.0**-33, (1 << 30, 1)),
    ],
)
def test_quantize_multiplier_edges(lib, scale, expected) -> None:
    assert lib.quantize_multiplier(scale) == expected


def test_quantize_multiplier_flushes_tiny_scales_to_zero(lib) -> None:
    # shift < -31 cannot be represented; TFLM returns (0, 0) and so must we.
    assert lib.quantize_multiplier(2.0**-40) == (0, 0)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_quantize_multiplier_rejects_invalid(lib, bad) -> None:
    with pytest.raises(b.ReferenceKernelError) as info:
        lib.quantize_multiplier(bad)
    assert info.value.code == b.E_PARAM


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_smaller_than_one_rejects_out_of_range(lib, bad) -> None:
    with pytest.raises(b.ReferenceKernelError):
        lib.quantize_multiplier_smaller_than_one_exp(bad)


def test_smaller_than_one_shift_is_non_positive(lib) -> None:
    m, s = lib.quantize_multiplier_smaller_than_one_exp(0.3)
    assert s <= 0 and (1 << 30) <= m < (1 << 31)


@pytest.mark.parametrize(
    "activation, scale, zp, expected",
    [
        ("NONE", 0.1, 3, (-128, 127)),
        ("RELU", 0.1, -5, (-5, 127)),
        ("RELU6", 0.05, -128, (-128, -8)),
        ("RELU6", 0.01, 0, (0, 127)),  # 6 / 0.01 = 600 clamps to qmax
        ("RELU_N1_TO_1", 0.02, 10, (-40, 60)),
        ("RELU", 0.1, -200, (-128, 127)),  # zp below qmin clamps
    ],
)
def test_activation_range(activation, scale, zp, expected) -> None:
    assert params.activation_range(activation, scale, zp, "s8") == expected


def test_activation_range_rounds_half_away_from_zero(lib) -> None:
    # 6 / 4.0 = 1.5 exactly: TfLiteRound gives 2, banker's rounding would give 2,
    # 0.5 / 1 -> 1 (away from zero) vs 0 (to even) is the discriminating case.
    assert lib.activation_range_quantized(b.ACT_RELU6, 4.0, 0, -128, 127) == (0, 2)
    assert lib.activation_range_quantized(b.ACT_RELU_N1_TO_1, 2.0, 0, -128, 127) == (-1, 1)


def test_activation_range_s16() -> None:
    assert params.activation_range("RELU", 1 / 4096, 0, "s16") == (0, 32767)
    assert params.activation_range("RELU6", 1 / 4096, 0, "s16") == (0, 24576)


def test_activation_range_rejects_bad_input(lib) -> None:
    with pytest.raises(ValueError):
        params.activation_range("TANH", 0.1, 0, "s8")
    with pytest.raises(ValueError):
        params.activation_range("RELU", 0.0, 0, "s8")
    with pytest.raises(b.ReferenceKernelError) as info:
        lib.activation_range_quantized(99, 0.1, 0, -128, 127)
    assert info.value.code == b.E_UNSUPPORTED
    with pytest.raises(b.ReferenceKernelError):
        lib.activation_range_quantized(b.ACT_RELU, 0.1, 0, 127, -128)


def test_per_channel_forms_the_scale_in_double_from_float32_scales() -> None:
    in_s, out_s = 0.0123, 0.0456
    w = [0.001, 0.0021, 0.00057]
    quant = params.per_channel(in_s, w, out_s)
    for c, w_s in enumerate(w):
        effective = float(np.float32(in_s)) * float(np.float32(w_s)) / float(np.float32(out_s))
        assert (int(quant.multiplier[c]), int(quant.shift[c])) == calculate_multiplier_shift(effective)


def test_per_channel_rejects_bad_scales() -> None:
    with pytest.raises(ValueError):
        params.per_channel(0.1, [], 0.1)
    with pytest.raises(ValueError):
        params.per_channel(0.1, [0.1, 0.0], 0.1)
    with pytest.raises(ValueError):
        params.per_channel(-0.1, [0.1], 0.1)


def test_addsub_params_follow_calculate_op_data_add() -> None:
    p = params.addsub_params("s8", 0.02, -3, 0.05, 7, 0.06, 2)
    twice_max = 2.0 * max(float(np.float32(0.02)), float(np.float32(0.05)))
    assert p.left_shift == 20
    assert (p.input1_offset, p.input2_offset, p.output_offset) == (3, -7, 2)
    lib = b.get_bindings()
    assert (p.input1_multiplier, p.input1_shift) == lib.quantize_multiplier_smaller_than_one_exp(float(np.float32(0.02)) / twice_max)
    assert (p.output_multiplier, p.output_shift) == lib.quantize_multiplier_smaller_than_one_exp(
        twice_max / ((1 << 20) * float(np.float32(0.06)))
    )
    assert params.addsub_params("s16", 0.02, 0, 0.05, 0, 0.06, 0).left_shift == 15
    with pytest.raises(ValueError):
        params.addsub_params("s4", 0.02, 0, 0.05, 0, 0.06, 0)


def test_mul_and_mean_params() -> None:
    lib = b.get_bindings()
    s1, s2, so = (float(np.float32(v)) for v in (0.02, 0.03, 0.001))
    assert params.mul_params(0.02, 0.03, 0.001) == lib.quantize_multiplier(s1 * s2 / so)
    assert params.mean_params(0.02, 0.03) == lib.quantize_multiplier(float(np.float32(0.02)) / float(np.float32(0.03)))


def test_softmax_params_s8_matches_tflm_shape() -> None:
    p = params.softmax_params_s8(1 / 16)
    assert p.input_multiplier > 0 and p.diff_min is not None and p.diff_min < 0
    # diff_min is the negated input radius at 5 integer bits.
    radius = b.get_bindings().calculate_input_radius(5, p.input_left_shift, 31)
    assert p.diff_min == -radius


def test_softmax_params_s16_rescale() -> None:
    p = params.softmax_params_s16(1 / 4096)
    expected = b.get_bindings().quantize_multiplier(float(np.float32(1 / 4096)) * 1.0 / (10.0 / 65535.0))
    assert (p.input_multiplier, p.input_left_shift) == expected and p.diff_min is None


def test_downscale_multiplier_to_s16() -> None:
    lib = b.get_bindings()
    assert lib.downscale_multiplier_to_s16(1 << 30) == 1 << 14
    assert lib.downscale_multiplier_to_s16(0x7FFFFFFF) == 0x7FFF
    with pytest.raises(b.ReferenceKernelError):
        lib.downscale_multiplier_to_s16(-1)


@pytest.mark.parametrize(
    "padding, in_size, k, stride, dilation, out, pad, offset",
    [
        ("SAME", 5, 3, 1, 1, 5, 1, 0),
        ("SAME", 6, 3, 2, 1, 3, 0, 1),
        ("SAME", 7, 3, 2, 2, 4, 2, 0),
        ("VALID", 7, 3, 2, 1, 3, 0, 0),
        ("VALID", 9, 3, 1, 3, 3, 0, 0),
        ("SAME", 5, 1, 3, 1, 2, 0, 0),
    ],
)
def test_conv_geometry(padding, in_size, k, stride, dilation, out, pad, offset) -> None:
    (oh, ow), ph, pw = params.conv_geometry(padding, (in_size, in_size), (k, k), (stride, stride), (dilation, dilation))
    assert (oh, ow) == (out, out)
    assert (ph.pad, ph.offset) == (pad, offset) == (pw.pad, pw.offset)


def test_conv_geometry_rejects_impossible_windows() -> None:
    with pytest.raises(ValueError):
        params.conv_geometry("VALID", (2, 2), (3, 3), (1, 1))
    with pytest.raises(ValueError):
        params.conv_geometry("FULL", (4, 4), (3, 3), (1, 1))
    with pytest.raises(ValueError):
        params.out_size("SAME", 4, 3, 0)


def test_tconv_padding_swaps_input_and_output() -> None:
    ph, pw = params.tconv_padding("SAME", (8, 8), (3, 3), (2, 2))
    assert (ph.pad, ph.offset) == (0, 1)
    ph, pw = params.tconv_padding("VALID", (9, 9), (3, 3), (2, 2))
    assert (ph.pad, ph.offset) == (0, 0)


def test_float_activation_range() -> None:
    assert params.float_activation_range("RELU6") == (0.0, 6.0)
    assert params.float_activation_range("none") == (-math.inf, math.inf)
    with pytest.raises(ValueError):
        params.float_activation_range("GELU")
