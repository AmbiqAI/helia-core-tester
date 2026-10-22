"""An s16 output calibrated on a fused clamp that zeroes (nearly) every output must still be a layer TFLite's
int16 requantization reaches (shift <= 7, scaled accumulator inside int32).

`--coverage --seed 724439025` drew all-negative biases for depthwise_conv_relu_s16: RELU zeroed every output,
the calibration fell to the policy's 1e-6 floor, the multipliers came out at shift 12 and the C reference
refused the layer with E_PARAM before the flat-golden redraw could see the case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.golden_check import DegenerateGoldenError
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.ops._shared.conv_reference import (
    S16_MAX_SHIFT,
    depthwise_reference_case,
    float_bounds,
    output_quant,
    per_channel_multipliers,
)
from helia_core_tester.generation.ops.ConvolutionFunctions.depthwise_conv import OpDepthwiseConv
from helia_core_tester.generation.reference import policy
from helia_core_tester.generation.test_ops import default_seed_for_case, generate_test, redraw_seed

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAME = "depthwise_conv_relu_s16"
RUN_SEED = 724439025
CPU = "cortex-m55"


@pytest.fixture(scope="module")
def desc() -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == NAME)


def _within_contract(in_scale: float, out_scale: float, filter_scales: np.ndarray, unclamped: np.ndarray) -> None:
    _, shift = per_channel_multipliers(in_scale, out_scale, filter_scales)
    assert int(np.max(shift)) <= S16_MAX_SHIFT
    assert float(np.max(np.abs(unclamped))) / out_scale < 2.0 ** 31


def test_pinned_seed_draws_a_layer_the_reference_accepts(desc) -> None:
    op = OpDepthwiseConv(dict(desc), seed=default_seed_for_case(NAME, RUN_SEED), target_cpu=CPU)
    case = depthwise_reference_case(op, {"input_c_type": "int16_t"}, False, np.float32)
    shift = np.array([int(v) for v in case["quant_params"]["shift_array"].strip("{} \n").split(",") if v.strip()])
    assert shift.size == 4 and int(shift.max()) <= S16_MAX_SHIFT
    # Still the draw that failed: every output clamped to zero, which the flat-golden check redraws.
    assert not np.any(case["output"])


def test_pinned_seed_reaches_the_flat_golden_redraw_and_the_redraw_generates(desc, tmp_path) -> None:
    case_seed = default_seed_for_case(NAME, RUN_SEED)
    failures: list = []
    with pytest.raises(DegenerateGoldenError, match=NAME):
        generate_test(dict(desc), str(tmp_path / "first"), seed=case_seed, cpu=CPU, kernel_numerics="bounded",
                      generation_failures=failures, run_seed=RUN_SEED)
    assert "E_PARAM" not in " ".join(f.get("exception", "") for f in failures)
    generate_test(dict(desc), str(tmp_path / "redraw"), seed=redraw_seed(case_seed, 1), cpu=CPU,
                  kernel_numerics="bounded", generation_failures=[], run_seed=RUN_SEED)
    assert any((tmp_path / "redraw").rglob(f"{NAME}*.c"))


class _Op:
    """The two OperationBase methods output_quant reads, over a bare descriptor."""

    activation_name = OperationBase.activation_name
    activation_quant = OperationBase.activation_quant

    def __init__(self, activation: str = "RELU", **desc) -> None:
        self.desc = {"name": "calibration_probe", "activation": activation, **desc}


IN_SCALE = 1.0 / 32767.0
FILTER_SCALES = np.array([0.003, 0.002, 0.0029, 0.0031], dtype=np.float32)


def _clamped(preact: np.ndarray):
    return lambda lo, hi: np.clip(np.asarray(preact, dtype=np.float32), lo, hi)


@pytest.mark.parametrize("preact", [
    np.full(16, -3.0),                                # every output clamped to zero
    np.zeros(16),                                     # all-zero accumulators
    np.r_[np.full(15, -3.5), 1e-9],                   # one output barely above the clamp
    np.r_[np.full(15, -2.0e4), 0.0],                  # huge negative accumulators
], ids=["all_clamped", "all_zero", "near_degenerate", "large_unclamped"])
def test_degenerate_s16_clamp_widens_into_the_contract(preact) -> None:
    quant = output_quant(_Op(), "s16", _clamped(preact), IN_SCALE, FILTER_SCALES)
    assert quant.zero_point == 0
    _within_contract(IN_SCALE, quant.scale, FILTER_SCALES, preact)


@pytest.mark.parametrize("activation, preact", [
    ("RELU", np.linspace(-1.0, 2.0, 16)),
    ("NONE", np.linspace(-0.5, 0.25, 16)),
    ("RELU6", np.linspace(-4.0, 9.0, 16)),
])
def test_a_non_degenerate_s16_draw_calibrates_as_before(activation, preact) -> None:
    op = _Op(activation)
    real_output = _clamped(preact)
    expected = policy.activation_quant("s16", policy.data_range(real_output(*float_bounds(op))))
    assert output_quant(op, "s16", real_output, IN_SCALE, FILTER_SCALES) == expected


def test_s8_keeps_the_policy_floor() -> None:
    quant = output_quant(_Op(), "s8", _clamped(np.full(16, -3.0)), IN_SCALE, FILTER_SCALES)
    assert quant == policy.activation_quant("s8", (0.0, 0.0))


def test_a_pinned_output_quantization_is_not_widened() -> None:
    op = _Op(quantization={"output": {"scale": 1e-12, "zero_point": 0}})
    quant = output_quant(op, "s16", _clamped(np.full(16, -3.0)), IN_SCALE, FILTER_SCALES)
    assert quant.scale == float(np.float32(1e-12))


@pytest.mark.parametrize("bad", [np.array([]), np.array([np.nan, 1.0]), np.array([np.inf, 0.0])],
                         ids=["empty", "nan", "inf"])
def test_non_finite_or_empty_output_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="non-empty and finite"):
        output_quant(_Op("NONE"), "s16", lambda lo, hi: bad, IN_SCALE, FILTER_SCALES)


def test_zero_filter_scales_still_bound_by_the_accumulator() -> None:
    zeros = np.zeros(4, dtype=np.float32)
    quant = output_quant(_Op(), "s16", _clamped(np.full(16, -3.0)), IN_SCALE, zeros)
    assert quant.scale > 0 and 3.0 / quant.scale < 2.0 ** 31
