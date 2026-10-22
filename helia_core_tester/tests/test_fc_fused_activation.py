"""FC fused RELU clamps at the output zero point."""

import pytest

from helia_core_tester.generation.ops._shared.conv_reference import quantized_bounds
from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import OpFullyConnected


@pytest.mark.parametrize(
    ("activation", "zero_point", "expected"),
    [
        ("RELU", -128, (-128, 127)),
        ("RELU", -5, (-5, 127)),
        ("RELU6", -128, (-128, -128 + 47)),
        ("NONE", -128, (-128, 127)),
    ],
)
def test_relu_clamp_uses_zero_point(activation, zero_point, expected):
    op = OpFullyConnected({"name": "fc_relu_s8", "operator": "FullyConnected", "activation": activation})
    assert quantized_bounds(op, "s8", 6.0 / 47, zero_point) == expected


def test_descriptor_bounds_narrow_the_fused_range_and_an_empty_one_fails():
    op = OpFullyConnected({"name": "fc_relu_s8", "operator": "FullyConnected", "activation": "RELU",
                           "activation_min": -3, "activation_max": 100})
    assert quantized_bounds(op, "s8", 0.1, -5) == (-3, 100)
    op = OpFullyConnected({"name": "fc_empty_s8", "operator": "FullyConnected", "activation": "RELU",
                           "activation_max": -10})
    with pytest.raises(ValueError, match="empty"):
        quantized_bounds(op, "s8", 0.1, -5)
