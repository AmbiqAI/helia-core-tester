"""FC fused RELU clamps at the output zero point."""

import numpy as np
import pytest

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
    quant = {"scale": 6.0 / 47, "zero_point": zero_point}
    assert op._compute_activation_range(quant, np.dtype(np.int8)) == expected
