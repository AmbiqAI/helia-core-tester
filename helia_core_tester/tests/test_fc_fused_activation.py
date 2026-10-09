"""FC fused RELU clamps at the output zero point (TFLite's CalculateActivationRangeQuantized)."""

import pytest

from helia_core_tester.generation.reference import params


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
    assert params.activation_range(activation, 6.0 / 47, zero_point, "s8") == expected


def test_fc_case_clamps_with_the_reference_range(tmp_path):
    from helia_core_tester.generation.ops.FullyConnectedFunctions.fully_connected import OpFullyConnected

    desc = {"name": "fc_relu6_s8", "operator": "FullyConnected", "activation": "RELU6", "activation_dtype": "S8",
            "weight_dtype": "S8", "input_shape": [2, 12], "filter_shape": [5, 12]}
    op = OpFullyConnected(desc, seed=4, target_cpu="cortex-m4")
    op.generate_c_files(tmp_path)
    act = op.reference.params["act"]
    out = op.reference.quant["output"]
    assert (act["min"], act["max"]) == params.activation_range("RELU6", out.scale, out.zero_point, "s8")
    assert op.golden().min() >= act["min"] and op.golden().max() <= act["max"]
