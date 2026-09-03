"""Regression coverage for the unified tolerance/comparison policy: asserts
that default_int_tolerance() is the single source of truth consumed by both
resolve_comparison() (hardware side) and
TemplateContextBuilder.infer_validation_tolerance() (FVP side).
"""

from helia_core_tester.generation.io.dtypes import default_int_tolerance, has_int_tolerance_override, resolve_comparison
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def test_resolve_comparison_preserves_prior_per_operator_overrides():
    cases = [
        ("PReLU", "S8", {"mode": "tolerant_int", "tolerance": 2}),
        ("LeakyRelu", "S8", {"mode": "tolerant_int", "tolerance": 1}),
        ("HardSwishCompat", "S8", {"mode": "tolerant_int", "tolerance": 1}),
        ("DepthwiseConv", "S8", {"mode": "exact_int"}),
        ("Abs", "S16", {"mode": "tolerant_int", "tolerance": 2}),
        ("Abs", "S8", {"mode": "exact_int"}),
        ("Add", "S16", {"mode": "tolerant_int", "tolerance": 3}),
        ("SquaredDifference", "S16", {"mode": "tolerant_int", "tolerance": 3}),
        ("Convolve", "S8", {"mode": "tolerant_int", "tolerance": 1}),
        ("Reshape", "S8", {"mode": "exact_int"}),
        ("Concatenation", "S8", {"mode": "exact_int"}),
        ("Split", "S8", {"mode": "exact_int"}),
        ("Pad", "S8", {"mode": "exact_int"}),
        ("Transpose", "S8", {"mode": "exact_int"}),
        ("StridedSlice", "S8", {"mode": "exact_int"}),
        ("Squeeze", "S8", {"mode": "exact_int"}),
        ("SpaceToDepth", "S8", {"mode": "exact_int"}),
        ("BatchToSpaceND", "S8", {"mode": "exact_int"}),
        ("SpaceToBatchND", "S8", {"mode": "exact_int"}),
    ]
    for operator, output_dtype, expected in cases:
        desc = {"operator": operator, "resolved_tensor_dtypes": {"input": output_dtype, "output": output_dtype}}
        assert resolve_comparison(desc) == expected, (operator, output_dtype)


def test_fvp_validation_tolerance_derives_from_same_operator_table_as_hardware_manifest():
    # FVP tolerance (infer_validation_tolerance) and hardware manifest
    # tolerance (resolve_comparison) must agree for every explicit override.
    overridden_operators = [
        ("PReLU", "S8", "int8_t"),
        ("LeakyRelu", "S8", "int8_t"),
        ("HardSwishCompat", "S8", "int8_t"),
        ("DepthwiseConv", "S8", "int8_t"),
        ("Abs", "S16", "int16_t"),
        ("Add", "S16", "int16_t"),
        ("SquaredDifference", "S16", "int16_t"),
    ]
    for operator, dtype, c_type in overridden_operators:
        desc = {"operator": operator, "resolved_tensor_dtypes": {"input": dtype, "output": dtype}}
        hardware_comparison = resolve_comparison(desc)
        hardware_tolerance = hardware_comparison.get("tolerance", 0)

        fvp_tolerance = TemplateContextBuilder.infer_validation_tolerance(
            "irrelevant/for/this/check.c.j2", {"output_dtype": c_type}, "tolerant_int", desc
        )
        assert fvp_tolerance == hardware_tolerance, (operator, dtype)


def test_default_int_tolerance_has_no_override_for_unlisted_operators():
    # An unlisted operator defaults to 0 (functionally exact).
    assert default_int_tolerance("SomeBrandNewOperator", "S8") == 0
    assert default_int_tolerance("SomeBrandNewOperator", "S16") == 0


# Operators from dtypes.py's KNOWN GAP comment that have been audited and now
# have an explicit _OPERATOR_TOLERANCE_OVERRIDES entry.
_AUDITED_KNOWN_GAP_OPERATORS = (
    "Convolve",
    "Reshape",
    "Concatenation",
    "Split",
    "Pad",
    "Transpose",
    "StridedSlice",
    "Squeeze",
    "SpaceToDepth",
    "BatchToSpaceND",
    "SpaceToBatchND",
    "TransposeConv",
    "FullyConnected",
    "BatchMatMul",
    "AvgPool",
    "MaxPool",
    "Softmax",
    "Quantize",
    "Logistic",
    "Relu",
    "Relu6",
    "Tanh",
    "HardSwish",
    "Mean",
    "MinMax",
    "ReduceMax",
    "ReduceMin",
    "Sub",
)

# Operators from the same KNOWN GAP list still NOT audited. When one gets
# audited, move it to _AUDITED_KNOWN_GAP_OPERATORS and update dtypes.py.
_STILL_UNAUDITED_KNOWN_GAP_OPERATORS: tuple[str, ...] = ()


def test_audited_known_gap_operators_all_have_explicit_overrides():
    for operator in _AUDITED_KNOWN_GAP_OPERATORS:
        assert has_int_tolerance_override(operator, "S8"), operator


def test_still_unaudited_known_gap_operators_have_no_explicit_override():
    for operator in _STILL_UNAUDITED_KNOWN_GAP_OPERATORS:
        assert not has_int_tolerance_override(operator, "S8"), operator
