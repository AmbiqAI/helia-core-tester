"""Regression coverage for the unified tolerance/comparison policy.

Prior to this, generation/utils/template_context.py maintained its own
hardcoded per-template-path tolerance override tables (used only to drive
what the FVP standalone harness validates), while
generation/io/dtypes.py's resolve_comparison() -- which the hardware
perf-stream bridge reads via descriptor["resolved_comparison"] -- had no
knowledge of those overrides and defaulted every int8/int16 output to
exact_int (zero tolerance). The two paths could therefore validate the same
generated golden data under silently different rules (see
convolve_grouped_conv_case_01_s8: FVP tolerant +-1 LSB, hardware bridge
exact-match). These tests assert that default_int_tolerance() is now the
single source of truth consumed by both resolve_comparison() (hardware side)
and TemplateContextBuilder.infer_validation_tolerance() (FVP side), and that
migrating the previously-hardcoded per-template overrides onto this
operator-keyed table did not change any previously-documented tolerance
value.
"""

from helia_core_tester.generation.io.dtypes import default_int_tolerance, resolve_comparison
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
        ("Convolve", "S8", {"mode": "exact_int"}),
    ]
    for operator, output_dtype, expected in cases:
        desc = {"operator": operator, "resolved_tensor_dtypes": {"input": output_dtype, "output": output_dtype}}
        assert resolve_comparison(desc) == expected, (operator, output_dtype)


def test_fvp_validation_tolerance_derives_from_same_operator_table_as_hardware_manifest():
    # For every operator with an explicit override, the FVP-side tolerance
    # (infer_validation_tolerance) and the hardware manifest's resolved
    # tolerance (resolve_comparison) must agree -- eliminating the class of
    # silent divergence that let convolve_grouped_conv_case_01_s8 pass FVP
    # while the hardware bridge required an exact match for the same golden.
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
    # Baseline sanity: an operator with no declared override resolves to 0
    # (functionally exact), which is the safe default absent evidence of a
    # genuine hardware rounding divergence.
    assert default_int_tolerance("SomeBrandNewOperator", "S8") == 0
    assert default_int_tolerance("SomeBrandNewOperator", "S16") == 0
