from __future__ import annotations

from pathlib import Path

import pytest

import helia_core_tester.reporting.models as reporting_models
import helia_core_tester.reporting.parser as reporting_parser


def test_parser_prefers_zero_failures_summary() -> None:
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="0 Failures\n",
        elf_path=Path("relu.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.PASS
    assert result.failure_reason is None


def test_parser_parses_nonzero_failures_summary() -> None:
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="Mismatch[3]: exp=7 got=9 (diff=2)\n2 Failures\n",
        elf_path=Path("relu.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.FAIL
    assert result.failure_reason == "Output mismatch: 2 element(s) differ from expected"
    assert result.error_type == "output_mismatch"


def test_parser_parses_generic_api_error_before_summary() -> None:
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="ReLU failed with status -1\n1 Failures\n",
        elf_path=Path("relu.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.FAIL
    assert result.failure_reason == "ReLU API error (status -1)"
    assert result.error_type == "api_error"


def test_parser_keeps_legacy_unity_fallback() -> None:
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="generated.c:test__relu_smoke:PASS\n",
        elf_path=Path("relu.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.PASS


def test_parser_extracts_float_maxdiff_headroom() -> None:
    # issue #53: HELIA_FLOAT_MAXDIFF is the mechanical headroom-measurement hook.
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="HELIA_FLOAT_MAXDIFF maxdiff=4.99963760e-04 maxfrac=0.499964 n=3\n0 Failures\n",
        elf_path=Path("nn_activation_float_tanh_f16.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.PASS
    assert result.max_diff == pytest.approx(4.99963760e-04)
    assert result.max_tolerance_fraction == pytest.approx(0.499964)


def test_parser_float_maxdiff_takes_worst_case_across_multiple_lines() -> None:
    parser = reporting_parser.TestResultParser()
    output = (
        "HELIA_FLOAT_MAXDIFF maxdiff=1.0e-05 maxfrac=0.1 n=4\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=2.0e-05 maxfrac=0.4 n=4\n"
        "0 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("multi_output.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.max_diff == pytest.approx(2.0e-05)
    assert result.max_tolerance_fraction == pytest.approx(0.4)


def test_parser_float_maxdiff_zero_tolerance_violation_is_sentinel() -> None:
    parser = reporting_parser.TestResultParser()
    output = (
        "Mismatch[0]: exp=0.000000 got=0.001000 (diff=0.001000, tol=0.000000)\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=1.00000005e-03 maxfrac=-1.000000 n=1\n"
        "1 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("edge_case.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.max_diff == pytest.approx(1.00000005e-03)
    assert result.max_tolerance_fraction == -1.0


def test_parser_float_maxdiff_nonfinite_is_sentinel() -> None:
    # issue #75: a NaN/Inf-producing kernel emits the -1.0 / -2.0 sentinel pair;
    # it must not be reported as benign near-zero headroom.
    parser = reporting_parser.TestResultParser()
    output = (
        "HELIA_NONFINITE_MISMATCH[7]: exp=0.500000 got=nan\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=64\n"
        "1 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("logistic_f16.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.max_diff == -1.0
    assert result.max_tolerance_fraction == -2.0


def test_parser_float_maxdiff_nonfinite_wins_over_finite_line() -> None:
    parser = reporting_parser.TestResultParser()
    output = (
        "HELIA_FLOAT_MAXDIFF maxdiff=2.0e-05 maxfrac=0.4 n=4\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=4\n"
        "1 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("multi_output.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.max_diff == -1.0
    assert result.max_tolerance_fraction == -2.0


def test_parser_float_maxdiff_literal_nan_token_is_sentinel() -> None:
    # Defensive: the macro substitutes negative sentinels and never prints a
    # literal nan/inf, but a stray token must still be caught, not dropped.
    parser = reporting_parser.TestResultParser()
    output = "HELIA_FLOAT_MAXDIFF maxdiff=nan maxfrac=nan n=16\n1 Failures\n"
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("stray.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.max_diff == -1.0
    assert result.max_tolerance_fraction == -2.0


def test_parser_no_maxdiff_line_yields_none() -> None:
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="0 Failures\n",
        elf_path=Path("int_case.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.max_diff is None
    assert result.max_tolerance_fraction is None


def test_parser_classifies_nonfinite_mismatch_distinctly() -> None:
    # issue #75: a NaN/Inf operand mismatch is a different defect from a
    # tolerance overrun and must not be flattened into "output_mismatch".
    parser = reporting_parser.TestResultParser()
    output = (
        "HELIA_NONFINITE_MISMATCH[3]: exp=+inf got=-inf\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=8\n"
        "1 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("logistic_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.FAIL
    assert result.error_type == "nonfinite_mismatch"
    assert result.failure_reason == (
        "Non-finite output mismatch: 1 element(s) differ from expected "
        "(first at [3]: expected +inf, got -inf)"
    )


def test_parser_keeps_finite_mismatch_classification() -> None:
    parser = reporting_parser.TestResultParser()
    output = (
        "Mismatch[3]: exp=1.000000 got=2.000000 (diff=1.000000, tol=0.000011)\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=1.00000000e+00 maxfrac=90909.090909 n=8\n"
        "1 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("logistic_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.error_type == "output_mismatch"
    assert result.max_diff == pytest.approx(1.0)
    assert result.max_tolerance_fraction == pytest.approx(90909.090909)


def test_parser_passes_matched_nonfinite_case_with_unmeasurable_headroom() -> None:
    # NaN in / NaN out is the propagation contract (AmbiqAI/ns-cmsis-nn#240):
    # the case passes, and only the headroom is unmeasurable.
    parser = reporting_parser.TestResultParser()
    output = (
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=8\n"
        "0 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("logistic_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.PASS
    assert result.error_type is None
    assert result.max_diff == -1.0
    assert result.max_tolerance_fraction == -2.0
