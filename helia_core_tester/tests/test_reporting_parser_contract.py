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
    # ns-cmsis-nn's Include/arm_nnfunctions_flt.h guarantees NaN-ness and not
    # payload (AmbiqAI/ns-cmsis-nn#333), so a matched NaN passes and only the
    # headroom is unmeasurable.
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


def test_parser_classifies_on_the_count_line_without_a_printed_element() -> None:
    # The runtime reports elements only while failures stay within the case's
    # report limit (20 by default), so finite overruns ahead of the non-finite
    # element consume the budget and the per-tensor count line is the only
    # evidence left.
    parser = reporting_parser.TestResultParser()
    reported = "\n".join(
        f"Mismatch[{i}]: exp=1.000000 got=2.000000 (diff=1.000000, tol=0.000011)"
        for i in range(20)
    )
    output = (
        f"{reported}\n"
        "HELIA_NONFINITE_MISMATCHES n=1\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=64\n"
        "21 Failures\n"
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
    assert "21 element(s)" in result.failure_reason
    assert "1 non-finite" in result.failure_reason
    assert result.max_diff == -1.0
    assert result.max_tolerance_fraction == -2.0


def test_parser_does_not_read_the_sentinel_as_a_nonfinite_mismatch() -> None:
    # A multi-output case shares one failure counter across tensors, so a
    # finite overrun in one tensor can coexist with a second tensor that had no
    # finite element to measure. The sentinel alone is not evidence of a
    # non-finite mismatch; the HELIA_NONFINITE_MISMATCHES line is.
    parser = reporting_parser.TestResultParser()
    output = (
        "Mismatch[2]: exp=1.000000 got=2.000000 (diff=1.000000, tol=0.000011)\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=1.00000000e+00 maxfrac=90909.090909 n=8\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=0\n"
        "1 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("split_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.FAIL
    assert result.error_type == "output_mismatch"


def test_parser_float_maxdiff_literal_inf_token_is_sentinel() -> None:
    # Same defensive path as the nan token: an inf maxdiff is unmeasurable
    # headroom, never a large-but-real measurement.
    parser = reporting_parser.TestResultParser()
    output = "HELIA_FLOAT_MAXDIFF maxdiff=inf maxfrac=inf n=16\n1 Failures\n"
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("stray_inf.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.max_diff == -1.0
    assert result.max_tolerance_fraction == -2.0


def test_parser_keeps_the_helia_evidence_lines_in_the_report() -> None:
    # The HELIA_* lines are the only record of which element went non-finite
    # and by how much the finite ones missed; the report is the artifact a
    # reader gets, so a verdict without them is not actionable.
    parser = reporting_parser.TestResultParser()
    output = (
        "HELIA_NONFINITE_MISMATCH[0]: exp=1 got=nan\n"
        "HELIA_NONFINITE_MISMATCHES n=1\n"
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
    assert "HELIA_NONFINITE_MISMATCH[0]: exp=1 got=nan" in result.output_lines
    assert "HELIA_NONFINITE_MISMATCHES n=1" in result.output_lines
    assert (
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=8"
        in result.output_lines
    )


def test_parser_keeps_the_per_element_mismatch_lines_in_the_report() -> None:
    # The per-element line carries the failing lane index, which is what separates a
    # kernel that spreads a token from one that merely misses a tolerance. It prints
    # ahead of every other keyword, so without its own keyword the retained section
    # starts after it and the index is lost.
    parser = reporting_parser.TestResultParser()
    output = (
        "Mismatch[309]: exp=1.234000 got=1.235000 (diff=0.001000, tol=0.000617)\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=1.00000000e-03 maxfrac=1.620746 n=1024\n"
        "1 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("depthwise_conv_float_nonfinite_nan_f16.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert (
        "Mismatch[309]: exp=1.234000 got=1.235000 (diff=0.001000, tol=0.000617)"
        in result.output_lines
    )


def test_parser_empty_tensor_does_not_void_sibling_tensor_headroom() -> None:
    # A split-style case validates several output tensors into one output
    # stream. A zero-length tensor compares nothing and always reports the
    # sentinel, which must not erase headroom the other tensors measured.
    parser = reporting_parser.TestResultParser()
    output = (
        "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=0\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=4.20000000e-04 maxfrac=0.420000 n=8\n"
        "HELIA_FLOAT_MAXDIFF maxdiff=1.00000000e-04 maxfrac=0.100000 n=8\n"
        "0 Failures\n"
    )
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("split_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.PASS
    assert result.max_diff == pytest.approx(4.2e-04)
    assert result.max_tolerance_fraction == pytest.approx(0.42)


def test_parser_lone_empty_tensor_still_reports_the_sentinel() -> None:
    # With nothing else to speak for the case, the empty tensor's sentinel is
    # the honest answer: no headroom was measured.
    parser = reporting_parser.TestResultParser()
    output = "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=0\n0 Failures\n"
    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("empty_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.PASS
    assert result.max_diff == -1.0
    assert result.max_tolerance_fraction == -2.0


def test_parser_records_masked_lane_count_and_still_requires_success() -> None:
    # A mask-policy case (issue #74) passes on the same terms as any other: the
    # kernel returned SUCCESS and the harness printed zero failures. The mask
    # only removes lanes from the comparison, so the count and the total it is
    # out of are both recorded: "passed with 3 of 128 masked" and "passed with
    # 3 of 4 masked" are different claims and k alone cannot tell them apart.
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output=(
            "HELIA_MASKED_LANES: 3 of 128\n"
            "HELIA_FLOAT_MAXDIFF maxdiff=1.00000000e-06 maxfrac=0.100000 n=128\n"
            "0 Failures\n"
        ),
        elf_path=Path("abs_float_nonfinite_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.PASS
    assert result.masked_lanes == 3
    assert result.masked_lanes_total == 128
    assert result.to_dict()["masked_lanes"] == 3
    assert result.to_dict()["masked_lanes_total"] == 128


def test_parser_sums_masked_lane_lines_across_outputs() -> None:
    # One summary line per validated output tensor, so a case with several outputs
    # prints several and both the counts and the totals have to add up. No shipped
    # mask-policy case has more than one output yet; the parser has to be right
    # before one does.
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output=(
            "HELIA_MASKED_LANES: 2 of 8\n"
            "HELIA_MASKED_LANES: 1 of 8\n"
            "0 Failures\n"
        ),
        elf_path=Path("max_pool_float_nonfinite_nan_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.masked_lanes == 3
    assert result.masked_lanes_total == 16


def test_parser_fails_a_case_reporting_more_masked_lanes_than_compared() -> None:
    # k > n cannot come from the runtime, so it means the capture is corrupt or
    # interleaved; silently recording a masked fraction above 1 would read as a
    # measurement. It is a failed result rather than an exception because the
    # serial and the parallel runner do not survive an exception the same way.
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="HELIA_MASKED_LANES: 9 of 8\n0 Failures\n",
        elf_path=Path("max_pool_float_nonfinite_nan_f32.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.status == reporting_models.TestStatus.FAIL
    assert result.error_type == "corrupted_capture"
    assert "Corrupted capture" in result.failure_reason
    assert result.masked_lanes is None
    assert result.masked_lanes_total is None


def test_parser_leaves_masked_lanes_unset_without_the_summary_line() -> None:
    parser = reporting_parser.TestResultParser()
    result = parser.parse_fvp_output(
        output="HELIA_FLOAT_MAXDIFF maxdiff=0.00000000e+00 maxfrac=0.000000 n=8\n0 Failures\n",
        elf_path=Path("relu.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )
    assert result.masked_lanes is None
    assert result.masked_lanes_total is None
    assert "masked_lanes" not in result.to_dict()
    assert "masked_lanes_total" not in result.to_dict()


def test_parser_fails_a_masked_case_that_faults_or_times_out() -> None:
    # Masking never rescues a case: status, HardFault and timeout are decided
    # before any lane is compared.
    parser = reporting_parser.TestResultParser()
    timed_out = parser.parse_fvp_output(
        output="HELIA_MASKED_LANES: 3 of 128\nTIMEOUT running abs\n",
        elf_path=Path("abs_float_nonfinite_f32.elf"),
        cpu="cortex-m0",
        duration=60.0,
        exit_code=124,
    )
    assert timed_out.status == reporting_models.TestStatus.TIMEOUT

    bad_status = parser.parse_fvp_output(
        output="HELIA_MASKED_LANES: 3 of 128\nAbs failed with status -1\n1 Failures\n",
        elf_path=Path("abs_float_nonfinite_f32.elf"),
        cpu="cortex-m0",
        duration=0.1,
        exit_code=0,
    )
    assert bad_status.status == reporting_models.TestStatus.FAIL
    assert bad_status.error_type == "api_error"


def test_parser_keeps_the_verdict_lines_past_the_truncation_cap() -> None:
    # A multi-output case emits enough per-element lines on its own to exhaust the
    # retained-line budget, which would otherwise cut the capture before the
    # summaries and the failure count that carry the verdict.
    parser = reporting_parser.TestResultParser()
    body = []
    for tensor in range(3):
        for lane in range(20):
            body.append(
                f"Mismatch[{tensor * 100 + lane}]: exp=1.234000 got=1.235000 "
                "(diff=0.001000, tol=0.000617)"
            )
        body.append("HELIA_MASKED_LANES: 4 of 1024")
        body.append(
            f"HELIA_FLOAT_MAXDIFF maxdiff=1.0000000{tensor}e-03 maxfrac=1.62074{tensor} n=1024"
        )
    body.append("HELIA_NONFINITE_MISMATCHES n=2")
    body.append("60 Failures")
    output = "\n".join(body) + "\n"

    result = parser.parse_fvp_output(
        output=output,
        elf_path=Path("depthwise_conv_float_nonfinite_nan_f16.elf"),
        cpu="cortex-m55",
        duration=0.1,
        exit_code=0,
    )

    assert "... (truncated)" in result.output_lines
    assert "60 Failures" in result.output_lines
    assert "HELIA_NONFINITE_MISMATCHES n=2" in result.output_lines
    assert result.output_lines.count("HELIA_MASKED_LANES: 4 of 1024") == 3
    for tensor in range(3):
        assert (
            f"HELIA_FLOAT_MAXDIFF maxdiff=1.0000000{tensor}e-03 maxfrac=1.62074{tensor} n=1024"
            in result.output_lines
        )
