from __future__ import annotations

from pathlib import Path

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
