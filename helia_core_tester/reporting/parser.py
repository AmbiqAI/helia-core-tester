"""Test result parser for standalone harness output with legacy Unity fallback."""

import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from helia_core_tester.reporting.models import TestResult, TestStatus


class TestResultParser:
    """Parser for standalone failure-count output with legacy Unity fallback."""
    
    def __init__(self):
        self.test_start_pattern = re.compile(r'Running test: (.+)')
        self.test_pass_pattern = re.compile(r'(.+):test__([^:]+):PASS')
        self.test_fail_pattern = re.compile(r'(.+):test__([^:]+):FAIL')
        self.assertion_pattern = re.compile(r'TEST_ASSERT_EQUAL.*?MESSAGE\([^,]+,\s*[^,]+,\s*"([^"]+)"\)')
        self.failure_reason_pattern = re.compile(r'Expected\s+(\S+)\s+Was\s+(\S+)')
        self.timeout_pattern = re.compile(r'TIMEOUT running (.+)')
        self.error_pattern = re.compile(r'ERROR:\s*(.+)')
        self.cycles_pattern = re.compile(r'\[PERF\]\s+(\w+):\s+(\d+)\s+cycles')
        self.memory_pattern = re.compile(r'Memory usage:\s+(\d+)\s+bytes')
        # Headroom instrumentation (issue #53): HELIA_FLOAT_MAXDIFF summary line
        # emitted once per float-validated case by HELIA_VALIDATE_FLOATS. The
        # numeric fields also accept a literal nan/inf defensively -- the macro
        # never prints those (it excludes non-finite elements from the
        # measurement, issue #75), but a stray one must not make the whole line
        # unmatchable and silently drop the headroom data.
        _num = r'[-+]?(?:[0-9.]+(?:[eE][-+]?[0-9]+)?|nan|inf)'
        self.float_maxdiff_pattern = re.compile(
            r'HELIA_FLOAT_MAXDIFF\s+maxdiff=(' + _num + r')\s+maxfrac=(' + _num + r')\s+n=(\d+)',
            re.IGNORECASE,
        )
        # Patterns for extracting output differences
        self.expected_pattern = re.compile(r'(?:Expected|Golden|Reference)[:\s]+([^\n]+)', re.IGNORECASE)
        self.actual_pattern = re.compile(r'(?:Actual|Got|Output|Result)[:\s]+([^\n]+)', re.IGNORECASE)
        self.difference_pattern = re.compile(r'(?:Difference|Delta|Diff)[:\s]+([^\n]+)', re.IGNORECASE)
        self.index_pattern = re.compile(r'(?:Index|Position|Element)\s*[\[\(]?\s*(\d+)\s*[\]\)]?', re.IGNORECASE)
        self.value_comparison_pattern = re.compile(r'(\d+)\s*[!=<>]+\s*(\d+)')
        # Pattern for exact "0 Failures" match (not substring)
        self.zero_failures_pattern = re.compile(r'^0\s+Failures\s*$', re.MULTILINE | re.IGNORECASE)
        # Pattern for "X Failures" where X > 0
        self.nonzero_failures_pattern = re.compile(r'^(\d+)\s+Failures\s*$', re.MULTILINE | re.IGNORECASE)
        # Non-finite operand mismatch (issue #75): emitted by
        # helia_test_nonfinite_mismatch() instead of the %f "Mismatch[...]"
        # line, because a NaN/Inf operand renders unhelpfully as a number.
        self.nonfinite_mismatch_pattern = re.compile(
            r'HELIA_NONFINITE_MISMATCH\[(\d+)\]:\s*exp=(\S+)\s+got=(\S+)'
        )
        # Per-tensor count of mismatched non-finite elements. Printed whenever
        # one occurred, including when the per-element reports above were used
        # up by earlier failures, so this is the reliable classifier.
        self.nonfinite_summary_pattern = re.compile(
            r'HELIA_NONFINITE_MISMATCHES\s+n=(\d+)'
        )
        # Pattern for "Convolution failed" or API errors
        self.api_error_pattern = re.compile(
            r'(?P<label>[A-Za-z][A-Za-z0-9 _-]*)\s+failed with status\s+(?P<status>-?\d+)',
            re.IGNORECASE,
        )
        
    def parse_fvp_output(self, 
                        output: str, 
                        elf_path: Path, 
                        cpu: str, 
                        duration: float,
                        exit_code: Optional[int] = None,
                        descriptor_name: Optional[str] = None) -> TestResult:
        """
        Parse FVP output to extract test result.
        
        Args:
            output: Raw FVP output
            elf_path: Path to the ELF file
            cpu: Target CPU
            duration: Test execution duration in seconds
            exit_code: Process exit code
            descriptor_name: Optional descriptor name to link test to descriptor
            
        Returns:
            TestResult object
        """
        lines = output.split('\n')
        test_name = self._extract_test_name(elf_path)
        
        if descriptor_name is None:
            descriptor_name = test_name
        
        status, failure_reason, skip_reason, error_type = self._determine_status(
            output, lines, exit_code
        )
        
        cycles = self._extract_cycles(output)
        memory_usage = self._extract_memory_usage(output)
        max_diff, max_tolerance_fraction = self._extract_float_maxdiff(output)
        
        relevant_lines = self._extract_relevant_lines(lines)
        
        # Extract output differences if test failed
        expected_output = None
        actual_output = None
        output_differences = []
        if status == TestStatus.FAIL:
            expected_output, actual_output, output_differences = self._extract_output_differences(output, lines)
        
        return TestResult(
            test_name=test_name,
            status=status,
            duration=duration,
            cpu=cpu,
            elf_path=elf_path,
            failure_reason=failure_reason,
            skip_reason=skip_reason,
            output_lines=relevant_lines,
            timestamp=datetime.now(),
            memory_usage=memory_usage,
            cycles=cycles,
            exit_code=exit_code,
            error_type=error_type,
            descriptor_name=descriptor_name,
            expected_output=expected_output,
            actual_output=actual_output,
            output_differences=output_differences,
            max_diff=max_diff,
            max_tolerance_fraction=max_tolerance_fraction,
        )
    
    def _extract_test_name(self, elf_path: Path) -> str:
        """Extract test name from ELF file path."""
        return elf_path.stem
    
    def _determine_status(self, 
                         output: str, 
                         lines: List[str], 
                         exit_code: Optional[int]) -> Tuple[TestStatus, Optional[str], Optional[str], Optional[str]]:
        """Determine test status and extract failure/skip reasons."""
        
        if exit_code == 124 or "TIMEOUT" in output:
            return TestStatus.TIMEOUT, "Test execution timed out", None, "timeout"
        
        api_error_match = self.api_error_pattern.search(output)
        if api_error_match:
            label = api_error_match.group("label").strip()
            status_code = api_error_match.group("status")
            return TestStatus.FAIL, f"{label} API error (status {status_code})", None, "api_error"

        if self.zero_failures_pattern.search(output):
            return TestStatus.PASS, None, None, None

        nonzero_match = self.nonzero_failures_pattern.search(output)
        if nonzero_match:
            failure_count = int(nonzero_match.group(1))
            nonfinite = self.nonfinite_mismatch_pattern.findall(output)
            if nonfinite:
                index, expected, actual = nonfinite[0]
                return (
                    TestStatus.FAIL,
                    f"Non-finite output mismatch: {failure_count} element(s) differ from expected "
                    f"(first at [{index}]: expected {expected}, got {actual})",
                    None,
                    "nonfinite_mismatch",
                )
            nonfinite_counts = self.nonfinite_summary_pattern.findall(output)
            if nonfinite_counts:
                # The runtime reports individual elements only while failures
                # stay within the case's report limit (20 by default), so a
                # tensor with enough finite overruns ahead of the non-finite
                # element carries only the per-tensor count. The headroom
                # sentinel cannot stand in for it: the sentinel also fires when
                # a tensor had no finite element to measure, which would
                # mislabel a finite overrun in a multi-output case.
                nonfinite_total = sum(int(count) for count in nonfinite_counts)
                return (
                    TestStatus.FAIL,
                    f"Non-finite output mismatch: {failure_count} element(s) differ from expected "
                    f"({nonfinite_total} non-finite, reported beyond the per-case report limit)",
                    None,
                    "nonfinite_mismatch",
                )
            return TestStatus.FAIL, f"Output mismatch: {failure_count} element(s) differ from expected", None, "output_mismatch"

        fail_matches = self.test_fail_pattern.findall(output)
        if fail_matches:
            failure_reason = self._extract_failure_reason(output, lines)
            return TestStatus.FAIL, failure_reason, None, "assertion"

        pass_matches = self.test_pass_pattern.findall(output)
        if pass_matches:
            return TestStatus.PASS, None, None, None

        if exit_code and exit_code != 0:
            error_msg = f"Process exited with code {exit_code}"
            return TestStatus.ERROR, error_msg, None, "crash"

        return TestStatus.ERROR, "Unknown test status", None, "unknown"
    
    def _extract_failure_reason(self, output: str, lines: List[str]) -> str:
        """Extract detailed failure reason from legacy assertion-style output."""
        assertion_matches = self.assertion_pattern.findall(output)
        if assertion_matches:
            return assertion_matches[-1]  # Get the last assertion message
        
        failure_matches = self.failure_reason_pattern.findall(output)
        if failure_matches:
            expected, actual = failure_matches[-1]
            return f"Expected {expected} but got {actual}"
        
        error_matches = self.error_pattern.findall(output)
        if error_matches:
            return error_matches[-1]
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['fail', 'error', 'assert']):
                return line.strip()
        
        return "Test failed (no specific reason found)"
    
    def _extract_cycles(self, output: str) -> Optional[int]:
        """Extract cycle count from performance output."""
        cycles_matches = self.cycles_pattern.findall(output)
        if cycles_matches:
            try:
                return int(cycles_matches[-1][1])
            except (ValueError, IndexError):
                pass
        return None
    
    def _extract_memory_usage(self, output: str) -> Optional[int]:
        """Extract memory usage from output."""
        memory_matches = self.memory_pattern.findall(output)
        if memory_matches:
            try:
                return int(memory_matches[-1])
            except (ValueError, IndexError):
                pass
        return None
    
    def _extract_float_maxdiff(self, output: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract the worst-case measured diff / tolerance-budget fraction (issue #53).

        A case can emit more than one HELIA_FLOAT_MAXDIFF line (multiple output
        tensors validated in one test); this returns the max raw diff and the max
        fraction across all of them.

        Negative values are sentinels, never headroom numbers, and the more
        severe one from any single line wins over any finite measurement:
          maxfrac == -1.0             a zero-width tolerance budget was violated
                                      (undefined/infinite fraction, not "small")
          maxdiff == -1.0, frac -2.0  headroom is unmeasurable (issue #75): a
                                      non-finite (NaN/Inf) element mismatched,
                                      or the tensor had no finite element to
                                      measure. Reported back as (-1.0, -2.0) so
                                      no consumer can read it as benign
                                      near-zero headroom. Matched non-finite
                                      elements do not raise it -- a tensor with
                                      NaN lanes and finite lanes reports real
                                      headroom for the finite ones.
        A literal nan/inf token (which the macro does not emit) is treated the
        same as the non-finite sentinel.
        """
        matches = self.float_maxdiff_pattern.findall(output)
        if not matches:
            return None, None

        max_diff = 0.0
        max_frac = 0.0
        saw_zero_tol_violation = False
        saw_nonfinite = False
        for diff_str, frac_str, _n_str in matches:
            try:
                diff_val = float(diff_str)
                frac_val = float(frac_str)
            except ValueError:
                continue
            if (not math.isfinite(diff_val) or not math.isfinite(frac_val)
                    or diff_val < 0.0 or frac_val <= -2.0):
                saw_nonfinite = True
                continue
            max_diff = max(max_diff, diff_val)
            if frac_val < 0.0:
                saw_zero_tol_violation = True
            else:
                max_frac = max(max_frac, frac_val)

        if saw_nonfinite:
            return -1.0, -2.0
        return max_diff, (-1.0 if saw_zero_tol_violation else max_frac)

    def _extract_relevant_lines(self, lines: List[str]) -> List[str]:
        """Extract relevant output lines for debugging."""
        relevant = []
        in_test_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(keyword in line.lower() for keyword in ['test', 'fail', 'pass', 'error', 'assert']):
                in_test_section = True
            
            if in_test_section:
                relevant.append(line)
                
                if len(relevant) > 50:
                    relevant.append("... (truncated)")
                    break
        
        return relevant
    
    def _extract_output_differences(self, output: str, lines: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Extract expected vs actual output differences from test output.
        
        Returns:
            Tuple of (expected_output, actual_output, differences_list)
        """
        expected = None
        actual = None
        differences = []
        
        # Look for explicit "Expected" and "Actual" patterns
        expected_matches = self.expected_pattern.findall(output)
        actual_matches = self.actual_pattern.findall(output)
        
        if expected_matches:
            expected = expected_matches[-1].strip()
        if actual_matches:
            actual = actual_matches[-1].strip()
        
        # Look for "Expected X Was Y" patterns (already handled by failure_reason_pattern)
        failure_matches = self.failure_reason_pattern.findall(output)
        if failure_matches:
            for exp, act in failure_matches:
                differences.append(f"Expected: {exp}, Actual: {act}")
        
        # Look for index-based differences (e.g., "Index [0]: Expected 5, Got 3")
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['expected', 'actual', 'got', 'golden', 'reference', 'difference']):
                # Check if this line contains value comparisons
                if self.value_comparison_pattern.search(line):
                    differences.append(line.strip())
                # Check if this line contains index information
                elif self.index_pattern.search(line):
                    differences.append(line.strip())
                # Check if this line contains difference information
                elif self.difference_pattern.search(line):
                    diff_match = self.difference_pattern.search(line)
                    if diff_match:
                        differences.append(f"Difference: {diff_match.group(1).strip()}")
        
        # Look for array/tensor output differences
        # Common patterns: "Output[0] = X (expected Y)" or "Element N: got X, expected Y"
        array_diff_pattern = re.compile(r'(?:Output|Element|Value|Index)\s*[\[\(]?\s*(\d+)\s*[\]\)]?\s*[:=]?\s*(?:got|was|actual)?\s*(\d+)\s*(?:expected|should be|golden)?\s*(\d+)', re.IGNORECASE)
        array_diffs = array_diff_pattern.findall(output)
        for idx, got, exp in array_diffs:
            differences.append(f"Index [{idx}]: Got {got}, Expected {exp}")
        
        # Limit differences to prevent overwhelming output
        if len(differences) > 50:
            differences = differences[:50]
            differences.append("... (more differences truncated)")
        
        return expected, actual, differences
    
    def parse_multiple_tests(self, 
                           outputs: List[Tuple[str, Path, str, float, Optional[int]]]) -> List[TestResult]:
        """
        Parse multiple test outputs.
        
        Args:
            outputs: List of tuples (output, elf_path, cpu, duration, exit_code)
            
        Returns:
            List of TestResult objects
        """
        results = []
        for output, elf_path, cpu, duration, exit_code in outputs:
            result = self.parse_fvp_output(output, elf_path, cpu, duration, exit_code)
            results.append(result)
        return results
