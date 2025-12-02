"""
Test result parser for Unity framework output.
"""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .models import TestResult, TestStatus


class TestResultParser:
    """Parser for Unity test framework output."""
    
    def __init__(self):
        # Regex patterns for Unity output
        self.test_start_pattern = re.compile(r'Running test: (.+)')
        self.test_pass_pattern = re.compile(r'(.+):test__([^:]+):PASS')
        self.test_fail_pattern = re.compile(r'(.+):test__([^:]+):FAIL')
        self.assertion_pattern = re.compile(r'TEST_ASSERT_EQUAL.*?MESSAGE\([^,]+,\s*[^,]+,\s*"([^"]+)"\)')
        self.failure_reason_pattern = re.compile(r'Expected\s+(\S+)\s+Was\s+(\S+)')
        self.timeout_pattern = re.compile(r'TIMEOUT running (.+)')
        self.error_pattern = re.compile(r'ERROR:\s*(.+)')
        self.cycles_pattern = re.compile(r'\[PERF\]\s+(\w+):\s+(\d+)\s+cycles')
        self.memory_pattern = re.compile(r'Memory usage:\s+(\d+)\s+bytes')
        
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
        
        # Use provided descriptor_name or extract from test_name
        if descriptor_name is None:
            descriptor_name = test_name
        
        # Determine test status
        status, failure_reason, skip_reason, error_type = self._determine_status(
            output, lines, exit_code
        )
        
        # Extract performance metrics
        cycles = self._extract_cycles(output)
        memory_usage = self._extract_memory_usage(output)
        
        # Extract relevant output lines
        relevant_lines = self._extract_relevant_lines(lines)
        
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
            descriptor_name=descriptor_name
        )
    
    def _extract_test_name(self, elf_path: Path) -> str:
        """Extract test name from ELF file path."""
        # Remove .elf extension and get the base name
        return elf_path.stem
    
    def _determine_status(self, 
                         output: str, 
                         lines: List[str], 
                         exit_code: Optional[int]) -> Tuple[TestStatus, Optional[str], Optional[str], Optional[str]]:
        """Determine test status and extract failure/skip reasons."""
        
        # Check for timeout
        if exit_code == 124 or "TIMEOUT" in output:
            return TestStatus.TIMEOUT, "Test execution timed out", None, "timeout"
        
        # Check for Unity test results
        pass_matches = self.test_pass_pattern.findall(output)
        fail_matches = self.test_fail_pattern.findall(output)
        
        if fail_matches:
            # Test failed - extract failure reason
            failure_reason = self._extract_failure_reason(output, lines)
            return TestStatus.FAIL, failure_reason, None, "assertion"
        elif pass_matches:
            # Test passed
            return TestStatus.PASS, None, None, None
        elif exit_code and exit_code != 0:
            # Non-zero exit code but no Unity failure message
            error_msg = f"Process exited with code {exit_code}"
            return TestStatus.ERROR, error_msg, None, "crash"
        else:
            # No clear indication - assume pass if no errors
            if "0 Failures" in output:
                return TestStatus.PASS, None, None, None
            else:
                return TestStatus.ERROR, "Unknown test status", None, "unknown"
    
    def _extract_failure_reason(self, output: str, lines: List[str]) -> str:
        """Extract detailed failure reason from Unity output."""
        # Look for assertion messages
        assertion_matches = self.assertion_pattern.findall(output)
        if assertion_matches:
            return assertion_matches[-1]  # Get the last assertion message
        
        # Look for expected vs actual values
        failure_matches = self.failure_reason_pattern.findall(output)
        if failure_matches:
            expected, actual = failure_matches[-1]
            return f"Expected {expected} but got {actual}"
        
        # Look for error messages
        error_matches = self.error_pattern.findall(output)
        if error_matches:
            return error_matches[-1]
        
        # Fallback: look for lines containing "fail" or "error"
        for line in lines:
            if any(keyword in line.lower() for keyword in ['fail', 'error', 'assert']):
                return line.strip()
        
        return "Test failed (no specific reason found)"
    
    def _extract_cycles(self, output: str) -> Optional[int]:
        """Extract cycle count from performance output."""
        cycles_matches = self.cycles_pattern.findall(output)
        if cycles_matches:
            try:
                return int(cycles_matches[-1][1])  # Get cycles from last match
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
    
    def _extract_relevant_lines(self, lines: List[str]) -> List[str]:
        """Extract relevant output lines for debugging."""
        relevant = []
        in_test_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Start capturing when we see test-related output
            if any(keyword in line.lower() for keyword in ['test', 'fail', 'pass', 'error', 'assert']):
                in_test_section = True
            
            if in_test_section:
                relevant.append(line)
                
                # Stop capturing after we've seen enough test output
                if len(relevant) > 50:  # Limit to prevent huge reports
                    relevant.append("... (truncated)")
                    break
        
        return relevant
    
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
