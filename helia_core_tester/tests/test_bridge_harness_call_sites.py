"""The hardware bridge reads the generic harness's call sites: every argument carries a
`/* name */` comment, and the contract parity assert spells `arm_cmsis_nn_status (` before the
first real kernel call."""

from __future__ import annotations

import pytest

from helia_core_tester.hardware.generated_test_bridge import (
    UnsupportedGeneratedTestError,
    _extract_all_call_args,
    _extract_call_args,
    _extract_first_cmsis_function_name,
)

_HARNESS_CALL = """
    return arm_equal_s8(
        input_1, /* input_1 */
        4, /* input_1_dims, a comment with , commas */
        NULL, // input_2
        -3968, /* diff_min */
        output /* output */
    );
"""

_PARITY = ("_Static_assert(__builtin_types_compatible_p(__typeof__(arm_equal_s8), "
           "arm_cmsis_nn_status (const int8_t *)), \"x\");\n")


def test_call_extraction_ignores_block_and_line_comments() -> None:
    expected = ["input_1", "4", "NULL", "-3968", "output"]
    assert _extract_call_args(_HARNESS_CALL, "arm_equal_s8", expected_count=5) == expected
    assert _extract_all_call_args(_HARNESS_CALL * 2, "arm_equal_s8", expected_count=5) == [expected, expected]


def test_call_extraction_still_counts_arguments() -> None:
    with pytest.raises(UnsupportedGeneratedTestError, match="has 5 arguments, expected 4"):
        _extract_call_args(_HARNESS_CALL, "arm_equal_s8", expected_count=4)
    with pytest.raises(UnsupportedGeneratedTestError, match="Could not find call"):
        _extract_call_args(_HARNESS_CALL, "arm_not_equal_s8", expected_count=5)


def test_the_kernel_name_skips_the_parity_asserts_status_type() -> None:
    assert _extract_first_cmsis_function_name(_PARITY + _HARNESS_CALL) == "arm_equal_s8"
    with pytest.raises(UnsupportedGeneratedTestError, match="Could not find"):
        _extract_first_cmsis_function_name("arm_cmsis_nn_status (int);")
