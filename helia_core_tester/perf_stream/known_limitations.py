"""Registry of generated test cases that are known-unbridgeable to real hardware
dispatch for reasons *other* than missing (family, operator) builder coverage or
FVP-gate rejection -- i.e. genuine, currently-unresolved correctness gaps between
the standalone-harness reference semantics and the real firmware dispatch path for
that specific case.

This exists so that case-specific "this one case is intentionally unbridged"
exceptions live in a single declarative table instead of being scattered as
ad hoc `if generated_test.name == "...":` checks inside individual per-operator
builder functions (as `convolve_grouped_conv_case_01_s8` previously was, inside
`_build_convolve_case`). Adding a new known limitation should mean adding one
entry here, not editing builder logic.

Entries should be removed once the underlying correctness gap is actually fixed
-- this registry documents currently-open gaps, not permanent exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownLimitation:
    """A generated test case name that is known-unbridgeable, with the reason why."""

    case_name: str
    reason: str


# Keyed by GeneratedTestCase.name. One entry per known-broken case; see each
# reason string for the specific correctness gap and investigation history.
_KNOWN_LIMITATIONS: dict[str, KnownLimitation] = {
    "convolve_grouped_conv_case_01_s8": KnownLimitation(
        case_name="convolve_grouped_conv_case_01_s8",
        reason=(
            "even after matching the standalone harness's first-slice (header n=1) "
            "semantics, the direct arm_convolve_wrapper_s8 path is still not "
            "exact-correct for this grouped-convolution artifact: a fresh host-side "
            "repro of the real generated header/kernel call still mismatched "
            "expected_output at index 68 (expected 77, got 78), and "
            "arm_convolve_weight_sum() reports ARG_ERROR on the same grouped dims. "
            "It therefore remains intentionally unbridged rather than shipping a "
            "known incorrect exact-match convolution result. Note: this case PASSES "
            "under FVP -- the gap is specific to the real-hardware direct-dispatch "
            "bridge path, not a general correctness issue with the generated case."
        ),
    ),
}


def lookup_known_limitation(case_name: str) -> KnownLimitation | None:
    """Return the KnownLimitation entry for `case_name`, or None if it has none."""
    return _KNOWN_LIMITATIONS.get(case_name)
