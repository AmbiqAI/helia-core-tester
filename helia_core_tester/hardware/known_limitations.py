"""Registry of generated test cases that are known-unbridgeable to real hardware
dispatch for reasons *other* than missing (family, operator) builder coverage or
FVP-gate rejection -- i.e. genuine, currently-unresolved correctness gaps between
the standalone-harness reference semantics and the real firmware dispatch path for
that specific case -- or that cannot fit the runtime arena of any board in
assets/hardware_boards.yaml, so that a hardware session plan skips them as
unsupported instead of rejecting the whole run.

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
# reason string for the specific correctness gap.
#
# convolve_grouped_conv_case_01_s8 was previously listed here (1 LSB mismatch
# on real hardware vs. FVP's tolerant fallback); fixed by adding an explicit
# "Convolve": 1 override in dtypes.py, so it was removed from this registry.
_WIDE_TRANSPOSE_REASON = (
    "needs {size} of input and output, more than any board's runtime arena: the "
    "v7.43.0 MVE gather-path boundary (ns-cmsis-nn #759) is past 9362 columns and a "
    "wrapped lane offset shows only with several rows, so the case runs on the FVP only"
)
_KNOWN_LIMITATIONS: dict[str, KnownLimitation] = {
    name: KnownLimitation(case_name=name, reason=_WIDE_TRANSPOSE_REASON.format(size=size))
    for name, size in (
        ("transpose_swap_wide_s8", "about 150 KB"),
        ("transpose_swap_wide_s16", "about 300 KB"),
    )
}


def lookup_known_limitation(case_name: str) -> KnownLimitation | None:
    """Return the KnownLimitation entry for `case_name`, or None if it has none."""
    return _KNOWN_LIMITATIONS.get(case_name)
