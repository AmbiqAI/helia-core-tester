"""`--suite both`: bridge int and float generated tests in one hardware session.

The two suites live side by side under artifacts/generated_tests/<suite>/<cpu>,
each case is bridged and FVP-gated against its own suite, and case_ids are
unique across suites -- so one flash and one result bundle can cover both
instead of paying for two full flash/stream cycles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.perf_stream.hardware_run import (
    build_generated_test_case_bundles,
    normalize_suites,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("int", ("int",)),
        ("float", ("float",)),
        ("both", ("int", "float")),
        ("  BOTH  ", ("int", "float")),
    ],
)
def test_normalize_suites(value: str, expected: tuple[str, ...]) -> None:
    assert normalize_suites(value) == expected


def test_normalize_suites_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Invalid suite"):
        normalize_suites("quad")


def _bundles(suite: str) -> list:
    bundles, _skipped = build_generated_test_case_bundles(
        PROJECT_ROOT, family="BasicMathFunctions", suite=suite, limit=3, fvp_gate="off"
    )
    return bundles


def test_both_is_the_union_of_int_and_float() -> None:
    int_ids = [b.case_id for b in _bundles("int")]
    float_ids = [b.case_id for b in _bundles("float")]
    both_ids = [b.case_id for b in _bundles("both")]

    assert int_ids and float_ids, "expected discoverable int and float BasicMath cases"
    assert both_ids == int_ids + float_ids


def test_case_ids_do_not_collide_across_suites() -> None:
    """One merged result bundle keys cases by case_id, so a collision between
    an int and a float case would silently overwrite one of them."""
    both_ids = [b.case_id for b in _bundles("both")]
    assert len(both_ids) == len(set(both_ids))
