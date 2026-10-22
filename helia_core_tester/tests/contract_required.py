"""Tests that need the checkout's planner skip with its reason on a checkout without one.

HELIA_CORE_TESTER_REQUIRE_CONTRACT (the contract CI leg that must see one) turns the skip into a failure.
The library itself still raises PlannerUnavailable; only tests turn it into a skip.
"""

from __future__ import annotations

import os

import pytest

from helia_core_tester.contract import planner

P1E_CALLS = ("buffer_bytes", "depthwise_conv_s8_opt")


def require_planner(*methods: str) -> None:
    try:
        _, loaded = planner.current_planner()
        missing = [method for method in methods if getattr(loaded, method, None) is None]
        if missing:
            raise planner.PlannerUnavailable(f"this checkout's planner has no {', '.join(missing)}")
    except planner.PlannerUnavailable as error:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {error}")
        pytest.skip(f"needs the checkout's planner: {error}")
