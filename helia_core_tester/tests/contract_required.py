"""Tests that need the checkout's planner skip with its reason on a checkout without one.

HELIA_CORE_TESTER_REQUIRE_CONTRACT (the contract CI leg that must see one) turns the skip into a failure.
The library itself still raises PlannerUnavailable; only tests turn it into a skip.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Sequence

import pytest

from helia_core_tester.contract import planner

P1E_CALLS = ("buffer_bytes", "depthwise_conv_s8_opt")
P1E_PLAN_FIELDS = ("route_value", "route_name")


@dataclasses.dataclass(frozen=True)
class P1dPlan:
    """A plan of a package older than P1e: no route_value or route_name."""

    kernel: str
    route: object
    implementation: object
    ctx_role: str
    ctx_bytes: int
    prepared_bytes: int


def require_planner(*methods: str, plan_fields: Sequence[str] = ()) -> None:
    try:
        package, loaded = planner.current_planner()
        missing = [method for method in methods if getattr(loaded, method, None) is None]
        fields = getattr(package.Plan, "__dataclass_fields__", {})
        missing += [f"Plan.{field}" for field in plan_fields if field not in fields]
        if missing:
            raise planner.PlannerUnavailable(f"this checkout's planner has no {', '.join(missing)}")
    except planner.PlannerUnavailable as error:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {error}")
        pytest.skip(f"needs the checkout's planner: {error}")
