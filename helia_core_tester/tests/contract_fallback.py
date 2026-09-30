"""Test-only kernel contract for rendering contract-bound templates without an ns-cmsis-nn checkout.

It holds every function of the operators whose templates bind their calls from the contract
(BOUND_PREFIXES), kernels and scratch-size queries alike, since the templates bind both.

Those templates render their kernel calls from the ns-cmsis-nn kernel contract and refuse
to render without one. The pure-Python pytest job has no checkout, so unit tests that render a
contract-bound case fall back to the committed fixture, but only when nothing better exists: a
checkout that ships the export always wins, and a test that points CMSIS_NN_ROOT at a real
directory keeps that directory's behaviour, including refusing to render.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from helia_core_tester.contract.ir import load_contract_set

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "contract" / "bound_operators"
BOUND_PREFIXES = ("arm_convolve_", "arm_depthwise_", "arm_fully_connected_", "arm_batch_matmul_")


def fallback_resolver(resolve: Callable[[], Optional[Path]]) -> Callable[[], Optional[Path]]:
    def resolver() -> Optional[Path]:
        root = resolve()
        if root is not None and (os.environ.get("CMSIS_NN_ROOT") or load_contract_set(root).present):
            return root
        return FIXTURE_ROOT

    return resolver
