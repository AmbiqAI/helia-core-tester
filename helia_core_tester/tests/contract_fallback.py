"""Test-only kernel contract for rendering the Convolve template without an ns-cmsis-nn checkout.

The Convolve template renders its kernel call from the ns-cmsis-nn kernel contract and refuses
to render without one. The pure-Python pytest job has no checkout, so unit tests that render a
Convolve case fall back to the committed fixture, but only when nothing better exists: a
checkout that ships the export always wins, and a test that points CMSIS_NN_ROOT at a real
directory keeps that directory's behaviour, including refusing to render.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from helia_core_tester.contract.ir import load_contract_set

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "contract" / "convolve_pilot"


def fallback_resolver(resolve: Callable[[], Optional[Path]]) -> Callable[[], Optional[Path]]:
    def resolver() -> Optional[Path]:
        root = resolve()
        if root is not None and (os.environ.get("CMSIS_NN_ROOT") or load_contract_set(root).present):
            return root
        return FIXTURE_ROOT

    return resolver
