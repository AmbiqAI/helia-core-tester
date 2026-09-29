from __future__ import annotations

import pytest

from helia_core_tester.tests.contract_fallback import fallback_resolver


@pytest.fixture(autouse=True)
def _kernel_contract_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """See contract_fallback: tests that render the Convolve template need a contract."""
    from helia_core_tester.contract import render

    monkeypatch.setattr(render, "resolve_cmsis_nn_root", fallback_resolver(render.resolve_cmsis_nn_root))
