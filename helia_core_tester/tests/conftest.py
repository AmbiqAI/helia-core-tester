from __future__ import annotations

import pytest

from helia_core_tester.tests.contract_fallback import fallback_resolver


@pytest.fixture(autouse=True, scope="session")
def _kernel_contract_fallback():
    """See contract_fallback: tests that render a contract-bound operator need a contract.

    Session scope, so fixtures of every scope (module-level generation fixtures included)
    render with it; the resolver still honours a CMSIS_NN_ROOT a test sets at call time."""
    from helia_core_tester.contract import render

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(render, "resolve_cmsis_nn_root", fallback_resolver(render.resolve_cmsis_nn_root))
        yield
