"""The unit-test fallback contract: used only without a usable checkout, and kept equal to
the real ns-cmsis-nn export when one is present."""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import CONTRACT_RELPATH, load_contract_set
from helia_core_tester.contract.render import ContractRenderError, contract_globals
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root
from helia_core_tester.tests.contract_fallback import FIXTURE_ROOT, fallback_resolver


def test_fixture_is_a_loadable_contract_of_convolve_kernels() -> None:
    contracts = load_contract_set(FIXTURE_ROOT)
    assert contracts.present
    assert contracts.functions and all(name.startswith("arm_convolve_") for name in contracts.functions)
    assert {"arm_convolve_wrapper_s8_get_buffer_size_mve", "arm_convolve_s8_get_buffer_size"} <= set(contracts.functions)
    assert {"arm_convolve_wrapper_s8", "arm_convolve_s8", "arm_convolve_f32", "arm_convolve_wrapper_f16"} <= set(contracts.functions)


def test_no_checkout_falls_back_to_the_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMSIS_NN_ROOT", raising=False)
    assert fallback_resolver(lambda: None)() == FIXTURE_ROOT


def test_a_checkout_without_the_export_falls_back_unless_a_test_chose_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = tmp_path / "old-checkout"
    (old / "Include").mkdir(parents=True)
    monkeypatch.delenv("CMSIS_NN_ROOT", raising=False)
    assert fallback_resolver(lambda: old)() == FIXTURE_ROOT
    # A test that points CMSIS_NN_ROOT at a checkout keeps it, and rendering fails closed there.
    monkeypatch.setenv("CMSIS_NN_ROOT", str(old))
    resolved = fallback_resolver(lambda: old)()
    assert resolved == old
    with pytest.raises(ContractRenderError, match="has no Tests/KernelContracts/kernel_contracts.json"):
        contract_globals(lambda: load_contract_set(resolved))["contract_call"]("arm_convolve_s8", {})


def test_a_checkout_with_the_export_always_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = tmp_path / "ns-cmsis-nn"
    shutil.copytree(FIXTURE_ROOT, real)
    monkeypatch.delenv("CMSIS_NN_ROOT", raising=False)
    assert fallback_resolver(lambda: real)() == real


def test_a_corrupt_export_in_a_real_checkout_is_not_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = tmp_path / "ns-cmsis-nn"
    shutil.copytree(FIXTURE_ROOT, real)
    (real / CONTRACT_RELPATH).write_text("{not json")
    monkeypatch.delenv("CMSIS_NN_ROOT", raising=False)
    with pytest.raises(Exception, match="not valid JSON"):
        fallback_resolver(lambda: real)()


def test_convolve_contract_fixture_matches_the_real_tree() -> None:
    root = resolve_cmsis_nn_root()
    real = load_contract_set(root)
    if not real.present:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {root} has no kernel contract")
        pytest.skip("no ns-cmsis-nn checkout with a kernel contract")
    fixture = load_contract_set(FIXTURE_ROOT)
    kernels = {name for name in real.functions if name.startswith("arm_convolve_")}
    assert set(fixture.functions) == kernels, (
        f"refresh {FIXTURE_ROOT} from {real.path}: missing {sorted(kernels - set(fixture.functions))}, "
        f"extra {sorted(set(fixture.functions) - kernels)}")
    for name in sorted(kernels):
        mine = fixture.require(name)
        assert mine == dataclasses.replace(real.require(name), line=mine.line), f"{name}: refresh the fixture from {real.path}"
