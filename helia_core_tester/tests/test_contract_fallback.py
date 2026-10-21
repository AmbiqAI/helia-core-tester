"""The unit-test fallback contract: used only without a usable checkout, and kept equal to
the real ns-cmsis-nn export when one is present."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import CONTRACT_RELPATH, MANIFEST_RELPATH, PACKAGE_RELPATH, load_contract_set
from helia_core_tester.contract.symbol_audit import gated_symbols_from_descriptors
from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.contract.render import ContractRenderError, contract_globals
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root
from helia_core_tester.tests.contract_checkout import install_reader
from helia_core_tester.tests.contract_fallback import BOUND_PREFIXES, FIXTURE_ROOT, fallback_resolver


def test_fixture_is_a_loadable_contract_of_the_bound_operators() -> None:
    contracts = load_contract_set(FIXTURE_ROOT)
    assert contracts.present
    assert contracts.functions and all(name.startswith(BOUND_PREFIXES) for name in contracts.functions)
    assert {"arm_depthwise_conv_s16", "arm_depthwise_conv_wrapper_s8_get_buffer_size"} <= set(contracts.functions)
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


def test_bound_operator_fixture_matches_the_real_tree() -> None:
    root = resolve_cmsis_nn_root()
    real = load_contract_set(root)
    if not real.present:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {root} has no kernel contract")
        pytest.skip("no ns-cmsis-nn checkout with a kernel contract")
    fixture = load_contract_set(FIXTURE_ROOT)
    kernels = {name for name in real.functions if name.startswith(BOUND_PREFIXES)}
    assert set(fixture.functions) == kernels, (
        f"refresh {FIXTURE_ROOT} from {real.path}: missing {sorted(kernels - set(fixture.functions))}, "
        f"extra {sorted(set(fixture.functions) - kernels)}")
    for name in sorted(kernels):
        mine = fixture.require(name)
        assert mine == dataclasses.replace(real.require(name), line=mine.line), f"{name}: refresh the fixture from {real.path}"


def _real_checkout_with(relative: Path) -> Path:
    root = resolve_cmsis_nn_root()
    if root is None or not (root / relative).exists():
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {root} has no {relative}")
        pytest.skip(f"no ns-cmsis-nn checkout with {relative}")
    return root


def test_vendored_reader_matches_the_real_tree() -> None:
    root = _real_checkout_with(PACKAGE_RELPATH / "__init__.py")
    real = {path.name: path.read_bytes() for path in sorted((root / PACKAGE_RELPATH).glob("*.py"))}
    mine = {path.name: path.read_bytes() for path in sorted((FIXTURE_ROOT / PACKAGE_RELPATH).glob("*.py"))}
    stale = sorted(name for name in real.keys() | mine.keys() if real.get(name) != mine.get(name))
    assert not stale, f"{stale} differ from {root / PACKAGE_RELPATH}; run scripts/refresh_contract_fixture.py {root}"


def test_fixture_manifest_matches_the_real_tree() -> None:
    root = _real_checkout_with(MANIFEST_RELPATH)
    real = json.loads((root / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    mine = json.loads((FIXTURE_ROOT / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    sliced = (FIXTURE_ROOT / CONTRACT_RELPATH).read_bytes()
    assert mine.pop("signature_sha256") == hashlib.sha256(sliced).hexdigest()
    real.pop("signature_sha256")
    assert mine == real, f"refresh the fixture manifest: scripts/refresh_contract_fixture.py {root}"


def test_fixture_carries_every_manifest_record() -> None:
    contracts = load_contract_set(FIXTURE_ROOT)
    assert contracts.implementations and contracts.library_version
    for impl in contracts.implementations.values():
        assert contracts.find(impl.entry) is not None, impl.id


def test_every_gated_symbol_of_the_real_tree_is_in_the_fixture() -> None:
    """A gated symbol the fixture lacks would skip, in the fallback, a case the real tree generates."""
    real = load_contract_set(resolve_cmsis_nn_root())
    if not real.present:
        pytest.skip("no ns-cmsis-nn checkout with a kernel contract")
    descriptors = load_all_descriptors(str(find_descriptors_dir()))
    gated = gated_symbols_from_descriptors(descriptors)
    outside = sorted(symbol for symbol in gated if symbol in real.functions and not symbol.startswith(BOUND_PREFIXES))
    assert not outside, f"add prefixes for {outside} to BOUND_PREFIXES and refresh the fixture"


def test_the_fixture_reader_is_the_one_tests_install(tmp_path: Path) -> None:
    root = install_reader(tmp_path / "ns-cmsis-nn")
    assert sorted(p.name for p in (root / PACKAGE_RELPATH).glob("*.py")) == \
        sorted(p.name for p in (FIXTURE_ROOT / PACKAGE_RELPATH).glob("*.py"))
    assert (FIXTURE_ROOT.parent / "hardware_pilot" / "python").resolve() == (FIXTURE_ROOT / "python").resolve()
