"""Every kernel symbol the tester hard-codes must be one the checkout declares."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import CONTRACT_RELPATH, load_contract_set
from helia_core_tester.contract.symbol_audit import (
    COMPLETED,
    GATED,
    PRESENT,
    UNDECLARED,
    audit_symbol_literals,
    collect_literals,
    symbol_source_files,
)
from helia_core_tester.core.discovery import find_descriptors_dir, find_repo_root
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root
from helia_core_tester.tests.test_contract_ir import FIXTURE, HEADERS

# Symbols the tester still hard-codes although no public header declares them. Each
# entry is a known defect with its tracking issue; the list may only shrink, and a
# symbol that becomes declared again must be removed from it (the test says so).
KNOWN_UNDECLARED = {
    # Defined in Source/PoolingFunctions/arm_{avg,max}_pool_f{16,32}.c but declared in no
    # header; pool_base.py emits them for the `kernel_variant: nhwc_alias` descriptors,
    # which carry no required_kernel_symbols gate.
    "arm_avg_pool_nhwc_": "helia-core-tester#205",
    "arm_max_pool_nhwc_": "helia-core-tester#205",
    "arm_avg_pool_nhwc_f16": "helia-core-tester#205",
    "arm_avg_pool_nhwc_f32": "helia-core-tester#205",
    "arm_max_pool_nhwc_f16": "helia-core-tester#205",
    "arm_max_pool_nhwc_f32": "helia-core-tester#205",
}


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "ns-cmsis-nn"
    for relative, text in HEADERS.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    (root / CONTRACT_RELPATH).parent.mkdir(parents=True)
    shutil.copy(FIXTURE, root / CONTRACT_RELPATH)
    return root


def _module(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_literals_are_collected_from_string_constants_only(tmp_path: Path) -> None:
    module = _module(tmp_path, "op.py", '''
"""Docstring mentioning arm_elementwise_add_s8 in prose."""
# comment: arm_fx_pool_s8
KERNELS = {"S8": "arm_elementwise_add_s8", "F16": "arm_fx_pool_f16"}
PREFIX = "arm_convolve_wrapper_"
STEM = "arm_fx_pool"
NOT_A_SYMBOL = "arm_Upper"
def f(kind, dtype): return f"arm_fx_po{kind}_{dtype}"
''')
    literals = collect_literals([module])
    assert sorted(literals) == ["arm_convolve_wrapper_", "arm_elementwise_add_s8", "arm_fx_po", "arm_fx_pool", "arm_fx_pool_f16"]
    assert literals["arm_fx_pool"].files == ("op.py",)
    assert literals["arm_fx_po"].fragment and not literals["arm_fx_pool"].fragment


def test_unparseable_module_is_an_error(tmp_path: Path) -> None:
    module = _module(tmp_path, "broken.py", "def (:\n")
    with pytest.raises(ValueError, match="cannot be parsed"):
        collect_literals([module])


def test_classification(checkout: Path, tmp_path: Path) -> None:
    module = _module(tmp_path, "op.py", '''
K = ["arm_elementwise_add_s8", "arm_fx_pool_s8_get_buffer_size_", "arm_fx_pool", "arm_fx_gone_s8", "arm_fx_future_s8",
     "arm_fx_po"]
def f(kind, gone): return (f"arm_fx_po{kind}_s8", f"arm_fx_gon{gone}")
''')
    descriptors = [{"name": "x", "required_kernel_symbols": ["arm_fx_future_s8"]}, {"name": "y"}]
    audit = audit_symbol_literals(load_contract_set(checkout), [module], descriptors)
    assert audit.classification == {
        "arm_elementwise_add_s8": PRESENT,
        "arm_fx_pool_s8_get_buffer_size_": COMPLETED,
        "arm_fx_pool": COMPLETED,
        "arm_fx_future_s8": GATED,
        "arm_fx_gone_s8": UNDECLARED,
        # "arm_fx_po" is an f-string fragment somewhere, so the plain prefix rule applies;
        # "arm_fx_gon" prefixes no declared symbol even as a fragment.
        "arm_fx_po": COMPLETED,
        "arm_fx_gon": UNDECLARED,
    }
    assert [lit.symbol for lit in audit.undeclared] == ["arm_fx_gon", "arm_fx_gone_s8"]
    assert audit.gated_symbols == frozenset({"arm_fx_future_s8"})


def test_audit_refuses_an_absent_contract(checkout: Path, tmp_path: Path) -> None:
    (checkout / CONTRACT_RELPATH).unlink()
    with pytest.raises(ValueError, match="no kernel contract"):
        audit_symbol_literals(load_contract_set(checkout), [], [])


def test_real_tree_symbols_are_declared_or_known_drift() -> None:
    """Runs whenever an ns-cmsis-nn checkout with the export is reachable
    (CMSIS_NN_ROOT or the nested layout); the self-validate CI leg asserts it ran.
    Fails on any hard-coded symbol the checkout does not declare unless it is
    listed in KNOWN_UNDECLARED, and fails when a KNOWN_UNDECLARED entry is stale."""
    root = resolve_cmsis_nn_root()
    contracts = load_contract_set(root)
    if not contracts.present:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {root} has no kernel contract")
        pytest.skip(f"no kernel contract at {root}; set CMSIS_NN_ROOT to an ns-cmsis-nn with the export")
    tester_root = find_repo_root()
    descriptors = load_all_descriptors(str(find_descriptors_dir(tester_root)))
    audit = audit_symbol_literals(contracts, symbol_source_files(tester_root), descriptors)
    undeclared = {lit.symbol: lit.files for lit in audit.undeclared}
    unexpected = {symbol: files for symbol, files in undeclared.items() if symbol not in KNOWN_UNDECLARED}
    assert not unexpected, (
        "tester hard-codes symbols no public ns-cmsis-nn header declares: "
        + "; ".join(f"{symbol} in {', '.join(files)}" for symbol, files in sorted(unexpected.items()))
    )
    stale = sorted(set(KNOWN_UNDECLARED) - set(undeclared))
    assert not stale, f"KNOWN_UNDECLARED entries no longer undeclared, remove them: {stale}"
    assert len(audit.of(PRESENT)) > 100, "audit saw suspiciously few declared symbols"
