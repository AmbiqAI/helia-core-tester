"""`helia_core_tester contract inventory` and the module behind it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.contract.inventory import (
    REPORT_SCHEMA,
    InventoryError,
    build_inventory,
    case_calls,
    discover_case_dirs,
)
from helia_core_tester.contract.ir import CONTRACT_RELPATH, load_contract_set
from helia_core_tester.tests.test_contract_ir import FIXTURE, HEADERS

runner = CliRunner()


def _text(result) -> str:
    return "".join(getattr(result, attr, "") or "" for attr in ("output", "stdout", "stderr"))


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "ns-cmsis-nn"
    for relative, text in HEADERS.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    (root / CONTRACT_RELPATH).parent.mkdir(parents=True)
    shutil.copy(FIXTURE, root / CONTRACT_RELPATH)
    return root


def _case(tree: Path, suite: str, cpu: str, family: str, name: str, *, c_source: str = "",
          sidecar_kernel: str | None = None) -> Path:
    case_dir = tree / suite / cpu / family / name
    case_dir.mkdir(parents=True)
    (case_dir / "descriptor.yaml").write_text(f"name: {name}\noperator: Fx\n")
    if c_source:
        (case_dir / f"{name}_fx.c").write_text(c_source)
    if sidecar_kernel is not None:
        (case_dir / f"{name}_fx.sidecar.json").write_text(json.dumps({"kernel_fn": sidecar_kernel}))
    return case_dir


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    tree = tmp_path / "generated_tests"
    # Sidecar only: the kernel is known from the sidecar, the sizer from the source.
    _case(tree, "int", "cortex-m55", "PoolingFunctions", "pool_a_s8", sidecar_kernel="arm_fx_pool_s8",
          c_source="int32_t n = arm_fx_pool_s8_get_buffer_size_mve(&dims);\n"
                   "status = arm_fx_pool_s8(&ctx, &dims, in, out);\n")
    # Source only, with a doc-comment mention that must not count as a call and a
    # preprocessor line naming another symbol that must not count either.
    _case(tree, "int", "cortex-m55", "BasicMathFunctions", "add_b_s8",
          c_source="/* see arm_fx_pool_f16(dims) */\n#define KERNEL arm_nn_fx_helper\n"
                   "status = arm_elementwise_add_s8(a, b, out, n); // arm_fx_pool_s8(...)\n")
    return tree


def test_case_calls_reads_sidecar_and_source(tree: Path) -> None:
    pool = case_calls(tree / "int" / "cortex-m55" / "PoolingFunctions" / "pool_a_s8")
    assert pool.symbols == ("arm_fx_pool_s8", "arm_fx_pool_s8_get_buffer_size_mve")
    assert pool.source == "both"
    add = case_calls(tree / "int" / "cortex-m55" / "BasicMathFunctions" / "add_b_s8")
    assert add.symbols == ("arm_elementwise_add_s8",)
    assert add.source == "source"


def test_inventory_splits_covered_uncovered_and_unknown(checkout: Path, tree: Path) -> None:
    _case(tree, "float", "cortex-m55", "PoolingFunctions", "ghost_f16",
          c_source="status = arm_fx_ghost_f16(out);\n")
    contracts = load_contract_set(checkout)
    inventory = build_inventory(contracts, tree)
    assert inventory.cases_scanned == 3
    assert inventory.covered == {
        "arm_elementwise_add_s8": 1,
        "arm_fx_pool_s8": 1,
        "arm_fx_pool_s8_get_buffer_size_mve": 1,
    }
    assert inventory.uncovered_by_kind("kernel") == ["arm_fx_pool_f16"]
    assert inventory.uncovered_by_kind("sizer") == ["arm_fx_pool_s8_get_buffer_size"]
    assert inventory.uncovered_by_kind("support") == ["arm_nn_fx_helper"]
    assert inventory.unknown == {"arm_fx_ghost_f16": 1}
    assert inventory.kinds == {"kernel": 3, "sizer": 2, "support": 1}
    document = inventory.to_document()
    assert document["schema"] == REPORT_SCHEMA
    assert document["uncovered"] == {"kernel": ["arm_fx_pool_f16"],
                                     "sizer": ["arm_fx_pool_s8_get_buffer_size"],
                                     "support": ["arm_nn_fx_helper"]}


def test_no_contract_and_no_cases_are_errors(checkout: Path, tree: Path, tmp_path: Path) -> None:
    (checkout / CONTRACT_RELPATH).unlink()
    with pytest.raises(InventoryError, match="no kernel contract"):
        build_inventory(load_contract_set(checkout), tree)
    shutil.copy(FIXTURE, checkout / CONTRACT_RELPATH)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InventoryError, match="no generated cases"):
        build_inventory(load_contract_set(checkout), empty)
    with pytest.raises(InventoryError, match="directory not found"):
        discover_case_dirs(tmp_path / "missing")


def test_unreadable_sidecar_is_an_error(tree: Path) -> None:
    case_dir = tree / "int" / "cortex-m55" / "PoolingFunctions" / "pool_a_s8"
    next(case_dir.glob("*.sidecar.json")).write_text("{not json")
    with pytest.raises(InventoryError, match="unreadable sidecar"):
        case_calls(case_dir)


def _invoke(checkout: Path, tree: Path, *extra: str):
    return runner.invoke(app, ["contract", "inventory", "--cmsis-nn-root", str(checkout),
                               "--generated-tests-dir", str(tree), *extra])


def test_cli_reports_and_writes_the_json(checkout: Path, tree: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("helia_core_tester.contract.cli.find_repo_root_or_cwd", lambda: tmp_path / "proj")
    result = _invoke(checkout, tree)
    assert result.exit_code == 0, _text(result)
    assert "kernel      2 covered /    3 public (1 uncovered)" in _text(result)
    report = tmp_path / "proj" / "artifacts" / "reports" / "contracts" / "inventory.json"
    assert report.is_file()
    assert json.loads(report.read_text())["cases_scanned"] == 2


def test_cli_gates(checkout: Path, tree: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("helia_core_tester.contract.cli.find_repo_root_or_cwd", lambda: tmp_path / "proj")
    result = _invoke(checkout, tree, "--fail-on-uncovered")
    assert result.exit_code == 1, _text(result)
    assert "1 public kernel(s) are called by no generated case" in _text(result)
    _case(tree, "float", "cortex-m55", "PoolingFunctions", "ghost_f16", c_source="arm_fx_ghost_f16(out);\n")
    result = _invoke(checkout, tree)
    assert result.exit_code == 1, _text(result)
    assert "not in the contract" in _text(result)
    assert "? arm_fx_ghost_f16 (1 cases)" in _text(result)
    result = _invoke(checkout, tree, "--allow-unknown")
    assert result.exit_code == 0, _text(result)


def test_cli_exits_2_without_a_contract_or_cases(checkout: Path, tree: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("helia_core_tester.contract.cli.find_repo_root_or_cwd", lambda: tmp_path / "proj")
    (checkout / CONTRACT_RELPATH).unlink()
    result = _invoke(checkout, tree)
    assert result.exit_code == 2, _text(result)
    assert "no kernel contract" in _text(result)
    shutil.copy(FIXTURE, checkout / CONTRACT_RELPATH)
    (checkout / "Include" / "arm_nnfunctions.h").write_text("/* prototypes gone */\n")
    result = _invoke(checkout, tree)
    assert result.exit_code == 2, _text(result)
    assert "not declared in Include/arm_nnfunctions.h" in _text(result)
    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copy(FIXTURE, checkout / CONTRACT_RELPATH)
    for relative, text in HEADERS.items():
        (checkout / relative).write_text(text)
    result = _invoke(checkout, empty)
    assert result.exit_code == 2, _text(result)
    assert "no generated cases" in _text(result)
