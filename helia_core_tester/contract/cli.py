"""`helia_core_tester contract ...`: read-only views over the kernel contract export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from helia_core_tester.contract.inventory import InventoryError, build_inventory, summary_lines, write_report
from helia_core_tester.contract.ir import ContractError, load_contract_set
from helia_core_tester.core.discovery import find_repo_root_or_cwd
from helia_core_tester.core.path_layout import generated_tests_root, reports_root
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

contract_app = typer.Typer(
    name="contract",
    help="Read-only views over the ns-cmsis-nn kernel contract export (Tests/KernelContracts).",
    add_completion=False,
    no_args_is_help=True,
)

EXIT_UNCOVERED = 1
EXIT_UNAVAILABLE = 2


def _fail(message: str, code: int) -> None:
    typer.echo(f"✗ {message}", err=True)
    raise typer.Exit(code=code)


@contract_app.command()
def inventory(
    cmsis_nn_root: Optional[Path] = typer.Option(
        None, "--cmsis-nn-root", help="ns-cmsis-nn checkout (default: $CMSIS_NN_ROOT, else the nested layout)."),
    generated_tests_dir: Optional[Path] = typer.Option(
        None, "--generated-tests-dir", help="Generated tests tree to scan (default: artifacts/generated_tests)."),
    fail_on_uncovered: bool = typer.Option(
        False, "--fail-on-uncovered", help="Exit 1 when any public kernel is called by no generated case."),
    fail_on_unknown: bool = typer.Option(
        True, "--fail-on-unknown/--allow-unknown",
        help="Exit 1 when a case calls an arm_* symbol the contract does not know (stale export or leaked helper)."),
) -> None:
    """Report which public functions the generated cases call, and which they never do.

    Exit 0 on a complete inventory, 1 when a gate flag trips, 2 when no inventory can be
    produced (checkout without a contract, tree without cases): an empty inventory reads
    like full coverage and is never reported as success.
    """
    root = Path(cmsis_nn_root) if cmsis_nn_root is not None else resolve_cmsis_nn_root()
    project_root = find_repo_root_or_cwd()
    tree = Path(generated_tests_dir) if generated_tests_dir is not None else generated_tests_root(project_root)
    try:
        contracts = load_contract_set(root)
        result = build_inventory(contracts, tree)
    except ContractError as error:
        _fail(str(error), EXIT_UNAVAILABLE)
    except InventoryError as error:
        _fail(str(error), EXIT_UNAVAILABLE)
    report = write_report(result, reports_root(project_root))
    for line in summary_lines(result):
        typer.echo(line)
    typer.echo(f"report: {report}")
    if fail_on_unknown and result.unknown:
        _fail(f"{len(result.unknown)} symbol(s) called by cases are not in the contract", EXIT_UNCOVERED)
    if fail_on_uncovered and result.uncovered_by_kind("kernel"):
        _fail(f"{len(result.uncovered_by_kind('kernel'))} public kernel(s) are called by no generated case",
              EXIT_UNCOVERED)
