"""`helia_core_tester contract ...`: read-only views over the kernel contract export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from helia_core_tester.contract.inventory import InventoryError, build_inventory, summary_lines, write_report
from helia_core_tester.contract.ir import ContractError, load_contract_set
from helia_core_tester.core.discovery import find_repo_root_or_cwd
from helia_core_tester.core.path_layout import generated_tests_root, reports_root
from helia_core_tester.core.config import configured_library
from helia_core_tester.core.errors import ConfigurationError
from helia_core_tester.core.library import LEGACY_LIBRARY, LibraryProfile, require_supported, resolve_library
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

contract_app = typer.Typer(
    name="contract",
    help="Read-only views over the library's kernel contract export and implementation manifest.",
    add_completion=False,
    no_args_is_help=True,
)

EXIT_UNCOVERED = 1
EXIT_INVALID = 1
EXIT_UNAVAILABLE = 2

_CMSIS_NN_ROOT_HELP = "ns-cmsis-nn checkout (default: $CMSIS_NN_ROOT, else the nested layout)."
_LIBRARY_HELP = ("Directory with the library's hct-library.yaml (default: $HELIA_CORE_TESTER_LIBRARY, else library "
                 "in helia_core_tester.toml). Not with a different --cmsis-nn-root.")


def _fail(message: str, code: int) -> None:
    typer.echo(f"✗ {message}", err=True)
    raise typer.Exit(code=code)


def _library_under_test(project_root: Path, library: Optional[Path], cmsis_nn_root: Optional[Path], *,
                        supported_only: bool) -> tuple[Path, Optional[LibraryProfile]]:
    """(checkout, profile) from --library and its configured forms, else (checkout, None) from
    --cmsis-nn-root or the resolved ns-cmsis-nn checkout. Exits 2 on a bad or conflicting library."""
    try:
        library_dir, source = configured_library(project_root, library)
        profile = resolve_library(library_dir) if library_dir is not None else None
        if profile is not None and supported_only:
            require_supported(profile)
    except ConfigurationError as error:
        _fail(str(error), EXIT_UNAVAILABLE)
    if profile is None:
        return (Path(cmsis_nn_root) if cmsis_nn_root is not None else resolve_cmsis_nn_root()), None
    if cmsis_nn_root is not None and Path(cmsis_nn_root).expanduser().resolve() != profile.source_root:
        _fail(f"{source} names {profile.describe()}, but --cmsis-nn-root names "
              f"{Path(cmsis_nn_root).expanduser().resolve()}; pass one of them", EXIT_UNAVAILABLE)
    return profile.source_root, profile


@contract_app.command()
def inventory(
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    library: Optional[Path] = typer.Option(None, "--library", help=_LIBRARY_HELP),
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
    project_root = find_repo_root_or_cwd()
    root, _profile = _library_under_test(project_root, library, cmsis_nn_root, supported_only=True)
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


@contract_app.command()
def validate(
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    library: Optional[Path] = typer.Option(None, "--library", help=_LIBRARY_HELP),
) -> None:
    """Check the library's kernel contract documents against hct's schemas.

    Reads the signature export and the implementation manifest the profile names (ns-cmsis-nn:
    Tests/KernelContracts/), with no library code run. Exit 0 when they validate, 1 when they do
    not, 2 when there is nothing to check (no library, no export).
    """
    project_root = find_repo_root_or_cwd()
    root, profile = _library_under_test(project_root, library, cmsis_nn_root, supported_only=False)
    named = profile is not None and profile.names_contracts
    if profile is not None:
        typer.echo(f"✓ Library: {profile.describe()}")
        if not named and profile.library != LEGACY_LIBRARY:
            _fail(f"{profile.profile_path}: names no contracts (contracts.signatures)", EXIT_UNAVAILABLE)
    try:
        contracts = load_contract_set(root, signatures=profile.signatures if named else None,
                                      implementations=profile.implementations if named else None)
    except ContractError as error:
        _fail(str(error), EXIT_INVALID)
    if not contracts.present:
        _fail(f"{contracts.root or 'no checkout resolved'}: has no signature export ({contracts.path}); nothing to "
              "validate", EXIT_UNAVAILABLE)
    if profile is not None and contracts.library != profile.library:
        _fail(f"{contracts.path} is for library {contracts.library!r}, but the profile is for {profile.library!r}",
              EXIT_INVALID)
    headers = len({decl.header for decl in contracts.functions.values()})
    typer.echo(f"✓ {contracts.path}: {contracts.schema}, {len(contracts.functions)} functions in {headers} header(s)")
    if contracts.manifest_path is None:
        typer.echo("⚠ implementation manifest: none")
        return
    claims: dict[str, int] = {}
    for record in contracts.implementations.values():
        claims[record.numerics.claim] = claims.get(record.numerics.claim, 0) + 1
    referenced = sum(1 for record in contracts.implementations.values() if record.numerics.reference)
    typer.echo(f"✓ {contracts.manifest_path}: {len(contracts.implementations)} implementations "
               f"({contracts.library} {contracts.library_version}); numerics: "
               f"{', '.join(f'{n} {claim}' for claim, n in sorted(claims.items()))}; {referenced} name a reference")
