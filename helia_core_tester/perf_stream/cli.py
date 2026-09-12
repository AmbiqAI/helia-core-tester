"""Typer wiring for the board-keyed hardware CLI.

Exposes three things the top-level `helia_core_tester` app mounts:

- `boards`            list the board table (assets/hardware_boards.yaml)
- `probes list|match` enumerate / resolve connected J-Link probes via pylink
- `hardware run|build|flash|stream|memory-report`

Every hardware command takes `--board` as its only identity flag; the CPU,
NSX board name, SEGGER device name, SWD speed and build dir are derived from
the board row. `--serial-no` is optional and resolves flag > $HPX_JLINK_SERIAL
> probe enumeration (see probes.py). The commands are thin adapters: the
behaviour lives in boards.py, probes.py, firmware_build.py, hardware_pipeline.py
and run_summary.py.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from .benchmark_firmware_report import generate_benchmark_server_memory_report
from .boards import BoardSpec, UnknownBoardError, default_board_id, load_board_table, resolve_board
from .phase0 import _repo_root
from .probes import ProbeResolutionError, list_probes, resolve_serial

hardware_app = typer.Typer(
    name="hardware",
    help="Run generated CMSIS-NN kernel tests on real Ambiq boards over SEGGER RTT (board-keyed).",
    add_completion=False,
    no_args_is_help=True,
)
probes_app = typer.Typer(
    name="probes",
    help="Enumerate and resolve connected J-Link probes.",
    add_completion=False,
    no_args_is_help=True,
)

_BOARD_HELP = "Board id from assets/hardware_boards.yaml (default: $HPX_BOARD, else apollo510_evb)."
_SERIAL_HELP = "J-Link probe serial number (default: $HPX_JLINK_SERIAL, else the single connected probe)."
_BUILD_DIR_HELP = "CMake build directory (default: build/perf_stream/<board>)."


def _fail(message: str) -> None:
    typer.echo(f"✗ {message}", err=True)
    sys.exit(1)


def _board(board_id: Optional[str]) -> BoardSpec:
    try:
        return resolve_board(board_id or default_board_id())
    except UnknownBoardError as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")


def _serial(explicit: Optional[int]) -> int:
    try:
        return resolve_serial(explicit)
    except ProbeResolutionError as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")


# --- boards / probes ---------------------------------------------------------------


def boards() -> None:
    """List the known hardware boards (assets/hardware_boards.yaml)."""
    table = load_board_table()
    header = ("id", "nsx_board", "cpu", "pmu_tier", "has_mve", "jlink_device", "swd_khz", "workspace_bytes")
    rows = [
        (b.id, b.nsx_board, b.cpu, b.pmu_tier, "yes" if b.has_mve else "no", b.jlink_device, str(b.swd_speed_khz), str(b.workspace_bytes))
        for b in table
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    typer.echo("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


@probes_app.command(name="list")
def probes_list() -> None:
    """Enumerate connected J-Link probes (serial number and product name) via pylink."""
    try:
        probes = list_probes()
    except ProbeResolutionError as exc:
        _fail(str(exc))
    if not probes:
        typer.echo("No connected J-Link probes detected.")
        return
    for probe in probes:
        typer.echo(f"{probe.serial}  {probe.product}".rstrip())


@probes_app.command(name="match")
def probes_match(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
) -> None:
    """Print the J-Link serial the hardware commands would use for --board
    ($HPX_JLINK_SERIAL, else the single connected probe). Exits 1 on 0 or >1 candidates."""
    spec = _board(board)
    serial = _serial(None)
    typer.echo(f"[probes] {spec.id} ({spec.jlink_device}) -> J-Link serial {serial}", err=True)
    typer.echo(str(serial))


# --- hardware -----------------------------------------------------------------------


@hardware_app.command()
def build(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel build jobs."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure even if the build dir already exists."),
) -> None:
    """Cross-compile the hct_benchmark_server firmware for --board (no flashing)."""
    from .firmware_build import build_firmware, resolve_build_dir

    spec = _board(board)
    elf = build_firmware(spec, build_dir=resolve_build_dir(_repo_root(), spec, build_dir), jobs=jobs, force_reconfigure=force_reconfigure)
    typer.echo(f"✓ Firmware build completed successfully: {elf}")


@hardware_app.command()
def flash(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    serial_no: Optional[int] = typer.Option(None, "--serial-no", help=_SERIAL_HELP),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel build jobs."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure even if the build dir already exists."),
    force: bool = typer.Option(False, "--force", help="Flash even if the ELF sha256 matches the last flash to this probe."),
) -> None:
    """Build (if needed) and flash the hct_benchmark_server firmware to --board via J-Link.
    Skipped when the ELF is unchanged since the last flash to the same probe."""
    from .firmware_build import flash_firmware, resolve_build_dir

    spec = _board(board)
    serial = _serial(serial_no)
    decision = flash_firmware(
        spec, serial, build_dir=resolve_build_dir(_repo_root(), spec, build_dir), jobs=jobs,
        force_reconfigure=force_reconfigure, force=force,
    )
    if decision.needed:
        typer.echo("✓ Firmware flashed successfully")
    else:
        typer.echo("✓ Firmware already up to date on the board (use --force to reflash)")


@hardware_app.command(name="memory-report")
def memory_report(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Directory to write memory_report.json into."),
) -> None:
    """Generate and print the flash/RAM memory_report.json for the linked firmware ELF."""
    from .firmware_build import resolve_build_dir

    spec = _board(board)
    path = generate_benchmark_server_memory_report(build_dir=resolve_build_dir(_repo_root(), spec, build_dir), output_root=output_root)
    typer.echo(json.dumps(json.loads(path.read_text()), indent=2))
    typer.echo(f"\n✓ Memory report written to {path}")


_SUITE_HELP = (
    "Generated-test suite to bridge: 'int' (S8/S16/S32/S4, default), 'float' (FP16/FP32), "
    "or 'both' to run int and float in one session -- one flash, one result bundle, with "
    "each case still gated and reported against its own suite."
)
_FVP_GATE_HELP = (
    "How much the FVP report is allowed to block a hardware run. 'advisory' (the default) "
    "skips only cases the FVP recorded as FAILING for these exact artifacts -- real evidence "
    "the kernel is wrong -- and runs cases whose report is merely stale or missing. 'strict' "
    "also skips stale/missing ones, for a CI job that just ran the FVP suite and wants full "
    "corroboration. 'off' consults the report for provenance only and never blocks. Every "
    "case's outcome is recorded in case_summary.csv's fvp_status column regardless."
)
_PRECISION_HELP = (
    "Float-suite shortcut: 'fp16' or 'fp32' forces --suite float and bridges only the "
    "_f16/_f32 generated cases. Cannot be combined with --suite both or --test-name."
)


def _stream_options(suite, family, test_name, limit, precision, pmu_groups, fvp_gate, session_id):
    from .hardware_pipeline import StreamOptions, apply_precision, parse_pmu_groups, validate_fvp_gate
    from .hardware_run import normalize_suites

    try:
        normalize_suites(suite)
        suite, test_name = apply_precision(precision, suite, test_name)
        validate_fvp_gate(fvp_gate)
    except ValueError as exc:
        _fail(str(exc))
    return StreamOptions(
        suite=suite, family=family, test_name=test_name, limit=limit,
        pmu_groups=parse_pmu_groups(pmu_groups), fvp_gate=fvp_gate, session_id=session_id,
    )


def _quiet_stdout(as_json: bool):
    """With --json, keep stdout clear for the one JSON document: the firmware build,
    J-Link flash and generation subprocesses all write to inherited stdout otherwise."""
    from .run_summary import stdout_to_stderr

    return stdout_to_stderr() if as_json else contextlib.nullcontext()


def _report(outcome, spec: BoardSpec, *, as_json: bool) -> None:
    from .run_summary import build_json_summary, print_run_report

    failed = print_run_report(outcome.result, outcome.skipped, outcome.bundle, err=as_json)
    if as_json:
        typer.echo(json.dumps(build_json_summary(
            outcome.result, outcome.skipped, session_id=outcome.session_id, board_id=spec.id, bundle=outcome.bundle,
        ), indent=2))
    if failed:
        typer.echo(typer.style("✗ One or more generated-test cases failed correctness", fg=typer.colors.RED, bold=True), err=True)
        sys.exit(1)
    typer.echo(
        typer.style(f"✓ {len(outcome.result.cases)} generated test case(s) passed on {spec.id}", fg=typer.colors.GREEN, bold=True),
        err=as_json,
    )


@hardware_app.command()
def stream(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    serial_no: Optional[int] = typer.Option(None, "--serial-no", help=_SERIAL_HELP),
    suite: str = typer.Option("int", "--suite", help=_SUITE_HELP),
    family: Optional[str] = typer.Option(None, "--family", help="Operator family under artifacts/generated_tests to bridge. Omit to bridge every family with real firmware dispatch support (see generated_test_bridge.bridged_families())."),
    test_name: Optional[str] = typer.Option(None, "--test-name", help="Only bridge generated tests whose directory name contains this substring."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only bridge the first N discovered generated tests (per suite/family)."),
    precision: Optional[str] = typer.Option(None, "--precision", help=_PRECISION_HELP),
    pmu_groups: str = typer.Option("cpu,memory,mve", "--pmu-groups", help="Comma-separated PMU counter groups to request."),
    fvp_gate: Optional[str] = typer.Option(None, "--fvp-gate", help=_FVP_GATE_HELP),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID; also the result-bundle directory name (default: <board>-<UTC timestamp>)."),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP + " Must hold the flashed firmware's ELF."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON summary document on stdout (human output goes to stderr)."),
) -> None:
    """Stream already-generated kernel tests to already-flashed firmware on --board over
    HCTP/RTT, check correctness, and write the result bundle. Run `generate` and
    `hardware flash` first, or use `hardware run` for the whole pipeline.

    Only kernels with real firmware dispatch support are bridged -- see the `_BUILDERS`
    dispatch table in `generated_test_bridge.py` (or call `bridged_families()` at
    runtime). Everything else is reported as skipped with the reason. Bridged cases are
    batched into groups of at most hardware_run.MAX_CASES_PER_SESSION, each run over its
    own fresh reset-on-open RTT session and merged into one result bundle.
    """
    from .firmware_build import resolve_build_dir
    from .hardware_pipeline import stream_generated_tests

    spec = _board(board)
    serial = _serial(serial_no)
    options = _stream_options(suite, family, test_name, limit, precision, pmu_groups, fvp_gate, session_id)
    echo = lambda msg: typer.echo(msg, err=as_json)  # noqa: E731
    try:
        with _quiet_stdout(as_json):
            outcome = stream_generated_tests(
                _repo_root(), spec, serial, build_dir=resolve_build_dir(_repo_root(), spec, build_dir),
                options=options, echo=echo, progress_to_stderr=as_json,
            )
    except RuntimeError as exc:
        _fail(str(exc))
    _report(outcome, spec, as_json=as_json)


@hardware_app.command()
def run(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    serial_no: Optional[int] = typer.Option(None, "--serial-no", help=_SERIAL_HELP),
    suite: str = typer.Option("int", "--suite", help=_SUITE_HELP),
    family: Optional[str] = typer.Option(None, "--family", help="Operator family under artifacts/generated_tests to bridge. Omit to bridge every family with real firmware dispatch support."),
    test_name: Optional[str] = typer.Option(None, "--test-name", help="Only bridge generated tests whose directory name contains this substring."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only bridge the first N discovered generated tests (per suite/family)."),
    precision: Optional[str] = typer.Option(None, "--precision", help=_PRECISION_HELP),
    pmu_groups: str = typer.Option("cpu,memory,mve", "--pmu-groups", help="Comma-separated PMU counter groups to request."),
    fvp_gate: Optional[str] = typer.Option(None, "--fvp-gate", help=_FVP_GATE_HELP),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID; also the result-bundle directory name (default: <board>-<UTC timestamp>)."),
    skip_generate: bool = typer.Option(False, "--skip-generate", help="Reuse existing artifacts/generated_tests instead of regenerating."),
    skip_flash: bool = typer.Option(False, "--skip-flash", help="Skip build+flash and reuse whatever firmware is already running on the board."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON summary document on stdout (human output goes to stderr)."),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel firmware build jobs."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure the CMake build dir even if it already exists."),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
) -> None:
    """The whole hardware pipeline: generate tests for the board's CPU, build the
    firmware, flash it if the ELF changed, stream the suite, write the result bundle,
    and print the summary."""
    from .hardware_pipeline import run_hardware_pipeline

    spec = _board(board)
    serial = _serial(serial_no)
    options = _stream_options(suite, family, test_name, limit, precision, pmu_groups, fvp_gate, session_id)
    echo = lambda msg: typer.echo(msg, err=as_json)  # noqa: E731
    try:
        with _quiet_stdout(as_json):
            outcome = run_hardware_pipeline(
                _repo_root(), spec, serial, options=options, build_dir=build_dir,
                skip_generate=skip_generate, skip_flash=skip_flash, jobs=jobs,
                force_reconfigure=force_reconfigure, echo=echo, progress_to_stderr=as_json,
            )
    except RuntimeError as exc:
        _fail(str(exc))
    _report(outcome, spec, as_json=as_json)
