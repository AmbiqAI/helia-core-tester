"""Typer wiring for the board-keyed hardware CLI.

Exposes three things the top-level `helia_core_tester` app mounts:

- `boards`            list the board table (assets/hardware_boards.yaml)
- `probes list|match` enumerate / resolve connected J-Link probes via pylink
- `hardware run|build|flash|stream|memory-report`

Every hardware command takes `--board` as its only identity flag; the CPU,
NSX board name, SEGGER device name, SWD speed and build dir are derived from
the board row. `--serial-no` is optional and resolves flag > $HPX_JLINK_SERIAL
> probe enumeration (see probes.py). Option combinations are validated before
any probe resolution or hardware I/O, so a bad flag fails with its own message
rather than a hardware error. The commands are thin adapters: the behaviour
lives in boards.py, probes.py, firmware_build.py, hardware_pipeline.py and
run_summary.py.

Failures inside the pipeline -- a cmake/J-Link subprocess exiting non-zero, a
pylink error, a stalled transport -- print one line and exit 1; the traceback
is shown with `--verbosity 1` or higher (also `$HELIA_CORE_TESTER_VERBOSITY`,
the same knob the generate/build/run commands use).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import typer

from .boards import BoardSpec, UnknownBoardError, default_board_id, load_board_table, repo_root, resolve_board
from .memory_report import generate_memory_report
from .probes import ProbeResolutionError, list_probes, resolve_serial

if TYPE_CHECKING:
    from .firmware_build import FirmwareOptions

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
_BUILD_DIR_HELP = (
    "Build directory: holds the generated NSX app (nsx_app/) and its build tree "
    "(default: build/hardware/<board>)."
)
_VERBOSITY_HELP = "Verbosity level (0-3); 1 or higher prints the full traceback on failure (default: $HELIA_CORE_TESTER_VERBOSITY, else 0)."
_VERBOSITY_ENV_VAR = "HELIA_CORE_TESTER_VERBOSITY"
_FORCE_FLASH_HELP = (
    "Flash even if the ELF is unchanged since the last flash to this probe and the board "
    "already reports this build's id."
)
_ALLOW_UNVERIFIED_HELP = (
    "Stream even when the build dir has no hct_build_id.txt (firmware built before build-id "
    "stamping), skipping the TARGET_INFO build-id check. Without it a missing stamp is an error."
)
_BASELINE_HELP = (
    "Dependency baseline JSON pinning every NSX project the firmware resolves (default: "
    "assets/dependency_baseline.json). heliaPROFILER's compatibility-baseline file is "
    "accepted too, so a run can be built against hpx's qualified pins."
)
_CMSIS_NN_ROOT_HELP = (
    "Build the kernels from this ns-cmsis-nn checkout instead of the baseline's pinned "
    "commit. The tree is declared to NSX as a local module source and mirrored into the "
    "app on every sync, and the generate step reads its schemas from the same tree. Use it "
    "to test uncommitted kernel work; the result is not a qualified build."
)
_REQUANTIZE_HELP = (
    "Compile the kernels' requantize routine as inline assembly "
    "(NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM). On by default -- the same setting "
    "heliaPROFILER builds with. --no-requantize-inline-asm is the A/B control."
)
_UPDATE_DEPS_HELP = (
    "Re-resolve nsx.lock even when the manifest and the baseline are unchanged."
)


def _firmware_options(baseline, cmsis_nn_root, requantize_inline_asm, update_dependencies, verbosity) -> "FirmwareOptions":
    from .firmware_build import FirmwareOptions

    return FirmwareOptions(
        baseline_path=baseline,
        cmsis_nn_root=cmsis_nn_root,
        requantize_inline_asm=requantize_inline_asm,
        update_dependencies=update_dependencies,
        verbose=verbosity,
    )


def _fail(message: str) -> None:
    typer.echo(f"✗ {message}", err=True)
    sys.exit(1)


def _verbosity(explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    raw = os.environ.get(_VERBOSITY_ENV_VAR, "").strip()
    return int(raw) if raw.isdigit() else 0


def _is_jlink_exception(exc: BaseException) -> bool:
    try:
        import pylink
    except ImportError:  # pragma: no cover - pylink is a hard dependency of the transport
        return False
    return isinstance(exc, pylink.JLinkException)


@contextlib.contextmanager
def _pipeline_errors(verbosity: int) -> Iterator[None]:
    """Turn the failures the hardware pipeline is known to raise into one-line errors.

    RuntimeError covers this package's own errors (probe resolution, J-Link library
    config, session/protocol failures); CalledProcessError is cmake or the J-Link
    flash target; pylink's JLinkException is the probe/RTT layer; TimeoutError and
    FileNotFoundError are the transport write timeout and a missing ELF. Anything
    else is a bug and keeps its traceback.
    """
    try:
        yield
    except subprocess.CalledProcessError as exc:
        if verbosity >= 1:
            traceback.print_exc()
        command = " ".join(str(part) for part in exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd)
        _fail(f"Command failed with exit status {exc.returncode}: {command}")
    except (RuntimeError, TimeoutError, FileNotFoundError) as exc:
        if verbosity >= 1:
            traceback.print_exc()
        _fail(str(exc))
    except Exception as exc:
        if not _is_jlink_exception(exc):
            raise
        if verbosity >= 1:
            traceback.print_exc()
        _fail(f"J-Link error: {exc}")


def _board(board_id: Optional[str]) -> BoardSpec:
    try:
        return resolve_board(board_id or default_board_id())
    except UnknownBoardError as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")


def _serial(explicit: Optional[int], board: Optional[BoardSpec] = None) -> int:
    """Resolve the probe: flag > $HPX_JLINK_SERIAL > enumeration, with the board
    (when known) used to break a multi-probe tie by the core each probe reaches."""
    try:
        return resolve_serial(explicit, board=board)
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
    serial = _serial(None, spec)
    typer.echo(f"[probes] {spec.id} ({spec.jlink_device}) -> J-Link serial {serial}", err=True)
    typer.echo(str(serial))


# --- hardware -----------------------------------------------------------------------


@hardware_app.command()
def build(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel build jobs."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure even if the build dir already exists."),
    baseline: Optional[Path] = typer.Option(None, "--baseline", help=_BASELINE_HELP),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    requantize_inline_asm: bool = typer.Option(True, "--requantize-inline-asm/--no-requantize-inline-asm", help=_REQUANTIZE_HELP),
    update_dependencies: bool = typer.Option(False, "--update-dependencies", help=_UPDATE_DEPS_HELP),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
) -> None:
    """Build the hct_benchmark_server firmware for --board as an NSX app (no flashing)."""
    from .firmware_build import build_firmware, resolve_build_dir

    spec = _board(board)
    level = _verbosity(verbosity)
    options = _firmware_options(baseline, cmsis_nn_root, requantize_inline_asm, update_dependencies, level)
    with _pipeline_errors(level):
        elf = build_firmware(
            spec, build_dir=resolve_build_dir(repo_root(), spec, build_dir), jobs=jobs,
            force_reconfigure=force_reconfigure, options=options,
        )
    typer.echo(f"✓ Firmware build completed successfully: {elf}")


@hardware_app.command()
def flash(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    serial_no: Optional[int] = typer.Option(None, "--serial-no", help=_SERIAL_HELP),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    force: bool = typer.Option(False, "--force", help=_FORCE_FLASH_HELP),
    baseline: Optional[Path] = typer.Option(None, "--baseline", help=_BASELINE_HELP),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    requantize_inline_asm: bool = typer.Option(True, "--requantize-inline-asm/--no-requantize-inline-asm", help=_REQUANTIZE_HELP),
    update_dependencies: bool = typer.Option(False, "--update-dependencies", help=_UPDATE_DEPS_HELP),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
) -> None:
    """Flash the already-built hct_benchmark_server firmware to --board through the
    J-Link recipe NSX generated for it.

    This never builds, renders or reconfigures: a build dir that is missing, or whose
    recorded render inputs differ from the ones in force now (a changed baseline,
    kernel source or build option), is an error naming `hardware build`. The flash
    itself is skipped only when the ELF is unchanged since this build dir last flashed
    the same probe *and* the board confirms (in TARGET_INFO) that it runs this build's
    id; `--force` flashes regardless. The build options below are accepted because
    they select *which* render must be on disk, not to build one.
    """
    from .firmware_build import flash_firmware, resolve_build_dir

    spec = _board(board)
    serial = _serial(serial_no, spec)
    level = _verbosity(verbosity)
    options = _firmware_options(baseline, cmsis_nn_root, requantize_inline_asm, update_dependencies, level)
    with _pipeline_errors(level):
        decision = flash_firmware(
            spec, serial, build_dir=resolve_build_dir(repo_root(), spec, build_dir),
            force=force, options=options,
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
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
) -> None:
    """Generate and print the flash/RAM memory_report.json for the linked firmware ELF."""
    from .firmware_build import resolve_build_dir

    spec = _board(board)
    # Same one-line failures as the other hardware commands: a missing ELF is a
    # FileNotFoundError, a missing/failing arm-none-eabi-* tool a FileNotFoundError
    # or CalledProcessError.
    with _pipeline_errors(_verbosity(verbosity)):
        path = generate_memory_report(spec, build_dir=resolve_build_dir(repo_root(), spec, build_dir), output_root=output_root)
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


_PMU_COUNTERS_HELP = (
    "PMU counters to capture, as GROUP:SELECTION (repeatable; hpx syntax). GROUP is cpu, "
    "memory or mve; SELECTION is 'all', 'default', or a comma-separated list of ARM_PMU_* "
    "names from assets/pmu/armv8m_pmu_events.json, e.g. --pmu-counters mve:all "
    "--pmu-counters cpu:ARM_PMU_INST_RETIRED,ARM_PMU_STALL. Each group is measured in "
    "passes of up to 4 chained 32-bit counters; ARM_PMU_CPU_CYCLES is always reported. "
    "One run takes at most 16 passes (HCT_SERVER_MAX_PASSES), so cpu:all memory:all mve:all "
    "(18) is refused before anything is built or flashed. "
    "Default: cpu:default memory:default mve:default."
)
_PMU_GROUPS_HELP = "Deprecated alias for --pmu-counters GROUP:default per listed group."


def _stream_options(suite, family, test_name, limit, precision, pmu_counters, pmu_groups, fvp_gate, session_id):
    from .hardware_pipeline import StreamOptions, apply_precision, float_precision_for, resolve_pmu_options, validate_fvp_gate
    from .session_runner import canonical_suite

    try:
        # Canonicalise before the precision rules so `--suite BOTH` is refused
        # exactly like `--suite both` rather than slipping through as float.
        suite = canonical_suite(suite)
        suite, test_name = apply_precision(precision, suite, test_name)
        validate_fvp_gate(fvp_gate)
        selection = resolve_pmu_options(pmu_counters or [], pmu_groups, warn=lambda msg: typer.echo(msg, err=True))
    except ValueError as exc:
        _fail(str(exc))
    return StreamOptions(
        suite=suite, family=family, test_name=test_name, limit=limit,
        pmu_counters=selection, fvp_gate=fvp_gate, session_id=session_id,
        float_precision=float_precision_for(precision),
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
            timing=outcome.timing, dependencies=outcome.dependencies,
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
    pmu_counters: Optional[list[str]] = typer.Option(None, "--pmu-counters", help=_PMU_COUNTERS_HELP),
    pmu_groups: Optional[str] = typer.Option(None, "--pmu-groups", help=_PMU_GROUPS_HELP, hidden=True),
    fvp_gate: Optional[str] = typer.Option(None, "--fvp-gate", help=_FVP_GATE_HELP),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID; also the result-bundle directory name (default: <board>-<UTC timestamp>)."),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP + " Must hold the flashed firmware's ELF."),
    allow_unverified_firmware: bool = typer.Option(False, "--allow-unverified-firmware", help=_ALLOW_UNVERIFIED_HELP),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON summary document on stdout (human output goes to stderr)."),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
) -> None:
    """Stream already-generated kernel tests to already-flashed firmware on --board over
    HCTP/RTT, check correctness, and write the result bundle. Run `generate` and
    `hardware flash` first, or use `hardware run` for the whole pipeline.

    Only kernels with real firmware dispatch support are bridged -- see the `_BUILDERS`
    dispatch table in `generated_test_bridge.py` (or call `bridged_families()` at
    runtime). Everything else is reported as skipped with the reason. Bridged cases are
    batched by the limits the target advertises (cases and PMU passes per plan, receive
    buffer), each batch run over its own fresh reset-on-open RTT session and merged into
    one result bundle.
    """
    from .firmware_build import resolve_build_dir
    from .hardware_pipeline import finalize_timing, stream_generated_tests

    # Options first, probe last: a bad flag combination must fail with its own
    # message, not with whatever probe enumeration happens to hit.
    spec = _board(board)
    options = _stream_options(suite, family, test_name, limit, precision, pmu_counters, pmu_groups, fvp_gate, session_id)
    serial = _serial(serial_no, spec)
    echo = lambda msg: typer.echo(msg, err=as_json)  # noqa: E731
    with _pipeline_errors(_verbosity(verbosity)), _quiet_stdout(as_json):
        outcome = stream_generated_tests(
            repo_root(), spec, serial, build_dir=resolve_build_dir(repo_root(), spec, build_dir),
            options=options, echo=echo, progress_to_stderr=as_json, allow_unverified_firmware=allow_unverified_firmware,
        )
        finalize_timing(outcome, echo=echo)
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
    pmu_counters: Optional[list[str]] = typer.Option(None, "--pmu-counters", help=_PMU_COUNTERS_HELP),
    pmu_groups: Optional[str] = typer.Option(None, "--pmu-groups", help=_PMU_GROUPS_HELP, hidden=True),
    fvp_gate: Optional[str] = typer.Option(None, "--fvp-gate", help=_FVP_GATE_HELP),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID; also the result-bundle directory name (default: <board>-<UTC timestamp>)."),
    skip_generate: bool = typer.Option(False, "--skip-generate", help="Reuse existing artifacts/generated_tests instead of regenerating."),
    skip_flash: bool = typer.Option(False, "--skip-flash", help="Skip build+flash and reuse whatever firmware is already running on the board (its TARGET_INFO build id is still checked against the build dir)."),
    force_flash: bool = typer.Option(False, "--force-flash", help=_FORCE_FLASH_HELP + " Mirror of `hardware flash --force`."),
    allow_unverified_firmware: bool = typer.Option(False, "--allow-unverified-firmware", help=_ALLOW_UNVERIFIED_HELP + " Only meaningful with --skip-flash."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON summary document on stdout (human output goes to stderr)."),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel firmware build jobs."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure the CMake build dir even if it already exists."),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    baseline: Optional[Path] = typer.Option(None, "--baseline", help=_BASELINE_HELP),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    requantize_inline_asm: bool = typer.Option(True, "--requantize-inline-asm/--no-requantize-inline-asm", help=_REQUANTIZE_HELP),
    update_dependencies: bool = typer.Option(False, "--update-dependencies", help=_UPDATE_DEPS_HELP),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
) -> None:
    """The whole hardware pipeline: build the firmware as an NSX app, flash it unless
    the board already runs this exact build, generate tests for the board's CPU from
    the same kernel checkout the firmware links, stream the suite, write the result
    bundle, and print the summary."""
    from .hardware_pipeline import run_hardware_pipeline

    if skip_flash and force_flash:
        _fail("--skip-flash and --force-flash cannot be combined.")
    spec = _board(board)
    options = _stream_options(suite, family, test_name, limit, precision, pmu_counters, pmu_groups, fvp_gate, session_id)
    serial = _serial(serial_no, spec)
    level = _verbosity(verbosity)
    firmware_options = _firmware_options(baseline, cmsis_nn_root, requantize_inline_asm, update_dependencies, level)
    echo = lambda msg: typer.echo(msg, err=as_json)  # noqa: E731
    with _pipeline_errors(level), _quiet_stdout(as_json):
        outcome = run_hardware_pipeline(
            repo_root(), spec, serial, options=options, build_dir=build_dir,
            skip_generate=skip_generate, skip_flash=skip_flash, force_flash=force_flash, jobs=jobs,
            force_reconfigure=force_reconfigure, firmware_options=firmware_options, echo=echo,
            progress_to_stderr=as_json, allow_unverified_firmware=allow_unverified_firmware,
        )
    _report(outcome, spec, as_json=as_json)
