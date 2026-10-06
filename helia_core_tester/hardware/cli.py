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

Exit codes, shared with `score`: 0 pass; 1 a case failed correctness; 2 bad
flags; 3 refused before running (dirty tester, a --skip-flash or --golden-from
mismatch, no case matches); 5 error (cmake, J-Link, transport, probe, or a
tester bug). Failures print one line;
the traceback is shown with `--verbosity 1` or higher (also
`$HELIA_CORE_TESTER_VERBOSITY`, the same knob the generate/build/run commands
use).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Iterator, Optional

import typer

from .boards import BoardSpec, UnknownBoardError, default_board_id, load_board_table, repo_root, resolve_board
from .memory_report import generate_memory_report
from .probes import ProbeResolutionError, list_probes, resolve_serial
from .score import EXIT_FAIL, EXIT_REFUSED
from .wire import clock_mhz

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
_BUILD_DIR_HELP = "CMake build directory (default: build/hardware/<board>)."
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


# Score's codes, plus usage and error.
EXIT_USAGE, EXIT_ERROR = 2, 5


def _fail(message: str, code: int = EXIT_USAGE) -> None:
    typer.echo(f"✗ {message}", err=True)
    sys.exit(code)


def _bugs_exit_error(command):
    """Unexpected errors exit 5, not 1."""

    @functools.wraps(command)
    def wrapper(*args, **kwargs):
        import click

        try:
            return command(*args, **kwargs)
        # Usage, refusal and normal exits.
        except (click.exceptions.ClickException, click.exceptions.Exit, click.exceptions.Abort):
            raise
        except Exception:
            traceback.print_exc()
            sys.exit(EXIT_ERROR)

    return wrapper


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

    RunRefused (a golden or case-selection misfit) exits EXIT_REFUSED.
    RuntimeError covers this package's own errors (probe resolution, J-Link library
    config, session/protocol failures); CalledProcessError is cmake or the J-Link
    flash target; pylink's JLinkException is the probe/RTT layer; TimeoutError and
    FileNotFoundError are the transport write timeout and a missing ELF. These
    exit EXIT_ERROR. Anything else is a bug: it prints its traceback and
    also exits EXIT_ERROR, never the correctness code.
    """
    from .errors import RunRefused

    try:
        yield
    except subprocess.CalledProcessError as exc:
        if verbosity >= 1:
            traceback.print_exc()
        command = " ".join(str(part) for part in exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd)
        _fail(f"Command failed with exit status {exc.returncode}: {command}", EXIT_ERROR)
    except RunRefused as exc:
        if verbosity >= 1:
            traceback.print_exc()
        _fail(str(exc), EXIT_REFUSED)
    except (RuntimeError, TimeoutError, FileNotFoundError) as exc:
        if verbosity >= 1:
            traceback.print_exc()
        _fail(str(exc), EXIT_ERROR)
    except Exception as exc:
        if not _is_jlink_exception(exc):
            # A bug: keep the traceback, exit 5.
            traceback.print_exc()
            sys.exit(EXIT_ERROR)
        if verbosity >= 1:
            traceback.print_exc()
        _fail(f"J-Link error: {exc}", EXIT_ERROR)


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
        _fail(str(exc), EXIT_ERROR)
        raise AssertionError("unreachable")


_DIRTY_TESTER_HELP = (
    "Run a --cmsis-nn-root candidate from an uncommitted tester. "
    "The bundle records harness.tester_dirty and a diff hash."
)
_CMSIS_NN_REF_HELP = "ns-cmsis-nn tag or commit to build (default: see --cmsis-nn-root)."
_CMSIS_NN_ROOT_HELP = (
    "Local ns-cmsis-nn checkout to build. Default: the last build's checkout or "
    "--cmsis-nn-ref in this build dir, else the enclosing checkout when the tester sits at "
    "ns-cmsis-nn/Tests/helia-core-tester, else the pinned release. "
    "Copies its Include/, Source/, cmake/ and nsx/ into the app."
)
_PLACEMENT_HELP = (
    "Operand memory: tcm (one workspace: DTCM on Apollo5, SRAM on Apollo3P) "
    "or mram (weights and bias in cached MRAM, evicted before each call; "
    "activations and scratch in DTCM). Default: tcm."
)
_JOBS_HELP = "Parallel build jobs (default: CPU count + 2, like ninja)."
_UPDATE_DEPS_HELP = "Re-resolve NSX modules and rewrite nsx.lock before building."
_INLINE_ASM_HELP = "Build requantize with or without inline assembly (default: on)."


def _check_placement(placement, spec: BoardSpec) -> None:
    from .nsx_app import PLACEMENTS

    if placement is not None and placement not in PLACEMENTS:
        _fail(f"--placement must be one of: {', '.join(PLACEMENTS)}.")
    if placement == "mram" and not spec.has_mram:
        _fail(f"{spec.id} has no cached MRAM; use tcm.")


def _check_tester_clean(allow: bool, echo) -> None:
    """Candidate runs need a committed tester."""
    from .harness_lock import tester_state

    # Unknown state counts as dirty.
    if tester_state(repo_root())["dirty"] is False:
        return
    if not allow:
        _fail("Tester worktree is dirty; commit or pass --allow-dirty-tester.", EXIT_REFUSED)
    echo("[hardware] WARNING: tester is dirty; bundle marks tester_dirty.")


def _app_options(build_dir: Path, cmsis_nn_ref, cmsis_nn_root, inline_asm, placement=None):
    """Kernel flags over the build dir's saved options."""
    from .firmware_build import nsx_app_dir
    from .nsx_app import AppRenderError, resolve_options, saved_options

    if cmsis_nn_ref and cmsis_nn_root:
        _fail("Pass --cmsis-nn-ref or --cmsis-nn-root, not both.")
    app_dir = nsx_app_dir(build_dir)
    try:
        options = resolve_options(
            app_dir, repo_root(), cmsis_nn_ref=cmsis_nn_ref, cmsis_nn_root=cmsis_nn_root, inline_asm=inline_asm,
            placement=placement,
        )
    except AppRenderError as exc:
        _fail(f"{exc}; pass --cmsis-nn-root or --cmsis-nn-ref.", EXIT_REFUSED)
    saved = saved_options(app_dir)
    # Compare values: templates embed paths.
    changes = options.changes_from(saved) if saved else []
    if changes:
        typer.echo(f"[hardware] Options changed, rebuilding: {'; '.join(changes)}", err=True)
    elif saved is None and (app_dir / "nsx.yml").is_file():
        typer.echo("[hardware] No saved build options; using defaults.", err=True)
    return options


def _built_options(build_dir: Path, cmsis_nn_ref, cmsis_nn_root, inline_asm, placement=None, stream_only=False):
    """The flashed build's options, unchanged."""
    from .firmware_build import nsx_app_dir
    from .nsx_app import AppRenderError, resolve_options, saved_options

    if cmsis_nn_ref and cmsis_nn_root:
        _fail("Pass --cmsis-nn-ref or --cmsis-nn-root, not both.")
    app_dir = nsx_app_dir(build_dir)
    saved = saved_options(app_dir)
    if saved is None:
        _fail("--skip-flash needs a saved build; run hardware build.", EXIT_REFUSED)
    try:
        wanted = resolve_options(
            app_dir, repo_root(), cmsis_nn_ref=cmsis_nn_ref, cmsis_nn_root=cmsis_nn_root, inline_asm=inline_asm,
            placement=placement, follow_pin=False,
        )
    except AppRenderError as exc:
        if not stream_only:
            _fail(f"{exc}; pass --skip-generate to stream only.", EXIT_REFUSED)
        # Streaming never reads the checkout.
        passed = {"requantize_inline_asm": inline_asm, "placement": placement}
        wanted = dataclasses.replace(saved, **{k: v for k, v in passed.items() if v is not None})
    # Generation must match the flashed firmware.
    changes = wanted.changes_from(saved)
    if changes:
        _fail(f"--skip-flash keeps the built kernels: {'; '.join(changes)}", EXIT_REFUSED)
    typer.echo(f"[hardware] Kernels: {saved.summary()}", err=True)
    return saved


def _saved_kernels(build_dir: Path, echo) -> None:
    """Print the kernels the build dir built."""
    from .firmware_build import nsx_app_dir
    from .nsx_app import saved_options

    saved = saved_options(nsx_app_dir(build_dir))
    echo(f"[hardware] Kernels: {saved.summary() if saved else 'unknown, no saved options'}")


# --- explain -----------------------------------------------------------------------


def explain(
    bundle: Path = typer.Argument(..., help="Bundle dir, or a dir holding bundles."),
    case: Optional[list[str]] = typer.Option(None, "--case", help="Case id substring (repeatable)."),
    op: Optional[list[str]] = typer.Option(
        None, "--op", help="conv, depthwise or fc; else a case or symbol substring (repeatable)."
    ),
    all_cases: bool = typer.Option(False, "--all", help="Include cases without MAC counts."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON document."),
) -> None:
    """Explain PMU counters per case: peak, stalls, hints."""
    from .pmu_explain import SCHEMA, SCHEMA_VERSION, explain_bundle

    dirs = sorted(path.parent for path in bundle.rglob("cases.json") if (path.parent / "session_manifest.json").is_file())
    if not dirs:
        raise typer.BadParameter(f"No result bundle under {bundle}")
    results = [explain_bundle(path, tuple(case or ()), tuple(op or ()), all_cases) for path in dirs]
    if as_json:
        # One flat case list; each names its bundle.
        cases = [
            {"bundle": result["bundle"], **explanation.to_dict()} for result in results for explanation in result["cases"]
        ]
        bundles = [{key: value for key, value in result.items() if key != "cases"} for result in results]
        typer.echo(json.dumps({"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "bundles": bundles, "cases": cases}, indent=2))
        return
    for result in results:
        typer.echo(f"# {result['board']} ({result['cpu']}, {result['placement']}): {result['bundle']}")
        for explanation in result["cases"]:
            typer.echo("\n".join(explanation.lines()))


# --- boards / probes ---------------------------------------------------------------


def boards() -> None:
    """List the known hardware boards (assets/hardware_boards.yaml)."""
    table = load_board_table()
    header = ("id", "nsx_board", "cpu", "pmu_tier", "has_mve", "jlink_device", "swd_khz", "workspace_bytes", "core_clock")
    rows = [
        (
            b.id, b.nsx_board, b.cpu, b.pmu_tier, "yes" if b.has_mve else "no", b.jlink_device, str(b.swd_speed_khz),
            str(b.workspace_bytes), clock_mhz(b.core_clock_hz) if b.core_clock_hz else "-",
        )
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
        _fail(str(exc), EXIT_ERROR)
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
    ($HPX_JLINK_SERIAL, else the single connected probe). Exits 5 on 0 or >1 candidates."""
    spec = _board(board)
    serial = _serial(None)
    typer.echo(f"[probes] {spec.id} ({spec.jlink_device}) -> J-Link serial {serial}", err=True)
    typer.echo(str(serial))


# --- hardware -----------------------------------------------------------------------


@hardware_app.command()
@_bugs_exit_error
def build(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help=_JOBS_HELP),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure even if the build dir already exists."),
    cmsis_nn_ref: Optional[str] = typer.Option(None, "--cmsis-nn-ref", help=_CMSIS_NN_REF_HELP),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    inline_asm: Optional[bool] = typer.Option(None, "--inline-asm/--no-inline-asm", help=_INLINE_ASM_HELP),
    placement: Optional[str] = typer.Option(None, "--placement", help=_PLACEMENT_HELP),
    update_dependencies: bool = typer.Option(False, "--update-dependencies", help=_UPDATE_DEPS_HELP),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
) -> None:
    """Cross-compile the hct_benchmark_server firmware for --board (no flashing)."""
    from .firmware_build import build_firmware, resolve_build_dir

    spec = _board(board)
    _check_placement(placement, spec)
    build_dir = resolve_build_dir(repo_root(), spec, build_dir)
    app_options = _app_options(build_dir, cmsis_nn_ref, cmsis_nn_root, inline_asm, placement)
    with _pipeline_errors(_verbosity(verbosity)):
        elf = build_firmware(
            spec, build_dir=build_dir, jobs=jobs,
            force_reconfigure=force_reconfigure, options=app_options, update_dependencies=update_dependencies,
        )
    typer.echo(f"✓ Firmware build completed successfully: {elf}")


@hardware_app.command()
@_bugs_exit_error
def flash(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    serial_no: Optional[int] = typer.Option(None, "--serial-no", help=_SERIAL_HELP),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help=_JOBS_HELP),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure even if the build dir already exists."),
    force: bool = typer.Option(False, "--force", help=_FORCE_FLASH_HELP),
    cmsis_nn_ref: Optional[str] = typer.Option(None, "--cmsis-nn-ref", help=_CMSIS_NN_REF_HELP),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    inline_asm: Optional[bool] = typer.Option(None, "--inline-asm/--no-inline-asm", help=_INLINE_ASM_HELP),
    placement: Optional[str] = typer.Option(None, "--placement", help=_PLACEMENT_HELP),
    update_dependencies: bool = typer.Option(False, "--update-dependencies", help=_UPDATE_DEPS_HELP),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
) -> None:
    """Build (if needed) and flash the hct_benchmark_server firmware to --board via J-Link.
    Skipped only when the ELF is unchanged since this build dir last flashed the same
    probe *and* the board confirms (in TARGET_INFO) that it runs this build's id."""
    from .firmware_build import flash_firmware, resolve_build_dir

    spec = _board(board)
    _check_placement(placement, spec)
    build_dir = resolve_build_dir(repo_root(), spec, build_dir)
    app_options = _app_options(build_dir, cmsis_nn_ref, cmsis_nn_root, inline_asm, placement)
    serial = _serial(serial_no)
    with _pipeline_errors(_verbosity(verbosity)):
        decision = flash_firmware(
            spec, serial, build_dir=build_dir, jobs=jobs,
            force_reconfigure=force_reconfigure, force=force, options=app_options,
            update_dependencies=update_dependencies,
        )
    if decision.needed:
        typer.echo("✓ Firmware flashed successfully")
    else:
        typer.echo("✓ Firmware already up to date on the board (use --force to reflash)")


@hardware_app.command(name="memory-report")
@_bugs_exit_error
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


_OP_HELP = (
    "Only bridge cases of this operator (repeatable), matched like "
    "`generate --op`: operator, descriptor stem or name prefix, e.g. DepthwiseConv."
)
_DTYPE_HELP = (
    "Only bridge cases of this dtype (repeatable), matched like `generate --dtype`: "
    "the activation dtype, or S4 for s4-weight cases, e.g. S8 or FP16."
)
_CASE_ID_HELP = "Only bridge this exact case id or test name (repeatable)."
_CASES_FROM_HELP = "File of case ids, one per line ('#' comments)."


_PMU_COUNTERS_HELP = (
    "PMU counters to capture, as GROUP:SELECTION (repeatable; hpx syntax). GROUP is cpu, "
    "memory or mve; SELECTION is 'all', 'default', or a comma-separated list of ARM_PMU_* "
    "names from assets/pmu/armv8m_pmu_events.json, e.g. --pmu-counters mve:all "
    "--pmu-counters cpu:ARM_PMU_INST_RETIRED,ARM_PMU_STALL. A bare 'all' is cpu:all "
    "memory:all mve:all: the full catalog in one run and one bundle. Each group is measured "
    "in passes of up to 4 chained 32-bit counters; ARM_PMU_CPU_CYCLES is always reported. "
    "Default: cpu:default memory:default mve:default."
)
_PMU_GROUPS_HELP = "Deprecated alias for --pmu-counters GROUP:default per listed group."
_STRICT_HELP = (
    "Require bit-exact integer outputs: ignore per-operator LSB tolerances. "
    "Every bundle records max_abs_diff and diff_count either way."
)
_GOLDEN_FROM_HELP = (
    "Result bundle dir whose outputs/ become the goldens (self-golden). "
    "Implies --strict-compare: every int case must match that run bit for bit; "
    "float cases keep their tolerance. Use it to judge kernel changes. "
    "Refuses a case the bundle lacks, one that run failed "
    "(see --golden-allow-failed), or one run on other inputs."
)
_GOLDEN_ALLOW_HELP = "With --golden-from, accept cases the golden run failed."


def _read_case_ids(case_ids: Optional[list[str]], cases_from: Optional[Path]) -> tuple[str, ...]:
    """Join --case-id with --cases-from lines."""
    ids = list(case_ids or [])
    if cases_from is not None:
        try:
            lines = cases_from.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            _fail(f"Cannot read --cases-from: {exc}")
        listed = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        # Empty would mean "run everything".
        if not listed:
            _fail(f"--cases-from lists no case ids: {cases_from}")
        ids += listed
    return tuple(ids)


def _check_ops(ops: tuple[str, ...]) -> None:
    """Fail on an op no descriptor matches."""
    from ..core.discovery import find_descriptors_dir
    from ..generation.io.descriptors import DescriptorLoadError, load_all_descriptors, unmatched_ops

    if not ops:
        return
    try:
        catalog = load_all_descriptors(str(find_descriptors_dir(repo_root())))
    except DescriptorLoadError as exc:
        raise ValueError(f"Descriptor catalog failed to load: {exc}") from exc
    unknown = unmatched_ops(catalog, list(ops))
    if unknown:
        raise ValueError(f"No descriptor matches --op: {', '.join(unknown)}")


def _stream_options(
    spec, suite, family, test_name, limit, precision, pmu_counters, pmu_groups, fvp_gate, session_id,
    ops, dtypes, case_ids, cases_from, strict_compare, golden_from, golden_allow_failed,
):
    from .hardware_pipeline import (
        StreamOptions, apply_precision, fit_to_board, float_precision_for, resolve_pmu_options, validate_fvp_gate,
    )
    from .generated_test_bridge import CaseSelection
    from .session_runner import canonical_suite

    try:
        # Canonicalise before the precision rules so `--suite BOTH` is refused
        # exactly like `--suite both` rather than slipping through as float.
        suite = canonical_suite(suite)
        suite, test_name = apply_precision(precision, suite, test_name)
        validate_fvp_gate(fvp_gate)
        cases = CaseSelection(tuple(ops or ()), tuple(dtypes or ()), _read_case_ids(case_ids, cases_from))
        _check_ops(cases.ops)
        # Exact ids already bound the run.
        if cases.case_ids and limit is not None:
            raise ValueError("--limit cannot combine with --case-id or --cases-from.")
        selection = resolve_pmu_options(pmu_counters or [], pmu_groups, warn=lambda msg: typer.echo(msg, err=True))
        options = StreamOptions(
            suite=suite, family=family, test_name=test_name, limit=limit,
            ops=cases.ops, dtypes=cases.dtypes, case_ids=cases.case_ids,
            pmu_counters=selection, fvp_gate=fvp_gate, session_id=session_id,
            float_precision=float_precision_for(precision), strict_compare=strict_compare, golden_from=golden_from,
            golden_allow_failed=golden_allow_failed,
        )
        return fit_to_board(spec, options, explicit_pmu=bool(pmu_counters) or pmu_groups is not None)
    except ValueError as exc:
        _fail(str(exc))


def _quiet_stdout(as_json: bool):
    """With --json, keep stdout clear for the one JSON document: the firmware build,
    J-Link flash and generation subprocesses all write to inherited stdout otherwise."""
    from .run_summary import stdout_to_stderr

    return stdout_to_stderr() if as_json else contextlib.nullcontext()


def _report(outcome, spec: BoardSpec, options, *, as_json: bool) -> None:
    from .hardware_pipeline import resolved_selection
    from .run_summary import build_json_summary, print_run_report

    failed = print_run_report(outcome.result, outcome.skipped, outcome.bundle, err=as_json, coverage=outcome.coverage)
    if as_json:
        typer.echo(json.dumps(build_json_summary(
            outcome.result, outcome.skipped, session_id=outcome.session_id, board_id=spec.id, bundle=outcome.bundle,
            selection=resolved_selection(repo_root(), spec, options), timing=outcome.timing,
            coverage=outcome.coverage,
        ), indent=2))
    if failed:
        typer.echo(typer.style("✗ One or more generated-test cases failed correctness", fg=typer.colors.RED, bold=True), err=True)
        sys.exit(EXIT_FAIL)
    typer.echo(
        typer.style(f"✓ {len(outcome.result.cases)} generated test case(s) passed on {spec.id}", fg=typer.colors.GREEN, bold=True),
        err=as_json,
    )


@hardware_app.command()
@_bugs_exit_error
def stream(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    serial_no: Optional[int] = typer.Option(None, "--serial-no", help=_SERIAL_HELP),
    suite: str = typer.Option("int", "--suite", help=_SUITE_HELP),
    family: Optional[str] = typer.Option(None, "--family", help="Operator family under artifacts/generated_tests to bridge. Omit to bridge every family with real firmware dispatch support (see generated_test_bridge.bridged_families())."),
    test_name: Optional[str] = typer.Option(None, "--test-name", help="Only bridge generated tests whose directory name contains this substring."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only bridge the first N discovered generated tests (per suite/family)."),
    op: Optional[list[str]] = typer.Option(None, "--op", help=_OP_HELP),
    dtype: Optional[list[str]] = typer.Option(None, "--dtype", help=_DTYPE_HELP),
    case_id: Optional[list[str]] = typer.Option(None, "--case-id", help=_CASE_ID_HELP),
    cases_from: Optional[Path] = typer.Option(None, "--cases-from", help=_CASES_FROM_HELP),
    precision: Optional[str] = typer.Option(None, "--precision", help=_PRECISION_HELP),
    pmu_counters: Optional[list[str]] = typer.Option(None, "--pmu-counters", help=_PMU_COUNTERS_HELP),
    pmu_groups: Optional[str] = typer.Option(None, "--pmu-groups", help=_PMU_GROUPS_HELP, hidden=True),
    fvp_gate: Optional[str] = typer.Option(None, "--fvp-gate", help=_FVP_GATE_HELP),
    strict_compare: bool = typer.Option(False, "--strict-compare", help=_STRICT_HELP),
    golden_from: Optional[Path] = typer.Option(
        None, "--golden-from", help=_GOLDEN_FROM_HELP, exists=True, file_okay=False, resolve_path=True,
    ),
    golden_allow_failed: bool = typer.Option(False, "--golden-allow-failed", help=_GOLDEN_ALLOW_HELP),
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
    from .hardware_pipeline import finalize_timing, prepare_bundles, stream_generated_tests

    # Options first, probe last: a bad flag combination must fail with its own
    # message, not with whatever probe enumeration happens to hit.
    spec = _board(board)
    options = _stream_options(
        spec, suite, family, test_name, limit, precision, pmu_counters, pmu_groups, fvp_gate, session_id,
        op, dtype, case_id, cases_from, strict_compare, golden_from, golden_allow_failed,
    )
    prepared = None
    if golden_from is not None:
        # Check goldens before probe access.
        with _pipeline_errors(_verbosity(verbosity)), _quiet_stdout(as_json):
            prepared = prepare_bundles(repo_root(), spec, options)
    serial = _serial(serial_no)
    echo = lambda msg: typer.echo(msg, err=as_json)  # noqa: E731
    build_dir = resolve_build_dir(repo_root(), spec, build_dir)
    _saved_kernels(build_dir, echo)
    with _pipeline_errors(_verbosity(verbosity)), _quiet_stdout(as_json):
        outcome = stream_generated_tests(
            repo_root(), spec, serial, build_dir=build_dir,
            options=options, echo=echo, progress_to_stderr=as_json, allow_unverified_firmware=allow_unverified_firmware,
            prepared=prepared,
        )
        finalize_timing(outcome, echo=echo)
    _report(outcome, spec, options, as_json=as_json)


@hardware_app.command()
@_bugs_exit_error
def run(
    board: Optional[str] = typer.Option(None, "--board", help=_BOARD_HELP),
    serial_no: Optional[int] = typer.Option(None, "--serial-no", help=_SERIAL_HELP),
    suite: str = typer.Option("int", "--suite", help=_SUITE_HELP),
    family: Optional[str] = typer.Option(None, "--family", help="Operator family under artifacts/generated_tests to bridge. Omit to bridge every family with real firmware dispatch support."),
    test_name: Optional[str] = typer.Option(None, "--test-name", help="Only bridge generated tests whose directory name contains this substring."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only bridge the first N discovered generated tests (per suite/family)."),
    op: Optional[list[str]] = typer.Option(None, "--op", help=_OP_HELP),
    dtype: Optional[list[str]] = typer.Option(None, "--dtype", help=_DTYPE_HELP),
    case_id: Optional[list[str]] = typer.Option(None, "--case-id", help=_CASE_ID_HELP),
    cases_from: Optional[Path] = typer.Option(None, "--cases-from", help=_CASES_FROM_HELP),
    precision: Optional[str] = typer.Option(None, "--precision", help=_PRECISION_HELP),
    pmu_counters: Optional[list[str]] = typer.Option(None, "--pmu-counters", help=_PMU_COUNTERS_HELP),
    pmu_groups: Optional[str] = typer.Option(None, "--pmu-groups", help=_PMU_GROUPS_HELP, hidden=True),
    fvp_gate: Optional[str] = typer.Option(None, "--fvp-gate", help=_FVP_GATE_HELP),
    strict_compare: bool = typer.Option(False, "--strict-compare", help=_STRICT_HELP),
    golden_from: Optional[Path] = typer.Option(
        None, "--golden-from", help=_GOLDEN_FROM_HELP, exists=True, file_okay=False, resolve_path=True,
    ),
    golden_allow_failed: bool = typer.Option(False, "--golden-allow-failed", help=_GOLDEN_ALLOW_HELP),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID; also the result-bundle directory name (default: <board>-<UTC timestamp>)."),
    skip_generate: bool = typer.Option(False, "--skip-generate", help="Reuse existing artifacts/generated_tests instead of regenerating."),
    skip_flash: bool = typer.Option(False, "--skip-flash", help="Skip build+flash and reuse whatever firmware is already running on the board (its TARGET_INFO build id is still checked against the build dir)."),
    force_flash: bool = typer.Option(False, "--force-flash", help=_FORCE_FLASH_HELP + " Mirror of `hardware flash --force`."),
    allow_unverified_firmware: bool = typer.Option(False, "--allow-unverified-firmware", help=_ALLOW_UNVERIFIED_HELP + " Only meaningful with --skip-flash."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON summary document on stdout (human output goes to stderr)."),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help=_JOBS_HELP),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure the CMake build dir even if it already exists."),
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help=_BUILD_DIR_HELP),
    cmsis_nn_ref: Optional[str] = typer.Option(None, "--cmsis-nn-ref", help=_CMSIS_NN_REF_HELP),
    cmsis_nn_root: Optional[Path] = typer.Option(None, "--cmsis-nn-root", help=_CMSIS_NN_ROOT_HELP),
    inline_asm: Optional[bool] = typer.Option(None, "--inline-asm/--no-inline-asm", help=_INLINE_ASM_HELP),
    placement: Optional[str] = typer.Option(None, "--placement", help=_PLACEMENT_HELP),
    update_dependencies: bool = typer.Option(False, "--update-dependencies", help=_UPDATE_DEPS_HELP),
    verbosity: Optional[int] = typer.Option(None, "--verbosity", "-v", help=_VERBOSITY_HELP),
    allow_dirty_tester: bool = typer.Option(False, "--allow-dirty-tester", help=_DIRTY_TESTER_HELP),
) -> None:
    """The whole hardware pipeline: generate tests for the board's CPU, build the
    firmware, flash it unless the board already runs this exact build, stream the
    suite, write the result bundle, and print the summary."""
    from .firmware_build import resolve_build_dir
    from .hardware_pipeline import run_hardware_pipeline

    if skip_flash and force_flash:
        _fail("--skip-flash and --force-flash cannot be combined.")
    spec = _board(board)
    _check_placement(placement, spec)
    options = _stream_options(
        spec, suite, family, test_name, limit, precision, pmu_counters, pmu_groups, fvp_gate, session_id,
        op, dtype, case_id, cases_from, strict_compare, golden_from, golden_allow_failed,
    )
    build_dir = resolve_build_dir(repo_root(), spec, build_dir)
    # Neither builds nor generates: nothing to resolve.
    streams_only = skip_generate and skip_flash
    if streams_only:
        # Passed build flags must match it.
        if any(flag is not None for flag in (cmsis_nn_ref, cmsis_nn_root, inline_asm, placement)):
            _built_options(build_dir, cmsis_nn_ref, cmsis_nn_root, inline_asm, placement, stream_only=True)
        app_options = None
    elif skip_flash:
        app_options = _built_options(build_dir, cmsis_nn_ref, cmsis_nn_root, inline_asm, placement)
    else:
        app_options = _app_options(build_dir, cmsis_nn_ref, cmsis_nn_root, inline_asm, placement)
    echo = lambda msg: typer.echo(msg, err=as_json)  # noqa: E731
    # Refuse before probing the board.
    if cmsis_nn_root is not None:
        _check_tester_clean(allow_dirty_tester, echo)
    serial = _serial(serial_no)
    if streams_only:
        _saved_kernels(build_dir, echo)
    with _pipeline_errors(_verbosity(verbosity)), _quiet_stdout(as_json):
        outcome = run_hardware_pipeline(
            repo_root(), spec, serial, options=options, build_dir=build_dir,
            skip_generate=skip_generate, skip_flash=skip_flash, force_flash=force_flash, jobs=jobs,
            force_reconfigure=force_reconfigure, echo=echo, progress_to_stderr=as_json,
            allow_unverified_firmware=allow_unverified_firmware, app_options=app_options,
            update_dependencies=update_dependencies,
        )
    _report(outcome, spec, options, as_json=as_json)
