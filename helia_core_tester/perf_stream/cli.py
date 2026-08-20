"""CLI wiring for the real streaming benchmark-server workflow (`perf-stream`).

This module intentionally exposes only the commands that are backed by a
real, verified implementation today: building/flashing the real
cross-compiled Apollo510/Cortex-M55 `hct_benchmark_server` firmware, running
a real RTT session against connected hardware, and generating the firmware
memory report. It deliberately does NOT expose the full CLI surface from
the design spec (--transport, --pmu-groups, baseline-vs-candidate, etc.)
until those are backed by real, tested behavior -- see
docs/performance-streaming-report.md for exactly what is real vs simulated.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from .benchmark_firmware_report import generate_benchmark_server_memory_report
from .phase0 import _repo_root

app = typer.Typer(
    name="perf-stream",
    help="Streaming hardware benchmark-server workflow (real Apollo510/Cortex-M55 vertical slice).",
    add_completion=False,
)

_DEFAULT_BUILD_DIR = "build/perf_stream/benchmark_server_gcc2"
_TOOLCHAIN_FILE = "cmake/nsx/toolchains/arm-none-eabi-gcc.cmake"


def _format_case_line(case, *, id_width: int = 0) -> str:
    passed = case.comparison.passed
    label = "PASS" if passed else "FAIL"
    # Pad the plain label to a fixed width *before* applying ANSI color styling --
    # styling first would make python's string padding count the (invisible)
    # escape codes as characters and silently break column alignment.
    status = typer.style(f"{label:<4}", fg=typer.colors.GREEN if passed else typer.colors.RED, bold=True)
    line = (
        f"  {case.case_bundle.case_id:<{id_width}}  {status}  "
        f"median_cycles={case.statistics.median_cycles:>10.1f}"
    )
    if not passed:
        line += f"  mismatches={case.comparison.mismatch_count}"
    return line


def _make_live_progress_printer(total_hint: int | None = None, *, id_width: int = 0):
    """Return an on_case_complete callback that prints one line per case as soon
    as it finishes running on hardware, so a long multi-batch suite shows visible
    progress instead of going silent until the very end (or until it errors out).

    id_width/total_hint (when known ahead of time) keep the case_id, PASS/FAIL,
    and progress-counter columns aligned across every printed line, regardless
    of how long individual case_ids are or how many cases/digits the total has.
    """
    count = 0
    total_width = len(str(total_hint)) if total_hint else 0

    def _on_case_complete(case) -> None:
        nonlocal count
        count += 1
        if total_hint:
            progress = f"[{count:>{total_width}}/{total_hint}]"
        else:
            progress = f"[{count}]"
        typer.echo(f"{progress} {_format_case_line(case, id_width=id_width)}")

    return _on_case_complete


def _print_case_results(cases) -> tuple[int, list[str]]:
    """Print a readable, colorized per-case listing grouped by operator family.

    Returns (passed_count, failed_case_ids) for use in a trailing summary.
    """
    passed_count = 0
    failed_case_ids: list[str] = []
    current_family: Optional[str] = None
    id_width = max((len(c.case_bundle.case_id) for c in cases), default=0)

    for case in cases:
        family = str(case.case_bundle.manifest.get("family", "?"))
        if family != current_family:
            typer.echo(typer.style(f"\n[{family}]", bold=True))
            current_family = family

        if case.comparison.passed:
            passed_count += 1
        else:
            failed_case_ids.append(case.case_bundle.case_id)

        typer.echo(_format_case_line(case, id_width=id_width))

    return passed_count, failed_case_ids


def _print_result_summary(total: int, passed_count: int, failed_case_ids: list[str]) -> None:
    """Print a compact, always-visible pass/fail summary at the end of a run."""
    typer.echo("\n" + "-" * 60)
    if failed_case_ids:
        typer.echo(
            typer.style(f"Summary: {passed_count}/{total} passed, {len(failed_case_ids)} failed", fg=typer.colors.RED, bold=True)
        )
        typer.echo("Failed cases:")
        for case_id in failed_case_ids:
            typer.echo(f"  - {case_id}")
    else:
        typer.echo(typer.style(f"Summary: {passed_count}/{total} passed", fg=typer.colors.GREEN, bold=True))
    typer.echo("-" * 60)


_BRIDGED_TODAY_RE = re.compile(r"\s*\(bridged today: \[.*?\]\)\.?$")
_NAMES_PER_LINE = 6


def _clean_skip_reason(test_name: str, reason: str) -> str:
    """Strip the redundant leading '{test_name}: ' (the CLI already prints the
    name once) and collapse the ever-growing 'bridged today: [...]' family/operator
    list -- otherwise identical for every not-yet-bridged case -- down to a
    pointer at the authoritative source, instead of repeating it per case.
    """
    prefix = f"{test_name}: "
    if reason.startswith(prefix):
        reason = reason[len(prefix):]
    return _BRIDGED_TODAY_RE.sub(
        " (see generated_test_bridge.bridged_families() for the current list).", reason
    )


def _print_skipped_summary(skipped: list[tuple]) -> None:
    """Print skipped generated-tests grouped by (deduplicated) reason instead of
    repeating an identical, verbose reason string once per case -- keeps the
    signal (why + how many + which cases) readable even with hundreds of
    not-yet-bridged cases.
    """
    groups: dict[str, list[str]] = {}
    for test, reason in skipped:
        cleaned = _clean_skip_reason(test.name, reason)
        groups.setdefault(cleaned, []).append(test.name)

    typer.echo(
        typer.style(
            f"\n  Skipped {len(skipped)} generated test(s) with no real firmware dispatch support yet, grouped by reason:",
            fg=typer.colors.YELLOW,
        )
    )
    for reason, names in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        typer.echo(typer.style(f"\n  [{len(names)}x] {reason}", fg=typer.colors.YELLOW))
        names = sorted(names)
        for i in range(0, len(names), _NAMES_PER_LINE):
            typer.echo("      " + ", ".join(names[i : i + _NAMES_PER_LINE]))


def _configure(build_dir: Path, cpu: str, board: str, force: bool) -> None:
    repo_root = _repo_root()
    cache = build_dir / "CMakeCache.txt"
    if cache.exists() and not force:
        typer.echo(f"[perf-stream] Reusing existing configured build dir: {build_dir}")
        return
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmake",
        "-S", str(repo_root),
        "-B", str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={_TOOLCHAIN_FILE}",
        "-DHELIA_BUILD_GENERATED_TESTS=OFF",
        "-DHELIA_BUILD_PERF_STREAM_BENCHMARK_SERVER=ON",
        "-DHELIA_HARDWARE_BUILD=ON",
        f"-DHELIA_HARDWARE_BOARD={board}",
        f"-DTARGET_CPU={cpu}",
        "-DARM_NN_ENABLE_F32=ON",
    ]
    typer.echo(f"[perf-stream] Configuring: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, check=True)


def _build(build_dir: Path, target: str, jobs: Optional[int]) -> None:
    cmd = ["cmake", "--build", str(build_dir), "--target", target]
    if jobs:
        cmd += ["-j", str(jobs)]
    typer.echo(f"[perf-stream] Building: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=_repo_root(), check=True)


@app.command()
def build_firmware(
    cpu: str = typer.Option("cortex-m55", help="Target CPU for the benchmark-server firmware."),
    board: str = typer.Option("apollo510_evb", help="NSX board name."),
    build_dir: Path = typer.Option(Path(_DEFAULT_BUILD_DIR), "--build-dir", help="CMake build directory."),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel build jobs."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure even if the build dir already exists."),
):
    """Cross-compile the real hct_benchmark_server firmware (no flashing)."""
    repo_root = _repo_root()
    resolved_build_dir = build_dir if build_dir.is_absolute() else repo_root / build_dir
    _configure(resolved_build_dir, cpu, board, force_reconfigure)
    _build(resolved_build_dir, "hct_benchmark_server", jobs)
    typer.echo("✓ Firmware build completed successfully")


@app.command()
def flash(
    cpu: str = typer.Option("cortex-m55", help="Target CPU for the benchmark-server firmware."),
    board: str = typer.Option("apollo510_evb", help="NSX board name."),
    build_dir: Path = typer.Option(Path(_DEFAULT_BUILD_DIR), "--build-dir", help="CMake build directory."),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel build jobs."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure even if the build dir already exists."),
):
    """Build (if needed) and flash the real hct_benchmark_server firmware to a connected Apollo510 board via J-Link."""
    repo_root = _repo_root()
    resolved_build_dir = build_dir if build_dir.is_absolute() else repo_root / build_dir
    _configure(resolved_build_dir, cpu, board, force_reconfigure)
    _build(resolved_build_dir, "hct_benchmark_server_flash", jobs)
    typer.echo("✓ Firmware flashed successfully")


@app.command(name="memory-report")
def memory_report(
    build_dir: Optional[Path] = typer.Option(None, "--build-dir", help="CMake build directory (defaults to the benchmark-server build dir)."),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Directory to write memory_report.json into."),
):
    """Generate and print the real flash/RAM memory_report.json for the linked firmware ELF."""
    path = generate_benchmark_server_memory_report(build_dir=build_dir, output_root=output_root)
    typer.echo(json.dumps(json.loads(path.read_text()), indent=2))
    typer.echo(f"\n✓ Memory report written to {path}")


@app.command()
def run(
    serial_no: int = typer.Option(..., "--serial-no", help="J-Link probe serial number (see `JLinkExe` -> ShowEmuList)."),
    chip_name: str = typer.Option("AP510NFA-CBR", "--chip-name", help="SEGGER device name for the target chip."),
    speed_khz: int = typer.Option(4000, "--speed-khz", help="SWD interface speed in kHz."),
    session_id: str = typer.Option("apollo510-live-session", "--session-id", help="Session ID; also the result-bundle directory name."),
    pmu_groups: str = typer.Option("cpu,memory,mve", "--pmu-groups", help="Comma-separated PMU counter groups to request."),
    build_dir: Path = typer.Option(Path(_DEFAULT_BUILD_DIR), "--build-dir", help="CMake build directory containing the flashed firmware's ELF."),
):
    """Run a real streaming session (arm_abs_s8 + arm_convolve_s8) against already-flashed hardware and write the result bundle."""
    from .hardware_run import run_apollo510_stream_session

    repo_root = _repo_root()
    resolved_build_dir = build_dir if build_dir.is_absolute() else repo_root / build_dir
    groups = tuple(g.strip() for g in pmu_groups.split(",") if g.strip())
    result, bundle = run_apollo510_stream_session(
        repo_root,
        serial_no=serial_no,
        chip_name=chip_name,
        speed_khz=speed_khz,
        requested_counter_groups=groups,
        session_id=session_id,
        build_dir=resolved_build_dir,
    )
    passed_count, failed_case_ids = _print_case_results(result.cases)
    _print_result_summary(len(result.cases), passed_count, failed_case_ids)
    typer.echo(f"\n✓ Session complete. Result bundle: {bundle}")


@app.command(name="run-generated")
def run_generated(
    serial_no: int = typer.Option(..., "--serial-no", help="J-Link probe serial number (see `JLinkExe` -> ShowEmuList)."),
    chip_name: str = typer.Option("AP510NFA-CBR", "--chip-name", help="SEGGER device name for the target chip."),
    speed_khz: int = typer.Option(4000, "--speed-khz", help="SWD interface speed in kHz."),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session ID; also the result-bundle directory name."),
    pmu_groups: str = typer.Option("cpu,memory,mve", "--pmu-groups", help="Comma-separated PMU counter groups to request."),
    build_dir: Path = typer.Option(Path(_DEFAULT_BUILD_DIR), "--build-dir", help="CMake build directory containing the flashed firmware's ELF."),
    cpu: str = typer.Option("cortex-m55", help="CPU whose generated tests to bridge (must match `helia_core_tester generate --cpu`)."),
    family: Optional[str] = typer.Option(None, "--family", help="Operator family under artifacts/generated_tests to bridge. Omit to bridge every family with real firmware dispatch support (see generated_test_bridge.bridged_families())."),
    test_name: Optional[str] = typer.Option(None, "--test-name", help="Only bridge generated tests whose directory name contains this substring."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only bridge the first N discovered generated tests."),
    suite: str = typer.Option("int", "--suite", help="Generated-test suite to bridge: 'int' (S8/S16/S32/S4, default) or 'float' (FP16/FP32)."),
):
    """Stream real `helia_core_tester generate`-produced kernel tests (real golden data,
    not synthetic demo data) to already-flashed hardware over HCTP/RTT and check correctness.

    Only kernels with real firmware dispatch support are bridged -- see the `_BUILDERS`
    dispatch table in `generated_test_bridge.py` (or call `bridged_families()` at
    runtime) for the up-to-date list. Everything else is reported as skipped with the
    reason, instead of silently fabricating a result. By default (no --family given) every bridged family is run,
    i.e. the complete hardware-supported suite. Run `helia_core_tester generate --cpu
    cortex-m55` first to produce bridgeable test data.

    Bridged cases are automatically batched into groups of at most
    hardware_run.MAX_CASES_PER_SESSION (matching firmware HCT_SERVER_MAX_CASES), each
    run over its own fresh reset-on-open RTT session and merged into one result bundle
    -- the firmware silently drops (and hangs the host on) a LOAD_PLAN naming more
    cases than that in a single session.
    """
    from .hardware_run import build_generated_test_case_bundles, run_apollo510_generated_test_session

    repo_root = _repo_root()
    resolved_build_dir = build_dir if build_dir.is_absolute() else repo_root / build_dir
    groups = tuple(g.strip() for g in pmu_groups.split(",") if g.strip())

    # Discover the bridgeable case count/case_ids up front (cheap: just descriptor/header
    # parsing, no hardware I/O) purely so the live progress printer can align its
    # [N/total] counter and case_id columns from the very first printed line instead of
    # widening them as longer names are discovered mid-run.
    preview_bundles, _preview_skipped = build_generated_test_case_bundles(
        repo_root, cpu=cpu, family=family, name_filter=test_name, limit=limit, suite=suite
    )
    id_width = max((len(b.case_id) for b in preview_bundles), default=0)

    typer.echo("Running full generated test suite over HCTP/RTT (progress below)...")
    on_case_complete = _make_live_progress_printer(len(preview_bundles), id_width=id_width)
    try:
        result, bundle, skipped = run_apollo510_generated_test_session(
            repo_root,
            serial_no=serial_no,
            chip_name=chip_name,
            speed_khz=speed_khz,
            requested_counter_groups=groups,
            session_id=session_id,
            build_dir=resolved_build_dir,
            cpu=cpu,
            family=family,
            name_filter=test_name,
            limit=limit,
            suite=suite,
            on_case_complete=on_case_complete,
        )
    except RuntimeError as exc:
        typer.echo(f"✗ {exc}", err=True)
        sys.exit(1)

    typer.echo("\nFinal per-case results:")
    passed_count, failed_case_ids = _print_case_results(result.cases)
    if skipped:
        _print_skipped_summary(skipped)

    _print_result_summary(len(result.cases), passed_count, failed_case_ids)
    typer.echo(f"\n✓ Result bundle: {bundle}")
    if failed_case_ids:
        typer.echo(typer.style("✗ One or more generated-test cases failed correctness", fg=typer.colors.RED, bold=True), err=True)
        sys.exit(1)
    typer.echo(typer.style(f"✓ {len(result.cases)} generated test case(s) passed on real hardware", fg=typer.colors.GREEN, bold=True))


@app.command()
def full(
    serial_no: int = typer.Option(..., "--serial-no", help="J-Link probe serial number (see `JLinkExe` -> ShowEmuList)."),
    cpu: str = typer.Option("cortex-m55", help="Target CPU for the benchmark-server firmware."),
    board: str = typer.Option("apollo510_evb", help="NSX board name."),
    chip_name: str = typer.Option("AP510NFA-CBR", "--chip-name", help="SEGGER device name for the target chip."),
    speed_khz: int = typer.Option(4000, "--speed-khz", help="SWD interface speed in kHz."),
    session_id: str = typer.Option("apollo510-live-session", "--session-id", help="Session ID; also the result-bundle directory name."),
    pmu_groups: str = typer.Option("cpu,memory,mve", "--pmu-groups", help="Comma-separated PMU counter groups to request."),
    build_dir: Path = typer.Option(Path(_DEFAULT_BUILD_DIR), "--build-dir", help="CMake build directory."),
    jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="Parallel build jobs."),
    skip_flash: bool = typer.Option(False, "--skip-flash", help="Skip build+flash and reuse whatever firmware is already running on the board."),
    force_reconfigure: bool = typer.Option(False, "--force-reconfigure", help="Reconfigure the CMake build dir even if it already exists."),
):
    """Build, flash (once), run the real streaming session, and print the memory report -- the full real-hardware pipeline in one command."""
    repo_root = _repo_root()
    resolved_build_dir = build_dir if build_dir.is_absolute() else repo_root / build_dir

    if not skip_flash:
        _configure(resolved_build_dir, cpu, board, force_reconfigure)
        _build(resolved_build_dir, "hct_benchmark_server_flash", jobs)
        typer.echo("✓ Firmware flashed successfully\n")
    else:
        typer.echo("[perf-stream] --skip-flash set; reusing firmware already running on the board.\n")

    typer.echo("[perf-stream] Memory report:")
    report_path = generate_benchmark_server_memory_report(build_dir=resolved_build_dir)
    usage = json.loads(report_path.read_text()).get("usage", {})
    typer.echo(
        f"  flash: {usage.get('flash_image_bytes')}/{usage.get('flash_capacity_bytes')} "
        f"bytes ({usage.get('flash_percent_used')}%), gate_pass={usage.get('flash_gate_pass')}"
    )
    typer.echo(
        f"  tcm:   {usage.get('tcm_static_bytes')}/{usage.get('tcm_capacity_bytes')} "
        f"bytes ({usage.get('tcm_percent_used_before_heap')}%), gate_pass={usage.get('tcm_gate_pass')}\n"
    )

    typer.echo("[perf-stream] Running live session against connected hardware...")
    from .hardware_run import run_apollo510_stream_session

    groups = tuple(g.strip() for g in pmu_groups.split(",") if g.strip())
    result, bundle = run_apollo510_stream_session(
        repo_root,
        serial_no=serial_no,
        chip_name=chip_name,
        speed_khz=speed_khz,
        requested_counter_groups=groups,
        session_id=session_id,
        build_dir=resolved_build_dir,
    )
    passed_count, failed_case_ids = _print_case_results(result.cases)
    _print_result_summary(len(result.cases), passed_count, failed_case_ids)
    typer.echo(f"\n✓ Result bundle: {bundle}")
    if failed_case_ids:
        typer.echo(typer.style("✗ One or more cases failed correctness", fg=typer.colors.RED, bold=True), err=True)
        sys.exit(1)
    typer.echo(typer.style("✓ perf-stream full pipeline completed successfully", fg=typer.colors.GREEN, bold=True))
