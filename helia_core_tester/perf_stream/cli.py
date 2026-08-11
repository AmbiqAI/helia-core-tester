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
    for case in result.cases:
        status = "PASS" if case.comparison.passed else "FAIL"
        typer.echo(
            f"  {case.case_bundle.case_id}: correctness={status} "
            f"median_cycles={case.statistics.median_cycles}"
        )
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
):
    """Stream real `helia_core_tester generate`-produced kernel tests (real golden data,
    not synthetic demo data) to already-flashed hardware over HCTP/RTT and check correctness.

    Only kernels with real firmware dispatch support are bridged (arm_convolve_s8,
    arm_add_s8, arm_sub_s8, arm_mul_s8; S8 activation + S8 weight, batch size 1).
    Everything else is reported as skipped with the reason, instead of silently
    fabricating a result. By default (no --family given) every bridged family is run,
    i.e. the complete hardware-supported suite. Run `helia_core_tester generate --cpu
    cortex-m55` first to produce bridgeable test data.

    Bridged cases are automatically batched into groups of at most
    hardware_run.MAX_CASES_PER_SESSION (matching firmware HCT_SERVER_MAX_CASES), each
    run over its own fresh reset-on-open RTT session and merged into one result bundle
    -- the firmware silently drops (and hangs the host on) a LOAD_PLAN naming more
    cases than that in a single session.
    """
    from .hardware_run import run_apollo510_generated_test_session

    repo_root = _repo_root()
    resolved_build_dir = build_dir if build_dir.is_absolute() else repo_root / build_dir
    groups = tuple(g.strip() for g in pmu_groups.split(",") if g.strip())
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
        )
    except RuntimeError as exc:
        typer.echo(f"✗ {exc}", err=True)
        sys.exit(1)

    all_passed = True
    for case in result.cases:
        status = "PASS" if case.comparison.passed else "FAIL"
        all_passed = all_passed and case.comparison.passed
        typer.echo(
            f"  {case.case_bundle.case_id}: correctness={status} "
            f"median_cycles={case.statistics.median_cycles}"
        )
    if skipped:
        typer.echo(f"\n  Skipped {len(skipped)} generated test(s) with no real firmware dispatch support yet:")
        for test, reason in skipped:
            typer.echo(f"    - {test.name}: {reason}")

    typer.echo(f"\n✓ Result bundle: {bundle}")
    if not all_passed:
        typer.echo("✗ One or more generated-test cases failed correctness", err=True)
        sys.exit(1)
    typer.echo(f"✓ {len(result.cases)} generated test case(s) passed on real hardware")


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
    all_passed = True
    for case in result.cases:
        status = "PASS" if case.comparison.passed else "FAIL"
        all_passed = all_passed and case.comparison.passed
        typer.echo(
            f"  {case.case_bundle.case_id}: correctness={status} "
            f"median_cycles={case.statistics.median_cycles}"
        )
    typer.echo(f"\n✓ Result bundle: {bundle}")
    if not all_passed:
        typer.echo("✗ One or more cases failed correctness", err=True)
        sys.exit(1)
    typer.echo("✓ perf-stream full pipeline completed successfully")
