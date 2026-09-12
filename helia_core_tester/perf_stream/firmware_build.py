"""Build and flash the real `hct_benchmark_server` firmware for a board.

Host-side glue only: the firmware under cmake/perf_stream/ and the NSX CMake
targets are untouched. This module owns the lazy dependency fetch, the CMake
configure/build invocations, and the "flash only if the ELF changed" stamp.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from .boards import BoardSpec
from .phase0 import _repo_root

TOOLCHAIN_FILE = "cmake/nsx/toolchains/arm-none-eabi-gcc.cmake"
DOWNLOADS_DIR = "artifacts/downloads"
SERVER_TARGET = "hct_benchmark_server"
FLASH_TARGET = "hct_benchmark_server_flash"


def ensure_hardware_dependencies(repo_root: Path) -> None:
    """Lazily fetch the real hardware-build dependencies (nsx-ambiq-sdk, neuralspotx,
    the generated NSX toolchain file) the first time any hardware command needs them,
    instead of requiring a separate manual bootstrap step.
    `helia_core_tester scripts.setup_dependencies --with-hardware` does the same
    thing ahead of time if you'd rather pre-fetch.
    """
    from ..scripts.setup_dependencies import (
        nsx_ambiq_sdk_dir,
        setup_arm_gcc,
        setup_cmsis5,
        setup_neuralspotx,
        setup_nsx_ambiq_sdk,
        setup_nsx_toolchain,
    )

    downloads_dir = repo_root / DOWNLOADS_DIR
    downloads_dir.mkdir(parents=True, exist_ok=True)

    sdk_modules_dir = nsx_ambiq_sdk_dir(repo_root, downloads_dir) / "modules"
    # Not just the SDK checkout: also the local symlinks that redirect
    # boards/apollo510_evb + cmake/socs + cmake/nsx_soc_facts.cmake into it (see
    # setup_nsx_ambiq_sdk()'s _ensure_nsx_sdk_symlinks() call) -- an SDK checkout
    # that predates those symlinks, or one whose symlinks got removed, needs
    # setup_nsx_ambiq_sdk() re-run too, not just skipped as "already installed".
    # Resolved, not just `.exists()`: a link left over from an earlier setup can
    # point at an SDK checkout outside the tester repo and would otherwise pass
    # as healthy, quietly building against that tree instead of the managed one.
    board_symlink = repo_root / "boards" / "apollo510_evb"
    board_link_ok = (
        board_symlink.exists()
        and board_symlink.resolve().is_relative_to(nsx_ambiq_sdk_dir(repo_root, downloads_dir).resolve())
    )
    neuralspotx_examples_dir = downloads_dir / "neuralspotx" / "examples"
    arm_gcc_dir = downloads_dir / "arm_gcc_download"
    # CMakeLists.txt's CMSIS_PATH default; the Cortex-M startup/system sources
    # and CMSIS core headers come from here for the hardware build too.
    cmsis5_core_dir = downloads_dir / "CMSIS_5" / "CMSIS" / "Core"
    toolchain_file = repo_root / TOOLCHAIN_FILE

    if (
        sdk_modules_dir.is_dir()
        and board_link_ok
        and neuralspotx_examples_dir.is_dir()
        and arm_gcc_dir.is_dir()
        and cmsis5_core_dir.is_dir()
        and toolchain_file.exists()
    ):
        return

    typer.echo("[hardware] Hardware-build dependencies not found -- fetching them now (first run only)...")
    if not sdk_modules_dir.is_dir() or not board_link_ok:
        setup_nsx_ambiq_sdk(repo_root, downloads_dir)
    if not neuralspotx_examples_dir.is_dir():
        setup_neuralspotx(downloads_dir)
    # The toolchain file bakes in the downloaded GCC's absolute path, so the GCC
    # download has to exist before the file can be generated -- a fresh clone
    # that never ran setup_dependencies.py has neither.
    if not arm_gcc_dir.is_dir():
        setup_arm_gcc(downloads_dir)
    if not cmsis5_core_dir.is_dir():
        setup_cmsis5(downloads_dir)
    if not toolchain_file.exists():
        setup_nsx_toolchain(repo_root, downloads_dir)
    typer.echo("[hardware] Hardware-build dependencies ready.")


def resolve_build_dir(repo_root: Path, board: BoardSpec, override: Optional[Path] = None) -> Path:
    """`--build-dir` if given (relative paths are repo-rooted), else the board-keyed default."""
    if override is None:
        return board.build_dir(repo_root)
    return override if override.is_absolute() else repo_root / override


def elf_path(build_dir: Path) -> Path:
    return build_dir / "perf_stream" / f"{SERVER_TARGET}.elf"


def _cached_var(cache_text: str, name: str) -> Optional[str]:
    """Return the cached value of a CMakeCache.txt entry (e.g. `NSX_JLINK_SERIAL`),
    or None if it isn't present. Cache lines look like `NAME:TYPE=value`."""
    match = re.search(rf"^{re.escape(name)}:[^=]*=(.*)$", cache_text, re.MULTILINE)
    return match.group(1) if match else None


def configure(build_dir: Path, board: BoardSpec, force: bool, serial_no: Optional[int] = None) -> None:
    repo_root = _repo_root()
    ensure_hardware_dependencies(repo_root)
    cache = build_dir / "CMakeCache.txt"
    if cache.exists() and not force:
        # A build dir configured before ARM_NN_ENABLE_F16 was added here would
        # otherwise silently keep compiling without FP16 kernel support.
        cache_text = cache.read_text(encoding="utf-8", errors="ignore")
        # NSX_JLINK_SERIAL is baked into the generated *_flash/_reset/_view
        # custom-target commands at configure time (see nsx_add_segger_targets()
        # in cmake/nsx/nsx_helpers.cmake), so switching --serial-no against an
        # already-configured build dir requires a reconfigure to take effect.
        serial_stale = serial_no is not None and _cached_var(cache_text, "NSX_JLINK_SERIAL") != str(serial_no)
        # Still re-run cmake below in every case (cheap, <1s) rather than skipping
        # outright when already-configured: relying on `cmake --build`'s own
        # internal cmake_check_build_system re-check to be the first
        # post-cache-write reconfigure has been observed to intermittently fail
        # on a relative-path EXISTS() check (CMakeLists.txt's CMSIS_PATH
        # validation) that a direct `cmake -S -B` invocation here never
        # reproduces -- doing that direct invocation unconditionally sidesteps
        # it instead of chasing the underlying CMake behavior.
        if "ARM_NN_ENABLE_F16:BOOL=ON" in cache_text and not serial_stale:
            typer.echo(f"[hardware] Reusing existing configured build dir: {build_dir}")
        elif serial_stale:
            typer.echo(f"[hardware] Requested --serial-no {serial_no} differs from configured build dir -- reconfiguring.")
        else:
            typer.echo(f"[hardware] Existing build dir at {build_dir} predates ARM_NN_ENABLE_F16 -- reconfiguring.")
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmake",
        "-S", str(repo_root),
        "-B", str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN_FILE}",
        "-DHELIA_BUILD_GENERATED_TESTS=OFF",
        "-DHELIA_BUILD_PERF_STREAM_BENCHMARK_SERVER=ON",
        "-DHELIA_HARDWARE_BUILD=ON",
        f"-DHELIA_HARDWARE_BOARD={board.nsx_board}",
        f"-DTARGET_CPU={board.cpu}",
        "-DARM_NN_ENABLE_F32=ON",
        "-DARM_NN_ENABLE_F16=ON",
        # Overrides CMakeLists.txt's fragile "3 levels up, outside the repo" default
        # (${CMAKE_CURRENT_SOURCE_DIR}/../../../neuralspotx) with the copy
        # ensure_hardware_dependencies() fetches into artifacts/downloads/.
        f"-DNEURALSPOTX_ROOT={repo_root / DOWNLOADS_DIR / 'neuralspotx'}",
    ]
    if serial_no is not None:
        cmd.append(f"-DNSX_JLINK_SERIAL={serial_no}")
    typer.echo(f"[hardware] Configuring: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, check=True)


def build(build_dir: Path, target: str, jobs: Optional[int]) -> None:
    cmd = ["cmake", "--build", str(build_dir), "--target", target]
    if jobs:
        cmd += ["-j", str(jobs)]
    typer.echo(f"[hardware] Building: {' '.join(cmd)}")
    # generate_kernel_symbol_refs.py (run as a build step) shells out to the
    # bare command name "arm-none-eabi-nm" -- the toolchain file points CMake's
    # own compiler/linker/objcopy invocations at absolute paths, but this one
    # still needs the toolchain's bin/ on PATH.
    env = os.environ.copy()
    toolchain_bin = str((_repo_root() / DOWNLOADS_DIR / "arm_gcc_download" / "bin").resolve())
    env["PATH"] = f"{toolchain_bin}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(cmd, cwd=_repo_root(), check=True, env=env)


# --- flash-only-if-changed stamp -------------------------------------------------


def elf_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flash_stamp_path(build_dir: Path, serial_no: int) -> Path:
    """Stamp recording the sha256 of the ELF last flashed to probe `serial_no` from
    this build dir. Keyed by serial because the same build can serve several boards."""
    return build_dir / f".flashed-{serial_no}.sha256"


@dataclass(frozen=True)
class FlashDecision:
    needed: bool
    digest: str
    reason: str
    # Wall-clock seconds spent in the cmake build and in the J-Link flash (0 when skipped).
    build_seconds: float = 0.0
    flash_seconds: float = 0.0


def decide_flash(build_dir: Path, serial_no: int, *, force: bool = False) -> FlashDecision:
    """Compare the built ELF against the stamp for this (build dir, serial)."""
    elf = elf_path(build_dir)
    if not elf.exists():
        raise FileNotFoundError(f"Built firmware ELF not found: {elf}")
    digest = elf_sha256(elf)
    if force:
        return FlashDecision(True, digest, "--force given")
    stamp = flash_stamp_path(build_dir, serial_no)
    if not stamp.exists():
        return FlashDecision(True, digest, f"no flash stamp for serial {serial_no} yet")
    previous = stamp.read_text(encoding="utf-8").strip()
    if previous == digest:
        return FlashDecision(False, digest, f"ELF sha256 unchanged since last flash to serial {serial_no}")
    return FlashDecision(True, digest, "ELF sha256 changed since last flash")


def record_flash(build_dir: Path, serial_no: int, digest: str) -> Path:
    stamp = flash_stamp_path(build_dir, serial_no)
    stamp.write_text(digest + "\n", encoding="utf-8")
    return stamp


# --- high-level entry points ------------------------------------------------------


def build_firmware(
    board: BoardSpec,
    *,
    build_dir: Path,
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    serial_no: Optional[int] = None,
) -> Path:
    """Cross-compile hct_benchmark_server for `board`; returns the ELF path."""
    configure(build_dir, board, force_reconfigure, serial_no=serial_no)
    build(build_dir, SERVER_TARGET, jobs)
    return elf_path(build_dir)


def flash_firmware(
    board: BoardSpec,
    serial_no: int,
    *,
    build_dir: Path,
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    force: bool = False,
) -> FlashDecision:
    """Build, then flash through the NSX-generated J-Link target only when the ELF
    differs from what was last flashed to this probe (or `force` is set)."""
    build_started = time.monotonic()
    build_firmware(board, build_dir=build_dir, jobs=jobs, force_reconfigure=force_reconfigure, serial_no=serial_no)
    build_seconds = time.monotonic() - build_started
    decision = decide_flash(build_dir, serial_no, force=force)
    if not decision.needed:
        typer.echo(f"[hardware] Skipping flash: {decision.reason}.")
        return FlashDecision(decision.needed, decision.digest, decision.reason, build_seconds=build_seconds)
    typer.echo(f"[hardware] Flashing {board.id} via J-Link serial {serial_no} ({decision.reason}).")
    flash_started = time.monotonic()
    build(build_dir, FLASH_TARGET, jobs)
    record_flash(build_dir, serial_no, decision.digest)
    return FlashDecision(
        decision.needed, decision.digest, decision.reason,
        build_seconds=build_seconds, flash_seconds=time.monotonic() - flash_started,
    )
