"""Build and flash the real `hct_benchmark_server` firmware for a board.

Host-side glue only: the firmware sources stay under cmake/hardware/. This
module renders them as an NSX app inside the build dir (nsx_app.py), drives
NSX lock/sync/configure/build through nsx_cli.py, and owns the "flash only if
the ELF changed" decision.

That decision has two halves. The host-side stamp
(`<build_dir>/.flashed-<serial>.sha256`) says whether *this build dir* last
flashed *this probe* with the current ELF. It cannot know what another build
dir (a second clone, `--build-dir`, a lab runner sharing the board) did since,
so a stamp match is only trusted after the board itself confirms it: every
firmware build carries a content-hash build id (`hct_build_id.txt`, stamped
into the linked image by scripts/patch_build_id.py as a CMake POST_BUILD step
and advertised by the firmware in TARGET_INFO), and the skip path opens one short
RTT session to read it.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import typer

from .boards import BoardSpec
from .boards import repo_root as tester_repo_root
from .jlink_library import JLinkLibraryError, find_jlink_exe
from .toolchain import DOWNLOADS_DIR, add_toolchain_to_path, toolchain_bin_dir

if TYPE_CHECKING:
    from .nsx_app import AppOptions

SERVER_TARGET = "hct_benchmark_server"
FLASH_TARGET = "hct_benchmark_server_flash"


def ensure_build_tools(repo_root: Path) -> None:
    """Fetch GCC and CMSIS_5; put GCC on PATH."""
    from ..scripts.setup_dependencies import setup_arm_gcc, setup_cmsis5

    downloads = repo_root / DOWNLOADS_DIR
    downloads.mkdir(parents=True, exist_ok=True)
    if not (downloads / "arm_gcc_download").is_dir():
        setup_arm_gcc(downloads)
    # Firmware still includes CMSIS_5's pmu_armv8.h.
    if not (downloads / "CMSIS_5" / "CMSIS" / "Core").is_dir():
        setup_cmsis5(downloads)
    # NSX's toolchain file finds GCC on PATH.
    add_toolchain_to_path(repo_root)


def resolve_build_dir(repo_root: Path, board: BoardSpec, override: Optional[Path] = None) -> Path:
    """`--build-dir` if given (relative paths are repo-rooted), else the board-keyed default."""
    if override is None:
        return board.build_dir(repo_root)
    return override if override.is_absolute() else repo_root / override


# Image subdir and build-id file under the build dir.
IMAGE_SUBDIR = "hardware"
BUILD_ID_TXT = "hct_build_id.txt"
NSX_APP_SUBDIR = "nsx_app"


def nsx_app_dir(build_dir: Path) -> Path:
    """The rendered NSX app lives inside the build dir."""
    return build_dir / NSX_APP_SUBDIR


def _artifact_path(build_dir: Path, suffix: str) -> Path:
    """The linked server image lives under `<build_dir>/hardware/`."""
    return build_dir / IMAGE_SUBDIR / f"{SERVER_TARGET}{suffix}"


def elf_path(build_dir: Path) -> Path:
    return _artifact_path(build_dir, ".elf")


def bin_path(build_dir: Path) -> Path:
    return _artifact_path(build_dir, ".bin")


def map_path(build_dir: Path) -> Path:
    return _artifact_path(build_dir, ".map")


def build_id_path(build_dir: Path) -> Path:
    """`hct_build_id.txt`, written next to the cache by the post-link stamp step (see
    scripts/patch_build_id.py); the same string the firmware advertises in TARGET_INFO."""
    return build_dir / BUILD_ID_TXT


def read_build_id(build_dir: Path) -> Optional[str]:
    """The build id of the firmware in `build_dir`, or None when the build predates
    build-id stamping (or the ELF has not been built at all)."""
    path = build_id_path(build_dir)
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _cached_var(cache_text: str, name: str) -> Optional[str]:
    """Return the cached value of a CMakeCache.txt entry (e.g. `NSX_JLINK_SERIAL`),
    or None if it isn't present. Cache lines look like `NAME:TYPE=value`."""
    match = re.search(rf"^{re.escape(name)}:[^=]*=(.*)$", cache_text, re.MULTILINE)
    return match.group(1) if match else None


def _configured_for(
    build_dir: Path, app_dir: Path, board: BoardSpec, serial_no: Optional[int], jlink_exe: Optional[str],
) -> bool:
    """The cache belongs to this app, board, probe."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file() or not (build_dir / "build.ninja").is_file():
        return False
    text = cache.read_text(encoding="utf-8", errors="ignore")
    wanted = {"CMAKE_HOME_DIRECTORY": str(app_dir.resolve()), "NSX_BOARD": board.nsx_board}
    # Flash target bakes in serial, JLinkExe.
    if serial_no is not None:
        wanted["NSX_JLINK_SERIAL"] = str(serial_no)
        if jlink_exe is not None:
            wanted["NSX_JLINK_EXE"] = jlink_exe
    return all(_cached_var(text, name) == value for name, value in wanted.items())


def _drop_foreign_cache(build_dir: Path, app_dir: Path) -> None:
    """Remove a cache another source tree wrote."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return
    home = _cached_var(cache.read_text(encoding="utf-8", errors="ignore"), "CMAKE_HOME_DIRECTORY")
    if home == str(app_dir.resolve()):
        return
    typer.echo(f"[hardware] Dropping CMake cache from {home}.")
    cache.unlink()
    shutil.rmtree(build_dir / "CMakeFiles", ignore_errors=True)


def _export_jlink_exe() -> Optional[str]:
    """Hand NSX the JLinkExe doctor reports."""
    # NSX reads $JLINK_PATH, then PATH.
    try:
        found = find_jlink_exe()
    except JLinkLibraryError as exc:
        typer.echo(f"[hardware] WARNING: {exc}", err=True)
        return None
    if found is None:
        return None
    os.environ["JLINK_PATH"] = found.path
    return found.path


def _sync_modules(app_dir: Path, relock: bool) -> None:
    """Sync modules/; relock if frozen sync fails."""
    from . import nsx_cli

    # Frozen sync cannot populate a tree.
    frozen = not relock and (app_dir / "modules").is_dir()
    try:
        nsx_cli.sync_app(app_dir, frozen=frozen)
    except nsx_cli.HardwareBuildError as exc:
        if not frozen:
            raise
        # Interrupted sync or NSX upgrade.
        typer.echo(f"[hardware] modules/ drifted from nsx.lock; relocking. ({exc})")
        nsx_cli.lock_app(app_dir)
        nsx_cli.sync_app(app_dir)


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
    toolchain_bin = str(toolchain_bin_dir(tester_repo_root()).resolve())
    env["PATH"] = f"{toolchain_bin}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(cmd, cwd=tester_repo_root(), check=True, env=env)


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
    build_id: Optional[str] = None
    """Build id of the firmware in the build dir (None when the build has no stamp)."""
    board_build_id: Optional[str] = None
    """What the board reported in TARGET_INFO when it was asked (None when it was not, or did not answer)."""
    # Wall-clock seconds spent in the cmake build and in the J-Link flash (0 when skipped).
    build_seconds: float = 0.0
    flash_seconds: float = 0.0


def decide_flash(build_dir: Path, serial_no: int, *, force: bool = False) -> FlashDecision:
    """Host-side half of the decision: compare the built ELF against the stamp for
    this (build dir, serial). A `needed=False` answer here is provisional -- see
    `confirm_board_build_id`."""
    elf = elf_path(build_dir)
    if not elf.exists():
        raise FileNotFoundError(f"Built firmware ELF not found: {elf}")
    digest = elf_sha256(elf)
    build_id = read_build_id(build_dir)
    if force:
        return FlashDecision(True, digest, "--force given", build_id)
    stamp = flash_stamp_path(build_dir, serial_no)
    if not stamp.exists():
        return FlashDecision(True, digest, f"no flash stamp for serial {serial_no} yet", build_id)
    previous = stamp.read_text(encoding="utf-8").strip()
    if previous == digest:
        return FlashDecision(
            False, digest, f"ELF sha256 unchanged since last flash to serial {serial_no} (stamp {stamp})", build_id
        )
    return FlashDecision(True, digest, "ELF sha256 changed since last flash", build_id)


BoardBuildIdReader = Callable[[BoardSpec, int, Path], str]


def board_build_id(board: BoardSpec, serial_no: int, build_dir: Path) -> str:
    """Ask the board which firmware it runs: one short reset-on-open RTT session
    that reads TARGET_INFO and closes without acknowledging it.

    Raises (RuntimeError, TimeoutError, pylink errors) when the board does not
    answer -- the RTT block address comes from this build dir's ELF, so unrelated
    firmware typically yields no TARGET_INFO at all, which callers treat as "flash".
    """
    from .session import read_target_info
    from .transport import JLinkRttTransport, symbol_address_from_elf

    rtt_address = symbol_address_from_elf(str(elf_path(build_dir)), "_SEGGER_RTT")
    transport = JLinkRttTransport(
        serial_no=serial_no,
        chip_name=board.jlink_device,
        speed_khz=board.swd_speed_khz,
        rtt_address=rtt_address,
        reset_on_open=True,
        read_timeout_s=5.0,
    )
    try:
        return read_target_info(transport).build_id
    finally:
        transport.close()


def confirm_board_build_id(
    board: BoardSpec,
    serial_no: int,
    build_dir: Path,
    decision: FlashDecision,
    *,
    reader: BoardBuildIdReader = board_build_id,
) -> FlashDecision:
    """Board-side half of the decision: turn a provisional "unchanged" into a real
    skip only if the board reports exactly this build dir's build id. A missing
    host build id, a different id on the board, or no TARGET_INFO at all means flash."""
    if decision.needed:
        return decision
    expected = decision.build_id
    if expected is None:
        return FlashDecision(
            True, decision.digest,
            f"{build_id_path(build_dir)} is missing, so what the board runs cannot be confirmed (rebuild stamps it)",
        )
    try:
        actual = reader(board, serial_no, build_dir)
    except Exception as exc:  # no TARGET_INFO, wrong RTT block, probe/DLL trouble: all mean "do not trust the stamp"
        return FlashDecision(
            True, decision.digest,
            f"board did not confirm build id {expected} ({type(exc).__name__}: {exc})", expected,
        )
    if actual != expected:
        return FlashDecision(
            True, decision.digest,
            f"board reports build id {actual}, expected {expected} (another build dir flashed serial {serial_no})",
            expected, actual,
        )
    return FlashDecision(False, decision.digest, f"{decision.reason}; board confirmed build id {expected}", expected, actual)


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
    options: Optional["AppOptions"] = None,
    update_dependencies: bool = False,
) -> Path:
    """Build hct_benchmark_server through NSX; returns the ELF path."""
    from . import nsx_cli
    from .nsx_app import AppOptions, render_app

    repo_root = tester_repo_root()
    ensure_build_tools(repo_root)
    options = options or AppOptions()
    app_dir = nsx_app_dir(build_dir)
    render_app(board, options, app_dir, repo_root=repo_root)
    # Local kernel edits skip nsx.yml.
    relock = (
        update_dependencies
        or options.cmsis_nn_root is not None
        or not nsx_cli.lock_is_current(app_dir, board.nsx_board)
    )
    if relock:
        typer.echo(f"[hardware] Locking NSX modules for {app_dir}")
        nsx_cli.lock_app(app_dir, update=update_dependencies)
    _sync_modules(app_dir, relock)
    jlink_exe = _export_jlink_exe()
    if force_reconfigure or not _configured_for(build_dir, app_dir, board, serial_no, jlink_exe):
        _drop_foreign_cache(build_dir, app_dir)
        nsx_cli.configure_app(
            app_dir, board.nsx_board, build_dir=build_dir, probe_serial=serial_no, frozen=True,
        )
    else:
        typer.echo(f"[hardware] Reusing configured build dir: {build_dir}")
    nsx_cli.build_app(app_dir, board=board.nsx_board, build_dir=build_dir, jobs=jobs, frozen=True)
    return elf_path(build_dir)


def flash_firmware(
    board: BoardSpec,
    serial_no: int,
    *,
    build_dir: Path,
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    force: bool = False,
    board_build_id_reader: BoardBuildIdReader = board_build_id,
    options: Optional["AppOptions"] = None,
    update_dependencies: bool = False,
) -> FlashDecision:
    """Build, then flash through the NSX-generated J-Link target unless the ELF is
    unchanged since this build dir last flashed this probe *and* the board confirms
    it is running this build's id (or `force` is set)."""
    build_started = time.monotonic()
    build_firmware(
        board, build_dir=build_dir, jobs=jobs, force_reconfigure=force_reconfigure, serial_no=serial_no,
        options=options, update_dependencies=update_dependencies,
    )
    build_seconds = time.monotonic() - build_started
    decision = decide_flash(build_dir, serial_no, force=force)
    if not decision.needed:
        typer.echo(f"[hardware] Stamp says {decision.reason}; asking the board which build it runs...")
        decision = confirm_board_build_id(board, serial_no, build_dir, decision, reader=board_build_id_reader)
    if not decision.needed:
        typer.echo(f"[hardware] Skipping flash: {decision.reason}.")
        return replace(decision, build_seconds=build_seconds)
    typer.echo(f"[hardware] Flashing {board.id} via J-Link serial {serial_no} ({decision.reason}).")
    flash_started = time.monotonic()
    build(build_dir, FLASH_TARGET, jobs)
    record_flash(build_dir, serial_no, decision.digest)
    return replace(decision, build_seconds=build_seconds, flash_seconds=time.monotonic() - flash_started)
