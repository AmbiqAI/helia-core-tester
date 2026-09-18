"""Build and flash the real `hct_benchmark_server` firmware for a board.

The firmware is an NSX app (see `nsx_app`), so this module owns the NSX
invocation sequence rather than a CMake command line:

    render nsx.yml -> nsx lock -> nsx sync --frozen -> nsx configure -> nsx build

Every step goes through `neuralspotx.api`, the same way heliaPROFILER drives it.
The kernels are the `nsx-cmsis-nn` module at the dependency baseline's pinned
commit (or a local checkout when `--cmsis-nn-root` is given); `CMSIS_NN_ROOT`
and the nested `<ns-cmsis-nn>/Tests/helia-core-tester` layout are not consulted
by any hardware command.

The lock is reused when the rendered manifest and the baseline are both
unchanged, so a repeat build pays neither a resolution round-trip nor a CMake
reconfigure. `nsx sync --frozen` still runs every time: it is cheap when the
tree already matches, and it is what proves the modules on disk are the commits
`nsx.lock` names.

Flashing does not go through the `<target>_flash` ninja target: that target is a
build-system entry point, so reaching it means owning a configured build tree at
flash time, and the previous revision of this module did exactly that -- it
re-rendered and rebuilt the app on every `hardware flash`. Instead `flash_firmware`
runs the recipe that target would have run, `<build>/jlink/<target>/flash_cmds.jlink`,
directly through JLinkExe (see `flash_recipe`), which is what heliaPROFILER does and
what lets flashing be a read-only consumer of the build output.

The "flash only if the ELF changed" decision has two halves. The host-side stamp
(`<build_dir>/.flashed-<serial>.sha256`) says whether *this build dir* last
flashed *this probe* with the current ELF. It cannot know what another build dir
(a second clone, `--build-dir`, a lab runner sharing the board) did since, so a
stamp match is only trusted after the board itself confirms it: every firmware
build carries a content-hash build id (`hct_build_id.txt`, stamped into the
linked image by scripts/patch_build_id.py as a POST_BUILD step and advertised by
the firmware in TARGET_INFO), and the skip path opens one short RTT session to
read it.
"""

from __future__ import annotations

import glob
import hashlib
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import typer

from . import flash_recipe, provenance
from .boards import BoardSpec
from .boards import repo_root as tester_repo_root
from .dependency_baseline import DependencyBaseline, resolve_baseline
from .jlink_library import JLinkLibraryError, find_jlink_exe
from .nsx_app import (
    SERVER_TARGET,
    AppRender,
    KernelOptions,
    app_dir_for,
    nsx_build_dir,
    plan_app,
    read_render_state,
    render_app,
    synced_kernel_dir,
)
from .toolchain import DOWNLOADS_DIR, add_toolchain_to_path

# NSX timeouts. Resolution reaches the network; a cold build compiles the whole
# kernel library plus the SDK.
_LOCK_TIMEOUT_S = 600
_SYNC_TIMEOUT_S = 900
_CONFIGURE_TIMEOUT_S = 300
_BUILD_TIMEOUT_S = 1800


@dataclass(frozen=True)
class FirmwareOptions:
    """What the CLI lets a caller change about how the firmware is built."""

    #: `--baseline FILE`; None means the repo's own assets/dependency_baseline.json.
    baseline_path: Optional[Path] = None
    #: `--cmsis-nn-root PATH`; None means the baseline's pinned nsx-cmsis-nn.
    cmsis_nn_root: Optional[Path] = None
    #: `--no-requantize-inline-asm` turns the kernels' inline-asm requantize off.
    requantize_inline_asm: bool = True
    #: Re-resolve nsx.lock even when the manifest and baseline are unchanged.
    update_dependencies: bool = False
    verbose: int = 0

    def kernel_options(self) -> KernelOptions:
        # The server dispatches every catalog kernel, so both float widths are
        # always compiled in; only the size probe varies them.
        return KernelOptions(
            requantize_inline_asm=self.requantize_inline_asm,
            enable_f32=True,
            enable_f16=True,
        )


def ensure_host_tools(repo_root: Path) -> None:
    """Fetch and expose the host-side build inputs the NSX app still needs.

    Two downloads survive the move to NSX, both shared with the FVP path:

    - ARM GCC, because NSX's toolchain file resolves the cross compiler with
      `find_program(... REQUIRED)` -- i.e. off PATH -- and because this process
      runs `arm-none-eabi-nm`/`-size`/`-objdump` itself for the RTT block
      address and the memory report.
    - CMSIS_5, for `pmu_armv8.h`: the session code drives the Armv8-M PMU
      registers directly and the SDK's nsx-cmsis-core does not carry that
      header.

    Everything else the old hardware build fetched (the nsx-ambiq-sdk checkout,
    a neuralspotx git clone, the generated toolchain file, the boards/ symlinks)
    is now NSX's business and comes from `nsx sync` into the app tree.
    """
    from ..scripts.setup_dependencies import setup_arm_gcc, setup_cmsis5

    downloads_dir = repo_root / DOWNLOADS_DIR
    downloads_dir.mkdir(parents=True, exist_ok=True)
    arm_gcc_dir = downloads_dir / "arm_gcc_download"
    cmsis5_core_dir = downloads_dir / "CMSIS_5" / "CMSIS" / "Core"

    if not arm_gcc_dir.is_dir() or not cmsis5_core_dir.is_dir():
        typer.echo("[hardware] Host build tools not found -- fetching them now (first run only)...")
        if not arm_gcc_dir.is_dir():
            setup_arm_gcc(downloads_dir)
        if not cmsis5_core_dir.is_dir():
            setup_cmsis5(downloads_dir)
    add_toolchain_to_path(repo_root)


def resolve_build_dir(repo_root: Path, board: BoardSpec, override: Optional[Path] = None) -> Path:
    """`--build-dir` if given (relative paths are repo-rooted), else the board-keyed default."""
    if override is None:
        return board.build_dir(repo_root)
    return override if override.is_absolute() else repo_root / override


def output_dir(build_dir: Path, board: BoardSpec) -> Path:
    """Where the linked firmware and its side artifacts land.

    NSX builds an app into `<app>/build/<board>`; the app itself lives in
    `<build_dir>/nsx_app`, so `--build-dir` keeps meaning what it always did.
    """
    return nsx_build_dir(app_dir_for(build_dir), board)


def elf_path(build_dir: Path, board: BoardSpec) -> Path:
    """The linked firmware image.

    NSX's GCC toolchain file links to `.axf`; a POST_BUILD step copies it to
    `.elf` so the host tooling (memory report, RTT block lookup, flash stamp)
    has one stable name regardless of toolchain.
    """
    return output_dir(build_dir, board) / f"{SERVER_TARGET}.elf"


def bin_path(build_dir: Path, board: BoardSpec) -> Path:
    return output_dir(build_dir, board) / f"{SERVER_TARGET}.bin"


def map_path(build_dir: Path, board: BoardSpec) -> Path:
    return output_dir(build_dir, board) / f"{SERVER_TARGET}.map"


def build_id_path(build_dir: Path, board: BoardSpec) -> Path:
    """`hct_build_id.txt`, written by the post-link stamp step (see
    scripts/patch_build_id.py); the same string the firmware advertises in TARGET_INFO."""
    return output_dir(build_dir, board) / "hct_build_id.txt"


def read_build_id(build_dir: Path, board: BoardSpec) -> Optional[str]:
    """The build id of the firmware in `build_dir`, or None when it has not been built."""
    path = build_id_path(build_dir, board)
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def lock_snapshot_path(build_dir: Path, board: BoardSpec) -> Path:
    """The `nsx.lock` copy kept next to the firmware it produced."""
    return output_dir(build_dir, board) / "nsx.lock"


def find_target_binary(build_root: Path, target_name: str) -> Optional[Path]:
    """Locate a built NSX target's executable under `build_root`.

    The suffix is the toolchain's (`.axf` for GCC/ATfE, none for a host build),
    and NSX is free to move where inside the build tree it lands, so this is a
    search rather than a fixed path. Deterministic pick -- shortest path first,
    ties lexicographic -- so a tree with a stale and a fresh candidate resolves
    the same way on every machine and run.
    """
    patterns = [
        str(build_root / f"{target_name}.elf"),
        str(build_root / f"{target_name}.axf"),
        str(build_root / target_name),
        str(build_root / "**" / f"{target_name}.elf"),
        str(build_root / "**" / f"{target_name}.axf"),
        str(build_root / "**" / target_name),
    ]
    for pattern in patterns:
        matches = sorted(
            (m for m in glob.glob(pattern, recursive=True) if Path(m).is_file()),
            key=lambda m: (len(Path(m).parts), m),
        )
        if matches:
            return Path(matches[0])
    return None


# --- NSX invocation ---------------------------------------------------------------


def _emitter(verbose: int):
    """NSX's output sink: silent below `-v`, its own default at `-v` and above."""
    if verbose >= 1:
        return None
    return lambda event: None


def load_baseline(repo_root: Path, options: FirmwareOptions) -> DependencyBaseline:
    return resolve_baseline(repo_root, options.baseline_path)


def prepare_app(
    board: BoardSpec,
    *,
    repo_root: Path,
    build_dir: Path,
    options: FirmwareOptions,
) -> AppRender:
    """Render the NSX app for `board` and make sure the host tools are present."""
    ensure_host_tools(repo_root)
    return render_app(
        board,
        repo_root=repo_root,
        build_dir=build_dir,
        baseline=load_baseline(repo_root, options),
        cmsis_nn_root=options.cmsis_nn_root,
        kernel_options=options.kernel_options(),
    )


def lock_reuse_reason(render: AppRender) -> Optional[str]:
    """Why the on-disk `nsx.lock` cannot be reused, or None when it can.

    A lock is reusable when it was produced from exactly this manifest (NSX's
    own manifest hash), it resolves the board this app targets, and the render
    it belongs to -- manifest text plus baseline fingerprint -- is the render
    being built now. The last check is what stops a baseline edit that happens
    not to change any pin this board resolves from leaving a stale claim behind.
    """
    from neuralspotx.nsx_lock import LOCK_SCHEMA_VERSION, hash_manifest, read_lock

    app_dir = render.app_dir
    if not (app_dir / "nsx.lock").is_file():
        return "nsx.lock is missing"
    state = read_render_state(app_dir)
    if state is None:
        return "this app has no recorded render state"
    if state.get("render_digest") != render.digest:
        return "the rendered manifest or the dependency baseline changed"
    try:
        lock = read_lock(app_dir, render.board.nsx_board)
    except Exception as exc:  # NSX raises for an incompatible on-disk schema
        return f"nsx.lock is unreadable or structurally incompatible: {exc}"
    if lock is None:
        return f"nsx.lock has no target section for board '{render.board.nsx_board}'"
    if lock.schema_version != LOCK_SCHEMA_VERSION:
        return (
            f"nsx.lock schema v{lock.schema_version} is incompatible "
            f"(the pinned neuralspotx requires v{LOCK_SCHEMA_VERSION})"
        )
    if lock.manifest_hash != hash_manifest(app_dir / "nsx.yml"):
        return "nsx.lock was produced from a different nsx.yml"
    if not lock.modules:
        return "nsx.lock resolved no modules"
    for name, module in lock.modules.items():
        if str(module.kind) != "git":
            continue
        commit = (module.commit or "").lower()
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            return f"nsx.lock module '{name}' has no exact peeled commit"
    return None


def lock_and_sync(render: AppRender, options: FirmwareOptions) -> str:
    """Resolve `nsx.lock` when it cannot be reused, then materialise `modules/`.

    The sync is always frozen: it must reproduce exactly what the lock names and
    fail on drift rather than quietly re-vendoring something else.

    Returns how the lock in force was obtained (`reused`, `resolved` or
    `updated`, hpx's `DependencyLockMode` vocabulary), which the build's
    provenance record carries.
    """
    from neuralspotx import api as nsx_api

    app_dir = render.app_dir
    emit = _emitter(options.verbose)
    reason = None if options.update_dependencies else lock_reuse_reason(render)
    if options.update_dependencies:
        typer.echo("[hardware] Re-resolving NSX dependencies (--update-dependencies).")
        nsx_api.lock_app(app_dir, update=True, quiet=True, timeout_s=_LOCK_TIMEOUT_S, emit=emit)
        mode = provenance.LOCK_UPDATED
    elif reason is None:
        typer.echo(f"[hardware] Reusing nsx.lock ({render.baseline.describe()}).")
        mode = provenance.LOCK_REUSED
    else:
        typer.echo(f"[hardware] Resolving NSX dependencies: {reason}.")
        nsx_api.lock_app(app_dir, update=False, quiet=True, timeout_s=_LOCK_TIMEOUT_S, emit=emit)
        mode = provenance.LOCK_RESOLVED

    remaining = lock_reuse_reason(render)
    if remaining is not None:
        raise RuntimeError(
            f"NSX produced a dependency lock this build cannot use: {remaining}. "
            f"Delete {app_dir} and retry, or re-run with --update-dependencies."
        )
    nsx_api.sync_app(app_dir, frozen=True, timeout_s=_SYNC_TIMEOUT_S, emit=emit)
    return mode


def _prepare_probe_env() -> None:
    """Point NSX's SEGGER lookup at the same JLinkExe the rest of the tool uses.

    NSX resolves Commander through `$JLINK_PATH` then PATH; this package also
    looks next to the resolved J-Link library. Exporting the richer answer keeps
    the flash target and the RTT transport on one binary instead of two that can
    disagree only at flash time, minutes into a run.
    """
    try:
        found = find_jlink_exe()
    except JLinkLibraryError as exc:
        typer.echo(f"[hardware] WARNING: {exc} -- NSX will look for JLinkExe on PATH.", err=True)
        return
    if found is None:
        typer.echo(
            "[hardware] WARNING: JLinkExe not found ($JLINK_PATH, next to the J-Link library, "
            "PATH); the flash target will need it.",
            err=True,
        )
        return
    os.environ["JLINK_PATH"] = found.path


def configure(render: AppRender, options: FirmwareOptions, serial_no: Optional[int] = None) -> None:
    """`nsx configure` the app into its build dir."""
    from neuralspotx import api as nsx_api

    _prepare_probe_env()
    build_root = render.build_dir
    build_root.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[hardware] Configuring NSX app: {render.app_dir} -> {build_root}")
    nsx_api.configure_app(
        render.app_dir,
        board=render.board.nsx_board,
        build_dir=build_root,
        probe_serial=str(serial_no) if serial_no is not None else None,
        frozen=True,
        timeout_s=_CONFIGURE_TIMEOUT_S,
        emit=_emitter(options.verbose),
    )


def build(render: AppRender, options: FirmwareOptions, target: str, jobs: Optional[int]) -> None:
    """`nsx build` one target of the configured app."""
    from neuralspotx import api as nsx_api

    typer.echo(f"[hardware] Building {target}")
    nsx_api.build_app(
        render.app_dir,
        board=render.board.nsx_board,
        build_dir=render.build_dir,
        target=target,
        jobs=jobs or 8,
        frozen=True,
        timeout_s=_BUILD_TIMEOUT_S,
        emit=_emitter(options.verbose),
    )


def _needs_configure(render: AppRender) -> bool:
    return not (render.build_dir / "build.ninja").exists()


# --- flash-only-if-changed stamp -------------------------------------------------


def elf_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flash_stamp_path(build_dir: Path, board: BoardSpec, serial_no: int) -> Path:
    """Stamp recording the sha256 of the ELF last flashed to probe `serial_no` from
    this build dir. Keyed by serial because the same build can serve several boards."""
    return output_dir(build_dir, board) / f".flashed-{serial_no}.sha256"


@dataclass(frozen=True)
class FlashDecision:
    needed: bool
    digest: str
    reason: str
    build_id: Optional[str] = None
    """Build id of the firmware in the build dir (None when the build has no stamp)."""
    board_build_id: Optional[str] = None
    """What the board reported in TARGET_INFO when it was asked (None when it was not, or did not answer)."""
    # Wall-clock seconds spent in the build and in the J-Link flash (0 when skipped).
    build_seconds: float = 0.0
    flash_seconds: float = 0.0


def decide_flash(
    build_dir: Path, board: BoardSpec, serial_no: int, *, force: bool = False
) -> FlashDecision:
    """Host-side half of the decision: compare the built ELF against the stamp for
    this (build dir, serial). A `needed=False` answer here is provisional -- see
    `confirm_board_build_id`."""
    elf = elf_path(build_dir, board)
    if not elf.exists():
        raise FileNotFoundError(f"Built firmware ELF not found: {elf}")
    digest = elf_sha256(elf)
    build_id = read_build_id(build_dir, board)
    if force:
        return FlashDecision(True, digest, "--force given", build_id)
    stamp = flash_stamp_path(build_dir, board, serial_no)
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

    rtt_address = symbol_address_from_elf(str(elf_path(build_dir, board)), "_SEGGER_RTT")
    transport = JLinkRttTransport(
        serial_no=serial_no,
        chip_name=board.jlink_device,
        speed_khz=board.swd_speed_khz,
        rtt_address=rtt_address,
        reset_on_open=True,
        read_timeout_s=5.0,
        scan_ranges=board.rtt_scan_ranges,
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
            f"{build_id_path(build_dir, board)} is missing, so what the board runs cannot be confirmed (rebuild stamps it)",
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


def record_flash(build_dir: Path, board: BoardSpec, serial_no: int, digest: str) -> Path:
    stamp = flash_stamp_path(build_dir, board, serial_no)
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
    options: Optional[FirmwareOptions] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """Render, lock, sync, configure and build the firmware; returns the ELF path."""
    options = options or FirmwareOptions()
    repo_root = repo_root or tester_repo_root()
    render = prepare_app(board, repo_root=repo_root, build_dir=build_dir, options=options)
    return build_rendered_firmware(
        render, build_dir=build_dir, jobs=jobs, force_reconfigure=force_reconfigure,
        serial_no=serial_no, options=options, repo_root=repo_root,
    )


def build_rendered_firmware(
    render: AppRender,
    *,
    build_dir: Path,
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    serial_no: Optional[int] = None,
    options: Optional[FirmwareOptions] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """Lock, sync, configure and build an already-rendered app."""
    options = options or FirmwareOptions()
    board = render.board
    lock_mode = lock_and_sync(render, options)
    if force_reconfigure or _needs_configure(render) or serial_no is not None:
        # A probe serial is baked into the generated J-Link targets at configure
        # time, so switching --serial-no against a configured build dir has to
        # reconfigure for it to take effect.
        configure(render, options, serial_no=serial_no)
    build(render, options, SERVER_TARGET, jobs)

    elf = elf_path(build_dir, board)
    if not elf.is_file():
        found = find_target_binary(render.build_dir, SERVER_TARGET)
        if found is None:
            raise FileNotFoundError(
                f"Build succeeded but no {SERVER_TARGET} image was found under {render.build_dir}"
            )
        elf = found
    # Keep the lock that produced this image next to it, so a result bundle can
    # record exactly what was built without re-reading a workspace that may have
    # moved on. Best effort: a missing lock means something upstream already
    # decided not to resolve one, and losing the copy must not fail a build that
    # otherwise produced a firmware image.
    lock = render.app_dir / "nsx.lock"
    if lock.is_file():
        snapshot = lock_snapshot_path(build_dir, board)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(lock.read_bytes())
    document = provenance.build_provenance(
        render,
        repo_root=repo_root or tester_repo_root(),
        build_dir=build_dir,
        lock_mode=lock_mode,
        update_requested=options.update_dependencies,
        binary=elf,
        build_id=read_build_id(build_dir, board),
    )
    path = provenance.write_provenance(build_dir, board, document)
    typer.echo(
        f"[hardware] Dependency provenance: {document['qualification']} "
        f"({provenance.summarize_kernels(document)}) -> {path}"
    )
    return elf


def kernel_source_root(build_dir: Path, options: Optional[FirmwareOptions] = None) -> Path:
    """The ns-cmsis-nn tree the firmware was built from, for the generate step.

    NSX vendors a git-backed module as a whole-repository clone, so the synced
    module is a complete checkout -- schemas and reference tables under `Tests/`
    included -- at exactly the commit the firmware's kernels came from.
    Generation reading a different checkout than the firmware links is the drift
    this replaces. With `--cmsis-nn-root` the override wins: that tree is the one
    being edited, and NSX only mirrors a copy of it into the app.
    """
    options = options or FirmwareOptions()
    if options.cmsis_nn_root is not None:
        return Path(options.cmsis_nn_root).expanduser().resolve()
    return synced_kernel_dir(app_dir_for(build_dir))


def check_build_current(
    board: BoardSpec,
    *,
    build_dir: Path,
    options: Optional[FirmwareOptions] = None,
    repo_root: Optional[Path] = None,
) -> None:
    """Refuse to flash a build dir that is missing or older than its render inputs.

    Flashing must never re-render or reconfigure the app. An earlier revision of
    this command re-rendered from the working tree on every flash, which silently
    rebuilt one leg of an A/B comparison from the other leg's sources: the flash
    step is exactly where a divergence between "what was measured" and "what was
    built" becomes invisible.

    So the render is computed in memory only (nothing is written) and compared
    against the state file the last `hardware build` left in the app tree. A
    mismatch -- an edited baseline, a different `--cmsis-nn-root`, a changed
    kernel option, an edited benchmark-server source list -- is an error naming
    `hardware build`, not an implicit rebuild.
    """
    options = options or FirmwareOptions()
    repo_root = repo_root or tester_repo_root()
    app_dir = app_dir_for(build_dir)
    elf = elf_path(build_dir, board)
    if not elf.is_file():
        raise FileNotFoundError(
            f"No firmware to flash: {elf} does not exist. Run `hardware build --board {board.id}` first."
        )
    state = read_render_state(app_dir)
    if state is None:
        raise RuntimeError(
            f"{app_dir} has no render state, so what {elf} was built from cannot be established. "
            f"Run `hardware build --board {board.id}` first."
        )
    planned = plan_app(
        board,
        repo_root=repo_root,
        build_dir=build_dir,
        baseline=load_baseline(repo_root, options),
        cmsis_nn_root=options.cmsis_nn_root,
        kernel_options=options.kernel_options(),
    )
    if state.get("render_digest") != planned.digest:
        raise RuntimeError(
            f"The NSX app in {app_dir} was rendered from different inputs than the ones in force "
            f"now (rendered {str(state.get('render_digest'))[:12]}, current {planned.digest[:12]}): "
            "the dependency baseline, the kernel source or a build option changed since the last "
            f"build. `hardware flash` never re-renders or rebuilds -- run "
            f"`hardware build --board {board.id}` (with the same options) and flash again."
        )


def flash_firmware(
    board: BoardSpec,
    serial_no: int,
    *,
    build_dir: Path,
    force: bool = False,
    options: Optional[FirmwareOptions] = None,
    repo_root: Optional[Path] = None,
    board_build_id_reader: BoardBuildIdReader = board_build_id,
    echo: Callable[[str], None] = typer.echo,
) -> FlashDecision:
    """Flash the firmware already built in `build_dir` through the NSX J-Link recipe.

    Nothing is rendered, configured or compiled here (see `check_build_current`):
    this runs `<build>/jlink/<target>/flash_cmds.jlink` verbatim through JLinkExe
    after vetting it, then verifies the flash bank J-Link reports.

    The flash is skipped when the ELF is unchanged since this build dir last
    flashed this probe *and* the board confirms it runs this build's id -- the
    tester's own optimisation, which hpx has no equivalent of. `force` (i.e.
    `--force` / `--force-flash`) always flashes.
    """
    options = options or FirmwareOptions()
    repo_root = repo_root or tester_repo_root()
    check_build_current(board, build_dir=build_dir, options=options, repo_root=repo_root)
    decision = decide_flash(build_dir, board, serial_no, force=force)
    if not decision.needed:
        echo(f"[hardware] Stamp says {decision.reason}; asking the board which build it runs...")
        decision = confirm_board_build_id(board, serial_no, build_dir, decision, reader=board_build_id_reader)
    if not decision.needed:
        echo(f"[hardware] Skipping flash: {decision.reason}.")
        return decision
    echo(f"[hardware] Flashing {board.id} via J-Link serial {serial_no} ({decision.reason}).")
    flash_started = time.monotonic()
    flash_recipe.flash_image(
        script_path=flash_recipe.recipe_path(build_dir, board),
        bin_path=bin_path(build_dir, board),
        device=board.jlink_device,
        serial_no=serial_no,
        speed_khz=board.swd_speed_khz,
        echo=echo,
    )
    record_flash(build_dir, board, serial_no, decision.digest)
    return replace(decision, flash_seconds=time.monotonic() - flash_started)
