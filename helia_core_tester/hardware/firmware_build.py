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

Flashing is unchanged from the CMake era: `nsx_finalize_app()` generates the
same `<target>_flash` J-Link target, and the "flash only if the ELF changed"
decision still has two halves. The host-side stamp
(`<build_dir>/nsx_app/build/<board>/.flashed-<serial>.sha256`, next to the image
it describes) says whether *this build dir* last flashed *this probe* with the
current ELF. It cannot know what another build dir
(a second clone, `--build-dir`, a lab runner sharing the board) did since, so a
stamp match is only trusted after the board itself confirms it: every firmware
build carries a content-hash build id (`hct_build_id.txt`, alongside the image in
the same directory, stamped into the linked image by scripts/patch_build_id.py as
a POST_BUILD step and advertised by the firmware in TARGET_INFO), and the skip
path opens one short RTT session to read it.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import typer

from .boards import BoardSpec
from .boards import repo_root as tester_repo_root
from .dependency_baseline import NON_NSX_CHECKOUTS, DependencyBaseline, resolve_baseline
from .jlink_library import JLinkLibraryError, find_jlink_exe
from .nsx_app import (
    SERVER_TARGET,
    AppRender,
    KernelOptions,
    app_dir_for,
    commit_render_state,
    nsx_build_dir,
    read_render_state,
    render_app,
    synced_kernel_dir,
)
from .toolchain import DOWNLOADS_DIR, add_toolchain_to_path

FLASH_TARGET = f"{SERVER_TARGET}_flash"

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


def ensure_host_tools(repo_root: Path, baseline: Optional[DependencyBaseline] = None) -> None:
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
    if baseline is not None:
        pin_optional_checkouts(repo_root, baseline)


#: How a non-NSX checkout stands relative to the baseline. The two non-clean
#: values use heliaPROFILER's qualification vocabulary, which the provenance
#: record this feeds into is built on.
PIN_MATCHED = "pinned"
PIN_DEVELOPMENT_OVERRIDES = "development-overrides"
PIN_UNVERIFIED = "unverified"


def pin_optional_checkouts(
    repo_root: Path, baseline: DependencyBaseline
) -> Dict[str, str]:
    """Put the non-NSX checkouts the firmware consumes on the baseline's commit.

    Everything else the firmware is built from is an NSX module, so `nsx lock`
    enforces its pin. CMSIS_5 is not: `setup_cmsis5()` shallow-clones the default
    branch for the FVP path, and the firmware takes `pmu_armv8.h` out of the same
    tree. Without this the baseline would *claim* a CMSIS_5 commit that nothing
    checked -- worse than not listing it, because the claim is recorded in every
    bundle.

    The working tree is inspected before `HEAD`, which is the whole point: a
    checkout sitting on the pinned commit *with uncommitted edits* is not the
    pinned content, and checking `HEAD` first would accept it as though it were.
    That is the one case where the build is silently unqualified and nothing
    says so -- a repointed checkout at least gets repointed, and a foreign
    directory is obviously foreign.

    Repointing therefore only ever happens on a clean checkout, and the pin is
    fetched first because the clone is shallow and will not have it as a local
    object. A dirty tree is left alone -- discarding someone's edits to enforce
    a pin would be worse than building unqualified -- and so is a directory that
    is not a git checkout at all (a vendored copy, a distro package).

    Returns the status per project so callers can record it rather than relying
    on having read stderr.
    """
    import subprocess

    statuses: Dict[str, str] = {}
    for project, dirname in NON_NSX_CHECKOUTS.items():
        entry = baseline.projects.get(project)
        if entry is None:
            continue
        pin, url = entry.ref, entry.url
        path = repo_root / DOWNLOADS_DIR / dirname
        if not (path / ".git").exists():
            statuses[project] = PIN_UNVERIFIED
            typer.echo(
                f"[hardware] WARNING: {path} is not a git checkout, so the baseline's {project} pin "
                f"{pin[:12]} cannot be verified; building against whatever is there.",
                err=True,
            )
            continue

        def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-C", str(path), *args], capture_output=True, text=True, check=check
            )
        try:
            head = _git("rev-parse", "HEAD").stdout.strip()
            if _git("status", "--porcelain").stdout.strip():
                statuses[project] = PIN_DEVELOPMENT_OVERRIDES
                at_pin = " (at the pinned commit, but with uncommitted changes)" if head == pin else ""
                typer.echo(
                    f"[hardware] WARNING: {path} has local changes{at_pin}, so this build does not "
                    f"match the baseline's {project} pin {pin[:12]}; leaving the checkout alone.",
                    err=True,
                )
                continue
            if head == pin:
                statuses[project] = PIN_MATCHED
                continue
            typer.echo(f"[hardware] Repointing {dirname} to the baseline pin {pin[:12]} from {url}...")
            # Fetched from the baseline's own URL rather than from whatever the
            # checkout's `origin` happens to be: a `--baseline` naming a fork
            # would otherwise ask upstream for a commit only the fork has (an
            # unhelpful failure) or, on a checkout someone had repointed, take
            # the pin from a repository the baseline never named while the log
            # line claimed the fork. Once fetched, checking the SHA out is
            # unambiguous -- a commit id is a content hash, so the tree it
            # names is the same whichever remote served it.
            _git("fetch", "--quiet", "--depth=1", url, pin)
            _git("checkout", "--quiet", "--detach", pin)
            statuses[project] = PIN_MATCHED
        except (OSError, subprocess.CalledProcessError) as exc:
            statuses[project] = PIN_UNVERIFIED
            typer.echo(
                f"[hardware] WARNING: could not put {path} on the baseline's {project} pin "
                f"{pin[:12]} ({exc}); building against whatever is there.",
                err=True,
            )
    return statuses


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
    baseline = load_baseline(repo_root, options)
    ensure_host_tools(repo_root, baseline)
    return render_app(
        board,
        repo_root=repo_root,
        build_dir=build_dir,
        baseline=baseline,
        cmsis_nn_root=options.cmsis_nn_root,
        kernel_options=options.kernel_options(),
    )


def lock_reuse_reason(render: AppRender) -> Optional[str]:
    """Why the on-disk `nsx.lock` cannot be *reused*, or None when it can.

    Two different questions get asked of a lock and they must not be conflated.
    This is the first: may the lock already on disk stand in for a resolution we
    would otherwise perform? That is only true if the last build recorded this
    exact render -- manifest text plus baseline fingerprint -- which is what
    catches a baseline edit that leaves `nsx.yml` byte-identical and so slips
    past NSX's own manifest hash.

    The second question, "is the lock NSX just wrote usable", is
    `lock_validity_reason`: it must not consult the render state, because the
    state describes the *previous* build and is only committed once this one
    succeeds.
    """
    app_dir = render.app_dir
    if not (app_dir / "nsx.lock").is_file():
        return "nsx.lock is missing"
    state = read_render_state(app_dir)
    if state is None:
        return "this app has no recorded render state"
    if state.get("render_digest") != render.digest:
        return "the rendered manifest or the dependency baseline changed"
    return lock_validity_reason(render)


def lock_validity_reason(render: AppRender) -> Optional[str]:
    """Why the on-disk `nsx.lock` is not usable at all, or None when it is.

    Structure only: the schema NSX requires, a section for this board, agreement
    with the manifest it was resolved from, and an exact peeled commit for every
    git module. Says nothing about which render it belongs to.
    """
    from neuralspotx.nsx_lock import LOCK_SCHEMA_VERSION, hash_manifest, read_lock

    app_dir = render.app_dir
    if not (app_dir / "nsx.lock").is_file():
        return "nsx.lock is missing"
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
    return baseline_resolution_reason(render, lock)


def baseline_resolution_reason(render: AppRender, lock: Any) -> Optional[str]:
    """Why the lock's resolved commits contradict the baseline, or None when they agree.

    The rendered manifest *asserts* the qualified pins; the lock is the
    *outcome*, and the claim "this firmware is built from the baseline" only
    means anything if the two agree. They can disagree without anything looking
    wrong: NSX gives a packaged registry's module-level revision precedence over
    an app's project-level override, so an alignment bug in the emitted
    `module_registry` resolves a different commit while every artifact still
    quotes the baseline -- heliaPROFILER found exactly that after eight hardware
    runs had silently built the wrong nsx-sensors, which is why it added
    `deps/dependencies.py::_verify_baseline_resolution`. A hand-edited or stale
    `nsx.lock` gets there too, and `sync_app(frozen=True)` would then faithfully
    materialise the wrong tree, because frozen verifies the modules against the
    lock, not the lock against the baseline.

    The URL is checked with the commit for the same reason it is emitted with
    it: a commit is only identified by the repository it is in. Modules that are
    not git-backed (`packaged`, or a `--cmsis-nn-root` local source) have no pin
    to contradict and are skipped.
    """
    baseline = render.baseline
    for name, module in sorted(lock.modules.items()):
        if str(module.kind) != "git":
            continue
        project = getattr(module, "project", None)
        if project is None:
            continue
        pinned = baseline.projects.get(str(project))
        if pinned is None:
            continue
        commit = (module.commit or "").lower()
        if commit != pinned.ref.lower():
            return (
                f"nsx.lock resolved module '{name}' to {commit or '<none>'}, but the baseline "
                f"pins project '{project}' at {pinned.ref}"
            )
        # Required, not merely checked when present: a truthiness guard would let
        # a lock with the url stripped out pass, which is the easiest edit to
        # make and the one that hides which repository a commit came from. A
        # baseline-pinned git module with no recorded url is unattributable, so
        # it is rejected like a mismatched one.
        url = (getattr(module, "url", None) or "").strip()
        if not url:
            return (
                f"nsx.lock records no repository for module '{name}', so the commit it pins "
                f"cannot be attributed to the baseline's '{project}' at {pinned.url}"
            )
        if url != pinned.url:
            return (
                f"nsx.lock fetched module '{name}' from {url}, but the baseline pins project "
                f"'{project}' in {pinned.url}"
            )
    return None


def lock_and_sync(render: AppRender, options: FirmwareOptions) -> None:
    """Resolve `nsx.lock` when it cannot be reused, then materialise `modules/`.

    The sync is always frozen: it must reproduce exactly what the lock names and
    fail on drift rather than quietly re-vendoring something else.
    """
    from neuralspotx import api as nsx_api

    app_dir = render.app_dir
    emit = _emitter(options.verbose)
    reason = None if options.update_dependencies else lock_reuse_reason(render)
    if options.update_dependencies:
        typer.echo("[hardware] Re-resolving NSX dependencies (--update-dependencies).")
        nsx_api.lock_app(app_dir, update=True, quiet=True, timeout_s=_LOCK_TIMEOUT_S, emit=emit)
    elif reason is None:
        typer.echo(f"[hardware] Reusing nsx.lock ({render.baseline.describe()}).")
    else:
        typer.echo(f"[hardware] Resolving NSX dependencies: {reason}.")
        nsx_api.lock_app(app_dir, update=False, quiet=True, timeout_s=_LOCK_TIMEOUT_S, emit=emit)

    # The lock just written describes this render by construction, so the
    # question here is only whether it is structurally usable -- asking
    # lock_reuse_reason would compare against the previous build's state, which
    # is exactly what a resolve was needed to move past.
    remaining = lock_validity_reason(render)
    if remaining is not None:
        raise RuntimeError(
            f"NSX produced a dependency lock this build cannot use: {remaining}. "
            f"Delete {app_dir} and retry, or re-run with --update-dependencies."
        )
    nsx_api.sync_app(app_dir, frozen=True, timeout_s=_SYNC_TIMEOUT_S, emit=emit)


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


#: What the last successful `nsx configure` of a build tree was configured from.
CONFIGURE_STATE = ".hct-configured.json"


def _configure_identity(render: AppRender, serial_no: Optional[int]) -> Dict[str, Any]:
    """Everything that decides what `nsx configure` would produce.

    The render digest covers the manifest and the app CMakeLists; the lock hash
    covers which module trees `nsx sync` materialised; and the board, probe
    serial and resolved JLinkExe are configure-time arguments that reach the
    CMake cache without touching any file CMake watches.
    """
    lock = render.app_dir / "nsx.lock"
    return {
        "render_digest": render.digest,
        "nsx_lock_sha256": (
            hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
        ),
        "board": render.board.nsx_board,
        "probe_serial": str(serial_no) if serial_no is not None else None,
        "jlink_exe": os.environ.get("JLINK_PATH"),
    }


def _configure_reason(render: AppRender, identity: Mapping[str, Any]) -> Optional[str]:
    """Why the build tree must be (re)configured, or None when it need not be.

    CMake re-runs itself when a file it listed as a configure input changes, and
    the app CMakeLists and `cmake/nsx/modules.cmake` are both on that list -- so
    a kernel-switch change or a re-synced module set would be picked up even
    without this check. It is not enough on its own, though: the probe serial,
    the resolved JLinkExe path and the board are configure *arguments*, so
    changing one of them changes the CMake cache with no input file touched and
    CMake would happily keep the stale cache. Deciding here, from a recorded
    identity, also stops the correctness of a build depending on a CMake
    implementation detail.
    """
    build_dir = render.build_dir
    if not (build_dir / "build.ninja").exists():
        return "the build tree is not configured yet"
    state_path = build_dir / CONFIGURE_STATE
    if not state_path.is_file():
        return "the build tree has no record of what it was configured from"
    try:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "the build tree's configure record is unreadable"
    if not isinstance(previous, dict):
        return "the build tree's configure record is malformed"
    changed = sorted(k for k in identity if previous.get(k) != identity[k])
    if changed:
        return f"the configured inputs changed ({', '.join(changed)})"
    return None


def _record_configure(render: AppRender, identity: Mapping[str, Any]) -> None:
    path = render.build_dir / CONFIGURE_STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(identity), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        serial_no=serial_no, options=options,
    )


def build_rendered_firmware(
    render: AppRender,
    *,
    build_dir: Path,
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    serial_no: Optional[int] = None,
    options: Optional[FirmwareOptions] = None,
) -> Path:
    """Lock, sync, configure and build an already-rendered app."""
    options = options or FirmwareOptions()
    board = render.board
    lock_and_sync(render, options)
    # The probe serial is baked into the generated J-Link targets at configure
    # time, and the resolved JLinkExe into the CMake cache, so both are part of
    # the configured identity rather than a blanket "always reconfigure".
    _prepare_probe_env()
    identity = _configure_identity(render, serial_no)
    reason = "--force-reconfigure given" if force_reconfigure else _configure_reason(render, identity)
    if reason is not None:
        typer.echo(f"[hardware] Reconfiguring: {reason}.")
        configure(render, options, serial_no=serial_no)
        _record_configure(render, identity)
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
    # Only now is the claim the state file makes true: a lock and a module tree
    # matching this render exist, and an image was built from them. Recording it
    # earlier is what would let a baseline change that leaves nsx.yml identical
    # slip past `lock_reuse_reason` (see nsx_app.commit_render_state).
    commit_render_state(render)
    return elf


def kernel_source_root(build_dir: Path, options: Optional[FirmwareOptions] = None) -> Path:
    """The ns-cmsis-nn tree the firmware in `build_dir` was built from.

    Generation reads kernel schemas and reference tables out of this tree, so it
    has to be the tree the *flashed image* linked, not the one a flag names now.
    Those differ in the case this exists to close: `--skip-flash` reuses firmware
    without building or syncing anything, so an explicit `--cmsis-nn-root` would
    otherwise silently point generation at a checkout that image was never built
    from -- the cases and the kernels under test back on separate commits, which
    is the drift the whole NSX-app change is meant to make impossible.

    So the build's own record wins: when the app tree says what it was built
    from, an override that disagrees is an error rather than a silent
    substitution, and the answer is the synced module tree. `nsx sync` mirrors a
    `--cmsis-nn-root` checkout into that same directory on every build, so for a
    matching override this is still the live tree's content -- as of the last
    build, which is precisely what was flashed.
    """
    options = options or FirmwareOptions()
    app_dir = app_dir_for(build_dir)
    synced = synced_kernel_dir(app_dir)
    requested = (
        str(Path(options.cmsis_nn_root).expanduser().resolve())
        if options.cmsis_nn_root is not None
        else None
    )
    recorded = (read_render_state(app_dir) or {}).get("kernel_source")

    if recorded is not None:
        if requested is not None and recorded != f"path:{requested}":
            raise RuntimeError(
                f"--cmsis-nn-root {requested} does not match what the firmware in {build_dir} was "
                f"built from ({recorded}). Generation would read a different kernel tree than the "
                f"image under test. Re-run `hardware build --board {board_name(app_dir)}` with this "
                f"--cmsis-nn-root, or drop the flag to use the tree the image was built from."
            )
        if not synced.is_dir():
            # Falling back to the live --cmsis-nn-root here would be the exact
            # substitution this function exists to prevent, just reached by a
            # different route: the recorded build used the mirror NSX synced,
            # and the working tree has been free to change since. There is no
            # honest answer without a rebuild, so say so.
            raise RuntimeError(
                f"The firmware in {build_dir} was built from {recorded}, but the synced kernel tree "
                f"{synced} is gone, so generation has nothing to read that matches the image. "
                f"Re-run `hardware build --board {board_name(app_dir)}` (which re-syncs it) before "
                f"generating."
            )
        return synced

    # No build to speak of: an override, or the place a build would put it.
    if requested is not None:
        return Path(requested)
    return synced


def board_name(app_dir: Path) -> str:
    """The board an app tree targets, for error messages (best effort)."""
    return str((read_render_state(app_dir) or {}).get("board") or "<board>")


def flash_firmware(
    board: BoardSpec,
    serial_no: int,
    *,
    build_dir: Path,
    jobs: Optional[int] = None,
    force_reconfigure: bool = False,
    force: bool = False,
    options: Optional[FirmwareOptions] = None,
    repo_root: Optional[Path] = None,
    board_build_id_reader: BoardBuildIdReader = board_build_id,
) -> FlashDecision:
    """Build, then flash through the NSX-generated J-Link target unless the ELF is
    unchanged since this build dir last flashed this probe *and* the board confirms
    it is running this build's id (or `force` is set)."""
    options = options or FirmwareOptions()
    repo_root = repo_root or tester_repo_root()
    build_started = time.monotonic()
    render = prepare_app(board, repo_root=repo_root, build_dir=build_dir, options=options)
    build_rendered_firmware(
        render, build_dir=build_dir, jobs=jobs, force_reconfigure=force_reconfigure,
        serial_no=serial_no, options=options,
    )
    build_seconds = time.monotonic() - build_started
    decision = decide_flash(build_dir, board, serial_no, force=force)
    if not decision.needed:
        typer.echo(f"[hardware] Stamp says {decision.reason}; asking the board which build it runs...")
        decision = confirm_board_build_id(board, serial_no, build_dir, decision, reader=board_build_id_reader)
    if not decision.needed:
        typer.echo(f"[hardware] Skipping flash: {decision.reason}.")
        return replace(decision, build_seconds=build_seconds)
    typer.echo(f"[hardware] Flashing {board.id} via J-Link serial {serial_no} ({decision.reason}).")
    flash_started = time.monotonic()
    build(render, options, FLASH_TARGET, jobs)
    record_flash(build_dir, board, serial_no, decision.digest)
    return replace(decision, build_seconds=build_seconds, flash_seconds=time.monotonic() - flash_started)
