"""Thin facade over the neuralspotx Python API.

`hardware build` goes through here instead of
``neuralspotx.api`` so that every lock/sync/configure/build call carries a
wall-clock timeout, drops NSX's own notes at verbosity 0, and fails as
:class:`HardwareBuildError` naming the step. Subprocess output (cmake,
ninja, git) still inherits the caller's stdio.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from neuralspotx import api as nsx_api
from neuralspotx._io import Emitter, Event
from neuralspotx.api import NSXError
from neuralspotx.nsx_lock import NsxLock, hash_manifest, read_lock

# Per-subprocess budgets, in seconds.
LOCK_TIMEOUT_S = 180
SYNC_TIMEOUT_S = 300
CONFIGURE_TIMEOUT_S = 120
BUILD_TIMEOUT_S = 300


class HardwareBuildError(RuntimeError):
    """An NSX lock, sync, configure or build step failed."""


def _quiet_emitter(event: Event) -> None:
    """Drop NSX notes at verbosity 0."""


def emitter_for_verbosity(verbosity: int) -> Optional[Emitter]:
    """None lets NSX print; otherwise swallow."""
    return None if verbosity >= 1 else _quiet_emitter


@contextmanager
def _nsx_errors(label: str) -> Iterator[None]:
    """Re-raise NSX failures as HardwareBuildError."""
    try:
        yield
    except (NSXError, subprocess.CalledProcessError) as exc:
        raise HardwareBuildError(f"{label} failed: {exc}") from exc


def lock_app(
    app_dir: Path,
    *,
    update: bool = False,
    timeout_s: float = LOCK_TIMEOUT_S,
    verbosity: int = 0,
) -> NsxLock:
    """Resolve module constraints and write nsx.lock."""
    with _nsx_errors("nsx lock"):
        return nsx_api.lock_app(
            app_dir,
            update=update,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
        )


def lock_is_current(app_dir: Path, board: str) -> bool:
    """nsx.lock matches the current nsx.yml."""
    try:
        lock = read_lock(app_dir, board)
    except NSXError:
        return False
    return lock is not None and lock.manifest_hash == hash_manifest(app_dir / "nsx.yml")


def sync_app(
    app_dir: Path,
    *,
    frozen: bool = False,
    force: bool = False,
    timeout_s: float = SYNC_TIMEOUT_S,
    verbosity: int = 0,
) -> None:
    """Materialise modules/ to match nsx.lock.

    ``frozen`` verifies an existing modules/ tree and refuses to
    modify it; it cannot populate a fresh checkout. Sync once
    unfrozen before syncing frozen.
    """
    with _nsx_errors("nsx sync"):
        nsx_api.sync_app(
            app_dir,
            frozen=frozen,
            force=force,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
        )


def configure_app(
    app_dir: Path,
    board: str,
    *,
    build_dir: Optional[Path] = None,
    toolchain: Optional[str] = None,
    probe_serial: Optional[int] = None,
    frozen: bool = False,
    timeout_s: float = CONFIGURE_TIMEOUT_S,
    verbosity: int = 0,
) -> None:
    """Run the CMake configure for one board.

    ``probe_serial`` lands in the generated flash target.
    ``frozen`` refuses to re-vendor modules/ if it drifts from nsx.lock.
    """
    with _nsx_errors("nsx configure"):
        nsx_api.configure_app(
            app_dir,
            board=board,
            build_dir=build_dir,
            toolchain=toolchain,
            probe_serial=None if probe_serial is None else str(probe_serial),
            frozen=frozen,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
        )


def build_app(
    app_dir: Path,
    *,
    board: Optional[str] = None,
    build_dir: Optional[Path] = None,
    toolchain: Optional[str] = None,
    target: Optional[str] = None,
    jobs: Optional[int] = None,
    frozen: bool = False,
    timeout_s: float = BUILD_TIMEOUT_S,
    verbosity: int = 0,
) -> None:
    """Build the app; jobs=None keeps NSX's default.

    ``frozen`` applies when the build triggers a reconfigure.
    """
    kwargs: dict[str, Any] = {} if jobs is None else {"jobs": jobs}
    with _nsx_errors("nsx build"):
        nsx_api.build_app(
            app_dir,
            board=board,
            build_dir=build_dir,
            toolchain=toolchain,
            target=target,
            frozen=frozen,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
            **kwargs,
        )


def starter_profile(board: str) -> Optional[dict[str, Any]]:
    """The board's minimal starter profile, or None."""
    return nsx_api.starter_profile(board)
