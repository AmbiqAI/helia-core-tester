"""Thin facade over the neuralspotx Python API.

Not wired into any command yet. Callers go through here instead of
``neuralspotx.api`` so that every lock/sync/configure/build call carries a
wall-clock timeout, stays quiet at verbosity 0, and fails as
:class:`HardwareBuildError` -- a RuntimeError, which ``cli._pipeline_errors``
already reports as a one-line error.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from neuralspotx import api as nsx_api
from neuralspotx._io import Emitter, Event
from neuralspotx.api import NSXError
from neuralspotx.nsx_lock import NsxLock

# Per-subprocess budgets, in seconds.
LOCK_TIMEOUT_S = 180
SYNC_TIMEOUT_S = 300
CONFIGURE_TIMEOUT_S = 120
BUILD_TIMEOUT_S = 300

T = TypeVar("T")


class HardwareBuildError(RuntimeError):
    """An NSX lock, sync, configure or build step failed."""


def _quiet_emitter(event: Event) -> None:
    """Drop NSX output at verbosity 0."""


def emitter_for_verbosity(verbosity: int) -> Optional[Emitter]:
    """None lets NSX print; otherwise swallow."""
    return None if verbosity >= 1 else _quiet_emitter


def _translate(label: str, func: Callable[[], T]) -> T:
    """Re-raise NSX failures as HardwareBuildError."""
    try:
        return func()
    except NSXError as exc:
        raise HardwareBuildError(f"{label} failed: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        # NSX's build runner raises this unwrapped.
        cmd = " ".join(str(a) for a in exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd)
        details = f"exit status {exc.returncode}: {cmd}"
        if exc.stderr:
            details += f"\n{exc.stderr}"
        raise HardwareBuildError(f"{label} failed: {details}") from exc


def lock_app(
    app_dir: Path,
    *,
    update: bool = False,
    timeout_s: float = LOCK_TIMEOUT_S,
    verbosity: int = 0,
) -> NsxLock:
    """Resolve module constraints and write nsx.lock."""
    return _translate(
        "nsx lock",
        lambda: nsx_api.lock_app(
            app_dir,
            update=update,
            quiet=verbosity == 0,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
        ),
    )


def sync_app(
    app_dir: Path,
    *,
    frozen: bool = True,
    force: bool = False,
    timeout_s: float = SYNC_TIMEOUT_S,
    verbosity: int = 0,
) -> None:
    """Materialise modules/ to match nsx.lock; frozen refuses drift."""
    _translate(
        "nsx sync",
        lambda: nsx_api.sync_app(
            app_dir,
            frozen=frozen,
            force=force,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
        ),
    )


def configure_app(
    app_dir: Path,
    board: str,
    *,
    build_dir: Optional[Path] = None,
    toolchain: Optional[str] = None,
    timeout_s: float = CONFIGURE_TIMEOUT_S,
    verbosity: int = 0,
) -> None:
    """Run the CMake configure for one board."""
    _translate(
        "nsx configure",
        lambda: nsx_api.configure_app(
            app_dir,
            board=board,
            build_dir=build_dir,
            toolchain=toolchain,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
        ),
    )


def build_app(
    app_dir: Path,
    *,
    board: Optional[str] = None,
    build_dir: Optional[Path] = None,
    toolchain: Optional[str] = None,
    target: Optional[str] = None,
    jobs: Optional[int] = None,
    timeout_s: float = BUILD_TIMEOUT_S,
    verbosity: int = 0,
) -> None:
    """Build the app; jobs=None keeps NSX's default."""
    kwargs: dict[str, Any] = {} if jobs is None else {"jobs": jobs}
    _translate(
        "nsx build",
        lambda: nsx_api.build_app(
            app_dir,
            board=board,
            build_dir=build_dir,
            toolchain=toolchain,
            target=target,
            timeout_s=timeout_s,
            emit=emitter_for_verbosity(verbosity),
            **kwargs,
        ),
    )


def starter_profile(board: str) -> Optional[dict[str, Any]]:
    """The board's minimal starter profile, or None."""
    return _translate("nsx starter profile", lambda: nsx_api.starter_profile(board))
