"""Advisory exclusive file lock shared by every on-disk cache that concurrent
runs (parallel pytest workers, two boards on one CPU) may fill at once."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive flock on `path` (created if missing) for the block.

    On platforms without fcntl the block runs unlocked: callers must still
    publish their results with an atomic rename so a racing writer can only
    produce a duplicate, never a torn, artifact.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - no flock on Windows
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
