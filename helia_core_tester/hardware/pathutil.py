"""Shared path helpers for the hardware package.

`is_relative_to()` predates the 3.11 floor as a 3.8 compatibility shim; it stays
because `display_path()` and the firmware build both want the same containment
check in one place.
"""

from __future__ import annotations

from pathlib import Path


def is_relative_to(path: Path, root: Path) -> bool:
    """True when `path` lies under `root` (both compared as given, no symlink resolution)."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def display_path(path: Path, root: Path) -> str:
    """`path` relative to `root` when it lies inside it, else the path as given.

    The form artifact manifests and reports use for file references: repo-relative
    for the common in-tree layout, absolute for an out-of-tree `--build-dir` or
    output root (which `relative_to()` alone would reject with ValueError).
    """
    return str(path.relative_to(root)) if is_relative_to(path, root) else str(path)


def write_text_lf(path: Path, text: str) -> None:
    """Write `text` as UTF-8 with LF newlines on every platform."""
    path.write_text(text, encoding="utf-8", newline="\n")
