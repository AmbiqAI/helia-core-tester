"""Path helpers that stay within the Python >=3.8 floor of pyproject.toml.

`Path.is_relative_to()` is 3.9+, so the containment check is spelled with
`relative_to()` here and reused wherever the package needs it.
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
