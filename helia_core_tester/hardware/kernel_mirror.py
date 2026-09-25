"""Mirror a local ns-cmsis-nn checkout for NSX.

NSX hashes and copies a module's local_path whole, nested tester
artifacts included. Mirror only what git would commit instead.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .pathutil import is_relative_to

KERNEL_SRC_SUBDIR = "kernel_src"


class KernelMirrorError(RuntimeError):
    """The kernel checkout could not be mirrored."""


@dataclass(frozen=True)
class KernelMirror:
    """Where the mirror lives and what changed."""

    path: Path
    files: int
    size: int
    changed: int
    stamp: str


def nested_kernel_root(repo_root: Path) -> Optional[Path]:
    """The enclosing ns-cmsis-nn checkout, if any."""
    # Layout: ns-cmsis-nn/Tests/helia-core-tester.
    root = repo_root.resolve().parent.parent
    markers = (root / "Include", root / "Source")
    if all(path.is_dir() for path in markers) and (root / "nsx" / "nsx-module.yaml").is_file():
        return root
    return None


def _git_files(root: Path) -> list[str]:
    """Paths git would commit under root."""
    cmd = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise KernelMirrorError(f"Kernel root is not a git checkout: {root}") from exc
    # A nested repo lists as "dir/".
    return sorted({name.rstrip("/") for name in os.fsdecode(out).split("\0") if name})


def _source_files(root: Path, excluded: Iterable[Path]) -> dict[str, os.stat_result]:
    """Regular files to mirror, with their stats."""
    prefixes = [path.relative_to(root).as_posix() for path in excluded if is_relative_to(path, root)]
    files: dict[str, os.stat_result] = {}
    for rel in _git_files(root):
        if any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes):
            continue
        try:
            info = (root / rel).stat()
        except OSError:
            # Deleted in the working tree.
            continue
        if stat.S_ISREG(info.st_mode):
            files[rel] = info
    return files


def _prune(dest: Path, keep: set[str]) -> int:
    """Delete mirror files no longer listed."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(dest, topdown=False):
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if path.relative_to(dest).as_posix() not in keep:
                path.unlink()
                removed += 1
        for name in dirnames:
            path = base / name
            if not any(path.iterdir()):
                path.rmdir()
    return removed


def _stamp(files: dict[str, os.stat_result]) -> str:
    """Hash of paths, sizes and mtimes."""
    digest = hashlib.sha256()
    for rel, info in sorted(files.items()):
        digest.update(f"{rel}\0{info.st_size}\0{info.st_mtime_ns}\n".encode("utf-8", "surrogateescape"))
    return "sha256:" + digest.hexdigest()


def mirror_kernels(root: Path, build_dir: Path, *, exclude: Iterable[Path] = ()) -> KernelMirror:
    """Sync `<build_dir>/kernel_src` with root."""
    root = root.expanduser().resolve()
    build_dir = build_dir.resolve()
    if is_relative_to(root, build_dir):
        raise KernelMirrorError(f"Kernel root {root} is inside build dir {build_dir}")
    dest = build_dir / KERNEL_SRC_SUBDIR
    files = _source_files(root, [build_dir, *(path.resolve() for path in exclude)])
    # Prune first: a file may become a dir.
    changed = _prune(dest, set(files)) if dest.is_dir() else 0
    for rel, info in files.items():
        target = dest / rel
        try:
            have = target.stat()
        except OSError:
            have = None
        if have is not None and (have.st_size, have.st_mtime_ns) == (info.st_size, info.st_mtime_ns):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # copy2 keeps mtimes, so ninja stays incremental.
        shutil.copy2(root / rel, target)
        changed += 1
    size = sum(info.st_size for info in files.values())
    return KernelMirror(dest, len(files), size, changed, _stamp(files))
