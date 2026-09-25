"""Mirror a local ns-cmsis-nn checkout for NSX.

NSX hashes and copies a module's local_path whole, nested tester
artifacts included. Mirror only what git would commit instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..generation.reuse import _is_git_toplevel
from .pathutil import is_relative_to

KERNEL_SRC_SUBDIR = "kernel_src"
# Source stats of the last mirror.
MIRROR_INDEX = "kernel_src.json"


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
        done = subprocess.run(cmd, capture_output=True)
    except OSError as exc:
        raise KernelMirrorError(f"Cannot run git: {exc}") from exc
    if done.returncode != 0:
        detail = os.fsdecode(done.stderr).strip()
        raise KernelMirrorError(f"git ls-files failed in {root}: {detail}")
    # git -C walks up to outer repos.
    if not _is_git_toplevel(root):
        raise KernelMirrorError(f"Kernel root is not a git checkout: {root}")
    # A nested repo lists as "dir/".
    return sorted({name.rstrip("/") for name in os.fsdecode(done.stdout).split("\0") if name})


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


def _stamp(root: Path, index: dict[str, list[int]]) -> str:
    """Hash of root, paths, sizes, mtimes."""
    digest = hashlib.sha256(f"{root}\n".encode("utf-8", "surrogateescape"))
    for rel, (size, mtime) in sorted(index.items()):
        digest.update(f"{rel}\0{size}\0{mtime}\n".encode("utf-8", "surrogateescape"))
    return "sha256:" + digest.hexdigest()


def _read_index(path: Path, root: Path) -> dict[str, list[int]]:
    """Last mirror's source stats, same root only."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved.get("files", {}) if saved.get("root") == str(root) else {}


def drop_mirror(build_dir: Path) -> None:
    """Remove the mirror and its index."""
    shutil.rmtree(build_dir / KERNEL_SRC_SUBDIR, ignore_errors=True)
    (build_dir / MIRROR_INDEX).unlink(missing_ok=True)


def mirror_kernels(root: Path, build_dir: Path, *, exclude: Iterable[Path] = ()) -> KernelMirror:
    """Sync `<build_dir>/kernel_src` with root."""
    root = root.expanduser().resolve()
    build_dir = build_dir.resolve()
    if is_relative_to(root, build_dir):
        raise KernelMirrorError(f"Kernel root {root} is inside build dir {build_dir}")
    dest = build_dir / KERNEL_SRC_SUBDIR
    files = _source_files(root, [build_dir, *(path.resolve() for path in exclude)])
    index = {rel: [info.st_size, info.st_mtime_ns] for rel, info in files.items()}
    last = _read_index(build_dir / MIRROR_INDEX, root) if dest.is_dir() else {}
    # Prune first: a file may become a dir.
    changed = _prune(dest, set(files)) if dest.is_dir() else 0
    for rel, stats in index.items():
        target = dest / rel
        if last.get(rel) == stats and target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Fresh mtime, so ninja recompiles it.
        shutil.copy(root / rel, target)
        changed += 1
    (build_dir / MIRROR_INDEX).write_text(json.dumps({"root": str(root), "files": index}), encoding="utf-8")
    size = sum(info.st_size for info in files.values())
    return KernelMirror(dest, len(files), size, changed, _stamp(root, index))
