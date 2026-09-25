"""Where the ARM GCC binutils come from for host-side ELF inspection.

The hardware commands fetch ARM GCC lazily into artifacts/downloads/ (see
`firmware_build.ensure_hardware_dependencies`). CMake is pointed at it through
the toolchain file, but the host also runs `arm-none-eabi-nm` (RTT block
address, memory report) and `-size`/`-objdump` (memory report) itself. Those
go through `arm_tool()` so a fresh clone that only just downloaded the
toolchain resolves them without the user editing PATH, and
`add_toolchain_to_path()` makes the same bin/ visible to anything that still
shells out to a bare tool name (the build's generate_kernel_symbol_refs.py).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

DOWNLOADS_DIR = "artifacts/downloads"


def _default_repo_root() -> Path:
    from .boards import repo_root

    return repo_root()


def toolchain_bin_dir(repo_root: Optional[Path] = None) -> Path:
    """`bin/` of the downloaded ARM GCC (may not exist yet)."""
    return (repo_root or _default_repo_root()) / DOWNLOADS_DIR / "arm_gcc_download" / "bin"


def arm_tool(name: str, repo_root: Optional[Path] = None) -> str:
    """Executable to run for `name` (e.g. `arm-none-eabi-nm`): the downloaded
    toolchain's copy when present, else whatever PATH offers (returned as the bare
    name so a missing tool still fails with a clear FileNotFoundError naming it)."""
    candidate = toolchain_bin_dir(repo_root) / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(name) or name


def add_toolchain_to_path(repo_root: Optional[Path] = None) -> bool:
    """Prepend the downloaded toolchain's bin/ to this process's PATH once.

    Returns True when it was added, False when it is already there or does not
    exist. Idempotent, so every hardware entry point can call it after the lazy
    dependency setup without stacking duplicates."""
    bin_dir = toolchain_bin_dir(repo_root)
    if not bin_dir.is_dir():
        return False
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(bin_dir) in entries:
        return False
    os.environ["PATH"] = os.pathsep.join([str(bin_dir), *entries]) if entries != [""] else str(bin_dir)
    return True
