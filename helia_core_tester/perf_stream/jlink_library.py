"""Locate the SEGGER J-Link shared library for pylink.

pylink's own default search (`pylink.library.Library()`) only asks the dynamic
loader (`ctypes.util.find_library`, i.e. the ldconfig cache / LD_LIBRARY_PATH)
and then walks `/opt/SEGGER`. Any install outside those -- a user-local unpack,
a Nix store path, the lab runners' `/var/lib/hpx-runner-tools/JLink` -- is
invisible to it even when `JLinkExe` from the same install is on PATH.

This module resolves the library the way the lab runners (and hpx) expect, in
this order:

1. `HPX_JLINK_DLL` -- explicit path to the shared library file.
2. `JLINK_PATH` -- the `JLinkExe` binary (hpx contract) or its directory; the
   library is looked up next to it.
3. `JLinkExe` on PATH -- same lookup next to the binary.
4. pylink's default search (nothing resolved here; `pylink.JLink()` does it).

Symlinks are resolved for steps 2 and 3, because packaged installs (Nix, the
runner tools dir) expose `bin/JLinkExe` as a symlink into the real SEGGER
directory that holds `libjlinkarm.so`.

The environment variable names are shared with hpx so one runner configuration
serves both tools; do not rename them.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional

DLL_ENV_VAR = "HPX_JLINK_DLL"
EXE_ENV_VAR = "JLINK_PATH"
JLINK_EXE_NAMES = ("JLinkExe", "JLink.exe")

SOURCE_DLL_ENV = f"${DLL_ENV_VAR}"
SOURCE_EXE_ENV = f"${EXE_ENV_VAR}"
SOURCE_PATH = "JLinkExe on PATH"
SOURCE_PYLINK_DEFAULT = "pylink default search"

WhichFn = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class JLinkLibrary:
    path: str
    source: str

    def describe(self) -> str:
        return f"{self.path} (via {self.source})"


def _library_patterns() -> tuple:
    if sys.platform.startswith("darwin"):
        return ("libjlinkarm.dylib", "libjlinkarm.*.dylib")
    if sys.platform.startswith("win"):
        return ("JLink_x64.dll", "JLinkARM.dll")
    return ("libjlinkarm.so", "libjlinkarm.so.*")


def _library_in_dir(directory: Path) -> Optional[Path]:
    """Return the J-Link shared library inside *directory*, preferring the unversioned name."""
    if not directory.is_dir():
        return None
    for pattern in _library_patterns():
        matches = sorted(p for p in directory.glob(pattern) if p.is_file())
        if matches:
            return matches[0]
    return None


def _library_next_to(exe_or_dir: Path) -> Optional[Path]:
    """Look for the library beside a JLinkExe path (or inside a directory), following symlinks."""
    candidates = []
    if exe_or_dir.is_dir():
        candidates.append(exe_or_dir)
    else:
        candidates.append(exe_or_dir.parent)
    resolved = Path(os.path.realpath(str(exe_or_dir)))
    candidates.append(resolved if resolved.is_dir() else resolved.parent)
    for directory in candidates:
        found = _library_in_dir(directory)
        if found is not None:
            return found
    return None


def find_jlink_library(env: Optional[Mapping[str, str]] = None, which: WhichFn = shutil.which) -> Optional[JLinkLibrary]:
    """Return the first J-Link library found by the env / PATH steps, or None for pylink's default."""
    env = os.environ if env is None else env

    explicit = (env.get(DLL_ENV_VAR) or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return JLinkLibrary(str(path), SOURCE_DLL_ENV)

    exe_hint = (env.get(EXE_ENV_VAR) or "").strip()
    if exe_hint:
        found = _library_next_to(Path(exe_hint).expanduser())
        if found is not None:
            return JLinkLibrary(str(found), SOURCE_EXE_ENV)

    for name in JLINK_EXE_NAMES:
        exe = which(name)
        if exe:
            found = _library_next_to(Path(exe))
            if found is not None:
                return JLinkLibrary(str(found), SOURCE_PATH)

    return None


def resolve_jlink_library(env: Optional[Mapping[str, str]] = None, which: WhichFn = shutil.which) -> Optional[str]:
    """Path of the J-Link shared library, or None to let pylink search its default locations."""
    found = find_jlink_library(env, which)
    return found.path if found is not None else None


def describe_search(env: Optional[Mapping[str, str]] = None, which: WhichFn = shutil.which) -> List[str]:
    """Human-readable summary of each resolution step, for error messages and doctor."""
    env = os.environ if env is None else env
    lines = []
    explicit = (env.get(DLL_ENV_VAR) or "").strip()
    lines.append(f"{SOURCE_DLL_ENV}={explicit}" if explicit else f"{SOURCE_DLL_ENV} unset")
    exe_hint = (env.get(EXE_ENV_VAR) or "").strip()
    lines.append(f"{SOURCE_EXE_ENV}={exe_hint}" if exe_hint else f"{SOURCE_EXE_ENV} unset")
    exe = next((which(name) for name in JLINK_EXE_NAMES if which(name)), None)
    lines.append(f"{SOURCE_PATH}: {exe}" if exe else f"{SOURCE_PATH}: not found")
    lines.append(f"{SOURCE_PYLINK_DEFAULT}: ldconfig/LD_LIBRARY_PATH, /opt/SEGGER")
    return lines


def missing_library_hint(env: Optional[Mapping[str, str]] = None, which: WhichFn = shutil.which) -> str:
    return (
        f"Set {DLL_ENV_VAR} to libjlinkarm.so (or {EXE_ENV_VAR} to JLinkExe), or put JLinkExe on PATH. "
        "Checked: " + "; ".join(describe_search(env, which)) + "."
    )


def open_jlink(pylink_module=None, env: Optional[Mapping[str, str]] = None):
    """Construct `pylink.JLink` with the resolved library (or pylink's default when none resolved).

    Raises whatever pylink raises when no library loads (a TypeError for its
    default search); callers wrap that into their own error types.
    """
    if pylink_module is None:
        import pylink as pylink_module  # noqa: N813 -- local alias of the module

    found = find_jlink_library(env)
    if found is None:
        return pylink_module.JLink()
    from pylink.library import Library

    return pylink_module.JLink(lib=Library(found.path))
