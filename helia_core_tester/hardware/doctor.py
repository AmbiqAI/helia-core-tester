"""Hardware-side checks for `helia_core_tester doctor`.

None of these are required for the FVP path, so a missing tool is reported as
missing rather than failing the whole doctor run.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .firmware_build import DOWNLOADS_DIR
from .jlink_library import (
    SOURCE_PYLINK_DEFAULT,
    JLinkLibraryError,
    find_jlink_exe,
    find_jlink_library,
    missing_library_hint,
)


@dataclass(frozen=True)
class HardwareCheck:
    label: str
    ok: bool
    detail: str


def _tool_check(tool: str, label: str) -> HardwareCheck:
    path = shutil.which(tool)
    return HardwareCheck(label, path is not None, path or "not found on PATH")


def _jlink_dll_check() -> HardwareCheck:
    label = "J-Link library (pylink)"
    try:
        found = find_jlink_library()
    except JLinkLibraryError as exc:
        return HardwareCheck(label, False, str(exc))
    try:
        from pylink.library import Library

        library = Library(found.path) if found is not None else Library()
        dll = library.dll()
    except Exception as exc:  # pylink import/DLL load failure
        where = f" from {found.describe()}" if found is not None else ""
        return HardwareCheck(label, False, f"not loadable{where}: {exc}")
    if dll is None:
        return HardwareCheck(label, False, f"SEGGER J-Link DLL not found. {missing_library_hint()}")
    if found is not None:
        return HardwareCheck(label, True, found.describe())
    path = getattr(library, "_path", None)
    return HardwareCheck(label, True, f"{path or 'loaded'} (via {SOURCE_PYLINK_DEFAULT})")


def _jlink_exe_check() -> HardwareCheck:
    """The `JLinkExe` binary the CMake flash target runs (`hardware flash` / `hardware run`)."""
    label = "JLinkExe (flash target)"
    try:
        found = find_jlink_exe()
    except JLinkLibraryError as exc:
        return HardwareCheck(label, False, str(exc))
    if found is None:
        return HardwareCheck(label, False, "not found: set $JLINK_PATH to JLinkExe (or its directory) or put JLinkExe on PATH")
    return HardwareCheck(label, True, found.describe())


def _git_head(path: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _checkout_check(label: str, path: Path) -> HardwareCheck:
    if not path.is_dir():
        return HardwareCheck(label, False, f"{path} missing (fetched lazily by `hardware build`)")
    head = _git_head(path)
    return HardwareCheck(label, True, f"{path} @ {head or 'unknown HEAD'}")


def _cmsis_nn_check(repo_root: Path) -> HardwareCheck:
    """The ns-cmsis-nn checkout `hardware build` compiles and `generate` reads."""
    from .dependency_sources import CmsisNnSourceError, resolve_cmsis_nn

    label = "ns-cmsis-nn checkout"
    try:
        resolved = resolve_cmsis_nn(repo_root)
    except CmsisNnSourceError as exc:
        return HardwareCheck(label, False, str(exc))
    head = _git_head(resolved.root)
    return HardwareCheck(label, True, f"{resolved.root} @ {head or 'unknown HEAD'} ({resolved.selector})")


def _board_table_check() -> HardwareCheck:
    try:
        from .boards import load_board_table

        table = load_board_table()
    except Exception as exc:
        return HardwareCheck("Board table (assets/hardware_boards.yaml)", False, f"unreadable: {exc}")
    return HardwareCheck("Board table (assets/hardware_boards.yaml)", True, f"{len(table)} board(s): {', '.join(b.id for b in table)}")


def hardware_checks(repo_root: Path) -> list[HardwareCheck]:
    from ..scripts.setup_dependencies import nsx_ambiq_sdk_dir

    downloads = repo_root / DOWNLOADS_DIR
    return [
        _tool_check("arm-none-eabi-gcc", "arm-none-eabi-gcc (firmware cross-compiler)"),
        _tool_check("cmake", "cmake (firmware build)"),
        _jlink_dll_check(),
        _jlink_exe_check(),
        _checkout_check("nsx-ambiq-sdk checkout", nsx_ambiq_sdk_dir(repo_root, downloads)),
        _checkout_check("neuralspotx checkout", downloads / "neuralspotx"),
        _cmsis_nn_check(repo_root),
        _board_table_check(),
    ]
