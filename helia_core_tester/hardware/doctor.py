"""Hardware-side checks for `helia_core_tester doctor`.

None of these are required for the FVP path, so a missing tool is reported as
missing rather than failing the whole doctor run.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def _toolchain_check(repo_root: Path) -> HardwareCheck:
    """The cross compiler NSX's toolchain file will find.

    That file uses `find_program(... REQUIRED)`, so what matters is what is on
    PATH once the checkout's downloaded toolchain has been prepended -- not what
    is on the login PATH. Report the same answer the build will get.
    """
    from .toolchain import arm_tool, toolchain_bin_dir

    label = "arm-none-eabi-gcc (NSX cross compiler)"
    resolved = arm_tool("arm-none-eabi-gcc", repo_root)
    if Path(resolved).is_file():
        return HardwareCheck(label, True, resolved)
    bin_dir = toolchain_bin_dir(repo_root)
    found = shutil.which("arm-none-eabi-gcc")
    if found:
        return HardwareCheck(label, True, f"{found} (PATH; {bin_dir} not downloaded yet)")
    return HardwareCheck(
        label, False, f"not found: neither {bin_dir} nor PATH has one (fetched by `hardware build`)"
    )


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


def _neuralspotx_check() -> HardwareCheck:
    """The pinned NSX build system: everything the firmware resolves comes from it."""
    label = "neuralspotx (NSX build system)"
    try:
        from importlib.metadata import version

        return HardwareCheck(label, True, f"{version('neuralspotx')} (packaged registry)")
    except Exception as exc:
        return HardwareCheck(label, False, f"not importable: {exc}")


def _baseline_check(repo_root: Path) -> tuple[HardwareCheck, Optional[object]]:
    """The dependency baseline in force, and its identity."""
    from .dependency_baseline import BaselineError, resolve_baseline

    label = "Dependency baseline"
    try:
        baseline = resolve_baseline(repo_root)
    except BaselineError as exc:
        return HardwareCheck(label, False, str(exc)), None
    return (
        HardwareCheck(label, True, f"{baseline.describe()} fingerprint {baseline.fingerprint[:16]}"),
        baseline,
    )


def _kernel_source_check(baseline) -> HardwareCheck:
    """Where the kernels under test come from: the pin, or a local override."""
    from .dependency_baseline import CMSIS_NN_PROJECT
    from .nsx_app import CMSIS_NN_MODULE

    label = f"Kernel source ({CMSIS_NN_MODULE})"
    if baseline is None:
        return HardwareCheck(label, False, "unknown: the dependency baseline did not load")
    try:
        project = baseline.project(CMSIS_NN_PROJECT)
    except Exception as exc:
        return HardwareCheck(label, False, str(exc))
    return HardwareCheck(label, True, f"{CMSIS_NN_PROJECT}@{project.ref} ({project.url})")


def _starter_profile_check(repo_root: Path) -> HardwareCheck:
    """Whether the pinned NSX registry actually knows every board in the table."""
    label = "NSX starter profiles"
    try:
        from .boards import load_board_table
        from .nsx_app import resolve_modules, KernelSource

        rows = []
        for board in load_board_table():
            modules = resolve_modules(board.nsx_board, KernelSource(ref="unused"))
            rows.append(f"{board.nsx_board}: {len(modules)} module(s)")
    except Exception as exc:
        return HardwareCheck(label, False, f"{type(exc).__name__}: {exc}")
    return HardwareCheck(label, True, "; ".join(rows))


def _board_table_check() -> HardwareCheck:
    try:
        from .boards import load_board_table

        table = load_board_table()
    except Exception as exc:
        return HardwareCheck("Board table (assets/hardware_boards.yaml)", False, f"unreadable: {exc}")
    return HardwareCheck("Board table (assets/hardware_boards.yaml)", True, f"{len(table)} board(s): {', '.join(b.id for b in table)}")


def hardware_checks(repo_root: Path) -> list[HardwareCheck]:
    baseline_check, baseline = _baseline_check(repo_root)
    return [
        _toolchain_check(repo_root),
        _tool_check("cmake", "cmake (NSX configure/build)"),
        _tool_check("ninja", "ninja (NSX build)"),
        _neuralspotx_check(),
        baseline_check,
        _kernel_source_check(baseline),
        _starter_profile_check(repo_root),
        _jlink_dll_check(),
        _jlink_exe_check(),
        _board_table_check(),
    ]
