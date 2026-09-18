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
    """The `JLinkExe` binary that flashes the recipe and resets the target.

    Reported with its version: `hardware flash` runs a commander script through
    it and `hardware run` resets through it before every RTT session, so a
    missing or mismatched commander is a hardware-run failure, not a nicety.
    """
    label = "JLinkExe (flash/reset)"
    try:
        found = find_jlink_exe()
    except JLinkLibraryError as exc:
        return HardwareCheck(label, False, str(exc))
    if found is None:
        return HardwareCheck(label, False, "not found: set $JLINK_PATH to JLinkExe (or its directory) or put JLinkExe on PATH")
    from .jlink_cli import version

    banner = version(exe=found.path)
    return HardwareCheck(label, True, f"{found.describe()}{f' -- {banner}' if banner else ''}")


def _flash_recipe_check(repo_root: Path) -> HardwareCheck:
    """Whether each board's build dir carries the NSX flash recipe `hardware flash` runs."""
    from .boards import load_board_table
    from .flash_recipe import describe_recipe

    label = "NSX flash recipe (hardware flash)"
    try:
        boards = load_board_table()
    except Exception as exc:
        return HardwareCheck(label, False, f"board table unreadable: {exc}")
    rows = []
    ok = True
    for board in boards:
        detail = describe_recipe(board.build_dir(repo_root), board)
        ok = ok and "missing" not in detail and "unreadable" not in detail
        rows.append(f"{board.id}: {detail}")
    return HardwareCheck(label, ok, "; ".join(rows))


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


def _provenance_check(repo_root: Path) -> HardwareCheck:
    """What each board's default build dir was actually built from, and whether it is qualified.

    Read out of the build dir rather than re-derived: the question is what the
    image sitting there would measure with, which a fresh render cannot answer.
    A board that has not been built yet is reported as such, not as a failure --
    doctor runs on hosts with no board attached.
    """
    from .boards import load_board_table
    from .provenance import describe_build

    label = "Build qualification (hardware run)"
    try:
        boards = load_board_table()
    except Exception as exc:
        return HardwareCheck(label, False, f"board table unreadable: {exc}")
    rows = [describe_build(board.build_dir(repo_root), board) for board in boards]
    # A development-overrides build is a legitimate state (it is what
    # --cmsis-nn-root is for), so it is reported, not failed; only an unreadable
    # record is a problem doctor should flag.
    ok = not any("unreadable" in row for row in rows)
    return HardwareCheck(label, ok, "; ".join(rows) if rows else "no boards in the table")


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
        _provenance_check(repo_root),
        _starter_profile_check(repo_root),
        _jlink_dll_check(),
        _jlink_exe_check(),
        _flash_recipe_check(repo_root),
        _board_table_check(),
    ]
