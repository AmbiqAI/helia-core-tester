"""Environment and dependency resolution for FVP orchestration."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from helia_core_tester.core.discovery import find_repo_root, find_setup_dependencies_script
from helia_core_tester.core.path_layout import artifacts_root

from .errors import FvpScriptError


FVP_EXE_NAME = "FVP_Corstone_SSE-300_Ethos-U55"
FVP_DIR_X86 = "Linux64_GCC-9.3"
FVP_DIR_AARCH64 = "Linux64_armv8l_GCC-9.3"

REPO_ROOT = find_repo_root()
ARTIFACTS_DIR = artifacts_root(REPO_ROOT)
DEFAULT_DL = ARTIFACTS_DIR / "downloads"
DEFAULT_SOURCE = REPO_ROOT


def _which_in_env(name: str, env: dict) -> Optional[str]:
    return shutil.which(name, path=env.get("PATH"))


def resolve_gcov_tool(env: dict) -> Optional[str]:
    for tool in ("arm-none-eabi-gcov-tool", "gcov-tool"):
        resolved = _which_in_env(tool, env)
        if resolved:
            return resolved
    return None


def resolve_gcov_executable(env: dict) -> Optional[str]:
    for exe in ("arm-none-eabi-gcov", "gcov"):
        resolved = _which_in_env(exe, env)
        if resolved:
            return resolved
    return None


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def arch_tag() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    raise FvpScriptError(f"Unsupported architecture: {machine}")


def ensure_exe_on_path(name: str) -> Optional[str]:
    return shutil.which(name)


def call_setup_dependencies(downloads_dir: Path) -> None:
    setup = find_setup_dependencies_script(REPO_ROOT)
    if not setup or not setup.exists():
        print("No setup_dependencies.py found; skipping dependency setup.")
        return
    print("Ensuring dependencies via setup_dependencies.py")
    rc = subprocess.call(
        [sys.executable, str(setup), "--downloads-dir", str(downloads_dir)],
        cwd=str(REPO_ROOT),
    )
    if rc != 0:
        raise FvpScriptError(f"Dependency setup failed (rc={rc})")


def prepend_path(path: Path, env: dict) -> None:
    env["PATH"] = str(path) + os.pathsep + env.get("PATH", "")


def _fvp_model_dirs_for_arch(arch: str) -> list[str]:
    if arch == "x86_64":
        return [FVP_DIR_X86, FVP_DIR_AARCH64]
    if arch == "aarch64":
        return [FVP_DIR_AARCH64, FVP_DIR_X86]
    return [FVP_DIR_X86, FVP_DIR_AARCH64]


def _resolve_downloaded_fvp_executable(downloads_dir: Path, arch: str) -> tuple[Optional[Path], list[Path]]:
    models_root = downloads_dir / "corstone300_download" / "models"
    checked: list[Path] = []
    for model_dir in _fvp_model_dirs_for_arch(arch):
        candidate = models_root / model_dir / FVP_EXE_NAME
        checked.append(candidate)
        if candidate.exists():
            return candidate, checked
    return None, checked


def detect_paths(args) -> dict:
    env = os.environ.copy()
    arch = arch_tag()

    dl = args.downloads_dir.resolve()

    ethos = Path(args.ethos_path).resolve() if args.ethos_path else (dl / "ethos-u-core-platform")
    if not ethos.exists():
        raise FvpScriptError(f"Ethos-U core platform not found: {ethos}. Run without -e or point -u to a valid path.")

    cmsis5 = Path(args.cmsis5_path).resolve() if args.cmsis5_path else (dl / "CMSIS_5")
    if not cmsis5.exists():
        raise FvpScriptError(f"CMSIS_5 not found: {cmsis5}. Run without -e or point -C to a valid path.")

    if args.use_arm_compiler:
        toolchain_file = ethos / "cmake" / "toolchain" / "armclang.cmake"
        compiler_tag = "arm-compiler"
    else:
        toolchain_file = ethos / "cmake" / "toolchain" / "arm-none-eabi-gcc.cmake"
        compiler_tag = "gcc"
        if not args.no_gcc_from_download:
            gcc_bin = dl / "arm_gcc_download" / "bin"
            if not gcc_bin.exists():
                raise FvpScriptError(f"GCC toolchain not found at {gcc_bin}. Run without -e or install gcc on PATH.")
            prepend_path(gcc_bin, env)

    if not toolchain_file.exists():
        raise FvpScriptError(f"Toolchain file missing: {toolchain_file}")

    if not args.no_fvp_from_download:
        fvp_exe_candidate, checked_paths = _resolve_downloaded_fvp_executable(dl, arch)
        if fvp_exe_candidate is None:
            checked = ", ".join(str(path) for path in checked_paths)
            raise FvpScriptError(
                f"FVP not found in downloads (checked: {checked}). "
                "Run with -f to use a system FVP on PATH."
            )
        prepend_path(fvp_exe_candidate.parent, env)
        fvp_exe = fvp_exe_candidate
    else:
        from_path = ensure_exe_on_path(FVP_EXE_NAME)
        if not from_path:
            raise FvpScriptError(f"{FVP_EXE_NAME} not on PATH (use downloads or add it).")
        fvp_exe = Path(from_path)

    return {
        "env": env,
        "dl": dl,
        "ethos": ethos,
        "cmsis5": cmsis5,
        "toolchain_file": toolchain_file,
        "compiler_tag": compiler_tag,
        "fvp_exe": fvp_exe,
    }


def get_git_sha(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception:
        return None
