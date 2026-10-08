"""Where the ARM GCC binutils come from for host-side ELF inspection.

The hardware commands fetch ARM GCC lazily into artifacts/downloads/ (see
`firmware_build.ensure_build_tools`), which puts it on PATH for the NSX
toolchain file. The host also runs `arm-none-eabi-nm` (RTT block
address, memory report) and `-size`/`-objdump` (memory report) itself. Those
go through `arm_tool()` so a fresh clone that only just downloaded the
toolchain resolves them without the user editing PATH, and
`add_toolchain_to_path()` makes the same bin/ visible to anything that still
shells out to a bare tool name (the build's generate_kernel_symbol_refs.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

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


def run_tool(tool: str, args: Iterable[str], repo_root: Optional[Path] = None) -> str:
    """Stdout of an ARM binutil; raises on failure."""
    return subprocess.run([arm_tool(tool, repo_root), *args], capture_output=True, text=True, check=True).stdout


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


GCC_NAME = "arm-none-eabi-gcc"


@dataclass(frozen=True)
class ToolchainSpec:
    """One firmware compiler and its names."""

    key: str
    # nsx.yml and provenance name.
    name: str
    # Compiler basename CMake configures.
    compiler: str
    # Default build dir suffix.
    dir_suffix: str

    def build_dir(self, base: Path) -> Path:
        """Default build dir for this toolchain."""
        return base.with_name(base.name + self.dir_suffix)

    def record(self, compiler: Optional[str]) -> Optional[dict[str, str]]:
        """Provenance for a configured compiler."""
        version = compiler_version(compiler)
        return {"name": self.name, "version": version} if version else None

    def require(self) -> None:
        """Fail fast without ATfE clang."""
        clang = atfe_clang()
        if self.key == "atfe" and not (clang and clang.is_file() and os.access(clang, os.X_OK)):
            raise FileNotFoundError(f"ATFE_ROOT has no bin/clang: {os.environ.get('ATFE_ROOT') or 'unset'}")

    def matches(self, compiler: Optional[str]) -> bool:
        """The configured compiler is this toolchain's."""
        if compiler is None:
            return True
        clang = atfe_clang()
        # A new ATFE_ROOT means a new clang.
        if self.key == "atfe" and clang:
            return Path(compiler).resolve() == clang.resolve()
        return Path(compiler).stem == self.compiler


def atfe_clang() -> Optional[Path]:
    """$ATFE_ROOT/bin/clang, or None."""
    root = os.environ.get("ATFE_ROOT")
    return Path(root) / "bin" / "clang" if root else None


DEFAULT_TOOLCHAIN = "gcc"
TOOLCHAINS = {
    "gcc": ToolchainSpec("gcc", GCC_NAME, GCC_NAME, ""),
    "atfe": ToolchainSpec("atfe", "atfe", "clang", "-atfe"),
}


def toolchain_spec(key: Optional[str] = None) -> ToolchainSpec:
    """Spec for key; None means gcc."""
    try:
        return TOOLCHAINS[key or DEFAULT_TOOLCHAIN]
    except KeyError:
        raise ValueError(f"toolchain must be one of: {', '.join(TOOLCHAINS)}") from None


def compiler_version(compiler: Optional[str]) -> Optional[str]:
    """A compiler's -dumpversion, or None."""
    if compiler is None:
        return None
    try:
        done = subprocess.run([compiler, "-dumpversion"], capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None
