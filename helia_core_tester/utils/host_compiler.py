"""Host C/C++ compiler discovery shared by the reference library, the host check
and the mutation harness.

Resolution order: the HCT_HOST_CC / HCT_HOST_CXX environment variable, then the
usual driver names on PATH. A missing compiler raises HostCompilerMissing naming
the variable and what to install, never a bare FileNotFoundError mid-build.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import Dict, Optional, Sequence

CC_ENV = "HCT_HOST_CC"
CXX_ENV = "HCT_HOST_CXX"

_CC_CANDIDATES = ("cc", "gcc", "clang")
_CXX_CANDIDATES = ("c++", "g++", "clang++")

_INSTALL_HINT = (
    "install a host toolchain (Debian/Ubuntu: `apt install build-essential`; "
    "macOS: `xcode-select --install`)"
)


class HostCompilerMissing(RuntimeError):
    """No usable host compiler was found."""


def _resolve(env_var: str, candidates: Sequence[str], kind: str, hint: str) -> str:
    override = os.environ.get(env_var, "").strip()
    if override:
        resolved = shutil.which(override)
        if resolved is None:
            raise HostCompilerMissing(f"{env_var}={override!r} is not an executable on PATH; {hint}")
        return resolved
    for name in candidates:
        resolved = shutil.which(name)
        if resolved is not None:
            return resolved
    raise HostCompilerMissing(
        f"No host {kind} found (tried {', '.join(candidates)}); set {env_var} or {hint}"
    )


def find_host_cc() -> str:
    """Absolute path of the host C compiler."""
    return _resolve(CC_ENV, _CC_CANDIDATES, "C compiler", _INSTALL_HINT)


def find_host_cxx() -> str:
    """Absolute path of the host C++ compiler."""
    return _resolve(CXX_ENV, _CXX_CANDIDATES, "C++ compiler", _INSTALL_HINT)


@lru_cache(maxsize=None)
def compiler_identity(compiler: str) -> str:
    """First line of `<compiler> --version`, which names the vendor and version.

    Folded into cache keys: two compilers can emit different code for the same
    source, so a library built by one must not be reused under the other.
    """
    try:
        proc = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HostCompilerMissing(f"cannot run {compiler} --version: {exc}") from exc
    if proc.returncode != 0:
        raise HostCompilerMissing(f"{compiler} --version exited {proc.returncode}: {proc.stderr.strip()}")
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    if not lines:
        raise HostCompilerMissing(f"{compiler} --version printed nothing")
    return lines[0].strip()


def describe_host_toolchain() -> Dict[str, Optional[str]]:
    """Best-effort report for `doctor`: never raises."""
    report: Dict[str, Optional[str]] = {}
    for key, finder in (("cc", find_host_cc), ("cxx", find_host_cxx)):
        try:
            path = finder()
            report[key] = f"{path} ({compiler_identity(path)})"
        except HostCompilerMissing as exc:
            report[key] = None
            report[f"{key}_error"] = str(exc)
    return report
