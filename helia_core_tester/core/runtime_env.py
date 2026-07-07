"""Runtime environment bootstrap helpers for pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from helia_core_tester.fvp.env import call_setup_dependencies, detect_paths


@dataclass(frozen=True)
class RuntimeEnvContext:
    """Resolved run-scoped environment used by build/run child invocations."""

    downloads_dir: Path
    ethos_path: Path
    cmsis5_path: Path
    toolchain_file: Path
    compiler_tag: str
    fvp_exe: Path
    child_env: dict[str, str]


def bootstrap_runtime_env(
    *,
    downloads_dir: Path,
    ensure_setup: bool,
) -> RuntimeEnvContext:
    """Resolve dependency/toolchain/FVP paths once and return a locked context."""
    resolved_downloads = Path(downloads_dir).resolve()
    if ensure_setup:
        call_setup_dependencies(resolved_downloads)

    args = SimpleNamespace(
        downloads_dir=resolved_downloads,
        ethos_path=None,
        cmsis5_path=None,
        use_arm_compiler=False,
        no_gcc_from_download=False,
        no_fvp_from_download=False,
    )

    ctx = detect_paths(args)
    return RuntimeEnvContext(
        downloads_dir=ctx["dl"],
        ethos_path=ctx["ethos"],
        cmsis5_path=ctx["cmsis5"],
        toolchain_file=ctx["toolchain_file"],
        compiler_tag=ctx["compiler_tag"],
        fvp_exe=ctx["fvp_exe"],
        child_env=dict(ctx["env"]),
    )


def build_locked_fvp_flags(runtime_env: Optional[RuntimeEnvContext], fallback_downloads_dir: Path) -> list[str]:
    """Return FVP flags that force deterministic dependency/path selection."""
    downloads_dir = Path(fallback_downloads_dir).resolve()
    ethos_path = downloads_dir / "ethos-u-core-platform"
    cmsis5_path = downloads_dir / "CMSIS_5"

    if runtime_env is not None:
        downloads_dir = runtime_env.downloads_dir
        ethos_path = runtime_env.ethos_path
        cmsis5_path = runtime_env.cmsis5_path

    return [
        "--no-setup",
        "--downloads-dir",
        str(downloads_dir),
        "--ethos-path",
        str(ethos_path),
        "--cmsis5-path",
        str(cmsis5_path),
        # Use the locked PATH from parent process instead of mutating PATH in child.
        "--no-gcc-from-download",
        "--no-fvp-from-download",
    ]
