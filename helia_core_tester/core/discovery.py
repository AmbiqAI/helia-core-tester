"""
Repository root discovery utilities.

This module provides robust repository root discovery that works from any
directory within the repository, without relying on __file__ paths.
"""

import os
from pathlib import Path
from typing import Optional

from helia_core_tester.core.errors import RepoRootNotFoundError
from helia_core_tester.core.path_layout import build_dir as canonical_build_dir
from helia_core_tester.core.path_layout import compiler_tag_for_toolchain
from helia_core_tester.core.path_layout import generated_tests_dir as canonical_generated_tests_dir


# Marker files/directories that indicate the repository root. All three are
# required (see _is_repo_root): pyproject.toml + CMakeLists.txt alone are not
# enough to identify *this* repo -- the sibling ns-cmsis-nn kernel checkout
# has both of those too (it's a CMake/scikit-build-core Python project in its
# own right), so CMSIS_NN_REPO_ROOT pointed at a kernel checkout used to
# falsely validate as this tool's own root (issue #65). helia_core_tester/ is
# this tool's own package directory and is what actually disambiguates it.
REPO_MARKERS = [
    "pyproject.toml",     # Tests/helia-core-tester/pyproject.toml
    "CMakeLists.txt",     # Tests/helia-core-tester/CMakeLists.txt
    "helia_core_tester",  # Tests/helia-core-tester/helia_core_tester/ (this tool's own package)
]


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the repository root by walking up from start_path looking for markers.

    The repository root is this tool's own checkout (helia-core-tester,
    normally nested at <ns-cmsis-nn>/Tests/helia-core-tester), identified by
    REPO_MARKERS: pyproject.toml, CMakeLists.txt, and the helia_core_tester/
    package directory.

    Args:
        start_path: Starting directory (default: current working directory)
        
    Returns:
        Path to the repository root (helia-core-tester's own checkout)

    Raises:
        RepoRootNotFoundError: If repository root cannot be found
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()
    
    # Also check environment variable override
    env_root = os.environ.get("CMSIS_NN_REPO_ROOT")
    if env_root:
        env_path = Path(env_root).resolve()
        if _is_repo_root(env_path):
            return env_path
        missing = _missing_repo_markers(env_path)
        raise RepoRootNotFoundError(
            f"Environment variable CMSIS_NN_REPO_ROOT points to invalid location: {env_root}\n"
            f"  Missing: {', '.join(missing)}.\n"
            f"  Note: CMSIS_NN_REPO_ROOT must point at this tool's own checkout "
            f"(helia-core-tester), not an ns-cmsis-nn kernel checkout -- despite the name. "
            f"If you meant to point at a kernel checkout, use --cmsis-nn-root / the "
            f"CMSIS_NN_ROOT CMake variable instead; that is unrelated to this override."
        )
    
    # Walk up from start_path
    current = start_path.resolve()
    visited = set()
    
    while current != current.parent:  # Stop at filesystem root
        if current in visited:
            break
        visited.add(current)
        
        if _is_repo_root(current):
            return current
        
        current = current.parent
    
    # If we didn't find it, try from the file's location as fallback
    # (but this should rarely be needed)
    raise RepoRootNotFoundError(
        f"Could not find repository root starting from {start_path}. "
        f"Looking for all of: {', '.join(REPO_MARKERS)}.\n"
        f"  This normally auto-resolves when helia-core-tester is checked out "
        f"nested at <ns-cmsis-nn>/Tests/helia-core-tester (no environment "
        f"variable needed). If it's checked out standalone elsewhere, set "
        f"CMSIS_NN_REPO_ROOT to point at that checkout directly."
    )


def _missing_repo_markers(path: Path) -> list:
    """Return the REPO_MARKERS entries not present under path, for error messages."""
    if not path.is_dir():
        return ["(not a directory)"]
    return [marker for marker in REPO_MARKERS if not (path / marker).exists()]


def _is_repo_root(path: Path) -> bool:
    """
    Check if a path is the repository root (Tests/helia-core-tester/).

    Requires all of REPO_MARKERS to be present, notably the tool's own
    helia_core_tester/ package directory -- pyproject.toml + CMakeLists.txt
    alone are not sufficient to identify this repo, since the sibling
    ns-cmsis-nn kernel checkout has both of those too (see issue #65).

    Args:
        path: Path to check

    Returns:
        True if path appears to be the repository root
    """
    if not path.is_dir():
        return False

    return not _missing_repo_markers(path)


def _resolve_repo_root(repo_root: Optional[Path] = None) -> Path:
    """Return repo_root if given, otherwise find via find_repo_root()."""
    return repo_root if repo_root is not None else find_repo_root()


def find_repo_root_or_cwd() -> Path:
    """
    Find repository root, or return current working directory if not found.
    
    This is a fallback function for cases where we want to be lenient.
    
    Returns:
        Repository root if found, otherwise current working directory
    """
    try:
        return find_repo_root()
    except RepoRootNotFoundError:
        return Path.cwd()


def find_descriptors_dir(repo_root: Optional[Path] = None) -> Path:
    """
    Find the descriptors directory.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
    
    Returns:
        Path to assets/descriptors/ directory at repo root
        
    Raises:
        RepoRootNotFoundError: If repo root cannot be found
        PathNotFoundError: If descriptors directory doesn't exist
    """
    repo_root = _resolve_repo_root(repo_root)
    descriptors_dir = repo_root / "assets" / "descriptors"
    if not descriptors_dir.exists():
        from .errors import PathNotFoundError
        raise PathNotFoundError(f"Descriptors directory not found: {descriptors_dir}")
    
    return descriptors_dir


def find_generated_tests_dir(
    repo_root: Optional[Path] = None,
    cpu: str = "cortex-m55",
    suite: str = "int",
    create: bool = True,
) -> Path:
    """
    Find or create the generated tests directory for a CPU.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        create: Create directory if it doesn't exist
        
    Returns:
        Path to artifacts/generated_tests/<cpu>/ directory at repo root
        
    Raises:
        RepoRootNotFoundError: If repo root cannot be found
    """
    repo_root = _resolve_repo_root(repo_root)
    generated_tests_dir = canonical_generated_tests_dir(repo_root, cpu, suite=suite)
    
    if create and not generated_tests_dir.exists():
        generated_tests_dir.mkdir(parents=True, exist_ok=True)
    
    return generated_tests_dir


def find_tester_templates_dir(repo_root: Optional[Path] = None) -> Path:
    """
    Find the tester templates directory (Jinja2 templates for C/H generation).
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        
    Returns:
        Path to templates/ under helia_core_tester/generation/
    """
    repo_root = _resolve_repo_root(repo_root)
    return repo_root / "assets" / "templates"


def find_build_dir(cpu: str, repo_root: Optional[Path] = None, suite: str = "int", toolchain: str = "gcc") -> Path:
    """Return build directory for a CPU under artifacts."""
    repo_root = _resolve_repo_root(repo_root)
    return canonical_build_dir(repo_root, cpu, compiler_tag=compiler_tag_for_toolchain(toolchain), suite=suite)


def find_fvp_script_path(repo_root: Optional[Path] = None) -> Path:
    """
    Find the FVP build-and-run script path.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        
    Returns:
        Path to build_and_run_fvp.py under helia_core_tester/fvp/
    """
    return _resolve_repo_root(repo_root) / "helia_core_tester" / "fvp" / "build_and_run_fvp.py"


def find_setup_dependencies_script(repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find the setup_dependencies.py script if it exists.
    
    Args:
        repo_root: Repository root (auto-discovered if None)
        
    Returns:
        Path to setup_dependencies.py under helia_core_tester/scripts/, or None if not found
    """
    repo_root = _resolve_repo_root(repo_root)
    script = repo_root / "helia_core_tester" / "scripts" / "setup_dependencies.py"
    return script if script.exists() else None


def ensure_arm_toolchain_on_path(repo_root: Optional[Path] = None) -> None:
    """Prepend the downloaded ARM GCC toolchain's bin/ to this process's PATH, once,
    for the lifetime of the whole `helia_core_tester` invocation.

    Several call sites across both the FVP and perf-stream/hardware paths (memory
    reports, kernel-symbol-ref generation, RTT address discovery) shell out to bare
    `arm-none-eabi-{nm,size,objdump}` -- CMake's own compiler/linker/objcopy
    invocations use absolute paths from the toolchain file, but these don't. Calling
    this once at CLI startup (see cli.py's module-level call) covers all of them,
    instead of patching each subprocess call's env individually. Silently a no-op if
    the toolchain hasn't been downloaded yet (setup_dependencies.py not run) --
    whichever command actually needs it will fail with its own clear error.
    """
    try:
        repo_root = _resolve_repo_root(repo_root)
    except RepoRootNotFoundError:
        return
    toolchain_bin = repo_root / "artifacts" / "downloads" / "arm_gcc_download" / "bin"
    if not toolchain_bin.is_dir():
        return
    toolchain_bin_str = str(toolchain_bin.resolve())
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if toolchain_bin_str not in path_entries:
        os.environ["PATH"] = os.pathsep.join([toolchain_bin_str, *path_entries])
