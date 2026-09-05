"""CMake configure/build helpers for FVP orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Set

from .env import REPO_ROOT
from .errors import FvpScriptError


def active_test_list(generated_tests_dir: Optional[Path]) -> Optional[Set[str]]:
    """Return the set of test-case names the last generation run for
    ``generated_tests_dir`` admitted, or ``None`` when there is no such list
    to constrain against.

    ``manifest.json`` is rewritten from scratch by every generation run
    (``generation/test_ops.py::test_generation``) to hold exactly the cases
    the active filter (``--op``, ``--name``, ``--float-precision``, ...)
    produced. Build and run steps key their working trees on cpu+suite only
    (see ``core/path_layout.py``), not on that filter, so a build/run against
    a narrower filter can still see binaries a previous, wider run left in the
    same tree -- CMake never deletes the output of a target it no longer
    defines. Reading the manifest back out here is how that active list is
    carried from generation into the build and run steps, which never see the
    filter directly (they run as separate subprocess invocations). ``None``
    preserves the historical "use whatever is on disk" behaviour when there is
    no manifest to consult (e.g. a tree never generated through this pipeline)
    -- but ``None`` is reserved strictly for that "no manifest file" case.

    A manifest that exists but admitted zero cases (every descriptor
    capability-skipped for this cpu, e.g. the f16 float suite on cortex-m0) is a
    real, empty active list, not "no constraint" -- it must return ``set()``,
    not ``None``, or build/run would fall back to unfiltered discovery and
    happily prune nothing / run every stale ``.elf`` in the tree.

    A manifest that exists but is unreadable, not valid JSON, or not shaped
    the way this pipeline always writes it -- a JSON object with a ``tests``
    list of objects, each carrying a non-empty ``name`` -- fails closed with
    ``FvpScriptError`` rather than falling back to ``None`` or quietly
    treating the gap as "no test here". A corrupt manifest is itself a sign
    something already went wrong (a crash mid-write, concurrent runs sharing
    a tree, schema drift, ...), and silently under-counting the active list at
    that moment -- pruning or refusing to run a case that was actually meant
    to be there -- would be worse than surfacing it loudly.
    See issue #66.
    """
    if generated_tests_dir is None:
        return None
    manifest_path = Path(generated_tests_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except OSError as exc:
        raise FvpScriptError(f"Could not read manifest {manifest_path}: {exc}") from exc
    except ValueError as exc:
        raise FvpScriptError(f"Manifest {manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FvpScriptError(
            f"Manifest {manifest_path} must be a JSON object, got {type(manifest).__name__}"
        )
    if "tests" not in manifest:
        raise FvpScriptError(f"Manifest {manifest_path} is missing required field 'tests'")
    tests = manifest["tests"]
    if not isinstance(tests, list):
        raise FvpScriptError(
            f"Manifest {manifest_path} field 'tests' must be a list, got {type(tests).__name__}"
        )
    names: Set[str] = set()
    for entry in tests:
        if not isinstance(entry, dict):
            raise FvpScriptError(
                f"Manifest {manifest_path} has a 'tests' entry that is not an object: {entry!r}"
            )
        name = entry.get("name")
        if not name:
            raise FvpScriptError(
                f"Manifest {manifest_path} has a 'tests' entry with no non-empty 'name': {entry!r}"
            )
        names.add(str(name))
    return names


def _prune_stale_test_elves(
    build_dir: Path, active_names: Optional[Set[str]], verbosity: int = 0
) -> List[Path]:
    """Delete ``build_dir/tests/**/*.elf`` files that fell out of the active
    test list before a (re)configure, so a filtered build in a tree a wider
    run already built does not leave excluded-case binaries behind for a
    later ``--no-build`` run step to discover and execute. No-op when
    ``active_names`` is ``None`` (no list to prune against). See issue #66.
    """
    tests_dir = build_dir / "tests"
    if active_names is None or not tests_dir.exists():
        return []
    removed: List[Path] = []
    for elf in tests_dir.rglob("*.elf"):
        if elf.is_file() and elf.stem not in active_names:
            elf.unlink()
            removed.append(elf)
    if removed and verbosity >= 1:
        print(f"Pruned {len(removed)} stale test ELF(s) outside the active test list under {tests_dir}")
    return removed


def _clear_stale_gcda(build_dir: Path) -> None:
    """Remove leftover .gcda files so a rebuild can never desync from stale
    coverage data. Recompiling a source file regenerates its .gcno with a
    new stamp; an old .gcda left next to it causes gcov to abort with
    'stamp mismatch with notes file'. Any real coverage is regenerated by
    the FVP run(s) that follow and re-merged via merge-stream, so it's safe
    to drop existing .gcda here.
    """
    if not build_dir.exists():
        return
    for gcda in build_dir.rglob("*.gcda"):
        gcda.unlink(missing_ok=True)


_COMPILER_LAUNCHER_ENV = "HELIA_CORE_TESTER_COMPILER_LAUNCHER"
_COMPILER_LAUNCHER_TOOLS = ("ccache", "sccache")
_COMPILER_LAUNCHER_OFF = ("", "none")


def resolve_compiler_launcher() -> Optional[str]:
    """The compiler cache to route this build through, or ``None``.

    Opt-in by presence only: nothing here installs a tool and no image ships
    one, so a host that already has ccache or sccache gets warm rebuilds and a
    host that has neither builds exactly as before. See issue #107.

    ``HELIA_CORE_TESTER_COMPILER_LAUNCHER`` names a specific tool, or is set to
    ``none`` (or empty) to disable caching outright even where ccache is on
    ``PATH`` -- a build that has to be reproducible wants the compiler invoked
    directly. A name that is not on ``PATH`` raises:
    silently building uncached under a launcher the caller explicitly asked for
    hides exactly the misconfiguration the override exists to express.
    """
    override = os.environ.get(_COMPILER_LAUNCHER_ENV)
    if override is not None:
        override = override.strip()
        if override.lower() in _COMPILER_LAUNCHER_OFF:
            return None
        resolved = shutil.which(override)
        if resolved is None:
            raise FvpScriptError(
                f"{_COMPILER_LAUNCHER_ENV}={override!r} is not on PATH. "
                f"Name a compiler launcher that is installed, or set it to "
                f"'none' to build without one."
            )
        return resolved
    for tool in _COMPILER_LAUNCHER_TOOLS:
        found = shutil.which(tool)
        if found:
            return found
    return None


def compiler_launcher_args(verbosity: int = 0) -> List[str]:
    """CMake defines routing compilation through an available compiler cache."""
    launcher = resolve_compiler_launcher()
    if not launcher:
        return []
    if verbosity >= 1:
        print(f"Compiler launcher: {launcher}")
    return [f"-DCMAKE_C_COMPILER_LAUNCHER={launcher}"]


def read_cmake_cache_path(build_dir: Path, key: str) -> Optional[Path]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    prefix = f"{key}:PATH="
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            if value:
                return Path(value).resolve()
    return None


def cmake_configure(
    source_dir: Path,
    build_dir: Path,
    toolchain_file: Path,
    cpu: str,
    cmsis5: Path,
    optimization: str,
    extra_defs: List[str],
    generator: Optional[str],
    generated_tests_dir: Optional[Path],
    enable_coverage: bool,
    verbosity: int,
    env: dict,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        cmake_cache.unlink()

    if enable_coverage:
        _clear_stale_gcda(build_dir)

    _prune_stale_test_elves(build_dir, active_test_list(generated_tests_dir), verbosity)

    cmd = [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DTARGET_CPU={cpu}",
        f"-DCMSIS_PATH={cmsis5}",
        f"-DCMSIS_OPTIMIZATION_LEVEL={optimization}",
    ] + compiler_launcher_args(verbosity) + [f"-D{item}" for item in extra_defs]
    if generated_tests_dir is not None:
        cmd.append(f"-DGENERATED_TESTS_DIR={generated_tests_dir}")
    if enable_coverage:
        cmd.append("-DENABLE_COVERAGE=ON")

    if generator:
        cmd += ["-G", generator]
    if verbosity >= 2:
        print(f"Configure: {' '.join(cmd)}")
    stdout = subprocess.DEVNULL if verbosity <= 1 else None
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env, stdout=stdout, stderr=None)
    if rc != 0:
        raise FvpScriptError(f"CMake configure failed for {cpu} (rc={rc})")


def cmake_build(build_dir: Path, verbosity: int, env: dict, jobs: Optional[int]) -> None:
    cmd = ["cmake", "--build", str(build_dir)]
    if jobs and jobs > 0:
        cmd += ["--", f"-j{jobs}"]
    if verbosity >= 2:
        print(f"Build: {' '.join(cmd)}")
    stdout = subprocess.DEVNULL if verbosity <= 1 else None
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env, stdout=stdout, stderr=None)
    if rc != 0:
        raise FvpScriptError(f"CMake build failed (rc={rc})")


def find_elves(build_dir: Path, active_names: Optional[Set[str]] = None) -> List[Path]:
    """Discover built test ELFs under ``build_dir``.

    ``active_names``, when given, restricts discovery to the current run's
    active test list (see ``active_test_list``) so a stale binary left behind
    by a wider previous run -- e.g. a ``--no-build`` run step that never gets
    a chance to prune -- is never picked up and executed, even though it is
    still on disk. Purely a discovery-time filter: it never deletes anything.
    """
    tests_dir = build_dir / "tests"
    root = tests_dir if tests_dir.exists() else build_dir
    elves = [path for path in root.rglob("*.elf") if path.is_file()]
    if active_names is not None:
        elves = [path for path in elves if path.stem in active_names]
    return elves
