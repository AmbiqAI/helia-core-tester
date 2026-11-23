"""A kernel library's host reference build: its plain-C configuration for the host CPU.

The library builds it with its own CMake (for ns-cmsis-nn, ``NS_CMSIS_NN_HOST_REFERENCE=ON``)
and describes the result in a manifest (schema ``hct/host-reference/1``): the archive, the
include directories and definitions a caller compiles against, the libraries to link, the
compiler and how it spells ``float16_t``. hct reads the manifest instead of guessing flags.

Builds are cached under ``<cache_root>/<key>/``, keyed by the library's build inputs, the
recipe and the compiler identity, built under a lock and marked complete by a stamp written
last. A CMake build directory cannot be moved after configure (the manifest holds absolute
paths), so an incomplete directory is wiped and rebuilt rather than renamed into place.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from helia_core_tester.utils.file_lock import exclusive_lock
from helia_core_tester.utils.host_compiler import compiler_identity, find_host_cc

SCHEMA = "hct/host-reference/1"
FLOAT16_TYPES = ("__fp16", "_Float16")
# Bump when what goes into a cached build changes without the key's other inputs changing.
_CACHE_SCHEMA = 1
_STAMP = "complete.json"
_LOG_TAIL_LINES = 40
_CONFIGURE_TIMEOUT_S = 600
_BUILD_TIMEOUT_S = 1800


class HostReferenceError(RuntimeError):
    """The host reference could not be built, or its manifest is unusable."""


@dataclass(frozen=True)
class HostReferenceRecipe:
    """How a library turns on its host reference build and names its manifest."""

    library: str
    option: str
    manifest: str
    # Paths, relative to the library root, whose content decides what the build produces.
    build_inputs: Tuple[str, ...]


NS_CMSIS_NN = HostReferenceRecipe(
    library="ns-cmsis-nn",
    option="NS_CMSIS_NN_HOST_REFERENCE",
    manifest="ns_cmsis_nn_host_reference.json",
    build_inputs=("CMakeLists.txt", "cmake", "Include", "Source"),
)


@dataclass(frozen=True)
class HostReference:
    manifest_path: Path
    library: str
    source_dir: Path
    archive: Path
    include_dirs: Tuple[Path, ...]
    compile_definitions: Tuple[str, ...]
    library_compile_options: Tuple[str, ...]
    link_libraries: Tuple[str, ...]
    float16_type: str
    compiler: Path
    compiler_id: str
    compiler_version: str
    key: str = ""

    def compile_flags(self) -> List[str]:
        """What a translation unit compiled against the library needs."""
        flags = [f"-D{d}" for d in self.compile_definitions]
        for directory in self.include_dirs:
            flags += ["-I", str(directory)]
        return flags

    def link_inputs(self) -> List[str]:
        return [str(self.archive), *[f"-l{name}" for name in self.link_libraries]]

    def to_json(self) -> Dict[str, object]:
        return {
            "library": self.library,
            "manifest": str(self.manifest_path),
            "archive": str(self.archive),
            "float16_type": self.float16_type,
            "compiler": f"{self.compiler_id} {self.compiler_version}",
            "key": self.key,
        }


def supports(tree: Path, recipe: HostReferenceRecipe = NS_CMSIS_NN) -> bool:
    """True when the checkout's top-level CMakeLists.txt offers the recipe's option."""
    cmakelists = Path(tree) / "CMakeLists.txt"
    try:
        return recipe.option in cmakelists.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _require(cond: bool, path: Path, what: str) -> None:
    if not cond:
        raise HostReferenceError(f"host reference manifest {path}: {what}")


def _string_list(data: dict, field: str, path: Path, allow_empty: bool = False) -> Tuple[str, ...]:
    value = data.get(field)
    if not isinstance(value, list):
        raise HostReferenceError(f"host reference manifest {path}: '{field}' must be a list")
    _require(all(isinstance(v, str) and v for v in value), path, f"'{field}' must hold non-empty strings")
    _require(allow_empty or bool(value), path, f"'{field}' is empty")
    _require(len(set(value)) == len(value), path, f"'{field}' has duplicates")
    return tuple(value)


def load_manifest(path: Path, library: Optional[str] = None) -> HostReference:
    """Read and validate a manifest; every problem is a HostReferenceError."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HostReferenceError(f"host reference manifest {path} does not exist") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostReferenceError(f"host reference manifest {path} is unreadable: {exc}") from exc
    _require(isinstance(data, dict), path, "top level must be an object")
    _require(data.get("schema") == SCHEMA, path, f"schema is {data.get('schema')!r}, expected {SCHEMA!r}")
    name = data.get("library")
    _require(isinstance(name, str) and bool(name), path, "'library' must be a non-empty string")
    _require(library is None or name == library, path, f"library is {name!r}, expected {library!r}")
    for field in ("source_dir", "archive", "float16_type"):
        _require(isinstance(data.get(field), str) and bool(data[field]), path, f"'{field}' must be a non-empty string")
    _require(data["float16_type"] in FLOAT16_TYPES, path, f"'float16_type' must be one of {FLOAT16_TYPES}")
    compiler = data.get("c_compiler")
    _require(isinstance(compiler, dict), path, "'c_compiler' must be an object")
    for field in ("path", "id", "version"):
        _require(isinstance(compiler.get(field), str) and bool(compiler[field]), path, f"'c_compiler.{field}' must be a non-empty string")

    archive = Path(data["archive"])
    _require(archive.is_absolute(), path, "'archive' must be an absolute path")
    _require(archive.is_file(), path, f"archive {archive} does not exist (is the library built?)")
    include_dirs = tuple(Path(d) for d in _string_list(data, "include_dirs", path))
    for directory in include_dirs:
        _require(directory.is_absolute() and directory.is_dir(), path, f"include dir {directory} is not an existing absolute directory")
    definitions = _string_list(data, "compile_definitions", path, allow_empty=True)
    _require(all(not d.startswith("-") for d in definitions), path, "'compile_definitions' must not carry flags")
    link_libraries = _string_list(data, "link_libraries", path, allow_empty=True)
    _require(all(not l.startswith("-") and "/" not in l for l in link_libraries), path, "'link_libraries' must be bare library names")
    return HostReference(
        manifest_path=path,
        library=name,
        source_dir=Path(data["source_dir"]),
        archive=archive,
        include_dirs=include_dirs,
        compile_definitions=definitions,
        library_compile_options=_string_list(data, "library_compile_options", path, allow_empty=True),
        link_libraries=link_libraries,
        float16_type=data["float16_type"],
        compiler=Path(compiler["path"]),
        compiler_id=compiler["id"],
        compiler_version=compiler["version"],
    )


def _input_digest(tree: Path, inputs: Sequence[str]) -> str:
    from helia_core_tester.generation.reuse import _subtree_digest

    return _subtree_digest(tree, inputs)


def _cmake() -> str:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise HostReferenceError("cmake is not on PATH; the host reference is built with the library's CMake")
    return cmake


def _tool_version(tool: str) -> str:
    try:
        proc = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HostReferenceError(f"`{tool} --version` failed: {exc}") from exc
    lines = (proc.stdout or "").strip().splitlines()
    return lines[0] if lines else ""


def host_reference_key(tree: Path, recipe: HostReferenceRecipe, cc: str, cmake: str) -> str:
    payload = {
        "schema": _CACHE_SCHEMA,
        "recipe": [recipe.library, recipe.option, recipe.manifest, list(recipe.build_inputs)],
        "inputs": _input_digest(tree, recipe.build_inputs),
        "compiler": compiler_identity(cc),
        "cmake": _tool_version(cmake),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def default_cache_root() -> Path:
    from helia_core_tester.core.discovery import find_repo_root

    return find_repo_root() / "artifacts" / "host_reference"


def _tail(text: str) -> str:
    return "\n".join((text or "").strip().splitlines()[-_LOG_TAIL_LINES:])


def _run(cmd: List[str], what: str, timeout: int) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise HostReferenceError(f"host reference {what} timed out after {timeout}s: {' '.join(cmd)}") from exc
    except OSError as exc:
        raise HostReferenceError(f"host reference {what} could not start: {exc}") from exc
    if proc.returncode != 0:
        raise HostReferenceError(
            f"host reference {what} failed (exit {proc.returncode}): {' '.join(cmd)}\n{_tail(proc.stdout + proc.stderr)}"
        )


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or os.path.samefile(a, b)
    except OSError:
        return False


def _validated(manifest: Path, tree: Path, recipe: HostReferenceRecipe, cc: str, key: str) -> HostReference:
    ref = load_manifest(manifest, library=recipe.library)
    if not _same_file(ref.source_dir, tree):
        raise HostReferenceError(f"host reference manifest {manifest} describes {ref.source_dir}, not {tree}")
    if not _same_file(ref.compiler, Path(cc)):
        raise HostReferenceError(f"host reference manifest {manifest} was built by {ref.compiler}, not {cc}")
    return HostReference(**{**ref.__dict__, "key": key})


def ensure_host_reference(
    tree: Path,
    recipe: HostReferenceRecipe = NS_CMSIS_NN,
    cache_root: Optional[Path] = None,
    cc: Optional[str] = None,
    jobs: Optional[int] = None,
) -> HostReference:
    """Build (or reuse) the library's host reference and return its validated manifest."""
    tree = Path(tree).resolve()
    if not tree.is_dir():
        raise HostReferenceError(f"{tree} is not a directory")
    if not supports(tree, recipe):
        raise HostReferenceError(
            f"{tree} has no host reference build: its CMakeLists.txt does not offer {recipe.option}"
        )
    if jobs is not None and jobs < 1:
        raise HostReferenceError(f"jobs must be >= 1, got {jobs}")
    cmake = _cmake()
    cc = cc or find_host_cc()
    key = host_reference_key(tree, recipe, cc, cmake)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    final = root / key
    build_dir = final / "build"
    manifest = build_dir / recipe.manifest
    stamp = final / _STAMP

    if stamp.is_file():
        return _validated(manifest, tree, recipe, cc, key)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / f".{key}.lock"):
        if stamp.is_file():
            return _validated(manifest, tree, recipe, cc, key)
        if final.exists():
            shutil.rmtree(final)
        build_dir.mkdir(parents=True)
        try:
            _run(
                [cmake, "-S", str(tree), "-B", str(build_dir), f"-D{recipe.option}=ON", f"-DCMAKE_C_COMPILER={cc}"],
                "configure",
                _CONFIGURE_TIMEOUT_S,
            )
            if not manifest.is_file():
                raise HostReferenceError(
                    f"{recipe.library} configured with {recipe.option}=ON but wrote no {recipe.manifest}"
                )
            _run(
                [cmake, "--build", str(build_dir), "--parallel", str(jobs or os.cpu_count() or 4)],
                "build",
                _BUILD_TIMEOUT_S,
            )
            ref = _validated(manifest, tree, recipe, cc, key)
            stamp.write_text(json.dumps({"key": key, **ref.to_json()}, indent=2) + "\n", encoding="utf-8")
        except BaseException:
            shutil.rmtree(final, ignore_errors=True)
            raise
    return ref
