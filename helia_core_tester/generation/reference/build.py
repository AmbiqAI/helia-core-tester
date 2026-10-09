"""Build (once per environment) and locate the host reference library.

The library is every C source under helia_core_tester/reference/src compiled
with the host C compiler into artifacts/host_ref/<key>/libhct_ref.{so,dylib}.
The key folds in everything that can change the compiled code: the spec, the
generated header, every source and private header, the flags and the compiler
identity. A build runs under an exclusive lock and is published by an atomic
rename, so concurrent workers build it at most once and never load a
half-written library.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from helia_core_tester.generation.reference.abi import HEADER_PATH, REFERENCE_ROOT, SPEC_PATH, load_spec, render_header
from helia_core_tester.utils.file_lock import exclusive_lock
from helia_core_tester.utils.host_compiler import compiler_identity, find_host_cc

SRC_DIR = REFERENCE_ROOT / "src"
INCLUDE_DIR = REFERENCE_ROOT / "include"
COMMON_DIR = SRC_DIR / "common"

CACHE_ENV = "HCT_HOST_REF_CACHE"
LIBRARY_STEM = "libhct_ref"

# Strict IEEE float: no fast-math, no a*b+c contraction into FMA, no excess
# precision, so a golden does not depend on the host's FPU or optimizer.
CFLAGS: Sequence[str] = (
    "-std=c11",
    "-O1",
    "-fPIC",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-strict-aliasing",
    "-fvisibility=hidden",
    "-Wall",
    "-Wextra",
    "-Werror",
)


class ReferenceBuildError(RuntimeError):
    """The reference library could not be built or its inputs are inconsistent."""


@dataclass(frozen=True)
class ReferenceLibrary:
    path: Path
    key: str
    compiler: str
    compiler_identity: str


def library_filename() -> str:
    return LIBRARY_STEM + (".dylib" if sys.platform == "darwin" else ".so")


def c_sources() -> List[Path]:
    return sorted(SRC_DIR.rglob("*.c"))


def build_inputs() -> List[Path]:
    """Every file the compiled library depends on, in a stable order."""
    return [SPEC_PATH, HEADER_PATH, *sorted(SRC_DIR.rglob("*.h")), *c_sources()]


def check_header_fresh() -> None:
    """The generated header must match the spec, or C and ctypes disagree."""
    current = HEADER_PATH.read_text() if HEADER_PATH.is_file() else ""
    if current != render_header(load_spec()):
        raise ReferenceBuildError(
            f"{HEADER_PATH} does not match {SPEC_PATH}; run "
            "`python -m helia_core_tester.generation.reference.abi --write`"
        )


def library_key(compiler: str, flags: Sequence[str] = CFLAGS) -> str:
    """16-hex digest of everything the compiled library depends on."""
    digest = hashlib.sha256()
    for path in build_inputs():
        digest.update(path.relative_to(REFERENCE_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    digest.update("\0".join(flags).encode())
    digest.update(b"\0")
    digest.update(compiler_identity(compiler).encode())
    digest.update(sys.platform.encode())
    return digest.hexdigest()[:16]


def default_cache_root() -> Path:
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        return Path(override)
    from helia_core_tester.core.discovery import find_repo_root

    return find_repo_root() / "artifacts" / "host_ref"


def compile_command(compiler: str, output: Path, flags: Sequence[str] = CFLAGS, shared: bool = True) -> List[str]:
    cmd = [compiler, *flags]
    if shared:
        cmd.append("-shared")
    cmd += ["-I", str(INCLUDE_DIR), "-I", str(COMMON_DIR)]
    return cmd + [str(p) for p in c_sources()] + ["-lm", "-o", str(output)]


def ensure_reference_library(cache_root: Optional[Path] = None, compiler: Optional[str] = None) -> ReferenceLibrary:
    """Return the cached library for this environment, building it if absent."""
    check_header_fresh()
    if not c_sources():
        raise ReferenceBuildError(f"no C sources under {SRC_DIR}")
    compiler = compiler or find_host_cc()
    key = library_key(compiler)
    identity = compiler_identity(compiler)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    final_dir = root / key
    library = final_dir / library_filename()
    if library.is_file():
        return ReferenceLibrary(library, key, compiler, identity)
    with exclusive_lock(root / f".{key}.lock"):
        if library.is_file():
            return ReferenceLibrary(library, key, compiler, identity)
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
        try:
            staged = staging / library_filename()
            cmd = compile_command(compiler, staged)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ReferenceBuildError(f"reference library build could not run: {exc}") from exc
            (staging / "build.log").write_text(" ".join(cmd) + "\n\n" + proc.stdout + proc.stderr, encoding="utf-8")
            if proc.returncode != 0 or not staged.is_file():
                tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-30:])
                raise ReferenceBuildError(f"reference library build failed (exit {proc.returncode}):\n{tail}")
            (staging / "flags.json").write_text(
                json.dumps({"compiler": compiler, "compiler_identity": identity, "cflags": list(CFLAGS), "key": key},
                           indent=2) + "\n",
                encoding="utf-8",
            )
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(staging, final_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    if not library.is_file():
        raise ReferenceBuildError(f"reference library missing after build: {library}")
    return ReferenceLibrary(library, key, compiler, identity)


def describe_cache(cache_root: Optional[Path] = None) -> Dict[str, object]:
    """Best-effort report for `doctor`: never raises."""
    try:
        check_header_fresh()
        compiler = find_host_cc()
        key = library_key(compiler)
        root = Path(cache_root) if cache_root is not None else default_cache_root()
        library = root / key / library_filename()
        return {"key": key, "library": str(library), "built": library.is_file()}
    except Exception as exc:  # doctor reports every failure instead of raising
        return {"error": str(exc)}
