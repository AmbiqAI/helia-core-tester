"""Build (once per environment) and locate the host reference-kernel library.

The library is the vendored TFLM reference kernels behind the hct_ref C shim
(helia_core_tester/reference_kernels), compiled with the host C++ compiler into
artifacts/host_ref/<key>/libhct_ref.{so,dylib}. The key folds in every input
that can change the compiled code: the vendored manifest (itself verified
against the files on disk), the shim and stub sources, the flags and the
compiler identity. A build runs under an exclusive lock and is published by an
atomic rename, so concurrent workers build it at most once and never load a
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
from typing import Dict, List, Optional

from helia_core_tester.utils.file_lock import exclusive_lock
from helia_core_tester.utils.host_compiler import compiler_identity, find_host_cxx

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "reference_kernels"
SHIM_DIR = REFERENCE_ROOT / "shim"
STUBS_DIR = SHIM_DIR / "stubs"
THIRD_PARTY = REFERENCE_ROOT / "third_party"
MANIFEST = THIRD_PARTY / "manifest.json"

CACHE_ENV = "HCT_HOST_REF_CACHE"

CXXFLAGS = (
    "-std=c++17",
    "-O2",
    "-fPIC",
    "-shared",
    "-fno-strict-aliasing",
    # Keep a*b+c as two rounded operations: the float entries must not change
    # with the host's FMA support.
    "-ffp-contract=off",
    # Export only the hct_ref_* entries (hct_ref.h): the vendored tflite:: code must not
    # bind to, or be bound by, another TFLite in the process.
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-DTF_LITE_STATIC_MEMORY",
    "-DTF_LITE_STRIP_ERROR_STRINGS",
)

LIBRARY_STEM = "libhct_ref"


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> Dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceBuildError(f"cannot read vendored manifest {MANIFEST}: {exc}") from exc


def verify_vendored_tree(manifest: Optional[Dict] = None) -> Dict:
    """Check every manifested file's hash and that nothing unmanifested is present.

    Returns the manifest. Raises ReferenceBuildError on the first mismatch: an
    edited vendored file would silently change every golden.
    """
    manifest = manifest if manifest is not None else load_manifest()
    files: Dict[str, str] = manifest.get("files", {})
    if not files:
        raise ReferenceBuildError(f"{MANIFEST} lists no files")
    for rel, digest in files.items():
        path = REFERENCE_ROOT / rel
        if not path.is_file():
            raise ReferenceBuildError(f"vendored file missing: {rel}")
        if _sha256(path) != digest:
            raise ReferenceBuildError(f"vendored file modified: {rel} (rerun scripts/vendor_tflm_reference.py)")
    allowed = set(files) | {"third_party/manifest.json", "third_party/VENDOR.md"}
    for path in THIRD_PARTY.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = path.relative_to(REFERENCE_ROOT).as_posix()
            if rel not in allowed:
                raise ReferenceBuildError(f"unmanifested file in vendored tree: {rel}")
    for rel in manifest.get("sources", []):
        if rel not in files:
            raise ReferenceBuildError(f"manifest source {rel} is not a manifested file")
    return manifest


def shim_sources() -> List[Path]:
    return sorted(SHIM_DIR.glob("*.cc"))


def _shim_inputs() -> List[Path]:
    paths = sorted(SHIM_DIR.glob("*.h")) + shim_sources()
    paths += sorted(p for p in STUBS_DIR.rglob("*") if p.is_file())
    return paths


def library_key(compiler: str) -> str:
    """16-hex digest of everything the compiled library depends on."""
    digest = hashlib.sha256()
    digest.update(MANIFEST.read_bytes())
    for path in _shim_inputs():
        digest.update(path.relative_to(REFERENCE_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    digest.update("\0".join(CXXFLAGS).encode())
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


def compile_command(compiler: str, output: Path, manifest: Dict) -> List[str]:
    include_dirs = [SHIM_DIR, STUBS_DIR, THIRD_PARTY / "tflite_micro", THIRD_PARTY / "gemmlowp"]
    sources = [str(p) for p in shim_sources()] + [str(REFERENCE_ROOT / rel) for rel in manifest.get("sources", [])]
    cmd = [compiler, *CXXFLAGS]
    for directory in include_dirs:
        cmd += ["-I", str(directory)]
    return cmd + sources + ["-o", str(output)]


def ensure_reference_library(cache_root: Optional[Path] = None, compiler: Optional[str] = None) -> ReferenceLibrary:
    """Return the cached library for this environment, building it if absent."""
    compiler = compiler or find_host_cxx()
    manifest = verify_vendored_tree()
    key = library_key(compiler)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    final_dir = root / key
    library = final_dir / library_filename()
    identity = compiler_identity(compiler)
    if library.is_file():
        return ReferenceLibrary(library, key, compiler, identity)
    with exclusive_lock(root / f".{key}.lock"):
        if library.is_file():
            return ReferenceLibrary(library, key, compiler, identity)
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
        try:
            staged_lib = staging / library_filename()
            cmd = compile_command(compiler, staged_lib, manifest)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ReferenceBuildError(f"reference library build could not run: {exc}") from exc
            (staging / "build.log").write_text(
                " ".join(cmd) + "\n\n" + proc.stdout + proc.stderr, encoding="utf-8"
            )
            if proc.returncode != 0 or not staged_lib.is_file():
                tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-30:])
                raise ReferenceBuildError(f"reference library build failed (exit {proc.returncode}):\n{tail}")
            (staging / "flags.json").write_text(
                json.dumps(
                    {"compiler": compiler, "compiler_identity": identity, "cxxflags": list(CXXFLAGS), "key": key},
                    indent=2,
                )
                + "\n",
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
    """Report for `doctor`: never raises."""
    report: Dict[str, object] = {}
    try:
        compiler = find_host_cxx()
        verify_vendored_tree()
        key = library_key(compiler)
        root = Path(cache_root) if cache_root is not None else default_cache_root()
        library = root / key / library_filename()
        report.update({"key": key, "library": str(library), "built": library.is_file()})
    except Exception as exc:  # doctor reports, it does not fail
        report["error"] = str(exc)
    return report
