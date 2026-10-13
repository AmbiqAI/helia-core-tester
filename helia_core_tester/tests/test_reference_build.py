"""Host build of the reference library: build, cache, invalidation, failure paths, export surface."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from helia_core_tester.generation.reference import abi, build
from helia_core_tester.utils.host_compiler import HostCompilerMissing, find_host_cc


@pytest.fixture
def tree(tmp_path: Path, monkeypatch) -> Path:
    """A private copy of the reference tree, so tests can edit sources freely."""
    root = tmp_path / "reference"
    shutil.copytree(abi.REFERENCE_ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(abi, "SPEC_PATH", root / "spec" / "entries.yaml")
    monkeypatch.setattr(abi, "HEADER_PATH", root / "include" / "hct_ref_abi.h")
    monkeypatch.setattr(build, "REFERENCE_ROOT", root)
    monkeypatch.setattr(build, "SPEC_PATH", root / "spec" / "entries.yaml")
    monkeypatch.setattr(build, "HEADER_PATH", root / "include" / "hct_ref_abi.h")
    monkeypatch.setattr(build, "SRC_DIR", root / "src")
    monkeypatch.setattr(build, "INCLUDE_DIR", root / "include")
    monkeypatch.setattr(build, "COMMON_DIR", root / "src" / "common")
    return root


def test_builds_once_and_reuses_the_cached_library(tree: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = build.ensure_reference_library(cache)
    assert first.path.is_file() and first.path.parent.name == first.key
    assert (first.path.parent / "flags.json").is_file() and (first.path.parent / "build.log").is_file()
    stamp = first.path.stat().st_mtime_ns
    second = build.ensure_reference_library(cache)
    assert second == first and second.path.stat().st_mtime_ns == stamp


def test_any_source_edit_changes_the_key(tree: Path) -> None:
    cc = find_host_cc()
    before = build.library_key(cc)
    add = tree / "src" / "basic_math" / "add.c"
    add.write_text(add.read_text() + "\n/* edited */\n")
    assert build.library_key(cc) != before
    after_source = build.library_key(cc)
    internal = tree / "src" / "common" / "hct_ref_internal.h"
    internal.write_text(internal.read_text() + "\n")
    assert build.library_key(cc) != after_source
    assert build.library_key(cc, flags=(*build.CFLAGS, "-O2")) != build.library_key(cc)


def test_a_stale_header_is_refused(tree: Path, tmp_path: Path) -> None:
    header = tree / "include" / "hct_ref_abi.h"
    header.write_text(header.read_text().replace("HCT_MAX_RANK 8", "HCT_MAX_RANK 9"))
    with pytest.raises(build.ReferenceBuildError, match="does not match"):
        build.ensure_reference_library(tmp_path / "cache")


def test_a_compile_error_is_reported_with_the_compiler_output(tree: Path, tmp_path: Path) -> None:
    (tree / "src" / "basic_math" / "broken.c").write_text("int broken(void) { return undeclared_symbol; }\n")
    with pytest.raises(build.ReferenceBuildError, match="undeclared_symbol"):
        build.ensure_reference_library(tmp_path / "cache")
    # Nothing half-built is left where a later run would load it.
    assert not list((tmp_path / "cache").rglob("libhct_ref.*"))


def test_warnings_are_errors(tree: Path, tmp_path: Path) -> None:
    (tree / "src" / "basic_math" / "warn.c").write_text("int warn(int x) { int unused; return x; }\n")
    with pytest.raises(build.ReferenceBuildError):
        build.ensure_reference_library(tmp_path / "cache")


def test_a_missing_compiler_names_the_override(tree: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HCT_HOST_CC", "definitely-not-a-compiler-hct")
    with pytest.raises(HostCompilerMissing, match="HCT_HOST_CC"):
        build.ensure_reference_library(tmp_path / "cache")


def test_no_sources_is_an_error(tree: Path, tmp_path: Path) -> None:
    shutil.rmtree(tree / "src")
    (tree / "src" / "common").mkdir(parents=True)
    with pytest.raises(build.ReferenceBuildError, match="no C sources"):
        build.ensure_reference_library(tmp_path / "cache")


def test_concurrent_builders_produce_one_library(tree: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    results, errors = [], []

    def worker() -> None:
        try:
            results.append(build.ensure_reference_library(cache))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len({r.path for r in results}) == 1
    assert len([p for p in cache.iterdir() if p.is_dir()]) == 1


def test_library_exports_only_the_abi(tmp_path: Path) -> None:
    lib = build.ensure_reference_library(tmp_path / "cache")
    nm = subprocess.run(["nm", "-gU" if sys.platform == "darwin" else "-D", "--defined-only"
                         if sys.platform != "darwin" else "-j", str(lib.path)],
                        capture_output=True, text=True)
    if nm.returncode != 0:
        pytest.skip(f"nm unavailable: {nm.stderr.strip()}")
    names = {line.split()[-1].lstrip("_") for line in nm.stdout.splitlines() if line.strip()}
    names = {n for n in names if n and not n.startswith((".", "dyld"))}
    spec = abi.load_spec()
    want = {"hct_ref_abi_version"} | {f"hct_ref_{n}" for n in (*spec.kernels, *spec.prepare)}
    assert want <= names
    assert {n for n in names if n.startswith("hct_")} == want
