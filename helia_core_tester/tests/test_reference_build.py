"""Host reference library: build, load, ABI and cache behaviour."""

from __future__ import annotations

import ctypes
import json
import shutil
import sys
import threading

import pytest

from helia_core_tester.generation.reference import bindings as b
from helia_core_tester.generation.reference import host_build
from helia_core_tester.utils import host_compiler


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> host_build.ReferenceLibrary:
    return host_build.ensure_reference_library(cache_root=tmp_path_factory.mktemp("host_ref"))


def test_builds_and_loads_with_matching_abi(library) -> None:
    assert library.path.is_file()
    assert library.path.name == host_build.library_filename()
    lib = b.Bindings(library.path)
    assert lib._lib.hct_ref_abi_version() == b.ABI_VERSION


def test_struct_sizes_match_the_header(library) -> None:
    raw = ctypes.CDLL(str(library.path))
    raw.hct_ref_sizeof.argtypes = [ctypes.c_char_p]
    raw.hct_ref_sizeof.restype = ctypes.c_int32
    for name, struct in b.STRUCTS.items():
        assert raw.hct_ref_sizeof(name.encode()) == ctypes.sizeof(struct), name
    assert raw.hct_ref_sizeof(b"NoSuchType") == -1
    assert raw.hct_ref_sizeof(None) == -1


def test_build_writes_log_and_flags(library) -> None:
    flags = json.loads((library.path.parent / "flags.json").read_text())
    assert flags["key"] == library.key
    assert flags["cxxflags"] == list(host_build.CXXFLAGS)
    assert "-ffp-contract=off" in flags["cxxflags"]
    assert (library.path.parent / "build.log").is_file()


def test_second_call_is_a_cache_hit(library, monkeypatch) -> None:
    def no_build(*args, **kwargs):
        raise AssertionError("a cached library must not be rebuilt")

    monkeypatch.setattr(host_build.subprocess, "run", no_build)
    again = host_build.ensure_reference_library(cache_root=library.path.parents[1])
    assert again.path == library.path and again.key == library.key


def test_key_moves_with_compiler_identity(monkeypatch) -> None:
    compiler = host_compiler.find_host_cxx()
    base = host_build.library_key(compiler)
    monkeypatch.setattr(host_build, "compiler_identity", lambda c: "other compiler 1.0")
    assert host_build.library_key(compiler) != base


def test_concurrent_callers_build_once(tmp_path, monkeypatch) -> None:
    calls = []
    real_run = host_build.subprocess.run

    def counting_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(host_build.subprocess, "run", counting_run)
    results, errors = [], []

    def worker():
        try:
            results.append(host_build.ensure_reference_library(cache_root=tmp_path))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len({r.path for r in results}) == 1
    assert len(calls) == 1
    # No staging directory is left behind.
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [results[0].key]


def test_missing_compiler_names_the_variable(monkeypatch) -> None:
    monkeypatch.setenv(host_compiler.CXX_ENV, "definitely-not-a-compiler-xyz")
    with pytest.raises(host_compiler.HostCompilerMissing, match=host_compiler.CXX_ENV):
        host_compiler.find_host_cxx()


def test_no_compiler_on_path_is_reported(monkeypatch) -> None:
    monkeypatch.delenv(host_compiler.CXX_ENV, raising=False)
    monkeypatch.setattr(host_compiler.shutil, "which", lambda name: None)
    with pytest.raises(host_compiler.HostCompilerMissing, match="install a host toolchain"):
        host_compiler.find_host_cxx()


def test_build_failure_is_reported_with_the_log_tail(tmp_path) -> None:
    fake = tmp_path / "fake-cxx"
    fake.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv: print('fake 1.0'); sys.exit(0)\nprint('boom: no such header', file=sys.stderr); sys.exit(3)\n")
    fake.chmod(0o755)
    with pytest.raises(host_build.ReferenceBuildError, match="boom: no such header"):
        host_build.ensure_reference_library(cache_root=tmp_path / "cache", compiler=str(fake))
    # The failed build leaves no library and no staging directory behind.
    assert not list((tmp_path / "cache").glob("*/libhct_ref*"))
    assert not [p for p in (tmp_path / "cache").iterdir() if p.is_dir()]


def test_tampered_vendored_file_refuses_to_build(tmp_path, monkeypatch) -> None:
    copy = tmp_path / "reference_kernels"
    shutil.copytree(host_build.REFERENCE_ROOT, copy)
    monkeypatch.setattr(host_build, "REFERENCE_ROOT", copy)
    monkeypatch.setattr(host_build, "THIRD_PARTY", copy / "third_party")
    monkeypatch.setattr(host_build, "MANIFEST", copy / "third_party" / "manifest.json")
    target = copy / "third_party" / "tflite_micro" / "tensorflow" / "lite" / "kernels" / "internal" / "common.h"
    target.write_text(target.read_text() + "\n// edited\n")
    with pytest.raises(host_build.ReferenceBuildError, match="modified"):
        host_build.verify_vendored_tree()


def test_unmanifested_vendored_file_is_rejected(tmp_path, monkeypatch) -> None:
    copy = tmp_path / "reference_kernels"
    shutil.copytree(host_build.REFERENCE_ROOT, copy)
    monkeypatch.setattr(host_build, "REFERENCE_ROOT", copy)
    monkeypatch.setattr(host_build, "THIRD_PARTY", copy / "third_party")
    monkeypatch.setattr(host_build, "MANIFEST", copy / "third_party" / "manifest.json")
    (copy / "third_party" / "tflite_micro" / "extra.h").write_text("int x;\n")
    with pytest.raises(host_build.ReferenceBuildError, match="unmanifested"):
        host_build.verify_vendored_tree()


def test_abi_version_mismatch_is_refused(library, monkeypatch) -> None:
    monkeypatch.setattr(b, "ABI_VERSION", b.ABI_VERSION + 1)
    with pytest.raises(b.ReferenceAbiMismatch, match="ABI"):
        b.Bindings(library.path)


def test_struct_size_drift_is_refused(library, monkeypatch) -> None:
    class Bigger(ctypes.Structure):
        _fields_ = [("rank", ctypes.c_int32), ("dims", ctypes.c_int32 * 7)]

    monkeypatch.setitem(b.STRUCTS, "HctShape", Bigger)
    with pytest.raises(b.ReferenceAbiMismatch, match="sizeof\\(HctShape\\)"):
        b.Bindings(library.path)


def test_unloadable_library_is_refused(tmp_path) -> None:
    bogus = tmp_path / host_build.library_filename()
    bogus.write_bytes(b"not a shared object")
    with pytest.raises(b.ReferenceAbiMismatch, match="cannot load"):
        b.Bindings(bogus)


def test_describe_cache_never_raises(monkeypatch) -> None:
    def broken():
        raise host_build.ReferenceBuildError("broken tree")

    monkeypatch.setattr(host_build, "verify_vendored_tree", broken)
    assert host_build.describe_cache()["error"] == "broken tree"


def test_toolchain_report_never_raises(monkeypatch) -> None:
    monkeypatch.setattr(host_compiler.shutil, "which", lambda name: None)
    monkeypatch.delenv(host_compiler.CC_ENV, raising=False)
    monkeypatch.delenv(host_compiler.CXX_ENV, raising=False)
    monkeypatch.delenv(host_compiler.FLATC_ENV, raising=False)
    report = host_compiler.describe_host_toolchain()
    assert report["cc"] is None and report["cxx"] is None and report["flatc"] is None
    assert "cc_error" in report and "flatc_error" in report


def test_library_exports_only_the_c_abi(library) -> None:
    # A second TFLite in the process (TensorFlow, ai_edge_litert) must not bind to the
    # vendored tflite:: code, nor it to theirs: on Linux that interposition corrupted the heap.
    import subprocess

    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm is not available")
    flags = ["-gU"] if sys.platform == "darwin" else ["-D", "--defined-only"]
    out = subprocess.run([nm, *flags, str(library.path)], check=True, capture_output=True, text=True).stdout
    names = [line.split()[-1].lstrip("_") for line in out.splitlines() if line.strip()]
    exported = [n for n in names if n.startswith("hct_ref_")]
    assert exported and "hct_ref_abi_version" in exported
    assert not [n for n in names if "tflite" in n or "gemmlowp" in n]
