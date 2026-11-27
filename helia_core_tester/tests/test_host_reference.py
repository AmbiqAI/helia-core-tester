"""Host reference builds: manifest validation, the cached CMake build, and host-check's use of it."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from helia_core_tester.core import host_reference as hr
from helia_core_tester.core.library import legacy_host_reference
from helia_core_tester.generation.reference import host_kernels as hk
from helia_core_tester.tests.library_stub import NN_STATUS_ENUM

needs_cmake = pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("cc") is None, reason="cmake and a host cc required"
)

NN = legacy_host_reference()

_FAKE_CMAKELISTS = r"""
cmake_minimum_required(VERSION 3.15)
project(fake_kernels C)
option(NS_CMSIS_NN_HOST_REFERENCE "host reference" OFF)
file(GLOB_RECURSE srcs "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.c")
add_library(cmsis-nn STATIC ${srcs})
target_include_directories(cmsis-nn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Include")
target_compile_definitions(cmsis-nn PUBLIC HCT_FAKE_REF=1)
if(NS_CMSIS_NN_HOST_REFERENCE AND NOT HCT_FAKE_SKIP_MANIFEST)
  set(source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
  if(HCT_FAKE_SOURCE_DIR)
    set(source_dir "${HCT_FAKE_SOURCE_DIR}")
  endif()
  string(CONFIGURE [=[{
  "schema": "hct/host-reference/1",
  "library": "ns-cmsis-nn",
  "source_dir": "@source_dir@",
  "archive": "$<TARGET_FILE:cmsis-nn>",
  "include_dirs": ["@CMAKE_CURRENT_SOURCE_DIR@/Include"],
  "compile_definitions": ["HCT_FAKE_REF=1"],
  "library_compile_options": ["-O2"],
  "link_libraries": ["m"],
  "float16_type": "_Float16",
  "c_compiler": {"path": "@CMAKE_C_COMPILER@", "id": "@CMAKE_C_COMPILER_ID@", "version": "@CMAKE_C_COMPILER_VERSION@"}
}
]=] content @ONLY)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/ns_cmsis_nn_host_reference.json" CONTENT "${content}")
endif()
"""


def _fake_tree(tmp_path: Path, kernel: str = "int32_t hct_kernel(void) { return HCT_FAKE_REF * 42; }") -> Path:
    root = tmp_path / "checkout"
    (root / "Include").mkdir(parents=True)
    (root / "Include" / "arm_nnfunctions.h").write_text(
        "#ifndef ARM_NNFUNCTIONS_H\n#define ARM_NNFUNCTIONS_H\n"
        + NN_STATUS_ENUM + "#endif\n"
    )
    basic = root / "Source" / "BasicMathFunctions"
    basic.mkdir(parents=True)
    (basic / "kernel_s8.c").write_text(f"#include <stdint.h>\n{kernel}\n")
    (root / "CMakeLists.txt").write_text(_FAKE_CMAKELISTS)
    return root


def _manifest(tmp_path: Path, **overrides) -> Path:
    archive = tmp_path / "lib.a"
    archive.write_bytes(b"!<arch>\n")
    include = tmp_path / "inc"
    include.mkdir(exist_ok=True)
    data = {
        "schema": hr.SCHEMA,
        "library": "ns-cmsis-nn",
        "source_dir": str(tmp_path),
        "archive": str(archive),
        "include_dirs": [str(include)],
        "compile_definitions": ["ARM_NN_ENABLE_F32=1", "ARM_NN_ENABLE_F16=1"],
        "library_compile_options": ["-O2", "-ffp-contract=off"],
        "link_libraries": ["m"],
        "float16_type": "_Float16",
        "c_compiler": {"path": "/usr/bin/cc", "id": "GNU", "version": "13.3.0"},
    }
    for key, value in overrides.items():
        if value is _DROP:
            data.pop(key)
        else:
            data[key] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


_DROP = object()


def test_manifest_round_trip(tmp_path) -> None:
    ref = hr.load_manifest(_manifest(tmp_path), library="ns-cmsis-nn")
    assert ref.float16_type == "_Float16"
    assert ref.compile_flags() == [
        "-DARM_NN_ENABLE_F32=1", "-DARM_NN_ENABLE_F16=1", "-I", str(tmp_path / "inc"),
    ]
    assert ref.link_inputs() == [str(tmp_path / "lib.a"), "-lm"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema": "hct/host-reference/2"}, "schema is 'hct/host-reference/2'"),
        ({"library": "cmsis-dsp"}, "library is 'cmsis-dsp', expected 'ns-cmsis-nn'"),
        ({"library": ""}, "'library' must be a non-empty string"),
        ({"archive": "lib.a"}, "'archive' must be an absolute path"),
        ({"archive": "/nonexistent/lib.a"}, "does not exist (is the library built?)"),
        ({"include_dirs": []}, "'include_dirs' is empty"),
        ({"include_dirs": ["/nonexistent/inc"]}, "is not an existing absolute directory"),
        ({"include_dirs": "inc"}, "'include_dirs' must be a list"),
        ({"compile_definitions": ["A=1", "A=1"]}, "'compile_definitions' has duplicates"),
        ({"compile_definitions": ["-DA=1"]}, "must not carry flags"),
        ({"compile_definitions": [""]}, "must hold non-empty strings"),
        ({"link_libraries": ["-lm"]}, "must be bare library names"),
        ({"link_libraries": ["/usr/lib/libm.a"]}, "must be bare library names"),
        ({"float16_type": "half"}, "'float16_type' must be one of"),
        ({"c_compiler": {"path": "/usr/bin/cc", "id": "GNU"}}, "'c_compiler.version' must be a non-empty string"),
        ({"c_compiler": _DROP}, "'c_compiler' must be an object"),
        ({"source_dir": _DROP}, "'source_dir' must be a non-empty string"),
    ],
)
def test_bad_manifest_is_refused(tmp_path, overrides, message) -> None:
    with pytest.raises(hr.HostReferenceError) as err:
        hr.load_manifest(_manifest(tmp_path, **overrides), library="ns-cmsis-nn")
    assert message in str(err.value)


@pytest.mark.parametrize("content, message", [(None, "does not exist"), ("{", "is unreadable"), ("[]", "top level must be an object")])
def test_unreadable_manifest_is_refused(tmp_path, content, message) -> None:
    path = tmp_path / "manifest.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(hr.HostReferenceError, match=message):
        hr.load_manifest(path)


def test_supports_reads_the_option_from_cmakelists(tmp_path) -> None:
    assert not hr.supports(tmp_path, NN)
    (tmp_path / "CMakeLists.txt").write_text("project(x C)\n")
    assert not hr.supports(tmp_path, NN)
    (tmp_path / "CMakeLists.txt").write_text('option(NS_CMSIS_NN_HOST_REFERENCE "" OFF)\n')
    assert hr.supports(tmp_path, NN)


def test_tree_without_the_option_is_refused(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    (tree / "CMakeLists.txt").write_text("project(x C)\n")
    with pytest.raises(hr.HostReferenceError, match="does not offer NS_CMSIS_NN_HOST_REFERENCE"):
        hr.ensure_host_reference(tree, NN, cache_root=tmp_path / "cache")


def test_bad_jobs_is_refused(tmp_path) -> None:
    with pytest.raises(hr.HostReferenceError, match="jobs must be >= 1"):
        hr.ensure_host_reference(_fake_tree(tmp_path), NN, cache_root=tmp_path / "cache", jobs=0)


def test_missing_cmake_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hr.shutil, "which", lambda name: None)
    with pytest.raises(hr.HostReferenceError, match="cmake is not on PATH"):
        hr.ensure_host_reference(_fake_tree(tmp_path), NN, cache_root=tmp_path / "cache", cc="/usr/bin/cc")


@needs_cmake
def test_builds_caches_and_rekeys(tmp_path, monkeypatch) -> None:
    tree = _fake_tree(tmp_path)
    cache = tmp_path / "cache"
    ref = hr.ensure_host_reference(tree, NN, cache_root=cache, jobs=2)
    assert ref.archive.is_file()
    assert (cache / ref.key / "complete.json").is_file()
    assert ref.compile_definitions == ("HCT_FAKE_REF=1",)

    calls = []
    real_run = hr._run
    monkeypatch.setattr(hr, "_run", lambda *a, **k: calls.append(a) or real_run(*a, **k))
    again = hr.ensure_host_reference(tree, NN, cache_root=cache)
    assert again.key == ref.key and calls == []

    (tree / "Source" / "BasicMathFunctions" / "kernel_s8.c").write_text("#include <stdint.h>\nint32_t hct_kernel(void) { return 7; }\n")
    edited = hr.ensure_host_reference(tree, NN, cache_root=cache)
    assert edited.key != ref.key and len(calls) == 2


@needs_cmake
def test_incomplete_cache_entry_is_rebuilt(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    cache = tmp_path / "cache"
    ref = hr.ensure_host_reference(tree, NN, cache_root=cache)
    (cache / ref.key / "complete.json").unlink()
    (ref.archive).unlink()
    rebuilt = hr.ensure_host_reference(tree, NN, cache_root=cache)
    assert rebuilt.archive.is_file() and (cache / ref.key / "complete.json").is_file()


@needs_cmake
def test_build_failure_leaves_no_cache_entry(tmp_path) -> None:
    tree = _fake_tree(tmp_path, kernel='#error "injected kernel build failure"')
    cache = tmp_path / "cache"
    with pytest.raises(hr.HostReferenceError) as err:
        hr.ensure_host_reference(tree, NN, cache_root=cache)
    assert "host reference build failed" in str(err.value)
    assert "injected kernel build failure" in str(err.value)
    assert [p for p in cache.iterdir() if not p.name.startswith(".")] == []


@needs_cmake
def test_configure_failure_is_reported(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    (tree / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.15)\nproject(x C)\noption(NS_CMSIS_NN_HOST_REFERENCE "" OFF)\n'
        'message(FATAL_ERROR "injected configure failure")\n'
    )
    with pytest.raises(hr.HostReferenceError) as err:
        hr.ensure_host_reference(tree, NN, cache_root=tmp_path / "cache")
    assert "host reference configure failed" in str(err.value)
    assert "injected configure failure" in str(err.value)


@needs_cmake
def test_option_without_a_manifest_is_refused(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    text = (tree / "CMakeLists.txt").read_text().replace("NOT HCT_FAKE_SKIP_MANIFEST", "FALSE")
    (tree / "CMakeLists.txt").write_text(text)
    with pytest.raises(hr.HostReferenceError, match="wrote no ns_cmsis_nn_host_reference.json"):
        hr.ensure_host_reference(tree, NN, cache_root=tmp_path / "cache")


@needs_cmake
def test_manifest_for_another_tree_is_refused(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    text = (tree / "CMakeLists.txt").read_text().replace("if(HCT_FAKE_SOURCE_DIR)", "if(TRUE)").replace(
        'set(source_dir "${HCT_FAKE_SOURCE_DIR}")', 'set(source_dir "/elsewhere")'
    )
    (tree / "CMakeLists.txt").write_text(text)
    with pytest.raises(hr.HostReferenceError, match="describes /elsewhere"):
        hr.ensure_host_reference(tree, NN, cache_root=tmp_path / "cache")
    assert not any((tmp_path / "cache").glob("*/complete.json"))


def _case(root: Path, name: str, expr: str) -> Path:
    case = root / "Fam" / name
    (case / "includes").mkdir(parents=True)
    (case / f"{name}.c").write_text(
        "#include <stdint.h>\n"
        "extern void helia_test_finish(int32_t failures);\n"
        "int32_t hct_kernel(void);\n"
        f"int main(void) {{ helia_test_finish(({expr}) ? 0 : 1); return 0; }}\n"
    )
    return case


@needs_cmake
def test_host_check_m0_uses_the_host_reference(tmp_path) -> None:
    tree = _fake_tree(tmp_path)
    cases = tmp_path / "cases"
    _case(cases, "kernel_ok", "hct_kernel() == 42 && HCT_FAKE_REF == 1")
    _case(cases, "kernel_wrong", "hct_kernel() == 41")
    report = hk.run_host_check([cases], tree, mode="m0", cache_root=tmp_path / "hk", jobs=2)
    assert report.provider == hk.PROVIDER_HOST_REFERENCE
    assert report.float16_type == "_Float16"
    assert report.total == 2 and report.passed == 1
    assert [f["name"] for f in report.failures] == ["kernel_wrong"]
    assert report.to_json()["provider"] == hk.PROVIDER_HOST_REFERENCE


def test_host_check_m0_falls_back_to_the_source_scan_and_says_so(tmp_path) -> None:
    tree = _fake_tree(tmp_path, kernel="int32_t hct_kernel(void) { return 42; }")
    (tree / "CMakeLists.txt").unlink()
    cases = tmp_path / "cases"
    _case(cases, "kernel_ok", "hct_kernel() == 42")
    report = hk.run_host_check([cases], tree, mode="m0", cache_root=tmp_path / "hk")
    assert report.provider == hk.PROVIDER_SOURCE_SCAN
    assert report.passed == 1


@needs_cmake
def test_host_reference_failure_is_a_host_check_error(tmp_path) -> None:
    tree = _fake_tree(tmp_path, kernel="#error injected")
    with pytest.raises(hk.HostCheckError, match="host reference build failed"):
        hk.ensure_host_kernels(tree, "m0", cache_root=tmp_path / "hk")


_REAL_TREE = os.environ.get("HCT_TEST_HOST_REFERENCE_TREE", "").strip()


@pytest.mark.skipif(not _REAL_TREE, reason="set HCT_TEST_HOST_REFERENCE_TREE to an ns-cmsis-nn checkout")
def test_real_ns_cmsis_nn_host_reference(tmp_path) -> None:
    ref = hr.ensure_host_reference(Path(_REAL_TREE), NN, cache_root=tmp_path / "cache")
    assert sorted(ref.compile_definitions) == ["ARM_NN_ENABLE_F16=1", "ARM_NN_ENABLE_F32=1"]
    assert ref.library_compile_options == ("-O2", "-ffp-contract=off", "-fno-fast-math")
    assert ref.float16_type in hr.FLOAT16_TYPES
