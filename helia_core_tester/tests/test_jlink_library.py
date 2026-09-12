"""J-Link library resolution order: $HPX_JLINK_DLL > $JLINK_PATH > JLinkExe on PATH > pylink default."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helia_core_tester.perf_stream import jlink_library
from helia_core_tester.perf_stream.jlink_library import (
    JLinkLibrary,
    describe_search,
    find_jlink_library,
    missing_library_hint,
    open_jlink,
    resolve_jlink_library,
)


def _no_which(_name: str):
    return None


def _segger_install(root: Path, name: str = "JLink_Linux_V960_x86_64") -> tuple[Path, Path]:
    """Create a SEGGER-style unpack: a versioned dir with JLinkExe + libjlinkarm.so*, plus a `JLink` symlink."""
    real = root / name
    real.mkdir()
    (real / "JLinkExe").write_bytes(b"#!/bin/sh\n")
    (real / "libjlinkarm.so.9.60.0").write_bytes(b"\x7fELF")
    os.symlink("libjlinkarm.so.9.60.0", real / "libjlinkarm.so.9")
    os.symlink("libjlinkarm.so.9", real / "libjlinkarm.so")
    (real / "libjlinkarm_x86.so").write_bytes(b"\x7fELF")
    link = root / "JLink"
    os.symlink(real, link, target_is_directory=True)
    return real, link


@pytest.fixture(autouse=True)
def _linux_patterns(monkeypatch):
    monkeypatch.setattr(jlink_library.sys, "platform", "linux")


def test_hpx_jlink_dll_wins_when_it_points_at_a_file(tmp_path: Path) -> None:
    real, _ = _segger_install(tmp_path)
    dll = real / "libjlinkarm.so.9.60.0"
    env = {"HPX_JLINK_DLL": str(dll), "JLINK_PATH": str(real / "JLinkExe")}
    found = find_jlink_library(env, which=lambda _n: str(real / "JLinkExe"))
    assert found == JLinkLibrary(str(dll), "$HPX_JLINK_DLL")
    assert resolve_jlink_library(env, which=_no_which) == str(dll)


def test_missing_hpx_jlink_dll_falls_through_to_jlink_path(tmp_path: Path) -> None:
    real, _ = _segger_install(tmp_path)
    env = {"HPX_JLINK_DLL": str(tmp_path / "nope.so"), "JLINK_PATH": str(real / "JLinkExe")}
    found = find_jlink_library(env, which=_no_which)
    assert found == JLinkLibrary(str(real / "libjlinkarm.so"), "$JLINK_PATH")


def test_jlink_path_accepts_the_exe_or_its_directory(tmp_path: Path) -> None:
    real, _ = _segger_install(tmp_path)
    expected = JLinkLibrary(str(real / "libjlinkarm.so"), "$JLINK_PATH")
    assert find_jlink_library({"JLINK_PATH": str(real / "JLinkExe")}, which=_no_which) == expected
    assert find_jlink_library({"JLINK_PATH": str(real)}, which=_no_which) == expected


def test_jlink_path_follows_symlinked_bin_wrapper(tmp_path: Path) -> None:
    """Nix / runner layouts expose bin/JLinkExe -> ../opt/SEGGER/JLink/JLinkExe; bin/ has no .so."""
    real, _ = _segger_install(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    os.symlink(real / "JLinkExe", bin_dir / "JLinkExe")
    found = find_jlink_library({"JLINK_PATH": str(bin_dir / "JLinkExe")}, which=_no_which)
    assert found == JLinkLibrary(str(real / "libjlinkarm.so"), "$JLINK_PATH")


def test_jlinkexe_on_path_is_used_when_no_env_vars(tmp_path: Path) -> None:
    real, link = _segger_install(tmp_path)
    asked = []

    def which(name: str):
        asked.append(name)
        return str(link / "JLinkExe") if name == "JLinkExe" else None

    found = find_jlink_library({}, which=which)
    assert found is not None
    assert found.source == "JLinkExe on PATH"
    # The symlinked `JLink` dir holds the library too, so it is found via the first (unresolved) directory.
    assert Path(found.path).resolve() == (real / "libjlinkarm.so.9.60.0").resolve()
    assert asked == ["JLinkExe"]


def test_prefers_unversioned_soname_and_ignores_x86_variant(tmp_path: Path) -> None:
    real, _ = _segger_install(tmp_path)
    found = find_jlink_library({"JLINK_PATH": str(real)}, which=_no_which)
    assert found is not None and found.path.endswith("/libjlinkarm.so")
    (real / "libjlinkarm.so").unlink()
    found = find_jlink_library({"JLINK_PATH": str(real)}, which=_no_which)
    assert found is not None and found.path.endswith("/libjlinkarm.so.9")


def test_nothing_found_returns_none_for_pylink_default(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {"HPX_JLINK_DLL": "", "JLINK_PATH": str(empty)}
    assert find_jlink_library(env, which=_no_which) is None
    assert resolve_jlink_library(env, which=_no_which) is None
    hint = missing_library_hint(env, which=_no_which)
    assert "HPX_JLINK_DLL" in hint and "JLINK_PATH" in hint and "JLinkExe on PATH: not found" in hint


def test_describe_search_lists_every_step() -> None:
    lines = describe_search({"HPX_JLINK_DLL": "/x/lib.so"}, which=lambda n: "/usr/bin/JLinkExe" if n == "JLinkExe" else None)
    assert lines[0] == "$HPX_JLINK_DLL=/x/lib.so"
    assert lines[1] == "$JLINK_PATH unset"
    assert lines[2] == "JLinkExe on PATH: /usr/bin/JLinkExe"
    assert "pylink default" in lines[3]


def test_open_jlink_passes_resolved_library_to_pylink(monkeypatch, tmp_path: Path) -> None:
    real, _ = _segger_install(tmp_path)
    dll = str(real / "libjlinkarm.so")
    monkeypatch.setattr(jlink_library, "find_jlink_library", lambda env=None, which=None: JLinkLibrary(dll, "$HPX_JLINK_DLL"))

    class _FakeLibrary:
        def __init__(self, path: str) -> None:
            self.path = path

    class _FakeJLink:
        def __init__(self, lib=None) -> None:
            self.lib = lib

    class _FakePylink:
        JLink = _FakeJLink

    import pylink.library

    monkeypatch.setattr(pylink.library, "Library", _FakeLibrary)
    jlink = open_jlink(_FakePylink)
    assert isinstance(jlink.lib, _FakeLibrary) and jlink.lib.path == dll


def test_open_jlink_uses_pylink_default_when_nothing_resolved(monkeypatch) -> None:
    monkeypatch.setattr(jlink_library, "find_jlink_library", lambda env=None, which=None: None)

    class _FakeJLink:
        def __init__(self, lib=None) -> None:
            self.lib = lib

    class _FakePylink:
        JLink = _FakeJLink

    assert open_jlink(_FakePylink).lib is None
