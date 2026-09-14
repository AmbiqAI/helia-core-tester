"""J-Link library resolution order: $HPX_JLINK_DLL > $JLINK_PATH > JLinkExe on PATH > pylink default."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helia_core_tester.perf_stream import jlink_library
from helia_core_tester.perf_stream.jlink_library import (
    JLinkExecutable,
    JLinkLibrary,
    JLinkLibraryError,
    describe_search,
    find_jlink_exe,
    find_jlink_library,
    missing_library_hint,
    open_jlink,
    resolve_jlink_exe,
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


def test_missing_hpx_jlink_dll_is_an_error_not_a_fallthrough(tmp_path: Path) -> None:
    """An explicitly configured library that does not exist must not be papered over by
    a different install found via $JLINK_PATH or PATH."""
    real, _ = _segger_install(tmp_path)
    env = {"HPX_JLINK_DLL": str(tmp_path / "nope.so"), "JLINK_PATH": str(real / "JLinkExe")}
    with pytest.raises(JLinkLibraryError, match=r"\$HPX_JLINK_DLL=.*nope\.so does not exist"):
        find_jlink_library(env, which=lambda _n: str(real / "JLinkExe"))
    # The flash-target resolver raises too once it needs the library step ($JLINK_PATH unset).
    with pytest.raises(JLinkLibraryError):
        find_jlink_exe({"HPX_JLINK_DLL": str(tmp_path / "nope.so")}, which=lambda _n: str(real / "JLinkExe"))
    # Blank is "unset", not "missing".
    assert find_jlink_library({"HPX_JLINK_DLL": "  ", "JLINK_PATH": str(real)}, which=_no_which) is not None


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


# --- JLinkExe (flash target) resolution ------------------------------------------------


def test_jlink_path_names_the_exe_or_its_directory_for_the_flash_target(tmp_path: Path) -> None:
    real, link = _segger_install(tmp_path)
    expected = JLinkExecutable(str(real / "JLinkExe"), "$JLINK_PATH")
    assert find_jlink_exe({"JLINK_PATH": str(real / "JLinkExe")}, which=_no_which) == expected
    assert find_jlink_exe({"JLINK_PATH": str(real)}, which=_no_which) == expected
    # A symlinked directory is returned as given (CMake gets a path that exists).
    assert find_jlink_exe({"JLINK_PATH": str(link)}, which=_no_which) == JLinkExecutable(str(link / "JLinkExe"), "$JLINK_PATH")
    assert resolve_jlink_exe({"JLINK_PATH": str(real)}, which=_no_which) == str(real / "JLinkExe")


def test_jlink_exe_is_found_next_to_hpx_jlink_dll_without_jlink_path(tmp_path: Path) -> None:
    """The runners export HPX_JLINK_DLL alone; the binary lives beside the library."""
    real, _ = _segger_install(tmp_path)
    found = find_jlink_exe({"HPX_JLINK_DLL": str(real / "libjlinkarm.so")}, which=_no_which)
    assert found == JLinkExecutable(str(real / "JLinkExe"), "next to the J-Link library from $HPX_JLINK_DLL")


def test_jlink_exe_follows_a_symlinked_library_into_the_real_install(tmp_path: Path) -> None:
    """Nix-style layout: lib/libjlinkarm.so -> <store>/JLink/libjlinkarm.so, no JLinkExe in lib/."""
    real, _ = _segger_install(tmp_path)
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    os.symlink(real / "libjlinkarm.so.9.60.0", lib_dir / "libjlinkarm.so")
    found = find_jlink_exe({"HPX_JLINK_DLL": str(lib_dir / "libjlinkarm.so")}, which=_no_which)
    assert found is not None and Path(found.path) == real / "JLinkExe"


def test_jlink_exe_falls_back_to_path_then_none(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {"HPX_JLINK_DLL": "", "JLINK_PATH": str(empty)}
    assert find_jlink_exe(env, which=lambda n: "/usr/bin/JLinkExe" if n == "JLinkExe" else None) == JLinkExecutable("/usr/bin/JLinkExe", "JLinkExe on PATH")
    assert find_jlink_exe(env, which=_no_which) is None
    assert resolve_jlink_exe(env, which=_no_which) is None


def test_jlink_path_wins_over_library_neighbour_and_path(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    real_a, _ = _segger_install(tmp_path / "a")
    real_b, _ = _segger_install(tmp_path / "b")
    env = {"HPX_JLINK_DLL": str(real_b / "libjlinkarm.so"), "JLINK_PATH": str(real_a / "JLinkExe")}
    found = find_jlink_exe(env, which=lambda _n: str(real_b / "JLinkExe"))
    assert found == JLinkExecutable(str(real_a / "JLinkExe"), "$JLINK_PATH")


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
