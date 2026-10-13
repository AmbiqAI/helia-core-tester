"""FVP build trees are cleaned when the configured compiler changes in place.

Ninja does not rebuild when the binary at a configured compiler path changes, so a
toolchain upgrade under artifacts/downloads/arm_gcc_download would otherwise link
objects the old compiler built.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from helia_core_tester.fvp import cmake as fvp_cmake
from helia_core_tester.fvp.errors import FvpScriptError


def _fake_gcc(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {version}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def cmake(monkeypatch, tmp_path: Path) -> dict:
    """Fake cmake: configure writes a cache naming `state['compiler']`; clean calls are recorded."""
    state: dict = {"compiler": None, "cleans": 0, "clean_rc": 0, "configures": 0}

    def fake_call(cmd, **kwargs):
        build_dir = Path(cmd[cmd.index("-B") + 1]) if "-B" in cmd else Path(cmd[2])
        if "--target" in cmd and cmd[cmd.index("--target") + 1] == "clean":
            state["cleans"] += 1
            return state["clean_rc"]
        state["configures"] += 1
        lines = ["CMAKE_HOME_DIRECTORY:INTERNAL=/src"]
        build_dir.mkdir(parents=True, exist_ok=True)
        if state["compiler"] is not None and state.get("via") == "probe":
            probe = build_dir / "CMakeFiles" / "3.31.0" / "CMakeCCompiler.cmake"
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text(f'set(CMAKE_C_COMPILER "{state["compiler"]}")\n')
        elif state["compiler"] is not None:
            lines.append(f"CMAKE_C_COMPILER:FILEPATH={state['compiler']}")
        (build_dir / "CMakeCache.txt").write_text("\n".join(lines) + "\n")
        return 0

    monkeypatch.setattr(fvp_cmake.subprocess, "call", fake_call)
    monkeypatch.delenv("HELIA_CORE_TESTER_COMPILER_LAUNCHER", raising=False)
    return state


def _configure(tmp_path: Path, env: dict | None = None) -> Path:
    build_dir = tmp_path / "build"
    fvp_cmake.cmake_configure(
        source_dir=tmp_path / "src",
        build_dir=build_dir,
        toolchain_file=tmp_path / "toolchain.cmake",
        cpu="cortex-m55",
        cmsis5=tmp_path / "cmsis5",
        optimization="-Ofast",
        extra_defs=[],
        generator=None,
        generated_tests_dir=None,
        enable_coverage=False,
        verbosity=0,
        env=env if env is not None else {"PATH": str(tmp_path / "empty")},
    )
    return build_dir


def _stamp(build_dir: Path) -> dict:
    return json.loads((build_dir / fvp_cmake.COMPILER_STAMP).read_text())


def test_first_configure_cleans_and_records_the_compiler(tmp_path: Path, cmake: dict) -> None:
    """No record: the tree may hold objects from an unknown compiler."""
    cmake["compiler"] = _fake_gcc(tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc", "14.3.1")
    build_dir = _configure(tmp_path)
    assert cmake["cleans"] == 1
    assert _stamp(build_dir)["version"] == "14.3.1"
    assert _stamp(build_dir)["path"] == str(cmake["compiler"].resolve())


def test_unchanged_compiler_does_not_clean(tmp_path: Path, cmake: dict) -> None:
    cmake["compiler"] = _fake_gcc(tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc", "14.3.1")
    _configure(tmp_path)
    _configure(tmp_path)
    assert cmake["cleans"] == 1


def test_toolchain_file_compiler_is_read_from_the_cmake_probe(tmp_path: Path, cmake: dict) -> None:
    """A toolchain file's plain set(CMAKE_C_COMPILER ...) never reaches the cache."""
    gcc = tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc"
    cmake["compiler"] = _fake_gcc(gcc, "14.2.1")
    cmake["via"] = "probe"
    build_dir = _configure(tmp_path)
    assert _stamp(build_dir)["version"] == "14.2.1"
    _fake_gcc(gcc, "14.3.1")
    _configure(tmp_path)
    assert cmake["cleans"] == 2


def test_version_upgrade_at_the_same_path_cleans(tmp_path: Path, cmake: dict) -> None:
    gcc = tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc"
    cmake["compiler"] = _fake_gcc(gcc, "14.2.1")
    build_dir = _configure(tmp_path)
    _fake_gcc(gcc, "14.3.1")
    _configure(tmp_path)
    assert cmake["cleans"] == 2
    assert _stamp(build_dir)["version"] == "14.3.1"


def test_same_version_binary_replaced_in_place_cleans(tmp_path: Path, cmake: dict) -> None:
    """A reinstall that changes the binary counts even when -dumpversion agrees."""
    gcc = tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc"
    cmake["compiler"] = _fake_gcc(gcc, "14.3.1")
    _configure(tmp_path)
    st = gcc.stat()
    os.utime(gcc, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _configure(tmp_path)
    assert cmake["cleans"] == 2


def test_bare_compiler_name_resolves_through_the_build_path(tmp_path: Path, cmake: dict) -> None:
    gcc = _fake_gcc(tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc", "14.3.1")
    cmake["compiler"] = "arm-none-eabi-gcc"
    build_dir = _configure(tmp_path, env={"PATH": str(gcc.parent)})
    assert cmake["cleans"] == 1
    assert _stamp(build_dir)["path"] == str(gcc.resolve())


@pytest.mark.parametrize("compiler", [None, "/no/such/arm-none-eabi-gcc", "not-on-path-gcc"])
def test_unresolvable_compiler_neither_cleans_nor_records(tmp_path: Path, cmake: dict, compiler) -> None:
    cmake["compiler"] = compiler
    build_dir = _configure(tmp_path)
    assert cmake["cleans"] == 0
    assert not (build_dir / fvp_cmake.COMPILER_STAMP).exists()


def test_failed_clean_fails_the_configure_and_retries_next_time(tmp_path: Path, cmake: dict) -> None:
    gcc = tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc"
    cmake["compiler"] = _fake_gcc(gcc, "14.2.1")
    build_dir = _configure(tmp_path)
    _fake_gcc(gcc, "14.3.1")
    cmake["clean_rc"] = 2
    with pytest.raises(FvpScriptError, match="compiler change"):
        _configure(tmp_path)
    assert not (build_dir / fvp_cmake.COMPILER_STAMP).exists()
    cmake["clean_rc"] = 0
    _configure(tmp_path)
    assert cmake["cleans"] == 3
    assert _stamp(build_dir)["version"] == "14.3.1"


@pytest.mark.parametrize("content", ["", "{not json", "[]", '{"path": "/other"}'])
def test_corrupt_or_foreign_stamp_cleans(tmp_path: Path, cmake: dict, content: str) -> None:
    cmake["compiler"] = _fake_gcc(tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc", "14.3.1")
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / fvp_cmake.COMPILER_STAMP).write_text(content)
    _configure(tmp_path)
    assert cmake["cleans"] == 1
    assert _stamp(build_dir)["version"] == "14.3.1"


def test_compiler_without_a_version_is_still_tracked_by_its_file(tmp_path: Path, cmake: dict) -> None:
    gcc = tmp_path / "gcc" / "bin" / "arm-none-eabi-gcc"
    gcc.parent.mkdir(parents=True)
    gcc.write_text("#!/bin/sh\nexit 1\n")
    gcc.chmod(0o755)
    cmake["compiler"] = gcc
    build_dir = _configure(tmp_path)
    assert _stamp(build_dir)["version"] is None
    _configure(tmp_path)
    assert cmake["cleans"] == 1
