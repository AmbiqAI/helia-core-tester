from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from helia_core_tester.fvp import cmake as fvp_cmake
from helia_core_tester.fvp.errors import FvpScriptError


def _fake_tool(tmp_path: Path, name: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / name
    tool.write_text("#!/bin/sh\nexec \"$@\"\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return tool


def _isolate_path(monkeypatch, *dirs: Path) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(str(d) for d in dirs))
    monkeypatch.delenv("HELIA_CORE_TESTER_COMPILER_LAUNCHER", raising=False)


def test_no_launcher_on_path_leaves_the_configure_untouched(monkeypatch, tmp_path: Path) -> None:
    _isolate_path(monkeypatch, tmp_path / "empty")

    assert fvp_cmake.compiler_launcher_args() == []


def test_ccache_on_path_is_passed_as_the_c_launcher(monkeypatch, tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, tool.parent)

    assert fvp_cmake.compiler_launcher_args() == [f"-DCMAKE_C_COMPILER_LAUNCHER={tool}"]


def test_sccache_is_accepted_when_ccache_is_absent(monkeypatch, tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "sccache")
    _isolate_path(monkeypatch, tool.parent)

    assert fvp_cmake.compiler_launcher_args() == [f"-DCMAKE_C_COMPILER_LAUNCHER={tool}"]


def test_env_override_beats_path_discovery(monkeypatch, tmp_path: Path) -> None:
    on_path = _fake_tool(tmp_path, "ccache")
    chosen = _fake_tool(tmp_path, "my-cache")
    _isolate_path(monkeypatch, on_path.parent)
    monkeypatch.setenv("HELIA_CORE_TESTER_COMPILER_LAUNCHER", "my-cache")

    assert fvp_cmake.compiler_launcher_args() == [f"-DCMAKE_C_COMPILER_LAUNCHER={chosen}"]


def test_override_naming_a_missing_tool_fails_the_configure(monkeypatch, tmp_path: Path) -> None:
    on_path = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, on_path.parent)
    monkeypatch.setenv("HELIA_CORE_TESTER_COMPILER_LAUNCHER", "no-such-cache")

    with pytest.raises(FvpScriptError, match="not on PATH"):
        fvp_cmake.compiler_launcher_args()


@pytest.mark.parametrize("value", ["none", "NONE", "", "  "])
def test_override_can_switch_off_a_launcher_that_is_on_path(
    monkeypatch, tmp_path: Path, value: str
) -> None:
    on_path = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, on_path.parent)
    monkeypatch.setenv("HELIA_CORE_TESTER_COMPILER_LAUNCHER", value)

    assert fvp_cmake.compiler_launcher_args() == []


def test_launcher_is_only_announced_at_verbosity(monkeypatch, tmp_path: Path, capsys) -> None:
    tool = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, tool.parent)

    fvp_cmake.compiler_launcher_args(0)
    assert "Compiler launcher" not in capsys.readouterr().out

    fvp_cmake.compiler_launcher_args(1)
    assert "Compiler launcher" in capsys.readouterr().out


def test_configure_command_carries_the_launcher_define(monkeypatch, tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, tool.parent)

    captured: dict = {}

    def fake_call(cmd, **kwargs):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(fvp_cmake.subprocess, "call", fake_call)

    fvp_cmake.cmake_configure(
        source_dir=tmp_path / "src",
        build_dir=tmp_path / "build",
        toolchain_file=tmp_path / "toolchain.cmake",
        cpu="cortex-m55",
        cmsis5=tmp_path / "cmsis5",
        optimization="-Ofast",
        extra_defs=["FOO=BAR"],
        generator=None,
        generated_tests_dir=None,
        enable_coverage=False,
        verbosity=0,
        env={},
    )

    cmd = captured["cmd"]
    assert f"-DCMAKE_C_COMPILER_LAUNCHER={tool}" in cmd
    # CMAKE_ASM_COMPILER_LAUNCHER is not a CMake variable: passing it only
    # produced an unused-variable warning on every configure.
    assert not any(item.startswith("-DCMAKE_ASM_COMPILER_LAUNCHER") for item in cmd)
    # Caller-supplied defines still win by coming later on the command line.
    assert cmd.index("-DFOO=BAR") > cmd.index(f"-DCMAKE_C_COMPILER_LAUNCHER={tool}")
