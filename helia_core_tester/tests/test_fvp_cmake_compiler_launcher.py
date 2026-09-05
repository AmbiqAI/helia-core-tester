from __future__ import annotations

import os
import stat
from pathlib import Path

from helia_core_tester.fvp import cmake as fvp_cmake


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
    monkeypatch.setattr(fvp_cmake, "_cmake_version", lambda: (3, 28))

    assert fvp_cmake.compiler_launcher_args() == []


def test_ccache_on_path_is_passed_as_the_c_and_asm_launcher(monkeypatch, tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, tool.parent)
    monkeypatch.setattr(fvp_cmake, "_cmake_version", lambda: (3, 28))

    assert fvp_cmake.compiler_launcher_args() == [
        f"-DCMAKE_C_COMPILER_LAUNCHER={tool}",
        f"-DCMAKE_ASM_COMPILER_LAUNCHER={tool}",
    ]


def test_asm_launcher_is_withheld_from_cmake_too_old_to_honour_it(monkeypatch, tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, tool.parent)
    monkeypatch.setattr(fvp_cmake, "_cmake_version", lambda: (3, 3))

    assert fvp_cmake.compiler_launcher_args() == [f"-DCMAKE_C_COMPILER_LAUNCHER={tool}"]


def test_sccache_is_accepted_when_ccache_is_absent(monkeypatch, tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "sccache")
    _isolate_path(monkeypatch, tool.parent)
    monkeypatch.setattr(fvp_cmake, "_cmake_version", lambda: (3, 28))

    assert fvp_cmake.compiler_launcher_args()[0] == f"-DCMAKE_C_COMPILER_LAUNCHER={tool}"


def test_env_override_beats_path_discovery(monkeypatch, tmp_path: Path) -> None:
    on_path = _fake_tool(tmp_path, "ccache")
    chosen = _fake_tool(tmp_path, "my-cache")
    _isolate_path(monkeypatch, on_path.parent)
    monkeypatch.setenv("HELIA_CORE_TESTER_COMPILER_LAUNCHER", "my-cache")
    monkeypatch.setattr(fvp_cmake, "_cmake_version", lambda: (3, 28))

    assert fvp_cmake.compiler_launcher_args()[0] == f"-DCMAKE_C_COMPILER_LAUNCHER={chosen}"


def test_configure_command_carries_the_launcher_defines(monkeypatch, tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "ccache")
    _isolate_path(monkeypatch, tool.parent)
    monkeypatch.setattr(fvp_cmake, "_cmake_version", lambda: (3, 28))

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
    assert f"-DCMAKE_ASM_COMPILER_LAUNCHER={tool}" in cmd
    # Caller-supplied defines still win by coming later on the command line.
    assert cmd.index("-DFOO=BAR") > cmd.index(f"-DCMAKE_C_COMPILER_LAUNCHER={tool}")
