"""Host-side helpers behind the hardware commands: where arm-none-eabi-* tools come
from (the lazily downloaded toolchain first), the Python 3.8-safe path helpers, and
the memory report's artifact paths for in-tree and external build dirs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from helia_core_tester.perf_stream import benchmark_firmware_report as report
from helia_core_tester.perf_stream import toolchain
from helia_core_tester.perf_stream.pathutil import display_path, is_relative_to


def _fake_tool(bin_dir: Path, name: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    tool = bin_dir / name
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    return tool


def test_arm_tool_prefers_the_downloaded_toolchain(tmp_path: Path, monkeypatch) -> None:
    bin_dir = toolchain.toolchain_bin_dir(tmp_path)
    assert bin_dir == tmp_path / "artifacts" / "downloads" / "arm_gcc_download" / "bin"

    # Not downloaded: PATH lookup, else the bare name (so the eventual failure names the tool).
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: "/usr/bin/" + name)
    assert toolchain.arm_tool("arm-none-eabi-nm", tmp_path) == "/usr/bin/arm-none-eabi-nm"
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: None)
    assert toolchain.arm_tool("arm-none-eabi-nm", tmp_path) == "arm-none-eabi-nm"

    nm = _fake_tool(bin_dir, "arm-none-eabi-nm")
    assert toolchain.arm_tool("arm-none-eabi-nm", tmp_path) == str(nm)
    # Only the tools that were actually downloaded are redirected.
    assert toolchain.arm_tool("arm-none-eabi-size", tmp_path) == "arm-none-eabi-size"

    # Callers that pass no repo root get the tester's own downloads dir.
    monkeypatch.setattr(toolchain, "_default_repo_root", lambda: tmp_path)
    assert toolchain.arm_tool("arm-none-eabi-nm") == str(nm)


def test_add_toolchain_to_path_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    bin_dir = toolchain.toolchain_bin_dir(tmp_path)
    assert toolchain.add_toolchain_to_path(tmp_path) is False  # nothing downloaded yet
    assert os.environ["PATH"] == "/usr/bin"

    bin_dir.mkdir(parents=True)
    assert toolchain.add_toolchain_to_path(tmp_path) is True
    assert os.environ["PATH"].split(os.pathsep) == [str(bin_dir), "/usr/bin"]
    assert toolchain.add_toolchain_to_path(tmp_path) is False
    assert os.environ["PATH"].split(os.pathsep) == [str(bin_dir), "/usr/bin"]


def test_symbol_lookup_and_memory_report_resolve_nm_through_the_helper(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.perf_stream import transport

    nm = _fake_tool(toolchain.toolchain_bin_dir(tmp_path), "arm-none-eabi-nm")
    monkeypatch.setattr(toolchain, "_default_repo_root", lambda: tmp_path)
    argv: list[list[str]] = []

    class _Done:
        stdout = "20000000 D _SEGGER_RTT\n"

    monkeypatch.setattr(transport.subprocess, "run", lambda cmd, **kwargs: argv.append(list(cmd)) or _Done())
    assert transport.symbol_address_from_elf("fw.elf", "_SEGGER_RTT") == 0x20000000
    monkeypatch.setattr(report.subprocess, "run", lambda cmd, **kwargs: argv.append(list(cmd)) or _Done())
    report._probe_binary("arm-none-eabi-nm", ["fw.elf"])
    assert [cmd[0] for cmd in argv] == [str(nm), str(nm)]


def test_path_helpers_are_python38_safe(tmp_path: Path) -> None:
    inside, outside = tmp_path / "repo" / "build" / "x.elf", Path("/elsewhere/build/x.elf")
    assert is_relative_to(inside, tmp_path / "repo") and not is_relative_to(outside, tmp_path / "repo")
    assert display_path(inside, tmp_path / "repo") == "build/x.elf"
    assert display_path(outside, tmp_path / "repo") == "/elsewhere/build/x.elf"
    # No Path.is_relative_to() (3.9+) anywhere in the package (pathutil.py is the
    # replacement and names it in its docstring).
    package = Path(toolchain.__file__).parent
    offenders = [
        p.name for p in package.glob("*.py")
        if p.name != "pathutil.py" and ".is_relative_to(" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.fixture
def report_env(tmp_path: Path, monkeypatch):
    """A fake repo root and stubbed binutils so the memory report runs without an ELF toolchain."""
    repo = tmp_path / "repo"
    (repo / "cmake" / "perf_stream").mkdir(parents=True)
    (repo / "cmake" / "perf_stream" / "kernel_catalog.json").write_text("[]")
    monkeypatch.setattr(report, "_repo_root", lambda: repo)
    monkeypatch.setattr(report, "_probe_binary", lambda tool, args: "")
    monkeypatch.setattr(report, "_parse_memory_regions", lambda path: [])
    return repo


def _fake_build(build_dir: Path) -> Path:
    elf = build_dir / "perf_stream" / "hct_benchmark_server.elf"
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"elf")
    return elf


def test_memory_report_paths_are_repo_relative_inside_and_absolute_outside(tmp_path: Path, report_env) -> None:
    repo = report_env
    inside = _fake_build(repo / "build" / "perf_stream" / "apollo510_evb")
    data = json.loads(report.generate_benchmark_server_memory_report(
        build_dir=inside.parent.parent, output_root=tmp_path / "out_in").read_text())
    assert data["artifacts"] == {
        "elf": "build/perf_stream/apollo510_evb/perf_stream/hct_benchmark_server.elf",
        "bin": "build/perf_stream/apollo510_evb/perf_stream/hct_benchmark_server.bin",
        "map": "build/perf_stream/apollo510_evb/perf_stream/hct_benchmark_server.map",
    }

    external = _fake_build(tmp_path / "extbuild")
    data = json.loads(report.generate_benchmark_server_memory_report(
        build_dir=external.parent.parent, output_root=tmp_path / "out_ext").read_text())
    assert data["artifacts"]["elf"] == str(external)
    assert data["artifacts"]["bin"] == str(external.with_suffix(".bin"))
    assert data["artifacts"]["map"] == str(external.with_suffix(".map"))


def test_memory_report_names_the_missing_elf(tmp_path: Path, report_env) -> None:
    with pytest.raises(FileNotFoundError, match="Built firmware ELF not found") as info:
        report.generate_benchmark_server_memory_report(build_dir=tmp_path / "never-built", output_root=tmp_path / "out")
    assert str(tmp_path / "never-built" / "perf_stream" / "hct_benchmark_server.elf") in str(info.value)
