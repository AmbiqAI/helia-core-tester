"""Host-side helpers behind the hardware commands: where arm-none-eabi-* tools come
from (the lazily downloaded toolchain first), the Python 3.8-safe path helpers, and
the memory report's artifact paths for in-tree and external build dirs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from helia_core_tester.hardware import memory_report as report
from helia_core_tester.hardware import toolchain
from helia_core_tester.hardware.boards import DEFAULT_BOARD_ID, resolve_board
from helia_core_tester.hardware.pathutil import display_path, is_relative_to


BOARD = resolve_board(DEFAULT_BOARD_ID)


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
    from helia_core_tester.hardware import transport

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
    (repo / "cmake" / "hardware").mkdir(parents=True)
    (repo / "cmake" / "hardware" / "kernel_catalog.json").write_text("[]")
    monkeypatch.setattr(report, "repo_root", lambda: repo)
    monkeypatch.setattr(report, "_probe_binary", lambda tool, args, project_root=None: "")
    monkeypatch.setattr(
        report, "parse_memory_regions",
        lambda path: [{"name": "MCU_MRAM", "capacity": 4128768}, {"name": "MCU_TCM", "capacity": 507904}],
    )
    return repo


def _fake_build(build_dir: Path) -> Path:
    elf = build_dir / "hardware" / "hct_benchmark_server.elf"
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"elf")
    return elf


def test_memory_report_paths_are_repo_relative_inside_and_absolute_outside(tmp_path: Path, report_env) -> None:
    repo = report_env
    inside = _fake_build(repo / "build" / "hardware" / "apollo510_evb")
    data = json.loads(report.generate_memory_report(BOARD, 
        build_dir=inside.parent.parent, output_root=tmp_path / "out_in").read_text())
    assert data["artifacts"] == {
        "elf": "build/hardware/apollo510_evb/hardware/hct_benchmark_server.elf",
        "bin": "build/hardware/apollo510_evb/hardware/hct_benchmark_server.bin",
        "map": "build/hardware/apollo510_evb/hardware/hct_benchmark_server.map",
    }

    external = _fake_build(tmp_path / "extbuild")
    data = json.loads(report.generate_memory_report(BOARD, 
        build_dir=external.parent.parent, output_root=tmp_path / "out_ext").read_text())
    assert data["artifacts"]["elf"] == str(external)
    assert data["artifacts"]["bin"] == str(external.with_suffix(".bin"))
    assert data["artifacts"]["map"] == str(external.with_suffix(".map"))


def test_memory_report_names_the_missing_elf(tmp_path: Path, report_env) -> None:
    with pytest.raises(FileNotFoundError, match="Built firmware ELF not found") as info:
        report.generate_memory_report(BOARD, build_dir=tmp_path / "never-built", output_root=tmp_path / "out")
    assert str(tmp_path / "never-built" / "hardware" / "hct_benchmark_server.elf") in str(info.value)


def test_memory_report_probes_use_the_requested_checkouts_toolchain(tmp_path: Path, monkeypatch) -> None:
    # analyze_elf(project_root=...) must reach arm_tool with that root for every binutils
    # call, so a custom checkout's downloaded toolchain is used rather than PATH.
    seen: list[tuple[str, Path | None]] = []

    def _fake_arm_tool(name: str, repo_root: Path | None = None) -> str:
        seen.append((name, repo_root))
        return "true"  # exits 0 with empty stdout

    monkeypatch.setattr(report, "arm_tool", _fake_arm_tool)
    assert report._probe_binary("arm-none-eabi-nm", ["ignored"], tmp_path) == ""
    assert seen == [("arm-none-eabi-nm", tmp_path)]
    import inspect
    source = inspect.getsource(report.analyze_elf)
    assert source.count("_probe_binary(") == 5 and source.count("], project_root)") == 5


def test_size_probe_is_board_keyed_and_builds_with_the_toolchain_on_path(report_env: Path, monkeypatch) -> None:
    # Two boards' probes in one checkout must not share a CMake cache, and the probe's
    # configure/build must see the downloaded ARM GCC on PATH (the build runs
    # generate_kernel_symbol_refs.py, whose arm-none-eabi-nm lookup is bare).
    runs: list[tuple[list[str], str]] = []

    def _fake_run(cmd, *, cwd, env=None):
        runs.append((cmd, (env or {}).get("PATH", "")))
        if cmd[:2] == ["cmake", "--build"]:
            out = Path(cmd[2]) / "probe"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{report.SIZE_PROBE_TARGET}.elf").write_bytes(b"elf")

    monkeypatch.setattr(report, "_run", _fake_run)
    variant = report.SIZE_PROBE_VARIANTS[0]
    out_dir = report.build_size_probe(resolve_board(DEFAULT_BOARD_ID), variant, project_root=report_env)
    board_keyed = report_env / "artifacts" / "hardware" / "size_probe" / DEFAULT_BOARD_ID / variant.name
    assert out_dir == board_keyed or board_keyed in out_dir.parents
    expected_bin = str(toolchain.toolchain_bin_dir(report_env).resolve())
    assert len(runs) == 2 and all(path.split(os.pathsep)[0] == expected_bin for _, path in runs)


def test_memory_report_fails_closed_when_a_board_region_is_missing(tmp_path: Path, report_env: Path, monkeypatch) -> None:
    # A mistyped flash_region/ram_region must be a configuration error, not a 0-byte
    # region that the 75 % gates silently pass.
    import dataclasses
    board = dataclasses.replace(resolve_board(DEFAULT_BOARD_ID), ram_region="MCU_TCM_TYPO")
    elf = tmp_path / "fw.elf"
    elf.write_bytes(b"elf")
    with pytest.raises(ValueError, match=r"defines no memory region\(s\) \['MCU_TCM_TYPO'\]; available regions: \['MCU_MRAM', 'MCU_TCM'\]"):
        report.analyze_elf(elf, board, report_env)
    # With both regions present the gates are computed against real capacities.
    usage = report.analyze_elf(elf, resolve_board(DEFAULT_BOARD_ID), report_env).usage
    assert usage["flash_capacity_bytes"] == 4128768 and usage["tcm_capacity_bytes"] == 507904
    assert usage["flash_gate_pass"] is True and usage["tcm_gate_pass"] is True


def test_write_text_lf_is_python38_safe_and_writes_lf(tmp_path: Path) -> None:
    # Path.write_text(newline=...) only exists from 3.10; the helper must not use it and
    # must still pin LF line endings.
    import inspect

    from helia_core_tester.hardware import pathutil

    body = inspect.getsource(pathutil.write_text_lf).replace(pathutil.write_text_lf.__doc__ or "", "")
    assert ".write_text(" not in body and 'newline="\\n"' in body
    target = tmp_path / "out.txt"
    pathutil.write_text_lf(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"
