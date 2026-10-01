"""Host-side helpers behind the hardware commands: where arm-none-eabi-* tools come
from (the lazily downloaded toolchain first), the shared path helpers, and
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
from helia_core_tester.tests.test_hardware_nsx_app import _write


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


def test_path_helpers_resolve_containment_and_display(tmp_path: Path) -> None:
    inside, outside = tmp_path / "repo" / "build" / "x.elf", Path("/elsewhere/build/x.elf")
    assert is_relative_to(inside, tmp_path / "repo") and not is_relative_to(outside, tmp_path / "repo")
    assert display_path(inside, tmp_path / "repo") == "build/x.elf"
    assert display_path(outside, tmp_path / "repo") == "/elsewhere/build/x.elf"
    # The containment check lives in pathutil.py alone so display_path() and the
    # firmware build agree on it; nothing else in the package rolls its own.
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
        lambda path: [
            {"name": "MCU_MRAM", "origin": 0x00410000, "capacity": 4128768},
            {"name": "MCU_TCM", "origin": 0x20000000, "capacity": 507904},
        ],
    )
    return repo


def _fake_build(build_dir: Path) -> Path:
    elf = build_dir / "hardware" / "hct_benchmark_server.elf"
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"elf")
    sdk = report.nsx_app_dir(build_dir) / "modules" / "nsx-ambiq-sdk"
    _write(report.linker_script_path(BOARD, sdk), "MEMORY {}\n")
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


@pytest.mark.parametrize("variant", report.SIZE_PROBE_VARIANTS, ids=lambda v: v.name)
def test_size_probe_builds_the_nsx_app(report_env: Path, monkeypatch, variant) -> None:
    # Board-keyed dir, probe render, probe target.
    from helia_core_tester.hardware import nsx_cli
    from helia_core_tester.hardware.nsx_app import SIZE_PROBE_TARGET

    calls: dict[str, object] = {}
    monkeypatch.setattr(report, "ensure_build_tools", lambda root: calls.setdefault("tools", root))

    def _stage(board, *, build_dir, options, repo_root):
        calls["stage"] = (build_dir, options)
        calls["repo_root"] = repo_root

    def _build(app_dir, *, board, build_dir, target, jobs, frozen):
        calls["build"] = (app_dir, build_dir, target)
        elf = build_dir / "probe" / f"{SIZE_PROBE_TARGET}.elf"
        elf.parent.mkdir(parents=True, exist_ok=True)
        elf.write_bytes(b"elf")

    monkeypatch.setattr(report, "stage_kernels", _stage)
    monkeypatch.setattr(nsx_cli, "configure_app", lambda app_dir, board, *, build_dir, frozen: calls.setdefault("configure", build_dir))
    monkeypatch.setattr(nsx_cli, "build_app", _build)
    monkeypatch.setattr(report, "app_linker_script", lambda board, build_dir: report_env / "fw.ld")

    out_dir = report.build_size_probe(BOARD, variant, project_root=report_env)

    assert out_dir == report_env / "artifacts" / "hardware" / "size_probe" / DEFAULT_BOARD_ID / variant.name
    build_dir = out_dir / "build"
    staged_dir, options = calls["stage"]
    assert staged_dir == build_dir and calls["configure"] == build_dir
    assert calls["repo_root"] == report_env
    assert options.build_size_probe and (options.enable_f32, options.enable_f16) == (variant.enable_f32, variant.enable_f16)
    assert calls["build"] == (report.nsx_app_dir(build_dir), build_dir, SIZE_PROBE_TARGET)
    data = json.loads((out_dir / "memory_report.json").read_text())
    assert data["variant"] == variant.name
    assert data["artifacts"]["elf"] == f"artifacts/hardware/size_probe/{DEFAULT_BOARD_ID}/{variant.name}/build/probe/{SIZE_PROBE_TARGET}.elf"


def test_size_probe_renders_from_the_requested_checkout(report_env: Path, monkeypatch) -> None:
    """The real staging path gets project_root."""
    from helia_core_tester.hardware import nsx_app, nsx_cli

    class _Rendered(Exception):
        pass

    def _render(board, options, app_dir, *, repo_root):
        raise _Rendered(repo_root)

    monkeypatch.setattr(report, "ensure_build_tools", lambda root: None)
    monkeypatch.setattr(nsx_app, "render_app", _render)
    monkeypatch.setattr(nsx_cli, "lock_is_current", lambda *a: True)
    with pytest.raises(_Rendered) as info:
        report.build_size_probe(BOARD, report.SIZE_PROBE_VARIANTS[0], project_root=report_env)
    assert info.value.args[0] == report_env


def test_memory_report_fails_closed_when_a_board_region_is_missing(tmp_path: Path, report_env: Path, monkeypatch) -> None:
    # A mistyped flash_region/ram_region must be a configuration error, not a 0-byte
    # region that the 75 % gates silently pass.
    import dataclasses
    board = dataclasses.replace(resolve_board(DEFAULT_BOARD_ID), ram_region="MCU_TCM_TYPO")
    elf = tmp_path / "fw.elf"
    elf.write_bytes(b"elf")
    with pytest.raises(ValueError, match=r"defines no memory region\(s\) \['MCU_TCM_TYPO'\]; available regions: \['MCU_MRAM', 'MCU_TCM'\]"):
        report.analyze_elf(elf, board, tmp_path / "fw.ld", report_env)
    # With both regions present the gates are computed against real capacities.
    usage = report.analyze_elf(elf, resolve_board(DEFAULT_BOARD_ID), tmp_path / "fw.ld", report_env).usage
    assert usage["flash_capacity_bytes"] == 4128768 and usage["ram_capacity_bytes"] == 507904
    assert usage["flash_gate_pass"] is True and usage["ram_gate_pass"] is True


def test_memory_regions_skip_block_comments(tmp_path: Path) -> None:
    # NSX scripts comment out whole rows.
    script = tmp_path / "fw.ld"
    script.write_text(
        "MEMORY\n{\n"
        "    MCU_TCM (rwx) : ORIGIN = 0x20000000, LENGTH = 245760\n"
        "    /* STACK (rw) : ORIGIN = 0x2007D000, LENGTH = 12288\n"
        "    HEAP (rw) : ORIGIN = 0x2007C000, LENGTH = 4096 */\n"
        "}\n",
        encoding="utf-8",
    )
    assert report.parse_memory_regions(script) == [{"name": "MCU_TCM", "origin": 0x20000000, "capacity": 245760}]


# Linked regions and `objdump -h` per board.
_REGIONS = {
    "apollo510_evb": [
        {"name": "MCU_ITCM", "origin": 0x00000000, "capacity": 262144},
        {"name": "MCU_MRAM", "origin": 0x00410000, "capacity": 4128768},
        {"name": "MCU_TCM", "origin": 0x20000000, "capacity": 507904},
        {"name": "SHARED_SRAM", "origin": 0x20080000, "capacity": 3145728},
    ],
    "apollo330mP_evb": [
        {"name": "MCU_MRAM", "origin": 0x00410000, "capacity": 2031616},
        {"name": "MCU_TCM", "origin": 0x20000000, "capacity": 245760},
        {"name": "SHARED_SRAM", "origin": 0x20080000, "capacity": 1835008},
    ],
    "apollo3p_evb": [
        {"name": "ROMEM", "origin": 0x0000C000, "capacity": 2048000},
        {"name": "RWMEM", "origin": 0x10011000, "capacity": 716800},
        {"name": "TCM", "origin": 0x10000000, "capacity": 65536},
        {"name": "STACKMEM", "origin": 0x10010000, "capacity": 4096},
    ],
}
_FIXTURES = Path(__file__).parent / "fixtures" / "memory_report"


def _analyze_board(board_id: str, report_env: Path, tmp_path: Path, monkeypatch, regions=None) -> report.ElfAnalysis:
    headers = (_FIXTURES / f"{board_id}.objdump_h.txt").read_text()
    monkeypatch.setattr(report, "_probe_binary", lambda tool, args, project_root=None: headers if args[0] == "-h" else "")
    monkeypatch.setattr(report, "parse_memory_regions", lambda path: regions or _REGIONS[board_id])
    elf = tmp_path / "fw.elf"
    elf.write_bytes(b"elf")
    return report.analyze_elf(elf, resolve_board(board_id), tmp_path / "fw.ld", report_env)


# Totals match the linked .bin and map.
@pytest.mark.parametrize(
    "board_id, region, flash, ram, heap, text",
    [
        # Vector table, ITCM code, exidx, .data.
        ("apollo510_evb", "MCU_TCM", 1024 + 28 + 672260 + 8 + 2096, 16384 + 2096 + 166852, 322572, 1024 + 672260),
        # DTCM code sits in flash and TCM.
        ("apollo330mP_evb", "MCU_TCM", 1024 + 28 + 650972 + 8 + 1960, 28 + 16388 + 1960 + 164180, 63196, 1024 + 650972),
        # Stack lives in STACKMEM, not RWMEM.
        ("apollo3p_evb", "RWMEM", 551884 + 8 + 1888, 1888 + 163988, 0, 551884),
    ],
)
def test_usage_matches_the_linked_image(board_id, region, flash, ram, heap, text, tmp_path, report_env, monkeypatch) -> None:
    analysis = _analyze_board(board_id, report_env, tmp_path, monkeypatch)
    usage = analysis.usage
    assert (usage["flash_image_bytes"], usage["ram_static_bytes"], usage["heap_available_bytes"]) == (flash, ram, heap)
    assert usage["ram_region"] == region and usage["flash_gate_pass"] and usage["ram_gate_pass"]
    assert not any(key.startswith("tcm_") for key in usage)
    # Both `.text` output sections count.
    assert analysis.sections[".text"] == text


def test_ram_gate_fails_over_75_percent(tmp_path, report_env, monkeypatch) -> None:
    # 182556 static bytes > 75 % of 240000.
    regions = [dict(row, capacity=240000) if row["name"] == "MCU_TCM" else row for row in _REGIONS["apollo330mP_evb"]]
    usage = _analyze_board("apollo330mP_evb", report_env, tmp_path, monkeypatch, regions).usage
    assert usage["ram_static_bytes"] == 182556 and usage["ram_gate_pass"] is False


def test_write_text_lf_writes_lf(tmp_path: Path) -> None:
    # Bundle artifacts are byte-compared across hosts, so the helper must pin LF
    # regardless of the platform's os.linesep.
    from helia_core_tester.hardware import pathutil

    target = tmp_path / "out.txt"
    pathutil.write_text_lf(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"


SCRIPT = "MEMORY {}\n"


def test_linker_script_comes_from_the_nsx_app(tmp_path: Path) -> None:
    build = tmp_path / "build"
    sdk = report.nsx_app_dir(build) / "modules" / "nsx-ambiq-sdk"
    expected = _write(report.linker_script_path(BOARD, sdk), SCRIPT)
    assert report.app_linker_script(BOARD, build) == expected


def test_linker_script_missing_names_the_fix(tmp_path: Path) -> None:
    # A pre-NSX build dir has no app.
    with pytest.raises(FileNotFoundError, match="rerun hardware build"):
        report.app_linker_script(BOARD, tmp_path / "old-build")
    # render_app made nsx_app/, sync failed.
    build = tmp_path / "build"
    report.nsx_app_dir(build).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="rerun hardware build"):
        report.app_linker_script(BOARD, build)


@pytest.mark.parametrize("compiler", [None, "/nonexistent/arm-none-eabi-gcc"])
def test_missing_gcc_reads_null(compiler) -> None:
    assert toolchain.gcc_version(compiler) is None
