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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    # The size probe renders a real app, which resolves the repo's own baseline.
    (repo / "assets").mkdir(parents=True)
    (repo / "assets" / "dependency_baseline.json").write_text(
        (PROJECT_ROOT / "assets" / "dependency_baseline.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "repo_root", lambda: repo)
    monkeypatch.setattr(report, "_probe_binary", lambda tool, args, project_root=None: "")
    # The real linker script comes out of the synced NSX SDK module, which only a
    # real build produces; the regions it would yield are supplied directly.
    monkeypatch.setattr(report, "linker_script_path", lambda board, build_dir: repo / "fake.ld")
    monkeypatch.setattr(
        report, "parse_memory_regions",
        lambda path: [{"name": "MCU_MRAM", "capacity": 4128768}, {"name": "MCU_TCM", "capacity": 507904}],
    )
    return repo


def _fake_build(build_dir: Path) -> Path:
    """A build dir with just the linked image NSX would have produced in it."""
    from helia_core_tester.hardware.firmware_build import elf_path

    elf = elf_path(build_dir, BOARD)
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"elf")
    return elf


def test_memory_report_paths_are_repo_relative_inside_and_absolute_outside(tmp_path: Path, report_env) -> None:
    repo = report_env
    inside_build_dir = repo / "build" / "hardware" / "apollo510_evb"
    _fake_build(inside_build_dir)
    data = json.loads(report.generate_memory_report(BOARD,
        build_dir=inside_build_dir, output_root=tmp_path / "out_in").read_text())
    nested = "build/hardware/apollo510_evb/nsx_app/build/apollo510_evb"
    assert data["artifacts"] == {
        "elf": f"{nested}/hct_benchmark_server.elf",
        "bin": f"{nested}/hct_benchmark_server.bin",
        "map": f"{nested}/hct_benchmark_server.map",
    }

    external = _fake_build(tmp_path / "extbuild")
    data = json.loads(report.generate_memory_report(BOARD,
        build_dir=tmp_path / "extbuild", output_root=tmp_path / "out_ext").read_text())
    assert data["artifacts"]["elf"] == str(external)
    assert data["artifacts"]["bin"] == str(external.with_suffix(".bin"))
    assert data["artifacts"]["map"] == str(external.with_suffix(".map"))


def test_memory_report_records_the_builds_own_provenance(tmp_path: Path, report_env) -> None:
    """The report must describe the ELF in front of it, so its baseline and kernel
    source come from the build dir's render state, not from a fresh resolve."""
    from helia_core_tester.hardware.nsx_app import RENDER_STATE, app_dir_for

    build_dir = report_env / "build" / "hardware" / "apollo510_evb"
    _fake_build(build_dir)
    state = {
        "baseline_id": "recorded-baseline",
        "baseline_fingerprint": "f" * 64,
        "kernel_source": "ns-cmsis-nn@" + "a" * 40,
        "kernel_options": {"NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM": "ON"},
    }
    app_dir = app_dir_for(build_dir)
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / RENDER_STATE).write_text(json.dumps(state), encoding="utf-8")

    data = json.loads(report.generate_memory_report(
        BOARD, build_dir=build_dir, output_root=tmp_path / "out").read_text())
    assert data["baseline_id"] == "recorded-baseline"
    assert data["kernel_source"] == state["kernel_source"]
    assert data["kernel_options"] == state["kernel_options"]


def test_memory_report_names_the_missing_elf(tmp_path: Path, report_env) -> None:
    with pytest.raises(FileNotFoundError, match="Built firmware ELF not found") as info:
        report.generate_memory_report(BOARD, build_dir=tmp_path / "never-built", output_root=tmp_path / "out")
    from helia_core_tester.hardware.firmware_build import elf_path

    assert str(elf_path(tmp_path / "never-built", BOARD)) in str(info.value)


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


def test_size_probe_is_board_keyed_and_reuses_the_app_render(report_env: Path, monkeypatch) -> None:
    # Two boards' probes in one checkout must not share a build tree, and the probe
    # must be built by the same renderer as the firmware -- with the variant's float
    # switches -- rather than a hand-mirrored -D list that can drift from it.
    from helia_core_tester.hardware import firmware_build, nsx_app

    built: list[tuple[Path, str]] = []
    monkeypatch.setattr(firmware_build, "ensure_host_tools", lambda repo_root, baseline=None: None)
    monkeypatch.setattr(firmware_build, "lock_and_sync", lambda render, options: None)
    monkeypatch.setattr(firmware_build, "configure", lambda render, options, **kw: None)

    def _fake_build(render, options, target, jobs):
        built.append((render.app_dir, target))
        render.build_dir.mkdir(parents=True, exist_ok=True)
        (render.build_dir / f"{target}.elf").write_bytes(b"elf")

    monkeypatch.setattr(firmware_build, "build", _fake_build)

    variant = report.SIZE_PROBE_VARIANTS[0]
    board = resolve_board(DEFAULT_BOARD_ID)
    out_dir = report.build_size_probe(board, variant, project_root=report_env)

    board_keyed = report_env / "artifacts" / "hardware" / "size_probe" / DEFAULT_BOARD_ID / variant.name
    assert out_dir == board_keyed or board_keyed in out_dir.parents
    [(app_dir, target)] = built
    assert target == report.SIZE_PROBE_TARGET
    # Its own app, a sibling of the firmware's under the same board build dir.
    probe_build_dir = report.size_probe_build_dir(board.build_dir(report_env), variant)
    assert app_dir == nsx_app.app_dir_for(probe_build_dir)
    assert app_dir != nsx_app.app_dir_for(board.build_dir(report_env))

    cmakelists = (app_dir / "CMakeLists.txt").read_text()
    assert f"add_executable({report.SIZE_PROBE_TARGET}" in cmakelists
    assert 'set(ARM_NN_ENABLE_F32 "OFF" CACHE STRING' in cmakelists

    data = json.loads((board_keyed / "memory_report.json").read_text())
    assert data["variant"] == variant.name
    assert data["feature_set"] == {"integer": True, "f32": False, "f16": False}


def test_memory_report_fails_closed_when_a_board_region_is_missing(tmp_path: Path, report_env: Path, monkeypatch) -> None:
    # A mistyped flash_region/ram_region must be a configuration error, not a 0-byte
    # region that the 75 % gates silently pass.
    import dataclasses
    board = dataclasses.replace(resolve_board(DEFAULT_BOARD_ID), ram_region="MCU_TCM_TYPO")
    elf = tmp_path / "fw.elf"
    elf.write_bytes(b"elf")
    with pytest.raises(ValueError, match=r"defines no memory region\(s\) \['MCU_TCM_TYPO'\]; available regions: \['MCU_MRAM', 'MCU_TCM'\]"):
        report.analyze_elf(elf, board, report_env / "bd", report_env)
    # With both regions present the gates are computed against real capacities.
    usage = report.analyze_elf(elf, resolve_board(DEFAULT_BOARD_ID), report_env / "bd", report_env).usage
    assert usage["flash_capacity_bytes"] == 4128768 and usage["tcm_capacity_bytes"] == 507904
    assert usage["flash_gate_pass"] is True and usage["tcm_gate_pass"] is True


def test_write_text_lf_writes_lf(tmp_path: Path) -> None:
    # Bundle artifacts are byte-compared across hosts, so the helper must pin LF
    # regardless of the platform's os.linesep.
    from helia_core_tester.hardware import pathutil

    target = tmp_path / "out.txt"
    pathutil.write_text_lf(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"


def test_size_probe_report_carries_the_same_provenance_as_the_firmware_report(
    report_env: Path, monkeypatch
) -> None:
    """A probe number is only comparable with a firmware number if both say which
    baseline and which kernel switches produced them."""
    from helia_core_tester.hardware import firmware_build, nsx_app

    monkeypatch.setattr(firmware_build, "ensure_host_tools", lambda repo_root, baseline=None: None)
    monkeypatch.setattr(firmware_build, "lock_and_sync", lambda render, options: None)
    monkeypatch.setattr(firmware_build, "configure", lambda render, options, **kw: None)

    def _fake_build(render, options, target, jobs):
        render.build_dir.mkdir(parents=True, exist_ok=True)
        (render.build_dir / f"{target}.elf").write_bytes(b"elf")

    monkeypatch.setattr(firmware_build, "build", _fake_build)

    variant = report.SIZE_PROBE_VARIANTS[0]
    board = resolve_board(DEFAULT_BOARD_ID)
    out_dir = report.build_size_probe(board, variant, project_root=report_env)
    data = json.loads((out_dir / "memory_report.json").read_text())

    for key in ("baseline_id", "baseline_fingerprint", "kernel_source", "kernel_options"):
        assert data.get(key), f"{key} missing from the size-probe report"
    assert data["kernel_options"]["ARM_NN_ENABLE_F32"] == "OFF"
    assert len(data["baseline_fingerprint"]) == 64

    # And the probe records its render, so the next run reuses the lock instead of
    # re-resolving it every time.
    probe_build_dir = report.size_probe_build_dir(board.build_dir(report_env), variant)
    state = nsx_app.read_render_state(nsx_app.app_dir_for(probe_build_dir))
    assert state is not None and state["kernel_options"]["ARM_NN_ENABLE_F32"] == "OFF"
