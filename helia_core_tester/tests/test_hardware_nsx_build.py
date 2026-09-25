"""`hardware build` drives NSX: render, lock, sync, configure, build.

Every nsx_cli step is monkeypatched; nothing here runs NSX or CMake.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from neuralspotx.nsx_lock import hash_manifest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware import firmware_build, hardware_pipeline, nsx_app, nsx_cli
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.jlink_library import JLinkExecutable, JLinkLibraryError
from helia_core_tester.hardware.nsx_app import AppOptions

BOARD = resolve_board("apollo510_evb")
SERIAL = 1160003180


def _write_lock(app_dir: Path) -> None:
    """Minimal nsx.lock for the current nsx.yml."""
    manifest = {"path": "nsx.yml", "hash": hash_manifest(app_dir / "nsx.yml")}
    lock = {"schema_version": 4, "targets": {BOARD.nsx_board: {"manifest": manifest, "modules": {}}}}
    (app_dir / "nsx.lock").write_text(yaml.safe_dump(lock), encoding="utf-8")


@pytest.fixture
def nsx(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record each NSX step; fake its on-disk effect."""
    calls: list[tuple] = []
    real_render = nsx_app.render_app

    def render_app(board, options, app_dir, **kwargs):
        calls.append(("render", options))
        return real_render(board, options, app_dir, **kwargs)

    def lock_app(app_dir, *, update=False):
        calls.append(("lock", update))
        _write_lock(app_dir)

    def sync_app(app_dir, *, frozen=False):
        calls.append(("sync", frozen))
        (app_dir / "modules").mkdir(exist_ok=True)

    def configure_app(app_dir, board, *, build_dir, probe_serial=None, frozen=False):
        calls.append(("configure", probe_serial, frozen))
        (build_dir / "build.ninja").write_text("", encoding="utf-8")
        (build_dir / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={app_dir.resolve()}\n"
            f"NSX_BOARD:STRING={board}\n"
            f"NSX_JLINK_SERIAL:UNINITIALIZED={probe_serial or ''}\n",
            encoding="utf-8",
        )

    def build_app(app_dir, *, board, build_dir, jobs=None, frozen=False):
        calls.append(("build", jobs, frozen))

    monkeypatch.setattr(firmware_build, "ensure_build_tools", lambda repo_root: None)
    monkeypatch.setattr(firmware_build, "find_jlink_exe", lambda: None)
    monkeypatch.setattr(nsx_cli, "starter_profile", lambda board: {"modules": ["nsx-core"]})
    monkeypatch.setattr(nsx_app, "render_app", render_app)
    for fake in (lock_app, sync_app, configure_app, build_app):
        monkeypatch.setattr(nsx_cli, fake.__name__, fake)
    return calls


def _steps(calls: list[tuple]) -> list[str]:
    return [call[0] for call in calls]


def test_first_build_runs_every_step_in_order(tmp_path: Path, nsx: list[tuple]) -> None:
    elf = firmware_build.build_firmware(BOARD, build_dir=tmp_path, jobs=4)
    assert elf == firmware_build.elf_path(tmp_path)
    assert nsx == [
        ("render", AppOptions()),
        ("lock", False),
        ("sync", False),
        ("configure", None, True),
        ("build", 4, True),
    ]
    assert (firmware_build.nsx_app_dir(tmp_path) / "nsx.yml").is_file()


def test_unchanged_rebuild_skips_lock_and_configure(tmp_path: Path, nsx: list[tuple]) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert _steps(nsx) == ["render", "sync", "build"]
    assert ("sync", True) in nsx


def test_manifest_change_relocks_and_syncs_unfrozen(tmp_path: Path, nsx: list[tuple]) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, options=AppOptions(cmsis_nn_ref="v1.2.3"))
    assert nsx[1:3] == [("lock", False), ("sync", False)]


def test_update_dependencies_forces_lock_update(tmp_path: Path, nsx: list[tuple]) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, update_dependencies=True)
    assert nsx[1:3] == [("lock", True), ("sync", False)]


def test_local_kernel_root_relocks_every_build(tmp_path: Path, nsx: list[tuple]) -> None:
    """Edits to a local checkout leave nsx.yml unchanged."""
    options = AppOptions(cmsis_nn_root=tmp_path / "kernels")
    build_dir = tmp_path / "build"
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    assert _steps(nsx) == ["render", "lock", "sync", "build"]
    assert ("sync", False) in nsx


def test_serial_change_or_force_reconfigures(tmp_path: Path, nsx: list[tuple]) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, serial_no=SERIAL)
    assert ("configure", SERIAL, True) in nsx
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, serial_no=SERIAL)
    assert "configure" not in _steps(nsx)
    # Build without a probe keeps the cached one.
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert "configure" not in _steps(nsx)
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, serial_no=7)
    assert ("configure", 7, True) in nsx
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, serial_no=7, force_reconfigure=True)
    assert ("configure", 7, True) in nsx


def test_old_path_cache_is_dropped(tmp_path: Path, nsx: list[tuple]) -> None:
    """The root-CMakeLists cache would block NSX's configure."""
    (tmp_path / "CMakeFiles").mkdir()
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/old/checkout\n", encoding="utf-8")
    (tmp_path / "build.ninja").write_text("", encoding="utf-8")
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert "configure" in _steps(nsx)
    assert not (tmp_path / "CMakeFiles").exists()


def test_configure_hands_nsx_the_resolved_jlinkexe(tmp_path: Path, nsx: list[tuple], monkeypatch) -> None:
    monkeypatch.delenv("JLINK_PATH", raising=False)
    found = JLinkExecutable("/opt/SEGGER/JLink/JLinkExe", "next to $HPX_JLINK_DLL")
    monkeypatch.setattr(firmware_build, "find_jlink_exe", lambda: found)
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert os.environ["JLINK_PATH"] == "/opt/SEGGER/JLink/JLinkExe"

    # A broken $HPX_JLINK_DLL must not stop a build.
    def _broken():
        raise JLinkLibraryError("$HPX_JLINK_DLL=/x/gone.so does not exist")

    monkeypatch.setattr(firmware_build, "find_jlink_exe", _broken)
    firmware_build.build_firmware(BOARD, build_dir=tmp_path / "other")


def test_flash_forwards_app_options(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(firmware_build, "build_firmware", lambda board, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(firmware_build, "build", lambda build_dir, target, jobs: None)
    elf = firmware_build.elf_path(tmp_path)
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"fw")
    options = AppOptions(enable_f32=False)
    firmware_build.flash_firmware(BOARD, SERIAL, build_dir=tmp_path, options=options, update_dependencies=True)
    assert seen["options"] is options and seen["update_dependencies"] is True and seen["serial_no"] == SERIAL


# --- CLI flags ---------------------------------------------------------------------

runner = CliRunner()


def test_build_flags_reach_app_options(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(firmware_build, "build_firmware", lambda board, **kwargs: seen.update(kwargs) or tmp_path)
    result = runner.invoke(app, [
        "hardware", "build", "--cmsis-nn-root", str(tmp_path), "--no-f16", "--no-inline-asm",
        "--update-dependencies", "--force-reconfigure", "-j", "3",
    ])
    assert result.exit_code == 0, result.output
    assert seen["options"] == AppOptions(cmsis_nn_root=tmp_path, enable_f16=False, requantize_inline_asm=False)
    assert seen["update_dependencies"] is True and seen["force_reconfigure"] is True and seen["jobs"] == 3


def test_run_flags_reach_the_pipeline(monkeypatch) -> None:
    seen: dict = {}

    def _pipeline(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setenv("HPX_JLINK_SERIAL", str(SERIAL))
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _pipeline)
    result = runner.invoke(app, ["hardware", "run", "--skip-generate", "--cmsis-nn-ref", "v1.0.0", "--no-f32"])
    assert result.exit_code == 1
    assert seen["app_options"] == AppOptions(cmsis_nn_ref="v1.0.0", enable_f32=False)
    assert seen["update_dependencies"] is False


def test_kernel_ref_and_root_are_exclusive(tmp_path: Path) -> None:
    result = runner.invoke(app, ["hardware", "build", "--cmsis-nn-ref", "v1", "--cmsis-nn-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "not both" in result.output
