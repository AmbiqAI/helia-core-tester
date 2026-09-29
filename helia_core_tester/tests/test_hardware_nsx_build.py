"""`hardware build` drives NSX: render, lock, sync, configure, build.

Every nsx_cli step is monkeypatched; nothing here runs NSX or CMake.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest
from neuralspotx.nsx_lock import LockKind, NsxLock, ResolvedModule, hash_manifest, hash_tree, write_lock
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware import cli as hardware_cli
from helia_core_tester.hardware import firmware_build, hardware_pipeline, nsx_app, nsx_cli
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.hardware_pipeline import HardwareRunOutcome, StreamOptions
from helia_core_tester.hardware.jlink_library import JLinkExecutable, JLinkLibraryError
from helia_core_tester.hardware.nsx_app import AppOptions
from helia_core_tester.tests.test_hardware_nsx_app import make_checkout

BOARD = resolve_board("apollo510_evb")
SERIAL = 1160003180


def _write_lock(app_dir: Path) -> None:
    """Minimal nsx.lock, vendored kernels hashed."""
    lock = NsxLock(manifest_hash=hash_manifest(app_dir / "nsx.yml"))
    lock.modules["nsx-core"] = ResolvedModule(
        project="nsx-ambiq-sdk", kind=LockKind.GIT, constraint="v1", vendored_at="modules/nsx-ambiq-sdk",
        content_hash="sha256:0", acquired_at="", url="https://example.invalid/sdk.git", commit="0" * 40,
    )
    kernels = app_dir / "modules" / nsx_app.CMSIS_NN_MODULE
    if kernels.is_dir():
        lock.modules[nsx_app.CMSIS_NN_MODULE] = ResolvedModule(
            project=nsx_app.CMSIS_NN_MODULE, kind=LockKind.VENDORED, constraint="vendored",
            vendored_at=f"modules/{nsx_app.CMSIS_NN_MODULE}", content_hash=hash_tree(kernels), acquired_at="",
        )
    write_lock(app_dir, lock, board=BOARD.nsx_board)


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
        (app_dir / "modules" / "nsx-ambiq-sdk").mkdir(parents=True, exist_ok=True)

    def configure_app(app_dir, board, *, build_dir, frozen=False):
        calls.append(("configure", frozen))
        (build_dir / "build.ninja").write_text("", encoding="utf-8")
        (build_dir / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={app_dir.resolve()}\nNSX_BOARD:STRING={board}\n", encoding="utf-8",
        )

    def build_app(app_dir, *, board, build_dir, jobs=None, frozen=False):
        calls.append(("build", jobs, frozen))

    monkeypatch.setattr(firmware_build, "ensure_build_tools", lambda repo_root: None)
    _use_jlink(monkeypatch, None)
    monkeypatch.setattr(nsx_cli, "starter_profile", lambda board: {"modules": ["nsx-core"]})
    monkeypatch.setattr(nsx_app, "render_app", render_app)
    for fake in (lock_app, sync_app, configure_app, build_app):
        monkeypatch.setattr(nsx_cli, fake.__name__, fake)
    return calls


def _use_jlink(monkeypatch, path: str | None) -> None:
    found = None if path is None else JLinkExecutable(path, "$JLINK_PATH")
    monkeypatch.setattr(firmware_build, "find_jlink_exe", lambda: found)


def _steps(calls: list[tuple]) -> list[str]:
    return [call[0] for call in calls]


def _cli(build_dir: Path, command: str, *flags: str) -> str:
    serial = [] if command == "build" else ["--serial-no", str(SERIAL)]
    result = runner.invoke(app, ["hardware", command, "--build-dir", str(build_dir), *serial, *flags])
    assert result.exit_code == 0, result.output
    return result.output


def test_first_build_runs_every_step_in_order(tmp_path: Path, nsx: list[tuple]) -> None:
    elf = firmware_build.build_firmware(BOARD, build_dir=tmp_path, jobs=4)
    assert elf == firmware_build.elf_path(tmp_path)
    assert nsx == [
        ("render", AppOptions()),
        ("lock", False),
        ("sync", False),
        ("configure", True),
        ("build", 4, True),
    ]
    assert (firmware_build.nsx_app_dir(tmp_path) / "nsx.yml").is_file()


def test_unchanged_rebuild_skips_lock_sync_configure(tmp_path: Path, nsx: list[tuple]) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert _steps(nsx) == ["render", "build"]
    # Ninja's default job count.
    assert nsx[-1] == ("build", (os.cpu_count() or 6) + 2, True)


def test_missing_module_resyncs(tmp_path: Path, nsx: list[tuple]) -> None:
    # The stamp alone cannot prove the tree.
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    shutil.rmtree(firmware_build.nsx_app_dir(tmp_path) / "modules" / "nsx-ambiq-sdk")
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert "sync" in _steps(nsx)
    assert (firmware_build.nsx_app_dir(tmp_path) / "modules" / "nsx-ambiq-sdk").is_dir()


def test_neuralspotx_upgrade_resyncs(tmp_path: Path, nsx: list[tuple], monkeypatch) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    monkeypatch.setattr(nsx_cli.metadata, "version", lambda name: "99.0.0")
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert _steps(nsx) == ["render", "sync", "configure", "build"] and ("sync", False) in nsx


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


def test_local_kernel_root_relocks_only_on_edits(tmp_path: Path, nsx: list[tuple]) -> None:
    """The lock's vendored hash drives relock."""
    kernels = make_checkout(tmp_path / "kernels")
    options = AppOptions(cmsis_nn_root=kernels)
    build_dir = tmp_path / "build"
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    vendored = firmware_build.nsx_app_dir(build_dir) / "modules" / "nsx-cmsis-nn"
    assert (vendored / "Source" / "arm_add.c").is_file()
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    assert _steps(nsx) == ["render", "build"]

    edited = kernels / "Source" / "arm_add.c"
    edited.write_text("int add; // edit\n", encoding="utf-8")
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    assert _steps(nsx) == ["render", "lock", "sync", "configure", "build"]
    assert ("sync", False) in nsx
    assert (vendored / "Source" / "arm_add.c").read_text(encoding="utf-8") == "int add; // edit\n"


def test_new_kernel_file_reconfigures(tmp_path: Path, nsx: list[tuple]) -> None:
    # ns-cmsis-nn globs sources at configure.
    kernels = make_checkout(tmp_path / "kernels")
    options = AppOptions(cmsis_nn_root=kernels)
    build_dir = tmp_path / "build"
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    (kernels / "Source" / "arm_zzz_s8.c").write_text("int zzz;\n", encoding="utf-8")
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    assert "configure" in _steps(nsx)
    vendored = firmware_build.nsx_app_dir(build_dir) / "modules" / "nsx-cmsis-nn"
    assert (vendored / "Source" / "arm_zzz_s8.c").is_file()


def test_build_dir_inside_kernel_root_builds(tmp_path: Path, nsx: list[tuple]) -> None:
    """The nested default build dir is fine."""
    kernels = make_checkout(tmp_path / "kernels")
    build_dir = kernels / "Tests" / "helia-core-tester" / "build" / "hardware" / BOARD.id
    options = AppOptions(cmsis_nn_root=kernels)
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=build_dir, options=options)
    assert _steps(nsx) == ["render", "build"]
    assert not (firmware_build.nsx_app_dir(build_dir) / "modules" / "nsx-cmsis-nn" / "Tests").exists()


def test_force_reconfigure_resyncs(tmp_path: Path, nsx: list[tuple]) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, force_reconfigure=True)
    assert _steps(nsx) == ["render", "sync", "configure", "build"]


def test_changed_options_are_named(tmp_path: Path, nsx: list[tuple]) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert nsx_app.saved_options(firmware_build.nsx_app_dir(tmp_path)) == AppOptions()
    out = _cli(tmp_path, "build")
    assert "Options changed" not in out and "inline asm on" in out
    out = _cli(tmp_path, "build", "--no-inline-asm")
    assert "Options changed, rebuilding: requantize inline asm on -> off" in out
    assert "Kernels: ns-cmsis-nn v7.35.1, inline asm off" in out
    assert nsx_app.saved_options(firmware_build.nsx_app_dir(tmp_path)) == AppOptions(requantize_inline_asm=False)


def test_template_edit_is_not_an_options_change(tmp_path: Path, nsx: list[tuple]) -> None:
    """Values, not rendered text, are compared."""
    _cli(tmp_path, "build")
    (firmware_build.nsx_app_dir(tmp_path) / "CMakeLists.txt").write_text("# older template\n", encoding="utf-8")
    out = _cli(tmp_path, "build")
    assert "Options changed" not in out and "No saved build options" not in out


def test_failed_build_keeps_saved_options(tmp_path: Path, nsx: list[tuple], monkeypatch) -> None:
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)

    def broken(app_dir, **kwargs):
        raise RuntimeError("compile error")

    monkeypatch.setattr(nsx_cli, "build_app", broken)
    with pytest.raises(RuntimeError):
        firmware_build.build_firmware(BOARD, build_dir=tmp_path, options=AppOptions(requantize_inline_asm=False))
    assert nsx_app.saved_options(firmware_build.nsx_app_dir(tmp_path)) == AppOptions()


def test_unsaved_build_dir_says_defaults(tmp_path: Path, nsx: list[tuple]) -> None:
    """Build dirs from before saved options."""
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    (firmware_build.nsx_app_dir(tmp_path) / nsx_app.OPTIONS_FILE).unlink()
    assert "No saved build options; using defaults." in _cli(tmp_path, "build")


def test_options_round_trip_json(tmp_path: Path) -> None:
    options = AppOptions(cmsis_nn_root=make_checkout(tmp_path / "k"), requantize_inline_asm=False)
    assert AppOptions.from_json(options.to_json()) == options
    assert AppOptions.from_json(AppOptions().to_json()) == AppOptions()


def test_old_path_cache_is_dropped(tmp_path: Path, nsx: list[tuple]) -> None:
    """The root-CMakeLists cache would block NSX's configure."""
    (tmp_path / "CMakeFiles").mkdir()
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/old/checkout\n", encoding="utf-8")
    (tmp_path / "build.ninja").write_text("", encoding="utf-8")
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert "configure" in _steps(nsx)
    assert not (tmp_path / "CMakeFiles").exists()


def test_flash_goes_through_nsx(tmp_path: Path, monkeypatch) -> None:
    """Scoped JLINK_PATH, like hpx's nsx flash."""
    seen: dict = {}
    monkeypatch.setattr(firmware_build, "build_firmware", lambda board, **kwargs: seen.update(kwargs))

    def flash_app(app_dir, **kwargs):
        seen["flash"] = dict(kwargs, app_dir=app_dir, jlink=os.environ.get("JLINK_PATH"))

    monkeypatch.setattr(nsx_cli, "flash_app", flash_app)
    monkeypatch.setenv("JLINK_PATH", "/etc/jlink/JLinkExe")
    _use_jlink(monkeypatch, "/opt/a/JLinkExe")
    elf = firmware_build.elf_path(tmp_path)
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"fw")
    options = AppOptions(enable_f32=False)
    firmware_build.flash_firmware(
        BOARD, SERIAL, build_dir=tmp_path, options=options, update_dependencies=True, jobs=5,
    )
    assert seen["options"] is options and seen["update_dependencies"] is True and seen["jobs"] == 5
    assert seen["flash"] == {
        "app_dir": firmware_build.nsx_app_dir(tmp_path), "board": BOARD.nsx_board, "build_dir": tmp_path,
        "target": firmware_build.SERVER_TARGET, "probe_serial": SERIAL, "jobs": 5, "jlink": "/opt/a/JLinkExe",
    }
    assert os.environ["JLINK_PATH"] == "/etc/jlink/JLinkExe"

    # A broken $HPX_JLINK_DLL must not stop a flash.
    def _broken():
        raise JLinkLibraryError("$HPX_JLINK_DLL=/x/gone.so does not exist")

    monkeypatch.setattr(firmware_build, "find_jlink_exe", _broken)
    firmware_build.flash_firmware(BOARD, SERIAL, build_dir=tmp_path, force=True)
    assert seen["flash"]["jlink"] == "/etc/jlink/JLinkExe"


# --- hardware run: kernel root for generation ---------------------------------------


def _run(tmp_path: Path, monkeypatch, nsx: list[tuple], repo_root: Path, **kwargs) -> Path:
    """Pipeline with NSX, generate, flash, stream faked."""
    build_dir = tmp_path / "build"

    def _generate(repo_root, spec, suite, float_precision=None, cmsis_nn_root=None):
        nsx.append(("generate", cmsis_nn_root))

    def _flash(spec, serial, *, force, **build_kwargs):
        firmware_build.build_firmware(spec, **build_kwargs)
        return firmware_build.FlashDecision(False, "digest", "test")

    def _stream(*args, **kw):
        return HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[])

    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", _generate)
    monkeypatch.setattr(hardware_pipeline, "flash_firmware", _flash)
    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _stream)
    hardware_pipeline.run_hardware_pipeline(
        repo_root, BOARD, SERIAL, options=StreamOptions(), build_dir=build_dir, echo=lambda _msg: None, **kwargs,
    )
    return build_dir


def test_run_generates_from_the_given_root(tmp_path: Path, nsx: list[tuple], monkeypatch) -> None:
    kernels = make_checkout(tmp_path / "kernels")
    options = AppOptions(cmsis_nn_root=kernels)
    _run(tmp_path, monkeypatch, nsx, tmp_path, app_options=options)
    # Build restages as a no-op.
    assert _steps(nsx) == ["render", "lock", "sync", "generate", "render", "configure", "build"]
    assert ("generate", kernels) in nsx

    # One forced sync, then reconfigure.
    nsx.clear()
    _run(tmp_path, monkeypatch, nsx, tmp_path, app_options=options, force_reconfigure=True, update_dependencies=True)
    assert _steps(nsx) == ["render", "lock", "sync", "generate", "render", "configure", "build"]
    assert ("lock", True) in nsx


def test_run_generates_from_the_synced_ref(tmp_path: Path, nsx: list[tuple], monkeypatch) -> None:
    build_dir = _run(tmp_path, monkeypatch, nsx, tmp_path, app_options=AppOptions(cmsis_nn_ref="v7.35.1"))
    clone = firmware_build.nsx_app_dir(build_dir) / "modules" / nsx_app.CMSIS_NN_PROJECT
    assert _steps(nsx) == ["render", "lock", "sync", "generate", "render", "configure", "build"]
    assert ("generate", clone) in nsx


def test_run_defaults_to_the_enclosing_checkout(tmp_path: Path, nsx: list[tuple], monkeypatch) -> None:
    kernels, tester = _nested_layout(tmp_path)
    _run(tmp_path, monkeypatch, nsx, tester)
    assert ("render", AppOptions(cmsis_nn_root=kernels.resolve())) in nsx
    assert ("generate", kernels.resolve()) in nsx


def test_skip_generate_stages_once(tmp_path: Path, nsx: list[tuple], monkeypatch) -> None:
    _run(tmp_path, monkeypatch, nsx, tmp_path, app_options=AppOptions(), skip_generate=True)
    assert _steps(nsx) == ["render", "lock", "sync", "configure", "build"]


# --- CLI flags ---------------------------------------------------------------------

runner = CliRunner()


def test_build_flags_reach_app_options(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(firmware_build, "build_firmware", lambda board, **kwargs: seen.update(kwargs) or tmp_path)
    result = runner.invoke(app, [
        "hardware", "build", "--cmsis-nn-root", str(tmp_path), "--no-inline-asm",
        "--update-dependencies", "--force-reconfigure", "-j", "3",
    ])
    assert result.exit_code == 0, result.output
    assert seen["options"] == AppOptions(cmsis_nn_root=tmp_path, requantize_inline_asm=False)
    assert seen["update_dependencies"] is True and seen["force_reconfigure"] is True and seen["jobs"] == 3


def test_run_flags_reach_the_pipeline(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}

    def _pipeline(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setenv("HPX_JLINK_SERIAL", str(SERIAL))
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _pipeline)
    result = runner.invoke(app, [
        "hardware", "run", "--skip-generate", "--cmsis-nn-ref", "v1.0.0", "--build-dir", str(tmp_path),
    ])
    assert result.exit_code == 1
    assert seen["app_options"] == AppOptions(cmsis_nn_ref="v1.0.0")
    assert seen["update_dependencies"] is False


def _nested_layout(tmp_path: Path) -> tuple[Path, Path]:
    """ns-cmsis-nn/Tests/helia-core-tester on disk."""
    kernels = make_checkout(tmp_path / "ns-cmsis-nn")
    return kernels, kernels / "Tests" / "helia-core-tester"


@pytest.mark.parametrize("nested", [True, False])
def test_default_kernels_follow_layout(tmp_path: Path, monkeypatch, nested: bool) -> None:
    kernels, tester = _nested_layout(tmp_path)
    if not nested:
        (kernels / "nsx" / "nsx-module.yaml").unlink()
    seen: dict = {}
    monkeypatch.setattr(firmware_build, "build_firmware", lambda board, **kwargs: seen.update(kwargs) or tmp_path)
    monkeypatch.setattr(hardware_cli, "repo_root", lambda: tester)
    result = runner.invoke(app, ["hardware", "build", "--build-dir", str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    assert seen["options"] == AppOptions(cmsis_nn_root=kernels.resolve() if nested else None)
    # An explicit ref still wins.
    runner.invoke(app, ["hardware", "build", "--build-dir", str(tmp_path / "b"), "--cmsis-nn-ref", "v1"])
    assert seen["options"] == AppOptions(cmsis_nn_ref="v1")


def test_kernel_ref_and_root_are_exclusive(tmp_path: Path) -> None:
    result = runner.invoke(app, ["hardware", "build", "--cmsis-nn-ref", "v1", "--cmsis-nn-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "not both" in result.output


# --- saved options: flash what was built --------------------------------------------


@pytest.fixture
def bench(tmp_path: Path, nsx: list[tuple], monkeypatch) -> dict:
    """Fake image per render; fake board."""
    board: dict = {}

    def build_app(app_dir, *, board, build_dir, jobs=None, frozen=False):
        nsx.append(("build", jobs, frozen))
        image = (app_dir / "CMakeLists.txt").read_bytes()
        firmware_build.elf_path(build_dir).parent.mkdir(parents=True, exist_ok=True)
        firmware_build.elf_path(build_dir).write_bytes(image)
        firmware_build.build_id_path(build_dir).write_text(f"hct-{hashlib.sha256(image).hexdigest()[:8]}\n", encoding="utf-8")

    def flash_app(app_dir, *, build_dir, **kwargs):
        board["id"] = firmware_build.read_build_id(build_dir)

    def reader(spec, serial, build_dir):
        return board["id"]

    real_flash = firmware_build.flash_firmware
    monkeypatch.setattr(nsx_cli, "build_app", build_app)
    monkeypatch.setattr(nsx_cli, "flash_app", flash_app)
    monkeypatch.setattr(
        firmware_build, "flash_firmware", lambda *a, **k: real_flash(*a, board_build_id_reader=reader, **k),
    )
    return board


def test_bare_flash_flashes_what_was_built(tmp_path: Path, nsx: list[tuple], bench: dict) -> None:
    build_dir = tmp_path / "build"
    _cli(build_dir, "build", "--no-inline-asm")
    built = firmware_build.read_build_id(build_dir)

    nsx.clear()
    out = _cli(build_dir, "flash")
    assert "inline asm off" in out and "Options changed" not in out
    assert "Firmware flashed successfully" in out and bench["id"] == built
    # No relock, resync or reconfigure.
    assert _steps(nsx) == ["render", "build"] and nsx[0][1].requantize_inline_asm is False

    out = _cli(build_dir, "flash")
    assert f"board confirmed build id {built}" in out and "already up to date" in out

    out = _cli(build_dir, "flash", "--inline-asm")
    assert "Options changed, rebuilding: requantize inline asm off -> on" in out
    assert "Firmware flashed successfully" in out and bench["id"] != built
    assert nsx_app.saved_options(firmware_build.nsx_app_dir(build_dir)).requantize_inline_asm is True


def test_flags_override_saved_options(tmp_path: Path) -> None:
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    kernels = make_checkout(tmp_path / "kernels")
    app_dir.mkdir(parents=True)
    nsx_app.save_options(app_dir, AppOptions(cmsis_nn_root=kernels, requantize_inline_asm=False))
    resolve = lambda **flags: nsx_app.resolve_options(app_dir, tmp_path, **flags)  # noqa: E731
    assert resolve() == AppOptions(cmsis_nn_root=kernels, requantize_inline_asm=False)
    assert resolve(cmsis_nn_ref="v2") == AppOptions(cmsis_nn_ref="v2", requantize_inline_asm=False)
    assert resolve(inline_asm=True) == AppOptions(cmsis_nn_root=kernels)


def test_no_saved_options_use_defaults(tmp_path: Path) -> None:
    kernels, tester = _nested_layout(tmp_path)
    app_dir = firmware_build.nsx_app_dir(tmp_path / "build")
    assert nsx_app.resolve_options(app_dir, tester) == AppOptions(cmsis_nn_root=kernels.resolve())
    # A corrupt file counts as absent.
    app_dir.mkdir(parents=True)
    (app_dir / nsx_app.OPTIONS_FILE).write_text("{not json", encoding="utf-8")
    assert nsx_app.resolve_options(app_dir, tester, inline_asm=False) == AppOptions(
        cmsis_nn_root=kernels.resolve(), requantize_inline_asm=False,
    )


def test_missing_saved_root_fails_clearly(tmp_path: Path) -> None:
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    app_dir.mkdir(parents=True)
    nsx_app.save_options(app_dir, AppOptions(cmsis_nn_root=tmp_path / "moved"))
    result = runner.invoke(app, ["hardware", "build", "--build-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "kernel root is gone" in result.output and "--cmsis-nn-root" in result.output
