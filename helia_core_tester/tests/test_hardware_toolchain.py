"""`--toolchain`: options, build dirs, cache, provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware import firmware_build, hardware_pipeline, nsx_app, toolchain
from helia_core_tester.hardware.boards import repo_root
from helia_core_tester.hardware.nsx_app import AppOptions
from helia_core_tester.hardware.result_bundle import build_provenance
from helia_core_tester.hardware.run_summary import bundle_toolchain
from helia_core_tester.tests.test_hardware_bundle_provenance import _fake_gcc
from helia_core_tester.tests.test_hardware_nsx_build import BOARD, SERIAL, nsx  # noqa: F401

runner = CliRunner()
ATFE = AppOptions(toolchain="atfe")


def _configured(build_dir: Path, compiler: Path) -> None:
    """A CMake probe naming compiler."""
    probe = build_dir / "CMakeFiles" / "4.4.3" / "CMakeCCompiler.cmake"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(f'set(CMAKE_C_COMPILER "{compiler}")\n', encoding="utf-8")


@pytest.fixture
def atfe_root(tmp_path: Path, monkeypatch) -> Path:
    """ATFE_ROOT with a fake clang."""
    root = tmp_path / "atfe"
    _fake_gcc(root / "bin" / "clang", "22.1.0")
    monkeypatch.setenv("ATFE_ROOT", str(root))
    return root


def test_spec_maps_each_toolchain(monkeypatch) -> None:
    monkeypatch.delenv("ATFE_ROOT", raising=False)
    gcc, atfe = toolchain.toolchain_spec(), toolchain.toolchain_spec("atfe")
    assert (gcc.name, gcc.dir_suffix) == ("arm-none-eabi-gcc", "")
    assert (atfe.name, atfe.dir_suffix) == ("atfe", "-atfe")
    assert atfe.matches("/nix/store/x-atfe/bin/clang") and not atfe.matches("/opt/bin/arm-none-eabi-gcc")
    assert gcc.matches(None) and gcc.matches("/opt/bin/arm-none-eabi-gcc")
    with pytest.raises(ValueError, match="gcc, atfe"):
        toolchain.toolchain_spec("armclang")


def test_new_atfe_root_drops_old_clang(atfe_root: Path, tmp_path: Path) -> None:
    atfe = toolchain.toolchain_spec("atfe")
    assert atfe.matches(str(atfe_root / "bin" / "clang"))
    assert not atfe.matches(str(tmp_path / "old-atfe" / "bin" / "clang"))


def test_default_build_dirs_differ() -> None:
    root = Path("/repo")
    gcc = firmware_build.resolve_build_dir(root, BOARD)
    assert gcc == firmware_build.resolve_build_dir(root, BOARD, toolchain="gcc") == BOARD.build_dir(root)
    assert firmware_build.resolve_build_dir(root, BOARD, toolchain="atfe") == gcc.with_name(f"{BOARD.id}-atfe")
    # An explicit dir wins.
    assert firmware_build.resolve_build_dir(root, BOARD, Path("b"), "atfe") == root / "b"


def test_options_persist_toolchain(tmp_path: Path) -> None:
    nsx_app.save_options(tmp_path, ATFE)
    assert nsx_app.saved_options(tmp_path) == ATFE
    # Older records lack the field.
    (tmp_path / nsx_app.OPTIONS_FILE).write_text(json.dumps({"placement": "tcm"}), encoding="utf-8")
    assert nsx_app.saved_options(tmp_path).toolchain == "gcc"
    (tmp_path / nsx_app.OPTIONS_FILE).write_text(json.dumps({"toolchain": "icc"}), encoding="utf-8")
    assert nsx_app.saved_options(tmp_path) is None


def test_toolchain_resets_unless_flashed(tmp_path: Path) -> None:
    nsx_app.save_options(tmp_path, ATFE)
    # Like placement: a switch.
    assert nsx_app.resolve_options(tmp_path, tmp_path).toolchain == "gcc"
    assert nsx_app.resolve_options(tmp_path, tmp_path, follow_pin=False).toolchain == "atfe"
    assert nsx_app.resolve_options(tmp_path, tmp_path, toolchain="atfe").toolchain == "atfe"
    assert "toolchain gcc -> atfe" in ATFE.changes_from(AppOptions())


def test_nsx_yml_names_the_toolchain(tmp_path: Path, nsx, atfe_root) -> None:  # noqa: F811
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, options=ATFE)
    assert "toolchain: atfe\n" in (firmware_build.nsx_app_dir(tmp_path) / "nsx.yml").read_text(encoding="utf-8")
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert "toolchain: arm-none-eabi-gcc\n" in (firmware_build.nsx_app_dir(tmp_path) / "nsx.yml").read_text(encoding="utf-8")
    # nsx.yml changed: relock.
    assert [step[0] for step in nsx][:2] == ["render", "lock"]


def test_atfe_needs_atfe_root(tmp_path: Path, nsx, monkeypatch) -> None:  # noqa: F811
    monkeypatch.delenv("ATFE_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="ATFE_ROOT has no bin/clang: unset"):
        firmware_build.build_firmware(BOARD, build_dir=tmp_path, options=ATFE)
    assert nsx == []


def test_other_compiler_cache_is_dropped(tmp_path: Path, nsx, atfe_root, capsys) -> None:  # noqa: F811
    firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    _configured(tmp_path, Path("/opt/gcc/bin/arm-none-eabi-gcc"))
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, options=ATFE)
    assert "configure" in [step[0] for step in nsx]
    assert "Dropping CMake cache built by /opt/gcc/bin/arm-none-eabi-gcc" in capsys.readouterr().out
    # A matching compiler keeps the cache.
    _configured(tmp_path, atfe_root / "bin" / "clang")
    nsx.clear()
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, options=ATFE)
    assert "configure" not in [step[0] for step in nsx]


def test_atfe_provenance(tmp_path: Path, nsx, atfe_root) -> None:  # noqa: F811
    firmware_build.build_firmware(BOARD, build_dir=tmp_path, options=ATFE)
    _configured(tmp_path, atfe_root / "bin" / "clang")
    firmware_build._record_built(tmp_path, ATFE)
    provenance = build_provenance(tmp_path)[0]
    assert provenance["toolchain"] == {"name": "atfe", "version": "22.1.0"}
    assert provenance["options"]["toolchain"] == "atfe"
    built = firmware_build.built_record(firmware_build.nsx_app_dir(tmp_path))
    assert built["harness"]["toolchain"] == {"name": "atfe", "version": "22.1.0"}


def test_run_json_reads_bundle_toolchain(tmp_path: Path) -> None:
    record = {"name": "atfe", "version": "22.1.0"}
    (tmp_path / "session_manifest.json").write_text(json.dumps({"build": {"toolchain": record}}), encoding="utf-8")
    assert bundle_toolchain(tmp_path) == record
    assert bundle_toolchain(tmp_path / "missing") is None


def test_cli_threads_toolchain(monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(firmware_build, "build_firmware", lambda board, **kwargs: seen.update(kwargs) or Path("x"))
    result = runner.invoke(app, ["hardware", "build", "--board", BOARD.id, "--toolchain", "atfe"])
    assert result.exit_code == 0, result.output
    assert seen["options"].toolchain == "atfe"
    assert seen["build_dir"] == BOARD.build_dir(repo_root()).with_name(f"{BOARD.id}-atfe")
    result = runner.invoke(app, ["hardware", "build", "--toolchain", "icc"])
    assert result.exit_code != 0 and "--toolchain must be one of: gcc, atfe" in result.output


def test_skip_flash_keeps_the_built_toolchain(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}

    def _pipeline(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setenv("HPX_JLINK_SERIAL", str(SERIAL))
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _pipeline)
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    app_dir.mkdir(parents=True)
    nsx_app.save_options(app_dir, AppOptions(cmsis_nn_ref="v1", cmsis_nn_ref_explicit=True, toolchain="atfe"))
    base = ["hardware", "run", "--build-dir", str(tmp_path), "--skip-flash"]
    runner.invoke(app, base)
    assert seen["app_options"].toolchain == "atfe"
    seen.clear()
    result = runner.invoke(app, [*base, "--skip-generate", "--toolchain", "gcc"])
    assert not seen and "toolchain atfe -> gcc" in result.output


def test_memory_report_picks_atfe_dir(monkeypatch, tmp_path: Path) -> None:
    from helia_core_tester.hardware import cli as hardware_cli

    seen: dict = {}
    report = tmp_path / "memory_report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hardware_cli, "generate_memory_report", lambda spec, **kwargs: seen.update(kwargs) or report)
    result = runner.invoke(app, ["hardware", "memory-report", "--board", BOARD.id, "--toolchain", "atfe"])
    assert result.exit_code == 0, result.output
    assert seen["build_dir"].name == f"{BOARD.id}-atfe"


def test_pipeline_defaults_to_the_toolchain_dir(monkeypatch, tmp_path: Path) -> None:
    from helia_core_tester.hardware.hardware_pipeline import StreamOptions, run_hardware_pipeline

    seen: dict = {}

    def _stream(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(hardware_pipeline, "flash_firmware", lambda *a, **k: None)
    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _stream)
    with pytest.raises(RuntimeError, match="stop here"):
        run_hardware_pipeline(
            tmp_path, BOARD, SERIAL, options=StreamOptions(), skip_generate=True, app_options=ATFE, echo=lambda _m: None,
        )
    assert seen["build_dir"] == BOARD.build_dir(tmp_path).with_name(f"{BOARD.id}-atfe")
