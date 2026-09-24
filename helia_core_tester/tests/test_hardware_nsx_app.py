"""The NSX app render: what nsx.yml and CMakeLists.txt say.

`nsx_cli.starter_profile` is monkeypatched; nothing here reads the
installed NSX registry.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from helia_core_tester.hardware import nsx_app, nsx_cli
from helia_core_tester.hardware.boards import resolve_board

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARD = resolve_board("apollo510_evb")
PROFILE_MODULES = ["nsx-ambiq-bsp", "nsx-board-apollo510-evb", "nsx-cmsis-core", "nsx-core"]
FILES = ("nsx.yml", "cmake/nsx/modules.cmake", "CMakeLists.txt")


@pytest.fixture(autouse=True)
def fake_profile(monkeypatch: pytest.MonkeyPatch) -> dict:
    profile = {"board": BOARD.nsx_board, "channel": "stable", "modules": list(PROFILE_MODULES)}
    monkeypatch.setattr(nsx_cli, "starter_profile", lambda board: profile)
    return profile


def _render(tmp_path: Path, **options) -> nsx_app.AppRender:
    return nsx_app.render_app(BOARD, nsx_app.AppOptions(**options), tmp_path / "app")


# --- nsx.yml -----------------------------------------------------------------------


def test_nsx_yml_declares_profile_modules_plus_extras(tmp_path: Path) -> None:
    manifest = yaml.safe_load(_render(tmp_path, cmsis_nn_ref="v9.9.9").nsx_yml)
    assert manifest["schema_version"] == 2
    assert manifest["project"]["name"] == "hct_benchmark_server"
    assert manifest["targets"] == {"default": "apollo510_evb", "supported": ["apollo510_evb"]}
    assert manifest["toolchain"] == "arm-none-eabi-gcc"
    assert manifest["channel"] == "stable"
    names = [entry["name"] for entry in manifest["modules"]]
    assert names == PROFILE_MODULES + ["nsx-cmsis-nn", "nsx-segger-rtt", "nsx-pmu-armv8m"]

    registry = manifest["module_registry"]
    assert registry["projects"]["ns-cmsis-nn"] == {"revision": "v9.9.9"}
    assert registry["modules"]["nsx-cmsis-nn"] == {
        "project": "ns-cmsis-nn",
        "revision": "v9.9.9",
        "metadata": "modules/ns-cmsis-nn/nsx/nsx-module.yaml",
    }
    # Not in the registry yet: declared inline.
    assert registry["projects"]["nsx-segger-rtt"] == {
        "url": "https://github.com/AmbiqAI/nsx-segger-rtt.git",
        "revision": "v0.1.1",
    }
    assert registry["modules"]["nsx-segger-rtt"] == {
        "project": "nsx-segger-rtt",
        "revision": "v0.1.1",
        "metadata": "nsx-module.yaml",
    }


def test_pmu_appended_once_and_channel_optional(tmp_path: Path, fake_profile: dict) -> None:
    fake_profile["modules"].append("nsx-pmu-armv8m")
    del fake_profile["channel"]
    render = _render(tmp_path)
    assert render.modules.count("nsx-pmu-armv8m") == 1
    assert "channel" not in yaml.safe_load(render.nsx_yml)


def test_cmsis_nn_root_is_a_local_path_without_git_coordinates(tmp_path: Path) -> None:
    checkout = tmp_path / "ns-cmsis-nn"
    registry = yaml.safe_load(_render(tmp_path, cmsis_nn_root=checkout).nsx_yml)["module_registry"]
    assert registry["projects"]["ns-cmsis-nn"] == {"local_path": str(checkout)}
    # A module revision would outrank the project's.
    assert registry["modules"]["nsx-cmsis-nn"] == {
        "project": "ns-cmsis-nn",
        "metadata": "modules/ns-cmsis-nn/nsx/nsx-module.yaml",
    }


def test_missing_profile_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nsx_cli, "starter_profile", lambda board: None)
    with pytest.raises(nsx_app.AppRenderError, match="No NSX starter profile"):
        _render(tmp_path)


# --- CMakeLists.txt ----------------------------------------------------------------


def test_cache_switches_precede_the_bootstrap(tmp_path: Path) -> None:
    """An option() default cannot be overridden later."""
    text = _render(tmp_path, enable_f16=False).cmakelists
    bootstrap = text.index("nsx_bootstrap_app(")
    for line in (
        'set(NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM "ON" CACHE STRING',
        'set(ARM_NN_ENABLE_F32 "ON" CACHE STRING',
        'set(ARM_NN_ENABLE_F16 "OFF" CACHE STRING',
        "set(NSX_SEGGER_RTT_BUFFER_SIZE_UP 8192 CACHE",
        "set(NSX_SEGGER_RTT_BUFFER_SIZE_DOWN 512 CACHE",
    ):
        assert text.index(line) < bootstrap, line
    assert "--enable-f16" not in text and "--enable-f32" in text


def test_cmakelists_uses_nsxs_own_bootstrap_and_finalize(tmp_path: Path) -> None:
    text = _render(tmp_path).cmakelists
    assert "include(${CMAKE_CURRENT_LIST_DIR}/cmake/nsx/modules.cmake)" in text
    assert "include(${CMAKE_CURRENT_LIST_DIR}/cmake/nsx/nsx_app_bootstrap.cmake)" in text
    assert "nsx_finalize_app(hct_benchmark_server)" in text
    assert "add_subdirectory" not in text
    for target in ("nsx::cmsis_nn", "nsx::segger_rtt", "nsx::pmu_armv8m"):
        assert target in text
    assert "SEGGER_RTT.c" not in text, "RTT comes only from the module"


def test_server_sources_compile_out_of_this_checkout(tmp_path: Path) -> None:
    text = _render(tmp_path).cmakelists
    assert f'set(HCT_HARDWARE_DIR "{PROJECT_ROOT / "cmake" / "hardware"}")' in text
    for name in ("benchmark_server_main.c", "benchmark_server_adapters.gen.c", "hct_build_id.c"):
        assert f'"${{HCT_HARDWARE_DIR}}/{name}"' in text
    assert "universal_size_probe.c" not in text
    assert f"HCT_SERVER_WORKSPACE_BYTES={BOARD.workspace_bytes}" in text
    assert f'HCT_BENCHMARK_SERVER_TARGET_CPU="{BOARD.cpu}"' in text
    assert "patch_build_id.py" in text, "the post-link build-id stamp must survive"
    assert 'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hardware"' in text
    assert '"$<TARGET_FILE_DIR:hct_benchmark_server>/hct_benchmark_server.elf"' in text


def test_size_probe_replaces_the_server(tmp_path: Path) -> None:
    text = _render(tmp_path, build_size_probe=True).cmakelists
    assert "add_executable(hct_universal_size_probe" in text
    assert "add_executable(hct_benchmark_server" not in text
    assert '"${HCT_HARDWARE_DIR}/universal_size_probe.c"' in text
    assert "patch_build_id.py" not in text
    assert "nsx_finalize_app(hct_universal_size_probe)" in text


# --- files on disk -----------------------------------------------------------------


def test_render_is_idempotent_and_keeps_mtimes(tmp_path: Path) -> None:
    first = _render(tmp_path)
    paths = [tmp_path / "app" / name for name in FILES]
    texts = (first.nsx_yml, first.modules_cmake, first.cmakelists)
    assert [path.read_text(encoding="utf-8") for path in paths] == list(texts)
    assert "    nsx-segger-rtt\n" in first.modules_cmake
    for path in paths:
        os.utime(path, (1_000_000_000, 1_000_000_000))
    second = _render(tmp_path)
    assert (second.nsx_yml, second.modules_cmake, second.cmakelists) == texts
    assert [path.stat().st_mtime for path in paths] == [1_000_000_000.0] * 3
