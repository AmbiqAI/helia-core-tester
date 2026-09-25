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


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_checkout(root: Path) -> Path:
    """ns-cmsis-nn stand-in with a nested tester."""
    _write(root / "Source" / "arm_add.c", "int add;\n")
    _write(root / "Source" / "sub" / "arm_sub.c", "int sub;\n")
    _write(root / "Include" / "arm_nn.h", "#pragma once\n")
    _write(root / "cmake" / "flags.cmake", "# flags\n")
    _write(root / "nsx" / "CMakeLists.txt", "add_library(nsx_cmsis_nn)\n")
    _write(root / "nsx" / "nsx-module.yaml", "module:\n  name: nsx-cmsis-nn\n")
    _write(root / "Tests" / "helia-core-tester" / "artifacts" / "big.bin", "x" * 1000)
    _write(root / "Documentation" / "index.md", "docs\n")
    return root


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

    assert "source" not in manifest["modules"][names.index("nsx-cmsis-nn")]
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
        "revision": nsx_app.SEGGER_RTT_REF,
    }
    assert registry["modules"]["nsx-segger-rtt"] == {
        "project": "nsx-segger-rtt",
        "revision": nsx_app.SEGGER_RTT_REF,
        "metadata": "nsx-module.yaml",
    }


def test_pmu_appended_once_and_channel_optional(tmp_path: Path, fake_profile: dict) -> None:
    fake_profile["modules"].append("nsx-pmu-armv8m")
    del fake_profile["channel"]
    render = _render(tmp_path)
    assert render.modules.count("nsx-pmu-armv8m") == 1
    assert "channel" not in yaml.safe_load(render.nsx_yml)


def test_cmsis_nn_root_is_vendored_in_the_app(tmp_path: Path) -> None:
    """Same as hpx: no local_path for NSX."""
    render = _render(tmp_path, cmsis_nn_root=make_checkout(tmp_path / "ns-cmsis-nn"))
    manifest = yaml.safe_load(render.nsx_yml)
    entry = next(module for module in manifest["modules"] if module["name"] == "nsx-cmsis-nn")
    assert entry == {"name": "nsx-cmsis-nn", "source": {"vendored": True}}
    registry = manifest["module_registry"]
    assert "ns-cmsis-nn" not in registry["projects"] and "nsx-cmsis-nn" not in registry["modules"]
    assert "local_path" not in render.nsx_yml
    assert 'HCT_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/modules/nsx-cmsis-nn"' in render.cmakelists
    pinned = _render(tmp_path / "pinned").cmakelists
    assert 'HCT_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/modules/ns-cmsis-nn"' in pinned


def _listing(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def test_local_kernels_are_written_as_hpx_writes_them(tmp_path: Path) -> None:
    checkout = make_checkout(tmp_path / "ns-cmsis-nn")
    _render(tmp_path, cmsis_nn_root=checkout)
    module = tmp_path / "app" / "modules" / "nsx-cmsis-nn"
    assert _listing(module) == {
        "nsx-module.yaml", "nsx/CMakeLists.txt", "CMakeLists.txt", "Include/arm_nn.h",
        "Source/arm_add.c", "Source/sub/arm_sub.c", "cmake/flags.cmake",
    }
    assert (module / "CMakeLists.txt").read_text(encoding="utf-8") == nsx_app.KERNEL_SHIM
    assert "add_subdirectory(nsx)" in nsx_app.KERNEL_SHIM
    assert (module / "nsx-module.yaml").read_bytes() == (checkout / "nsx" / "nsx-module.yaml").read_bytes()
    # Mtimes survive; the shim is kept.
    os.utime(checkout / "Source" / "arm_add.c", (1, 1))
    os.utime(module / "CMakeLists.txt", (2, 2))
    (checkout / "Source" / "sub" / "arm_sub.c").unlink()
    _render(tmp_path, cmsis_nn_root=checkout)
    assert (module / "Source" / "arm_add.c").stat().st_mtime == 1
    assert (module / "CMakeLists.txt").stat().st_mtime == 2
    assert not (module / "Source" / "sub" / "arm_sub.c").exists()


def test_a_non_checkout_is_rejected(tmp_path: Path) -> None:
    checkout = make_checkout(tmp_path / "ns-cmsis-nn")
    (checkout / "nsx" / "nsx-module.yaml").unlink()
    with pytest.raises(nsx_app.AppRenderError, match="lacks nsx/nsx-module.yaml"):
        _render(tmp_path, cmsis_nn_root=checkout)
    assert not (tmp_path / "app" / "nsx.yml").exists()


@pytest.mark.parametrize(
    "module",
    [
        lambda root: root,
        lambda root: root.parent,
        lambda root: root / "Source" / "build" / "nsx-cmsis-nn",
    ],
    ids=["same-dir", "root-inside-module", "module-inside-source"],
)
def test_overlapping_kernel_root_is_refused(tmp_path: Path, module) -> None:
    # Refuse before rmtree can delete sources.
    checkout = make_checkout(tmp_path / "ns-cmsis-nn")
    before = sorted(p.relative_to(checkout) for p in checkout.rglob("*"))
    with pytest.raises(nsx_app.AppRenderError, match="overlaps"):
        nsx_app.write_kernels(checkout, module(checkout))
    assert sorted(p.relative_to(checkout) for p in checkout.rglob("*")) == before


def test_module_under_the_checkout_is_allowed(tmp_path: Path) -> None:
    # The nested layout builds inside the checkout.
    checkout = make_checkout(tmp_path / "ns-cmsis-nn")
    module = checkout / "Tests" / "helia-core-tester" / "build" / "nsx-cmsis-nn"
    nsx_app.write_kernels(checkout, module)
    assert (module / "Include").is_dir()


def test_nested_kernel_root(tmp_path: Path) -> None:
    root = make_checkout(tmp_path / "ns-cmsis-nn")
    tester = root / "Tests" / "helia-core-tester"
    assert nsx_app.nested_kernel_root(tester) == root.resolve()
    (root / "nsx" / "nsx-module.yaml").unlink()
    assert nsx_app.nested_kernel_root(tester) is None
    assert nsx_app.nested_kernel_root(tmp_path / "standalone") is None


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
    # MVE in the harness skews MVE counters.
    assert "-fno-tree-vectorize" in text
    assert 'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hardware"' in text
    assert '"$<TARGET_FILE_DIR:hct_benchmark_server>/hct_benchmark_server.elf"' in text


def test_flash_recipe_loads_the_stamped_bin(tmp_path: Path) -> None:
    """NSX's flash recipe reads <build>/<target>.bin."""
    text = _render(tmp_path).cmakelists
    copy = text.index('"${CMAKE_BINARY_DIR}/hct_benchmark_server.bin"')
    assert text.index("patch_build_id.py") < copy


def test_kernel_source_is_a_define(tmp_path: Path) -> None:
    """A source switch recompiles every kernel."""
    define = 'target_compile_definitions(nsx_cmsis_nn PRIVATE HCT_KERNEL_SOURCE="{}")'
    texts = {
        options.kernel_id(): _render(tmp_path / options.kernel_id(), **vars(options)).cmakelists
        for options in (
            nsx_app.AppOptions(),
            nsx_app.AppOptions(cmsis_nn_ref="v1.2.3"),
            nsx_app.AppOptions(cmsis_nn_root=make_checkout(tmp_path / "a")),
            nsx_app.AppOptions(cmsis_nn_root=make_checkout(tmp_path / "b")),
        )
    }
    assert len(texts) == 4
    for kernel_id, text in texts.items():
        assert define.format(kernel_id) in text
        assert text.index("nsx_bootstrap_app(") < text.index(define.format(kernel_id))


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


def test_synced_modules_cmake_is_left_alone(tmp_path: Path) -> None:
    # NSX owns cmake/nsx/modules.cmake after sync.
    first = _render(tmp_path)
    synced = first.app_dir / "cmake" / "nsx" / "modules.cmake"
    synced.write_text("# written by nsx sync\n", encoding="utf-8")
    _render(tmp_path)
    assert synced.read_text(encoding="utf-8") == "# written by nsx sync\n"
