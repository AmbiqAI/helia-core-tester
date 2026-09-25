"""Render the hardware firmware as an NSX app.

Renderer behind `hardware build`: writes ``nsx.yml``, a placeholder
``cmake/nsx/modules.cmake`` and ``CMakeLists.txt`` into an app directory,
plus ``modules/nsx-cmsis-nn`` for a local kernel checkout, vendored the way
helia-profiler does it. ``nsx lock``/``nsx sync`` own ``cmake/nsx/`` and the
other ``modules/`` from there; the firmware sources stay in this checkout.
NSX copies the rest of ``cmake/nsx/`` (bootstrap, helpers, toolchain flags)
out of its own wheel on every lock and sync, so the app never ships them.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jinja2
import yaml

from ..core.discovery import find_tester_templates_dir
from . import nsx_cli
from .boards import BoardSpec
from .boards import repo_root as tester_repo_root
from .firmware_build import BUILD_ID_TXT, IMAGE_SUBDIR, SERVER_TARGET
from .toolchain import DOWNLOADS_DIR

APP_NAME = "hct_benchmark_server"
SIZE_PROBE_TARGET = "hct_universal_size_probe"
TOOLCHAIN = "arm-none-eabi-gcc"

# Kernel identity per neuralspotx registry.lock.yaml.
CMSIS_NN_MODULE = "nsx-cmsis-nn"
CMSIS_NN_PROJECT = "ns-cmsis-nn"
CMSIS_NN_METADATA = "modules/ns-cmsis-nn/nsx/nsx-module.yaml"
CMSIS_NN_REF = "v7.35.1"

# Not in the registry yet; declared inline.
SEGGER_RTT_MODULE = "nsx-segger-rtt"
SEGGER_RTT_URL = "https://github.com/AmbiqAI/nsx-segger-rtt.git"
SEGGER_RTT_METADATA = "nsx-module.yaml"
SEGGER_RTT_REF = "v0.1.2"

PMU_MODULE = "nsx-pmu-armv8m"

# Copied from a local checkout, like hpx.
KERNEL_TREES = ("Include", "Source", "cmake")
# What makes a dir a checkout.
CHECKOUT_FILES = ("Include", "Source", "nsx/CMakeLists.txt", "nsx/nsx-module.yaml")
KERNEL_SHIM = "# Shim: delegates to the native ns-cmsis-nn NSX build.\nadd_subdirectory(nsx)\n"

RTT_BUFFER_SIZE_UP = 8192
RTT_BUFFER_SIZE_DOWN = 512

_SERVER_SOURCES = (
    "benchmark_server_main.c",
    "hctp_protocol.c",
    "benchmark_server_catalog.c",
    "benchmark_server_messages.c",
    "benchmark_server_adapter.c",
    "benchmark_server_session.c",
    "benchmark_server_adapters.gen.c",
    "benchmark_server_transport_rtt.c",
    "hct_build_id.c",
)


class AppRenderError(RuntimeError):
    """The app could not be rendered."""


@dataclass(frozen=True)
class AppOptions:
    """Kernel source, kernel switches, target choice."""

    cmsis_nn_ref: str = CMSIS_NN_REF
    cmsis_nn_root: Optional[Path] = None
    # ON matches hpx and shipping builds.
    requantize_inline_asm: bool = True
    enable_f32: bool = True
    enable_f16: bool = True
    build_size_probe: bool = False

    def kernel_source(self) -> str:
        """Kernel source, as printed."""
        return str(self.cmsis_nn_root or f"ns-cmsis-nn {self.cmsis_nn_ref}")

    def cache_vars(self) -> dict[str, str]:
        """Switches forced before the NSX bootstrap."""
        switches = {
            "NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM": self.requantize_inline_asm,
            "ARM_NN_ENABLE_F32": self.enable_f32,
            "ARM_NN_ENABLE_F16": self.enable_f16,
        }
        return {name: "ON" if on else "OFF" for name, on in switches.items()}


@dataclass(frozen=True)
class AppRender:
    """The rendered texts and where they went."""

    app_dir: Path
    modules: tuple[str, ...]
    nsx_yml: str
    modules_cmake: str
    cmakelists: str
    # Existing files this render rewrote.
    changed: tuple[str, ...] = ()


def module_names(board: BoardSpec, profile: dict[str, Any]) -> list[str]:
    """Profile modules, then kernels, RTT, PMU."""
    names = [str(name) for name in profile.get("modules") or []]
    extras = [CMSIS_NN_MODULE, SEGGER_RTT_MODULE]
    if board.pmu_tier == "armv8m":
        extras.append(PMU_MODULE)
    names.extend(name for name in extras if name not in names)
    return names


def module_registry(options: AppOptions) -> dict[str, Any]:
    """Overrides for the modules the profile lacks."""
    projects: dict[str, Any] = {SEGGER_RTT_MODULE: {"url": SEGGER_RTT_URL, "revision": SEGGER_RTT_REF}}
    modules: dict[str, Any] = {
        SEGGER_RTT_MODULE: {"project": SEGGER_RTT_MODULE, "revision": SEGGER_RTT_REF, "metadata": SEGGER_RTT_METADATA},
    }
    # A local checkout is vendored instead.
    if options.cmsis_nn_root is None:
        ref = options.cmsis_nn_ref
        projects[CMSIS_NN_PROJECT] = {"revision": ref}
        modules[CMSIS_NN_MODULE] = {"project": CMSIS_NN_PROJECT, "revision": ref, "metadata": CMSIS_NN_METADATA}
    return {"projects": projects, "modules": modules}


def nested_kernel_root(repo_root: Path) -> Optional[Path]:
    """The enclosing ns-cmsis-nn checkout, if any."""
    # Layout: ns-cmsis-nn/Tests/helia-core-tester.
    root = repo_root.resolve().parent.parent
    return root if all((root / name).exists() for name in CHECKOUT_FILES) else None


def kernel_dir(app_dir: Path, options: AppOptions) -> Path:
    """Where the build reads kernels."""
    # NSX vendors by module, clones by project.
    name = CMSIS_NN_MODULE if options.cmsis_nn_root else CMSIS_NN_PROJECT
    return app_dir / "modules" / name


def _write_if_absent(path: Path, text: str) -> None:
    """NSX rewrites this file on sync; seed it once."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_if_changed(path: Path, text: str) -> bool:
    """Skip unchanged files; True if rewritten."""
    existed = path.exists()
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except (OSError, UnicodeDecodeError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return existed


def write_kernels(root: Path, module_dir: Path) -> None:
    """Vendor a local checkout, as hpx does."""
    missing = [name for name in CHECKOUT_FILES if not (root / name).exists()]
    if missing:
        raise AppRenderError(f"Not an ns-cmsis-nn checkout: {root} lacks {missing[0]}")
    (module_dir / "nsx").mkdir(parents=True, exist_ok=True)
    # Native manifest at the module root.
    shutil.copy2(root / "nsx" / "nsx-module.yaml", module_dir / "nsx-module.yaml")
    shutil.copy2(root / "nsx" / "CMakeLists.txt", module_dir / "nsx" / "CMakeLists.txt")
    # A fresh mtime would rerun CMake.
    _write_if_changed(module_dir / "CMakeLists.txt", KERNEL_SHIM)
    # copytree keeps mtimes: ninja skips unchanged.
    for name in KERNEL_TREES:
        shutil.rmtree(module_dir / name, ignore_errors=True)
        if (root / name).is_dir():
            shutil.copytree(root / name, module_dir / name)


def render_app(
    board: BoardSpec,
    options: AppOptions,
    app_dir: Path,
    *,
    repo_root: Optional[Path] = None,
) -> AppRender:
    """Write nsx.yml, modules.cmake, CMakeLists.txt, local kernels."""
    repo_root = (repo_root or tester_repo_root()).resolve()
    profile = nsx_cli.starter_profile(board.nsx_board)
    if profile is None:
        raise AppRenderError(f"No NSX starter profile for {board.nsx_board}")
    modules = module_names(board, profile)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(find_tester_templates_dir(repo_root) / "hardware" / "nsx")),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )

    registry_yaml = yaml.safe_dump({"module_registry": module_registry(options)}, sort_keys=False)
    nsx_yml = env.get_template("nsx.yml.j2").render(
        app_name=APP_NAME,
        board=board.nsx_board,
        toolchain=TOOLCHAIN,
        channel=profile.get("channel"),
        modules=modules,
        vendored=[CMSIS_NN_MODULE] if options.cmsis_nn_root else [],
        module_registry_yaml=registry_yaml,
        kernel_source=options.kernel_source(),
    )
    modules_cmake = env.get_template("modules.cmake.j2").render(modules=modules)

    probe = options.build_size_probe
    cmakelists = env.get_template("CMakeLists.txt.j2").render(
        app_name=APP_NAME,
        board=board,
        target=SIZE_PROBE_TARGET if probe else SERVER_TARGET,
        build_size_probe=probe,
        sources=("universal_size_probe.c",) if probe else _SERVER_SOURCES,
        cache_vars=options.cache_vars(),
        enable_f32=options.enable_f32,
        enable_f16=options.enable_f16,
        hardware_dir=repo_root / "cmake" / "hardware",
        scripts_dir=repo_root / "scripts",
        cmsis_core_include=repo_root / DOWNLOADS_DIR / "CMSIS_5" / "CMSIS" / "Core" / "Include",
        kernel_dir=kernel_dir(app_dir, options).name,
        image_dir="probe" if probe else IMAGE_SUBDIR,
        build_id_txt=BUILD_ID_TXT,
        link_pmu=PMU_MODULE in modules,
        rtt_buffer_size_up=RTT_BUFFER_SIZE_UP,
        rtt_buffer_size_down=RTT_BUFFER_SIZE_DOWN,
    )

    if options.cmsis_nn_root is not None:
        write_kernels(options.cmsis_nn_root, kernel_dir(app_dir, options))
    changed = tuple(
        name for name, text in (("nsx.yml", nsx_yml), ("CMakeLists.txt", cmakelists))
        if _write_if_changed(app_dir / name, text)
    )
    _write_if_absent(app_dir / "cmake" / "nsx" / "modules.cmake", modules_cmake)
    return AppRender(app_dir, tuple(modules), nsx_yml, modules_cmake, cmakelists, changed)
