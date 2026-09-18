"""Render the benchmark-server firmware as an NSX app.

The hardware firmware is an ordinary NSX application: a generated `nsx.yml`
manifest naming the modules it needs, a `cmake/nsx/modules.cmake` module list,
and a `CMakeLists.txt` that includes NSX's own bootstrap and calls
`nsx_bootstrap_app()` / `nsx_finalize_app()`. `neuralspotx.api` then locks,
syncs, configures and builds it (see `firmware_build`). This mirrors what
heliaPROFILER does, so the two tools build ns-cmsis-nn the same way and their
numbers are comparable.

What is *not* here, deliberately:

- No hand-reconstructed bootstrap. `cmake/nsx/` inside the app is reproduced by
  `nsx lock`/`nsx sync` from the pinned neuralspotx wheel on every run; this
  repo vendors no copy of it.
- No `CMSIS_NN_ROOT`, and no nested `<ns-cmsis-nn>/Tests/helia-core-tester`
  layout. The kernels are the NSX module `nsx-cmsis-nn`, resolved from the
  registry at the baseline's pinned commit, or from a local checkout when
  `--cmsis-nn-root` asks for one.

The app root is `<build dir>/nsx_app`, with NSX's own build directory nested
inside it at `build/<board>` per NSX convention. Everything the app compiles
that belongs to this repo (the benchmark-server sources, the generated adapter
table, the build-id stamp) is referenced by absolute path out of the checkout
that rendered it, so the app tree stays pure generated output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from .boards import BoardSpec
from .dependency_baseline import CMSIS_NN_PROJECT, DependencyBaseline

APP_NAME = "hct_benchmark_server"
SERVER_TARGET = "hct_benchmark_server"
SIZE_PROBE_TARGET = "hct_universal_size_probe"

#: NSX registry identity of the kernels under test.
CMSIS_NN_MODULE = "nsx-cmsis-nn"

#: Starter-profile entries this app never consumes directly (same two
#: heliaPROFILER drops): legacy helpers that would drag in their own closure.
_UNUSED_PROFILE_MODULES = frozenset({"nsx-harness", "nsx-utils"})

#: Name of the render state file written next to the manifest.
RENDER_STATE = ".hct-nsx-app.json"

RENDER_SCHEMA_VERSION = 1


class AppRenderError(RuntimeError):
    """The app could not be rendered (no starter profile, bad override path...)."""


@dataclass(frozen=True)
class ModuleSpec:
    """One entry of the app's module list: the module and the project owning it."""

    name: str
    project: str
    #: Set for a module vendored from a local directory instead of the registry.
    local_path: Optional[Path] = None


@dataclass(frozen=True)
class KernelSource:
    """Where `nsx-cmsis-nn` comes from for this render."""

    ref: Optional[str] = None
    path: Optional[Path] = None

    def describe(self) -> str:
        if self.path is not None:
            return f"path:{self.path}"
        return f"{CMSIS_NN_PROJECT}@{self.ref}"


@dataclass(frozen=True)
class KernelOptions:
    """The ns-cmsis-nn cache variables this app forces before any module loads.

    An `option()` default cannot be overridden once the module has been added,
    so these are written into the app CMakeLists ahead of `nsx_bootstrap_app()`
    rather than passed as `-D` at configure time. That also means they are part
    of the rendered text, so a change to any of them changes the render digest
    and cannot silently reuse a build dir configured with the other setting.
    """

    requantize_inline_asm: bool = True
    enable_f32: bool = True
    enable_f16: bool = True

    def cache_vars(self) -> Dict[str, str]:
        return {
            "NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM": _on_off(self.requantize_inline_asm),
            "ARM_NN_ENABLE_F32": _on_off(self.enable_f32),
            "ARM_NN_ENABLE_F16": _on_off(self.enable_f16),
        }


def _on_off(value: bool) -> str:
    return "ON" if value else "OFF"


@dataclass(frozen=True)
class AppRender:
    """The rendered app: where it lives, what it says, and what identifies it."""

    app_dir: Path
    board: BoardSpec
    modules: Tuple[ModuleSpec, ...]
    kernel_source: KernelSource
    kernel_options: KernelOptions
    baseline: DependencyBaseline
    nsx_yml: str
    modules_cmake: str
    cmakelists: str

    @property
    def build_dir(self) -> Path:
        """NSX's build directory for this app (`<app>/build/<board>`)."""
        return nsx_build_dir(self.app_dir, self.board)

    @property
    def digest(self) -> str:
        """Identity of this render: the three rendered files plus the baseline.

        The baseline fingerprint is folded in even though every pin it carries
        already appears in `nsx.yml`, so that editing an unrelated part of the
        baseline file still invalidates a reused lock rather than leaving a
        bundle claiming a baseline the lock never saw.
        """
        digest = hashlib.sha256()
        for text in (self.nsx_yml, self.modules_cmake, self.cmakelists):
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        digest.update(self.baseline.fingerprint.encode("utf-8"))
        return digest.hexdigest()

    def state(self) -> Dict[str, Any]:
        return {
            "schema": "hct.nsx-app-render",
            "schema_version": RENDER_SCHEMA_VERSION,
            "render_digest": self.digest,
            "baseline_id": self.baseline.baseline_id,
            "baseline_fingerprint": self.baseline.fingerprint,
            "board": self.board.id,
            "nsx_board": self.board.nsx_board,
            "kernel_source": self.kernel_source.describe(),
            "kernel_options": self.kernel_options.cache_vars(),
            "modules": [m.name for m in self.modules],
        }


# --- paths -------------------------------------------------------------------------


def app_dir_for(build_dir: Path) -> Path:
    """The NSX app root inside a board's build directory."""
    return build_dir / "nsx_app"


def nsx_build_dir(app_dir: Path, board: BoardSpec) -> Path:
    """NSX's own build directory, nested in the app root per NSX convention."""
    return app_dir / "build" / board.nsx_board


def read_render_state(app_dir: Path) -> Optional[Dict[str, Any]]:
    """The state written by the last successful render of `app_dir`, if any."""
    path = app_dir / RENDER_STATE
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


# --- module resolution --------------------------------------------------------------


def starter_profile(nsx_board: str) -> Dict[str, Any]:
    """The NSX starter profile for `nsx_board`.

    The profile is the single source of truth for which modules the board stack
    needs and which project owns each one, so neither is maintained here. A
    board with no profile is a board NSX cannot build for at all.
    """
    from neuralspotx import api as nsx_api

    profile = nsx_api.starter_profile(nsx_board)
    if profile is None:
        raise AppRenderError(
            f"No NSX starter profile for board '{nsx_board}'. The board must be registered "
            f"in the NSX registry that ships with neuralspotx; check nsx_board in "
            f"assets/hardware_boards.yaml, or update the pinned neuralspotx."
        )
    return profile


def profile_module_names(profile: Mapping[str, Any]) -> List[str]:
    """The profile's declared module list, in order."""
    modules = profile.get("modules")
    if not isinstance(modules, list) or not all(isinstance(name, str) for name in modules):
        raise AppRenderError(
            "The NSX starter profile declares no valid module list; update neuralspotx."
        )
    return [name for name in modules if name not in _UNUSED_PROFILE_MODULES]


def module_project(name: str, profile: Mapping[str, Any]) -> str:
    """The NSX project owning `name`: the profile's repoint first, then the registry.

    A profile's `module_overrides` entry is authoritative -- it is how a module
    that has migrated into the nsx-ambiq-sdk monorepo stops resolving to its old
    standalone project.
    """
    from neuralspotx import api as nsx_api

    override = (profile.get("module_overrides") or {}).get(name)
    if isinstance(override, Mapping) and override.get("project"):
        return str(override["project"])
    return nsx_api.registry_module_project(name) or name


def resolve_modules(nsx_board: str, kernel_source: KernelSource) -> Tuple[ModuleSpec, ...]:
    """The app's module list: the board's starter profile plus the kernels."""
    profile = starter_profile(nsx_board)
    specs = [ModuleSpec(name, module_project(name, profile)) for name in profile_module_names(profile)]
    names = {spec.name for spec in specs}
    if CMSIS_NN_MODULE not in names:
        specs.append(
            ModuleSpec(CMSIS_NN_MODULE, CMSIS_NN_PROJECT, local_path=kernel_source.path)
        )
    return tuple(specs)


# --- nsx.yml ------------------------------------------------------------------------


def project_ref_overrides(
    modules: Sequence[ModuleSpec], baseline: DependencyBaseline
) -> Dict[str, str]:
    """Baseline pin per project the module list resolves to.

    A module vendored from a local path has no project to pin, and a project the
    baseline does not name is left to the packaged registry.
    """
    refs: Dict[str, str] = {}
    for spec in modules:
        if spec.local_path is not None:
            continue
        ref = baseline.pin(spec.project)
        if ref is not None:
            refs[spec.project] = ref
    return dict(sorted(refs.items()))


def render_module_registry(
    profile: Mapping[str, Any], ref_overrides: Mapping[str, str]
) -> str:
    """The `module_registry:` block that holds every pin in force.

    Emitting the profile's whole `project_overrides` / `module_overrides` (not
    just the projects this app names) is what makes the app's effective registry
    agree with itself: NSX gives a *module*-level `revision` precedence over its
    project's, and the starter profile pins each migrated module to a tag, so a
    project override alone would be silently overruled and the app would build
    something other than the commit nsx.yml claims. Every module owned by an
    overridden project is therefore realigned onto the same pin, and modules the
    profile never mentions (the kernels) get an explicit entry of their own.
    """
    from neuralspotx import api as nsx_api

    registry = nsx_api.load_registry()
    base_projects = registry.get("projects") or {}
    base_modules = registry.get("modules") or {}

    projects: Dict[str, Any] = {
        str(name): dict(entry)
        for name, entry in (profile.get("project_overrides") or {}).items()
        if isinstance(entry, Mapping)
    }
    modules: Dict[str, Any] = {
        str(name): dict(entry)
        for name, entry in (profile.get("module_overrides") or {}).items()
        if isinstance(entry, Mapping)
    }

    for project, ref in ref_overrides.items():
        entry = projects.get(project) or dict(base_projects.get(project) or {"name": project})
        entry["revision"] = ref
        projects[project] = entry

    for name, entry in list(modules.items()):
        ref = ref_overrides.get(str(entry.get("project", "")))
        if ref is not None:
            aligned = dict(entry)
            aligned["revision"] = ref
            modules[name] = aligned

    # Modules this app names that the profile's override map never mentions --
    # the kernels -- would otherwise fall back to the packaged registry's own
    # module-level revision while nsx.yml still claims the baseline commit.
    for name, base in base_modules.items():
        if name in modules or not isinstance(base, Mapping):
            continue
        ref = ref_overrides.get(str(base.get("project", "")))
        if ref is None:
            continue
        aligned = dict(base)
        aligned["revision"] = ref
        modules[str(name)] = aligned

    if not projects and not modules:
        return ""
    block: Dict[str, Any] = {}
    if projects:
        block["projects"] = dict(sorted(projects.items()))
    if modules:
        block["modules"] = dict(sorted(modules.items()))
    return yaml.safe_dump(
        {"module_registry": block}, sort_keys=False, default_flow_style=False
    )


def render_nsx_yml(
    board: BoardSpec,
    modules: Sequence[ModuleSpec],
    ref_overrides: Mapping[str, str],
    module_registry_yaml: str,
    *,
    toolchain: str,
    channel: str,
) -> str:
    lines = [
        "# nsx.yml -- generated by helia-core-tester. Do not edit.",
        "schema_version: 2",
        "project:",
        f"  name: {APP_NAME}",
        "targets:",
        f"  default: {board.nsx_board}",
        "  supported:",
        f"    - {board.nsx_board}",
        f"toolchain: {toolchain}",
        f"channel: {channel}",
        "modules:",
    ]
    for spec in modules:
        lines.append(f"  - name: {spec.name}")
        if spec.local_path is not None:
            lines.append("    source:")
            lines.append(f"      path: {spec.local_path}")
            continue
        lines.append(f"    project: {spec.project}")
        ref = ref_overrides.get(spec.project)
        if ref is not None:
            lines.append(f"    revision: {ref}")
    lines.append("features: {}")
    text = "\n".join(lines) + "\n"
    if module_registry_yaml:
        text += module_registry_yaml
    return text


def render_modules_cmake(modules: Sequence[ModuleSpec]) -> str:
    """A bare module list, used only until NSX writes the real one.

    `nsx lock` / `nsx sync` regenerate `cmake/nsx/modules.cmake` themselves, in
    dependency order and with the `NSX_APP_MODULE_DIR_*` / `NSX_APP_PROJECT_DIRS`
    mappings that point each module at its vendored location. This version has
    the names only, so it is written just once, to give a freshly rendered app a
    syntactically complete tree before the first lock (see `write_app`).
    """
    lines = [
        "# modules.cmake -- placeholder written by helia-core-tester;",
        "# `nsx lock` / `nsx sync` replace it with the resolved module list.",
        "set(NSX_APP_MODULES",
    ]
    lines += [f"    {spec.name}" for spec in modules]
    lines.append(")")
    return "\n".join(lines) + "\n"


# --- CMakeLists.txt -----------------------------------------------------------------

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


def render_cmakelists(
    board: BoardSpec,
    *,
    repo_root: Path,
    kernel_options: KernelOptions,
    kernel_source_dir: str,
    cmsis_core_include: Path,
    build_size_probe: bool = False,
) -> str:
    """The app's top-level CMakeLists.

    `kernel_source_dir` is a CMake expression (not a resolved path) for the
    synced ns-cmsis-nn tree: the header scan that generates the symbol-reference
    table has to read the same kernels the module compiles, and where NSX
    materialises them is only known after the sync.
    """
    hardware = repo_root / "cmake" / "hardware"
    scripts = repo_root / "scripts"
    cache_vars = kernel_options.cache_vars()
    target = SIZE_PROBE_TARGET if build_size_probe else SERVER_TARGET

    out: List[str] = [
        "# CMakeLists.txt -- generated by helia-core-tester. Do not edit.",
        "#",
        "# An NSX app whose sources live in the helia-core-tester checkout that",
        "# rendered it. cmake/nsx/ below is NSX's own build glue, reproduced from the",
        "# pinned neuralspotx package by `nsx lock` / `nsx sync` -- not a copy kept in",
        "# this repo.",
        "cmake_minimum_required(VERSION 3.24)",
        f"project({APP_NAME} LANGUAGES C ASM)",
        "",
        f'set(NSX_BOARD "{board.nsx_board}" CACHE STRING "Target board")',
        "",
        "# compile_commands.json is the evidence for \"what did the kernels actually",
        "# compile with\" -- the question every hardware-number regression starts from.",
        "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)",
        "",
        "# The board fragment sets this too, but a BSP module added before it would",
        "# fall back to a bare \"evb\" default and resolve the prebuilt lib path wrong.",
        f'set(NSX_AMBIQ_BSP_LIB_SUBDIR "{board.nsx_board}" CACHE STRING "AmbiqSuite BSP lib subdir")',
        "",
        "# ns-cmsis-nn kernel switches. These must be set before the module is added:",
        "# an option() default cannot be overridden afterwards.",
    ]
    for name, value in cache_vars.items():
        out.append(f'set({name} "{value}" CACHE STRING "helia-core-tester kernel option" FORCE)')
    out += [
        "",
        "include(${CMAKE_CURRENT_LIST_DIR}/cmake/nsx/modules.cmake)",
        "include(${CMAKE_CURRENT_LIST_DIR}/cmake/nsx/nsx_app_bootstrap.cmake)",
        "",
        "nsx_bootstrap_app(",
        '    APP_ROOT "${CMAKE_CURRENT_LIST_DIR}"',
        '    BOARD "${NSX_BOARD}"',
        "    MODULES ${NSX_APP_MODULES}",
        ")",
        "",
        "# --- helia-core-tester sources ---",
        f'set(HCT_REPO_ROOT "{repo_root}")',
        f'set(HCT_HARDWARE_DIR "{hardware}")',
        f'set(HCT_SCRIPTS_DIR "{scripts}")',
        f'set(HCT_KERNEL_SOURCE_DIR "{kernel_source_dir}")',
        "# pmu_armv8.h: the Armv8-M PMU register interface the session code drives",
        "# directly. The SDK's nsx-cmsis-core does not carry it.",
        f'set(HCT_CMSIS_CORE_INCLUDE "{cmsis_core_include}")',
        "",
        "find_package(Python3 COMPONENTS Interpreter REQUIRED)",
        "",
        "# --- compile-time kernel symbol reference table ---",
        "# One weak reference per public kernel the archive actually defines, so the",
        "# linker retains them for the firmware's runtime dispatch table.",
        'set(HCT_SYMBOL_REFS_DIR "${CMAKE_BINARY_DIR}/hardware")',
        'set(HCT_SYMBOL_REFS_INC "${HCT_SYMBOL_REFS_DIR}/kernel_symbol_refs.inc")',
        "set(HCT_SYMBOL_REF_ARGS",
        '    --cmsis-nn-root "${HCT_KERNEL_SOURCE_DIR}"',
        '    --archive "$<TARGET_FILE:nsx_cmsis_nn>"',
        '    --output "${HCT_SYMBOL_REFS_INC}")',
    ]
    if kernel_options.enable_f32:
        out.append("list(APPEND HCT_SYMBOL_REF_ARGS --enable-f32)")
    if kernel_options.enable_f16:
        out.append("list(APPEND HCT_SYMBOL_REF_ARGS --enable-f16)")
    out += [
        "",
        "add_custom_command(",
        '    OUTPUT "${HCT_SYMBOL_REFS_INC}"',
        '    COMMAND "${CMAKE_COMMAND}" -E make_directory "${HCT_SYMBOL_REFS_DIR}"',
        '    COMMAND "${Python3_EXECUTABLE}" "${HCT_SCRIPTS_DIR}/generate_kernel_symbol_refs.py"',
        "            ${HCT_SYMBOL_REF_ARGS}",
        "    DEPENDS",
        "        nsx_cmsis_nn",
        '        "${HCT_SCRIPTS_DIR}/generate_kernel_symbol_refs.py"',
        '        "${HCT_KERNEL_SOURCE_DIR}/Include/arm_nnfunctions.h"',
        '        "${HCT_KERNEL_SOURCE_DIR}/Include/arm_nnfunctions_flt.h"',
        "    VERBATIM)",
        "",
        'add_custom_target(hct_kernel_symbol_refs DEPENDS "${HCT_SYMBOL_REFS_INC}")',
        "",
    ]

    if build_size_probe:
        out += [
            "# --- universal size probe ---",
            "# Links the whole retained kernel library for one feature set, to prove it",
            "# fits the board before any firmware work.",
            f"add_executable({SIZE_PROBE_TARGET}",
            '    "${HCT_HARDWARE_DIR}/universal_size_probe.c")',
            "",
        ]
    else:
        out += [
            "# --- benchmark server ---",
            "# HCT_BENCHMARK_SERVER_BUILD_ID_PATCHED compiles out the constant build-id",
            "# fallback in benchmark_server_catalog.c (kept for host-side unit builds);",
            "# the firmware's id comes from the slot in hct_build_id.c, filled in after",
            "# the link below.",
            f"add_executable({SERVER_TARGET}",
        ]
        out += [f'    "${{HCT_HARDWARE_DIR}}/{name}"' for name in _SERVER_SOURCES]
        out += ['    "${HCT_HARDWARE_DIR}/rtt/RTT/SEGGER_RTT.c")', ""]

    out += [
        f"set_target_properties({target} PROPERTIES",
        '    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")',
        "",
        f"add_dependencies({target} hct_kernel_symbol_refs)",
        "",
        f"target_include_directories({target} PRIVATE",
        '    "${HCT_SYMBOL_REFS_DIR}"',
        '    "${HCT_HARDWARE_DIR}"',
        '    "${HCT_CMSIS_CORE_INCLUDE}"',
    ]
    if not build_size_probe:
        out += [
            '    "${HCT_HARDWARE_DIR}/rtt/RTT"',
            '    "${HCT_HARDWARE_DIR}/rtt/Config"',
        ]
    out += [
        ")",
        "",
        f"target_compile_definitions({target} PRIVATE",
        "    HELIA_HARDWARE_BUILD",
    ]
    if not build_size_probe:
        out += [
            f'    HCT_BENCHMARK_SERVER_BOARD_ID="{board.nsx_board}"',
            f'    HCT_BENCHMARK_SERVER_TARGET_CPU="{board.cpu}"',
            f"    HCT_SERVER_WORKSPACE_BYTES={board.workspace_bytes}",
            "    HCT_BENCHMARK_SERVER_BUILD_ID_PATCHED",
        ]
    out += [
        ")",
        "",
        f"target_compile_options({target} PRIVATE -Wno-cast-function-type)",
        "",
        f"target_link_libraries({target} PRIVATE",
        "    nsx::board",
        "    nsx::core",
        "    nsx::cmsis_core",
        "    nsx::cmsis_nn",
        ")",
        "",
        f"nsx_finalize_app({target})",
        "",
    ]

    if not build_size_probe:
        out += [
            "# Per-build firmware identity, stamped after the link: the whole flash image",
            "# is hashed with the id slot zeroed and `hct-<sha256[:48]>` written into that",
            "# slot in the .axf and the .bin in place. The firmware advertises it in",
            "# TARGET_INFO; the host reads the same string from hct_build_id.txt. Hashing",
            "# the linked image rather than a subset of its inputs is what makes two builds",
            "# that differ anywhere get different ids. POST_BUILD commands run in the order",
            "# they are added, so this follows nsx_finalize_app()'s objcopy (the .bin",
            "# exists) and precedes the copies below (they copy the patched files).",
            f"add_custom_command(TARGET {SERVER_TARGET} POST_BUILD",
            '    COMMAND "${Python3_EXECUTABLE}" "${HCT_SCRIPTS_DIR}/patch_build_id.py"',
            f'            --elf "$<TARGET_FILE:{SERVER_TARGET}>"',
            f'            --bin "$<TARGET_FILE_DIR:{SERVER_TARGET}>/{SERVER_TARGET}.bin"',
            f'            --output-txt "${{CMAKE_BINARY_DIR}}/hct_build_id.txt"',
            f'    COMMENT "Stamping {SERVER_TARGET} build id (post-link image hash)"',
            "    VERBATIM)",
            "",
            "# NSX links to <target>.axf; the host tooling (memory report, RTT block",
            "# lookup, flash stamp) reads <target>.elf. Same file, stable name.",
            f"add_custom_command(TARGET {SERVER_TARGET} POST_BUILD",
            "    COMMAND ${CMAKE_COMMAND} -E copy_if_different",
            f'        "$<TARGET_FILE:{SERVER_TARGET}>"',
            f'        "${{CMAKE_BINARY_DIR}}/{SERVER_TARGET}.elf"',
            "    VERBATIM)",
            "",
        ]
    else:
        out += [
            f"add_custom_command(TARGET {SIZE_PROBE_TARGET} POST_BUILD",
            "    COMMAND ${CMAKE_COMMAND} -E copy_if_different",
            f'        "$<TARGET_FILE:{SIZE_PROBE_TARGET}>"',
            f'        "${{CMAKE_BINARY_DIR}}/{SIZE_PROBE_TARGET}.elf"',
            "    VERBATIM)",
            "",
        ]
    return "\n".join(out)


# --- top-level render ---------------------------------------------------------------


def kernel_source_for(
    baseline: DependencyBaseline, cmsis_nn_root: Optional[Path] = None
) -> KernelSource:
    """Resolve where the kernels come from: `--cmsis-nn-root`, else the pin.

    A local override is mirrored into `modules/` by every `nsx sync`, so the
    working tree stays the source of truth while it is in force -- that is the
    point of using it (editing kernels and rebuilding). It also makes the build
    unqualified, which the bundle and `doctor` both say out loud.
    """
    if cmsis_nn_root is None:
        return KernelSource(ref=baseline.project(CMSIS_NN_PROJECT).ref)
    path = Path(cmsis_nn_root).expanduser().resolve()
    for required in ("Include", "Source", "nsx"):
        if not (path / required).is_dir():
            raise AppRenderError(
                f"--cmsis-nn-root {path} does not look like an ns-cmsis-nn checkout "
                f"(missing {required}/). It must be a repository with a native nsx/ "
                f"module (ns-cmsis-nn >= v7.23.0)."
            )
    if not (path / "nsx" / "nsx-module.yaml").is_file():
        raise AppRenderError(
            f"--cmsis-nn-root {path} has no nsx/nsx-module.yaml, so NSX cannot use it "
            f"as a module. Use ns-cmsis-nn >= v7.23.0."
        )
    return KernelSource(path=path)


def synced_kernel_dir(app_dir: Path) -> Path:
    """Where `nsx sync` materialises the kernels for this app.

    NSX vendors a git-backed module as a whole-repository clone under
    `modules/<project>/`, so this is a complete ns-cmsis-nn tree -- including
    `Tests/`, which the generation step reads schemas and reference tables from.
    A `source: {path: ...}` module is mirrored to the same place on every sync.
    """
    return app_dir / "modules" / CMSIS_NN_PROJECT


def render_app(
    board: BoardSpec,
    *,
    repo_root: Path,
    build_dir: Path,
    baseline: DependencyBaseline,
    cmsis_nn_root: Optional[Path] = None,
    kernel_options: Optional[KernelOptions] = None,
    cmsis_core_include: Optional[Path] = None,
    build_size_probe: bool = False,
    toolchain: str = "arm-none-eabi-gcc",
    channel: str = "stable",
) -> AppRender:
    """Render the NSX app for `board` under `build_dir` and write it to disk."""
    render = plan_app(
        board,
        repo_root=repo_root,
        build_dir=build_dir,
        baseline=baseline,
        cmsis_nn_root=cmsis_nn_root,
        kernel_options=kernel_options,
        cmsis_core_include=cmsis_core_include,
        build_size_probe=build_size_probe,
        toolchain=toolchain,
        channel=channel,
    )
    write_app(render)
    return render


def plan_app(
    board: BoardSpec,
    *,
    repo_root: Path,
    build_dir: Path,
    baseline: DependencyBaseline,
    cmsis_nn_root: Optional[Path] = None,
    kernel_options: Optional[KernelOptions] = None,
    cmsis_core_include: Optional[Path] = None,
    build_size_probe: bool = False,
    toolchain: str = "arm-none-eabi-gcc",
    channel: str = "stable",
) -> AppRender:
    """The app this render *would* write, computed without touching the app tree.

    `hardware flash` uses this to compare the inputs in force now against the
    render state the last `hardware build` recorded, without becoming a build
    step itself (see `firmware_build.check_build_current`).
    """
    options = kernel_options or KernelOptions()
    app_dir = app_dir_for(build_dir)
    kernel_source = kernel_source_for(baseline, cmsis_nn_root)
    modules = resolve_modules(board.nsx_board, kernel_source)
    profile = starter_profile(board.nsx_board)
    ref_overrides = project_ref_overrides(modules, baseline)

    render = AppRender(
        app_dir=app_dir,
        board=board,
        modules=modules,
        kernel_source=kernel_source,
        kernel_options=options,
        baseline=baseline,
        nsx_yml=render_nsx_yml(
            board,
            modules,
            ref_overrides,
            render_module_registry(profile, ref_overrides),
            toolchain=toolchain,
            channel=channel,
        ),
        modules_cmake=render_modules_cmake(modules),
        cmakelists=render_cmakelists(
            board,
            repo_root=repo_root,
            kernel_options=options,
            kernel_source_dir=str(synced_kernel_dir(app_dir)),
            cmsis_core_include=(
                cmsis_core_include
                or repo_root / "artifacts" / "downloads" / "CMSIS_5" / "CMSIS" / "Core" / "Include"
            ),
            build_size_probe=build_size_probe,
        ),
    )
    return render


def write_app(render: AppRender) -> None:
    """Write the rendered files, touching only those whose content changed.

    `cmake/nsx/` belongs to NSX: it is reproduced wholesale from the pinned
    neuralspotx package on every lock/sync, `modules.cmake` included. So the
    placeholder module list is written only when the directory has none yet --
    overwriting NSX's resolved one with the bare name list would strip the
    per-module directory mappings and break the next configure.
    """
    app_dir = render.app_dir
    (app_dir / "cmake" / "nsx").mkdir(parents=True, exist_ok=True)
    _write_if_changed(app_dir / "nsx.yml", render.nsx_yml)
    modules_cmake = app_dir / "cmake" / "nsx" / "modules.cmake"
    if not modules_cmake.is_file():
        _write_if_changed(modules_cmake, render.modules_cmake)
    _write_if_changed(app_dir / "CMakeLists.txt", render.cmakelists)
    _write_if_changed(
        app_dir / RENDER_STATE, json.dumps(render.state(), indent=2, sort_keys=True) + "\n"
    )


def _write_if_changed(path: Path, text: str) -> bool:
    """Write `text` unless the file already holds exactly it.

    Not an optimisation: rewriting an unchanged CMakeLists.txt makes CMake
    re-run its whole configure on the next build, which would defeat the lock
    and build-dir reuse this module exists to enable.
    """
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def describe_modules(modules: Iterable[ModuleSpec], ref_overrides: Mapping[str, str]) -> str:
    """One-line-per-module summary, for `doctor` and verbose build output."""
    rows = []
    for spec in modules:
        if spec.local_path is not None:
            rows.append(f"{spec.name} <- path:{spec.local_path}")
        else:
            ref = ref_overrides.get(spec.project, "registry default")
            rows.append(f"{spec.name} <- {spec.project}@{ref}")
    return "\n".join(rows)
