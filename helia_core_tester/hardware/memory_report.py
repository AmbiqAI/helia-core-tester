"""Flash/RAM memory report for a board's linked firmware image.

One analysis (`arm-none-eabi-size/nm/objdump` on the ELF plus the board's NSX linker
script memory regions) feeds two reports:

- `generate_memory_report(board)`: the real `hct_benchmark_server` image. This is what
  `hardware memory-report` prints and what every result bundle's `memory_report.json`
  contains.
- `build_size_probe(board, variant)`: the universal size probe, which links the whole
  retained ns-cmsis-nn library for one integer/F16/F32 feature set to prove it fits
  the board before any firmware work.

Everything board-specific -- the SoC directory the linker script lives in and the
names of the flash and RAM regions in it -- comes from the board table.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from .boards import DEFAULT_BOARD_ID, BoardSpec, repo_root, resolve_board
from .nsx_app import SERVER_TARGET, SIZE_PROBE_TARGET, KernelOptions, app_dir_for, read_render_state
from .pathutil import display_path, write_text_lf
from .toolchain import arm_tool

if TYPE_CHECKING:
    from .firmware_build import FirmwareOptions

# Catalog kernels whose symbols are checked for retention in the linked server image.
_SELECTED_ADAPTERS = (
    "arm_abs_s8",
    "arm_convolve_s8",
    "arm_add_s8",
    "arm_sub_s8",
    "arm_mul_s8",
    "arm_minimum_s8",
    "arm_maximum_s8",
)

_MEMORY_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*\([^)]*\)\s*:\s*ORIGIN\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*,\s*LENGTH\s*=\s*(0x[0-9A-Fa-f]+|\d+)"
)
_SYMBOL_RE = re.compile(r"^[0-9a-fA-F]+\s+[A-Za-z]\s+(arm_[A-Za-z0-9_]+)$")


_LINK_SCRIPT_RE = re.compile(r"-T\s*(\S+\.ld)")


def linker_script_path(board: BoardSpec, build_dir: Path) -> Path:
    """The NSX linker script the firmware in `build_dir` was actually linked with.

    The board fragment selects it (`NSX_LINKER_SCRIPT`) and puts it on the board
    flags target as a `-T` link option; it never reaches the CMake cache. So it
    is read back out of the generated Ninja file -- the same string the linker
    saw -- rather than reconstructed from a path convention that the SDK is free
    to change under us. A build tree that has no Ninja file yet (or an NSX that
    lays the flag out differently) falls back to the SoC's default script in the
    synced SDK module.
    """
    from .firmware_build import output_dir

    ninja = output_dir(build_dir, board) / "build.ninja"
    if ninja.is_file():
        for match in _LINK_SCRIPT_RE.finditer(ninja.read_text(encoding="utf-8", errors="ignore")):
            candidate = Path(match.group(1))
            if candidate.is_file():
                return candidate
    from .nsx_app import app_dir_for

    fallback = (
        app_dir_for(build_dir)
        / "modules" / "nsx-ambiq-sdk" / "modules" / "nsx-core"
        / "src" / board.soc / "gcc" / "linker_script_sbl.ld"
    )
    if not fallback.is_file():
        raise FileNotFoundError(
            f"Cannot determine the linker script for board {board.id!r}: neither "
            f"{ninja} names one nor does {fallback} exist. Run `hardware build --board "
            f"{board.id}` first."
        )
    return fallback


def parse_memory_regions(linker_script: Path) -> list[dict[str, int | str]]:
    """`NAME (attrs) : ORIGIN = ..., LENGTH = ...` rows of the linker script's MEMORY block."""
    regions: list[dict[str, int | str]] = []
    for line in linker_script.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        match = _MEMORY_RE.match(line)
        if match is None:
            continue
        name, origin, length = match.groups()
        regions.append({"name": name, "origin": int(origin, 0), "capacity": int(length, 0)})
    return regions


def _parse_size_a(output: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("."):
            try:
                sections[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return sections


def _parse_top_symbols(output: str, *, limit: int = 20) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for line in output.splitlines()[-limit:]:
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        size_hex, _kind, name = parts[1], parts[2], parts[3]
        try:
            size_value = int(size_hex, 16)
        except ValueError:
            continue
        rows.append({"name": name, "size": size_value})
    return list(reversed(rows))


def _retained_kernel_count(output: str) -> int:
    count = 0
    for line in output.splitlines():
        match = _SYMBOL_RE.match(line.strip())
        if match and "_get_buffer_size" not in match.group(1):
            count += 1
    return count


def _probe_binary(tool: str, args: Iterable[str], project_root: Optional[Path] = None) -> str:
    # `project_root` picks that checkout's downloaded toolchain (arm_tool), so a custom
    # checkout does not fall back to whatever binutils happen to be on PATH.
    return subprocess.run([arm_tool(tool, project_root), *args], capture_output=True, text=True, check=True).stdout


@dataclass(frozen=True)
class ElfAnalysis:
    """Everything measured from one linked ELF against its board's memory map."""

    sections: dict[str, int]
    memory_regions: list[dict[str, int | str]]
    usage: dict[str, object]
    size_summary: str
    size_sections: str
    nm_size_sort: str
    nm_symbols: str
    objdump_headers: str

    @property
    def retained_public_kernel_count(self) -> int:
        return _retained_kernel_count(self.nm_symbols)

    @property
    def largest_symbols(self) -> list[dict[str, int | str]]:
        return _parse_top_symbols(self.nm_size_sort)

    @property
    def symbols(self) -> set[str]:
        names = set()
        for line in self.nm_symbols.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                names.add(parts[-1])
        return names

    def write_tool_outputs(self, out_root: Path) -> None:
        write_text_lf(out_root / "size.txt", self.size_summary)
        write_text_lf(out_root / "size_A.txt", self.size_sections)
        write_text_lf(out_root / "symbols.txt", self.nm_symbols)
        write_text_lf(out_root / "symbols_size_sort.txt", self.nm_size_sort)
        write_text_lf(out_root / "objdump_h.txt", self.objdump_headers)


def analyze_elf(
    elf: Path, board: BoardSpec, build_dir: Path, project_root: Optional[Path] = None
) -> ElfAnalysis:
    size_default = _probe_binary("arm-none-eabi-size", [str(elf)], project_root)
    size_sections = _probe_binary("arm-none-eabi-size", ["-A", str(elf)], project_root)
    nm_size_sort = _probe_binary("arm-none-eabi-nm", ["-S", "--size-sort", str(elf)], project_root)
    nm_symbols = _probe_binary("arm-none-eabi-nm", [str(elf)], project_root)
    objdump_headers = _probe_binary("arm-none-eabi-objdump", ["-h", str(elf)], project_root)

    sections = _parse_size_a(size_sections)
    memory_regions = parse_memory_regions(linker_script_path(board, build_dir))
    region_map = {str(row["name"]): int(row["capacity"]) for row in memory_regions}
    flash_image_bytes = sections.get(".text", 0) + sections.get(".itcm_text", 0) + sections.get(".data", 0)
    tcm_static_bytes = sections.get(".stack", 0) + sections.get(".data", 0) + sections.get(".bss", 0)
    heap_available_bytes = sections.get(".heap", 0)
    # Fail closed: a missing or mistyped board region is a configuration error, not a
    # zero-capacity region that the gates below would wave through.
    missing = [name for name in (board.flash_region, board.ram_region) if name not in region_map]
    if missing:
        raise ValueError(
            f"Linker script for board {board.id!r} defines no memory region(s) {missing}; "
            f"available regions: {sorted(region_map)}. Check flash_region/ram_region in the board table."
        )
    flash_capacity = region_map[board.flash_region]
    tcm_capacity = region_map[board.ram_region]
    usage = {
        "flash_image_bytes": flash_image_bytes,
        "flash_capacity_bytes": flash_capacity,
        "flash_free_bytes": max(0, flash_capacity - flash_image_bytes),
        "flash_percent_used": round((flash_image_bytes / flash_capacity) * 100, 2) if flash_capacity else None,
        "tcm_static_bytes": tcm_static_bytes,
        "tcm_capacity_bytes": tcm_capacity,
        "tcm_free_bytes_before_heap": max(0, tcm_capacity - tcm_static_bytes),
        "tcm_percent_used_before_heap": round((tcm_static_bytes / tcm_capacity) * 100, 2) if tcm_capacity else None,
        "heap_available_bytes": heap_available_bytes,
        "flash_gate_pass": flash_image_bytes <= int(flash_capacity * 0.75),
        "tcm_gate_pass": tcm_static_bytes <= int(tcm_capacity * 0.75),
    }
    return ElfAnalysis(
        sections=sections,
        memory_regions=memory_regions,
        usage=usage,
        size_summary=size_default,
        size_sections=size_sections,
        nm_size_sort=nm_size_sort,
        nm_symbols=nm_symbols,
        objdump_headers=objdump_headers,
    )


def generate_memory_report(
    board: BoardSpec,
    *,
    project_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Path:
    """Write `memory_report.json` (plus the raw size/nm/objdump outputs and the kernel
    catalog) for the board's linked benchmark-server firmware and return its path."""
    from .firmware_build import bin_path, elf_path, map_path

    project_root = project_root or repo_root()
    build_root = build_dir or board.build_dir(project_root)
    out_root = output_root or project_root / "artifacts" / "hardware" / "benchmark_server"
    out_root.mkdir(parents=True, exist_ok=True)

    elf = elf_path(build_root, board)
    if not elf.is_file():
        raise FileNotFoundError(f"Built firmware ELF not found: {elf} -- run `hardware build` for this board/build dir first.")
    analysis = analyze_elf(elf, board, build_root, project_root)
    symbols = analysis.symbols
    retained = {name: name in symbols for name in _SELECTED_ADAPTERS}
    catalog = json.loads((project_root / "cmake" / "hardware" / "kernel_catalog.json").read_text(encoding="utf-8"))

    # What this image was built from, taken from the build dir's own render state
    # rather than re-resolved: the report has to describe the ELF in front of it,
    # not what a fresh render would produce today.
    render_state = read_render_state(app_dir_for(build_root)) or {}
    # The same reasoning one level deeper: the lock-derived provenance the build
    # wrote next to this ELF. `memory_report.json` is a bundle artifact of its
    # own and is read on its own (`hardware memory-report`), so it carries the
    # block rather than pointing at a sibling file.
    from .provenance import read_provenance

    dependencies = read_provenance(build_root, board)

    report = {
        "schema": "hct.memory_report",
        "schema_version": 1,
        "artifact": SERVER_TARGET,
        "target": {"board": board.id, "cpu": board.cpu},
        "baseline_id": render_state.get("baseline_id"),
        "baseline_fingerprint": render_state.get("baseline_fingerprint"),
        "kernel_source": render_state.get("kernel_source"),
        "kernel_options": render_state.get("kernel_options"),
        # Repo-relative for the default in-tree build dir, absolute for an external --build-dir.
        "artifacts": {
            "elf": display_path(elf, project_root),
            "bin": display_path(bin_path(build_root, board), project_root),
            "map": display_path(map_path(build_root, board), project_root),
        },
        "memory_regions": analysis.memory_regions,
        "sections": analysis.sections,
        "usage": analysis.usage,
        "retained_public_kernel_count": analysis.retained_public_kernel_count,
        "verified_catalog_entries": retained,
        "kernel_catalog": catalog,
        "largest_symbols": analysis.largest_symbols,
        "size_summary": analysis.size_summary,
    }
    if dependencies is not None:
        report["dependencies"] = dependencies

    write_text_lf(out_root / "memory_report.json", json.dumps(report, indent=2))
    analysis.write_tool_outputs(out_root)
    write_text_lf(out_root / "kernel_catalog.json", json.dumps(catalog, indent=2))
    return out_root / "memory_report.json"


# --- universal size probe --------------------------------------------------------------


@dataclass(frozen=True)
class SizeProbeVariant:
    name: str
    enable_f32: bool
    enable_f16: bool

    def kernel_options(self, *, requantize_inline_asm: bool = True) -> KernelOptions:
        return KernelOptions(
            requantize_inline_asm=requantize_inline_asm,
            enable_f32=self.enable_f32,
            enable_f16=self.enable_f16,
        )


SIZE_PROBE_VARIANTS: tuple[SizeProbeVariant, ...] = (
    SizeProbeVariant("int", False, False),
    SizeProbeVariant("int_f32", True, False),
    SizeProbeVariant("int_f16", False, True),
    SizeProbeVariant("int_f16_f32", True, True),
)


def size_probe_build_dir(build_dir: Path, variant: SizeProbeVariant) -> Path:
    """The probe's own build dir, a sibling of the server's under the same board dir.

    The float switches are ns-cmsis-nn cache options baked into the rendered app
    (an `option()` default cannot be overridden after the module is added), so a
    variant is a different render and needs its own app -- it cannot be a second
    target in the server's. `nsx sync` copies each module out of the shared
    content-addressed cache, so the extra apps cost disk, not network.
    """
    return build_dir / f"probe_{variant.name}"


def build_size_probe(
    board: BoardSpec,
    variant: SizeProbeVariant,
    *,
    project_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    options: Optional["FirmwareOptions"] = None,
) -> Path:
    """Configure, build and measure one size-probe variant for `board`; returns the
    directory holding its `memory_report.json` and raw tool outputs."""
    from .firmware_build import (
        FirmwareOptions,
        build as nsx_build,
        configure as nsx_configure,
        load_baseline,
        lock_and_sync,
        prepare_app,
    )
    from .nsx_app import render_app

    project_root = project_root or repo_root()
    options = options or FirmwareOptions()
    build_root = build_dir or board.build_dir(project_root)
    probe_build_dir = size_probe_build_dir(build_root, variant)
    # Board-keyed so two boards' probes in one checkout never share a build tree or
    # overwrite each other's report and raw tool outputs.
    probe_root = project_root / "artifacts" / "hardware" / "size_probe" / board.id / variant.name
    probe_root.mkdir(parents=True, exist_ok=True)

    # Same renderer as the firmware, with the probe's target and this variant's
    # kernel switches -- so the probe measures the library the firmware would get,
    # not a hand-mirrored approximation of it.
    prepare_app(board, repo_root=project_root, build_dir=probe_build_dir, options=options)
    render = render_app(
        board,
        repo_root=project_root,
        build_dir=probe_build_dir,
        baseline=load_baseline(project_root, options),
        cmsis_nn_root=options.cmsis_nn_root,
        kernel_options=variant.kernel_options(
            requantize_inline_asm=options.requantize_inline_asm
        ),
        build_size_probe=True,
    )
    lock_and_sync(render, options)
    nsx_configure(render, options)
    nsx_build(render, options, SIZE_PROBE_TARGET, None)

    elf = render.build_dir / f"{SIZE_PROBE_TARGET}.elf"
    analysis = analyze_elf(elf, board, probe_build_dir, project_root)

    report = {
        "schema": "hct.memory_report",
        "schema_version": 1,
        "variant": variant.name,
        "target": {"board": board.id, "cpu": board.cpu},
        "feature_set": {
            "integer": True,
            "f32": variant.enable_f32,
            "f16": variant.enable_f16,
        },
        "kernel_source": render.kernel_source.describe(),
        "baseline_id": render.baseline.baseline_id,
        "artifacts": {
            "elf": display_path(elf, project_root),
            "bin": display_path(render.build_dir / f"{SIZE_PROBE_TARGET}.bin", project_root),
            "map": display_path(render.build_dir / f"{SIZE_PROBE_TARGET}.map", project_root),
        },
        "memory_regions": analysis.memory_regions,
        "sections": analysis.sections,
        "usage": analysis.usage,
        "size_summary": analysis.size_summary,
        "retained_public_kernel_count": analysis.retained_public_kernel_count,
        "largest_symbols": analysis.largest_symbols,
    }
    write_text_lf(probe_root / "memory_report.json", json.dumps(report, indent=2))
    analysis.write_tool_outputs(probe_root)
    return probe_root


def main() -> int:
    board = resolve_board(DEFAULT_BOARD_ID)
    for variant in SIZE_PROBE_VARIANTS:
        print(build_size_probe(board, variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
