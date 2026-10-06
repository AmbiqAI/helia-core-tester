"""Flash/RAM memory report for a board's linked firmware image.

One analysis (`arm-none-eabi-size/nm/objdump` on the ELF plus the board's NSX linker
script memory regions) feeds two reports:

- `generate_memory_report(board)`: the real `hct_benchmark_server` image. This is what
  `hardware memory-report` prints and what every result bundle's `memory_report.json`
  contains.
- `build_size_probe(board, variant)`: the universal size probe, which links the whole
  retained ns-cmsis-nn library for one integer/F16/F32 feature set to prove it fits
  the board before any firmware work. It builds as the same NSX app as the server,
  rendered with the probe as its only target.

Everything board-specific -- the SoC directory the linker script lives in and the
names of the flash and RAM regions in it -- comes from the board table.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .boards import DEFAULT_BOARD_ID, BoardSpec, repo_root, resolve_board
from .firmware_build import (
    SERVER_TARGET,
    _drop_foreign_cache,
    _jobs,
    bin_path,
    elf_path,
    ensure_build_tools,
    map_path,
    nsx_app_dir,
    stage_kernels,
)
from .pathutil import display_path, write_text_lf
from .toolchain import run_tool

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
# `objdump -h`: idx, name, size, VMA, LMA.
_HEADER_RE = re.compile(r"^\s*\d+\s+(\S+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s")
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SYMBOL_RE = re.compile(r"^[0-9a-fA-F]+\s+[A-Za-z]\s+(arm_[A-Za-z0-9_]+)$")


def linker_script_path(board: BoardSpec, sdk_root: Path) -> Path:
    """The board's SoC linker script under an NSX SDK tree.

    The SDK's `cmake/socs/<soc>.cmake` selects it (`NSX_LINKER_SCRIPT`) but never
    exports it to the CMake cache, so the same default path is rebuilt here from the
    board's SoC directory.
    """
    gcc_dir = sdk_root / "modules" / "nsx-core" / "src" / board.soc / "gcc"
    sbl, plain = gcc_dir / "linker_script_sbl.ld", gcc_dir / "linker_script.ld"
    # Apollo3/4 ship one script, no SBL.
    return plain if plain.is_file() and not sbl.is_file() else sbl


def app_linker_script(board: BoardSpec, build_dir: Path) -> Path:
    """The linker script the app was linked with."""
    from . import nsx_cli

    app_dir = nsx_app_dir(build_dir)
    sdk = nsx_cli.module_project("nsx-core")
    if app_dir.is_dir() and sdk:
        nsx_script = linker_script_path(board, app_dir / "modules" / sdk)
        # A failed first sync leaves none.
        if nsx_script.is_file():
            return nsx_script
    raise FileNotFoundError(f"No linker script for {build_dir}; rerun hardware build.")


def parse_memory_regions(linker_script: Path) -> list[dict[str, int | str]]:
    """`NAME (attrs) : ORIGIN = ..., LENGTH = ...` rows of the linker script's MEMORY block."""
    regions: list[dict[str, int | str]] = []
    # Drop block comments that span lines.
    text = _COMMENT_RE.sub("", linker_script.read_text(encoding="utf-8"))
    for line in text.splitlines():
        match = _MEMORY_RE.match(line)
        if match is None:
            continue
        name, origin, length = match.groups()
        regions.append({"name": name, "origin": int(origin, 0), "capacity": int(length, 0)})
    return regions


@dataclass(frozen=True)
class SectionHeader:
    name: str
    size: int
    vma: int
    lma: int
    flags: frozenset[str]


def parse_section_headers(output: str) -> list[SectionHeader]:
    """`objdump -h` rows with their flags."""
    lines = output.splitlines()
    rows: list[SectionHeader] = []
    for line, flags in zip(lines, lines[1:]):
        match = _HEADER_RE.match(line)
        if match is None:
            continue
        name, size, vma, lma = match.groups()
        rows.append(SectionHeader(
            name, int(size, 16), int(vma, 16), int(lma, 16),
            frozenset(flag.strip() for flag in flags.split(",")),
        ))
    return rows


def _in_region(address: int, region: dict[str, int | str]) -> bool:
    origin = int(region["origin"])
    return origin <= address < origin + int(region["capacity"])


def _section_totals(headers: list[SectionHeader]) -> dict[str, int]:
    # NSX scripts emit two `.text` sections.
    totals: dict[str, int] = {}
    for header in headers:
        totals[header.name] = totals.get(header.name, 0) + header.size
    return totals


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
    return run_tool(tool, args, project_root)


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


def analyze_elf(elf: Path, board: BoardSpec, linker_script: Path, project_root: Optional[Path] = None) -> ElfAnalysis:
    size_default = _probe_binary("arm-none-eabi-size", [str(elf)], project_root)
    size_sections = _probe_binary("arm-none-eabi-size", ["-A", str(elf)], project_root)
    nm_size_sort = _probe_binary("arm-none-eabi-nm", ["-S", "--size-sort", str(elf)], project_root)
    nm_symbols = _probe_binary("arm-none-eabi-nm", [str(elf)], project_root)
    objdump_headers = _probe_binary("arm-none-eabi-objdump", ["-h", str(elf)], project_root)

    headers = parse_section_headers(objdump_headers)
    sections = _section_totals(headers)
    memory_regions = parse_memory_regions(linker_script)
    region_map = {str(row["name"]): row for row in memory_regions}
    # Fail closed: a missing or mistyped board region is a configuration error, not a
    # zero-capacity region that the gates below would wave through.
    missing = [name for name in (board.flash_region, board.ram_region) if name not in region_map]
    if missing:
        raise ValueError(
            f"Linker script for board {board.id!r} defines no memory region(s) {missing}; "
            f"available regions: {sorted(region_map)}. Check flash_region/ram_region in the board table."
        )
    flash, ram = region_map[board.flash_region], region_map[board.ram_region]
    # Flash holds every loaded section's image.
    flash_image_bytes = sum(h.size for h in headers if "LOAD" in h.flags and _in_region(h.lma, flash))
    # RAM counts allocated sections placed there.
    ram_static_bytes = sum(
        h.size for h in headers if "ALLOC" in h.flags and h.name != ".heap" and _in_region(h.vma, ram)
    )
    heap_available_bytes = sections.get(".heap", 0)
    flash_capacity = int(flash["capacity"])
    ram_capacity = int(ram["capacity"])
    usage = {
        "flash_region": board.flash_region,
        "flash_image_bytes": flash_image_bytes,
        "flash_capacity_bytes": flash_capacity,
        "flash_free_bytes": max(0, flash_capacity - flash_image_bytes),
        "flash_percent_used": round((flash_image_bytes / flash_capacity) * 100, 2) if flash_capacity else None,
        "ram_region": board.ram_region,
        "ram_static_bytes": ram_static_bytes,
        "ram_capacity_bytes": ram_capacity,
        "ram_free_bytes_before_heap": max(0, ram_capacity - ram_static_bytes),
        "ram_percent_used_before_heap": round((ram_static_bytes / ram_capacity) * 100, 2) if ram_capacity else None,
        "heap_available_bytes": heap_available_bytes,
        "flash_gate_pass": flash_image_bytes <= int(flash_capacity * 0.75),
        "ram_gate_pass": ram_static_bytes <= int(ram_capacity * 0.75),
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
    project_root = project_root or repo_root()
    build_root = build_dir or board.build_dir(project_root)
    out_root = output_root or project_root / "artifacts" / "hardware" / "benchmark_server"
    out_root.mkdir(parents=True, exist_ok=True)

    elf = elf_path(build_root)
    if not elf.is_file():
        raise FileNotFoundError(f"Built firmware ELF not found: {elf} -- run `hardware build` for this board/build dir first.")
    analysis = analyze_elf(elf, board, app_linker_script(board, build_root), project_root)
    symbols = analysis.symbols
    retained = {name: name in symbols for name in _SELECTED_ADAPTERS}
    catalog = json.loads((project_root / "cmake" / "hardware" / "kernel_catalog.json").read_text(encoding="utf-8"))

    report = {
        "schema": "hct.memory_report",
        "schema_version": 2,
        "artifact": SERVER_TARGET,
        "target": {"board": board.id, "cpu": board.cpu},
        # Repo-relative for the default in-tree build dir, absolute for an external --build-dir.
        "artifacts": {
            "elf": display_path(elf, project_root),
            "bin": display_path(bin_path(build_root), project_root),
            "map": display_path(map_path(build_root), project_root),
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


SIZE_PROBE_VARIANTS: tuple[SizeProbeVariant, ...] = (
    SizeProbeVariant("int", False, False),
    SizeProbeVariant("int_f32", True, False),
    SizeProbeVariant("int_f16", False, True),
    SizeProbeVariant("int_f16_f32", True, True),
)


def build_size_probe(board: BoardSpec, variant: SizeProbeVariant, *, project_root: Optional[Path] = None) -> Path:
    """Configure, build and measure one size-probe variant for `board`; returns the
    directory holding its `memory_report.json` and raw tool outputs."""
    from . import nsx_cli
    from .nsx_app import SIZE_PROBE_TARGET, AppOptions, nested_kernel_root

    project_root = project_root or repo_root()
    # Board-keyed so two boards' probes in one checkout never share a CMake cache or
    # overwrite each other's report and raw tool outputs.
    probe_root = project_root / "artifacts" / "hardware" / "size_probe" / board.id / variant.name
    build_dir = probe_root / "build"
    # Same kernel default as hardware build.
    options = AppOptions(
        cmsis_nn_root=nested_kernel_root(project_root),
        enable_f32=variant.enable_f32,
        enable_f16=variant.enable_f16,
        build_size_probe=True,
    )
    ensure_build_tools(project_root)
    stage_kernels(board, build_dir=build_dir, options=options, repo_root=project_root)
    app_dir = nsx_app_dir(build_dir)
    # Pre-NSX probe caches name the repo.
    _drop_foreign_cache(build_dir, app_dir)
    nsx_cli.configure_app(app_dir, board.nsx_board, build_dir=build_dir, frozen=True)
    nsx_cli.build_app(
        app_dir, board=board.nsx_board, build_dir=build_dir, target=SIZE_PROBE_TARGET, jobs=_jobs(None), frozen=True,
    )

    out_dir = build_dir / "probe"
    elf = out_dir / f"{SIZE_PROBE_TARGET}.elf"
    analysis = analyze_elf(elf, board, app_linker_script(board, build_dir), project_root)

    report = {
        "schema": "hct.memory_report",
        "schema_version": 2,
        "variant": variant.name,
        "target": {"board": board.id, "cpu": board.cpu},
        "feature_set": {
            "integer": True,
            "f32": variant.enable_f32,
            "f16": variant.enable_f16,
        },
        "artifacts": {
            "elf": display_path(elf, project_root),
            "bin": display_path(out_dir / f"{SIZE_PROBE_TARGET}.bin", project_root),
            "map": display_path(out_dir / f"{SIZE_PROBE_TARGET}.map", project_root),
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
