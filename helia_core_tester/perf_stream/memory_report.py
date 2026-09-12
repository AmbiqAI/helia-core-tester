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
from typing import Iterable, Optional

from .boards import DEFAULT_BOARD_ID, BoardSpec, repo_root, resolve_board
from ..scripts.setup_dependencies import nsx_ambiq_sdk_dir

SERVER_TARGET = "hct_benchmark_server"
SIZE_PROBE_TARGET = "hct_universal_size_probe"

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


def linker_script_path(board: BoardSpec, project_root: Optional[Path] = None) -> Path:
    """The NSX SDK linker script the firmware for `board` is linked with.

    The SDK's `cmake/socs/<soc>.cmake` selects it (`NSX_LINKER_SCRIPT`) but never
    exports it to the CMake cache, so the same default path is rebuilt here from the
    board's SoC directory.
    """
    sdk = nsx_ambiq_sdk_dir(project_root or repo_root())
    return sdk / "modules" / "nsx-core" / "src" / board.soc / "gcc" / "linker_script_sbl.ld"


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


def _probe_binary(tool: str, args: Iterable[str]) -> str:
    return subprocess.run([tool, *args], capture_output=True, text=True, check=True).stdout


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
        (out_root / "size.txt").write_text(self.size_summary, encoding="utf-8", newline="\n")
        (out_root / "size_A.txt").write_text(self.size_sections, encoding="utf-8", newline="\n")
        (out_root / "symbols.txt").write_text(self.nm_symbols, encoding="utf-8", newline="\n")
        (out_root / "symbols_size_sort.txt").write_text(self.nm_size_sort, encoding="utf-8", newline="\n")
        (out_root / "objdump_h.txt").write_text(self.objdump_headers, encoding="utf-8", newline="\n")


def analyze_elf(elf: Path, board: BoardSpec, project_root: Optional[Path] = None) -> ElfAnalysis:
    size_default = _probe_binary("arm-none-eabi-size", [str(elf)])
    size_sections = _probe_binary("arm-none-eabi-size", ["-A", str(elf)])
    nm_size_sort = _probe_binary("arm-none-eabi-nm", ["-S", "--size-sort", str(elf)])
    nm_symbols = _probe_binary("arm-none-eabi-nm", [str(elf)])
    objdump_headers = _probe_binary("arm-none-eabi-objdump", ["-h", str(elf)])

    sections = _parse_size_a(size_sections)
    memory_regions = parse_memory_regions(linker_script_path(board, project_root))
    region_map = {str(row["name"]): int(row["capacity"]) for row in memory_regions}
    flash_image_bytes = sections.get(".text", 0) + sections.get(".itcm_text", 0) + sections.get(".data", 0)
    tcm_static_bytes = sections.get(".stack", 0) + sections.get(".data", 0) + sections.get(".bss", 0)
    heap_available_bytes = sections.get(".heap", 0)
    flash_capacity = region_map.get(board.flash_region, 0)
    tcm_capacity = region_map.get(board.ram_region, 0)
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
        "flash_gate_pass": flash_capacity == 0 or flash_image_bytes <= int(flash_capacity * 0.75),
        "tcm_gate_pass": tcm_capacity == 0 or tcm_static_bytes <= int(tcm_capacity * 0.75),
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
    out_root = output_root or project_root / "artifacts" / "perf_stream" / "benchmark_server"
    out_root.mkdir(parents=True, exist_ok=True)

    elf = build_root / "perf_stream" / f"{SERVER_TARGET}.elf"
    analysis = analyze_elf(elf, board, project_root)
    symbols = analysis.symbols
    retained = {name: name in symbols for name in _SELECTED_ADAPTERS}
    catalog = json.loads((project_root / "cmake" / "perf_stream" / "kernel_catalog.json").read_text(encoding="utf-8"))

    report = {
        "schema": "hct.memory_report",
        "schema_version": 1,
        "artifact": SERVER_TARGET,
        "target": {"board": board.id, "cpu": board.cpu},
        "artifacts": {
            "elf": str(elf.relative_to(project_root)),
            "bin": str((build_root / "perf_stream" / f"{SERVER_TARGET}.bin").relative_to(project_root)),
            "map": str((build_root / "perf_stream" / f"{SERVER_TARGET}.map").relative_to(project_root)),
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

    (out_root / "memory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    analysis.write_tool_outputs(out_root)
    (out_root / "kernel_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8", newline="\n")
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


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


def _configure_size_probe(project_root: Path, build_dir: Path, board: BoardSpec, variant: SizeProbeVariant) -> None:
    toolchain = project_root / "cmake" / "nsx" / "toolchains" / "arm-none-eabi-gcc.cmake"
    cmd = [
        "cmake",
        "-S",
        str(project_root),
        "-B",
        str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        "-DHELIA_HARDWARE_BUILD=ON",
        "-DHELIA_BUILD_GENERATED_TESTS=OFF",
        "-DHELIA_BUILD_UNIVERSAL_SIZE_PROBE=ON",
        f"-DHELIA_HARDWARE_BOARD={board.nsx_board}",
        f"-DTARGET_CPU={board.cpu}",
        f"-DARM_NN_ENABLE_F32={'ON' if variant.enable_f32 else 'OFF'}",
        f"-DARM_NN_ENABLE_F16={'ON' if variant.enable_f16 else 'OFF'}",
    ]
    _run(cmd, cwd=project_root)


def build_size_probe(board: BoardSpec, variant: SizeProbeVariant, *, project_root: Optional[Path] = None) -> Path:
    """Configure, build and measure one size-probe variant for `board`; returns the
    directory holding its `memory_report.json` and raw tool outputs."""
    project_root = project_root or repo_root()
    probe_root = project_root / "artifacts" / "perf_stream" / "size_probe" / variant.name
    build_dir = probe_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _configure_size_probe(project_root, build_dir, board, variant)
    _run(["cmake", "--build", str(build_dir), "--target", SIZE_PROBE_TARGET], cwd=project_root)

    out_dir = build_dir / "probe"
    elf = out_dir / f"{SIZE_PROBE_TARGET}.elf"
    analysis = analyze_elf(elf, board, project_root)

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
        "artifacts": {
            "elf": str(elf.relative_to(project_root)),
            "bin": str((out_dir / f"{SIZE_PROBE_TARGET}.bin").relative_to(project_root)),
            "map": str((out_dir / f"{SIZE_PROBE_TARGET}.map").relative_to(project_root)),
        },
        "memory_regions": analysis.memory_regions,
        "sections": analysis.sections,
        "usage": analysis.usage,
        "size_summary": analysis.size_summary,
        "retained_public_kernel_count": analysis.retained_public_kernel_count,
        "largest_symbols": analysis.largest_symbols,
    }
    (probe_root / "memory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    analysis.write_tool_outputs(probe_root)
    return probe_root


def main() -> int:
    board = resolve_board(DEFAULT_BOARD_ID)
    for variant in SIZE_PROBE_VARIANTS:
        print(build_size_probe(board, variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
