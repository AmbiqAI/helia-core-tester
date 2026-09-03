"""Phase-0 universal firmware size probe for streaming performance work."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..scripts.setup_dependencies import nsx_ambiq_sdk_dir


@dataclass(frozen=True)
class Variant:
    name: str
    enable_f32: bool
    enable_f16: bool


VARIANTS: tuple[Variant, ...] = (
    Variant("int", False, False),
    Variant("int_f32", True, False),
    Variant("int_f16", False, True),
    Variant("int_f16_f32", True, True),
)

_MEMORY_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*\([^)]*\)\s*:\s*ORIGIN\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*,\s*LENGTH\s*=\s*(0x[0-9A-Fa-f]+|\d+)"
)
_SYMBOL_RE = re.compile(r"^[0-9a-fA-F]+\s+[A-Za-z]\s+(arm_[A-Za-z0-9_]+)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


def _parse_memory_regions(linker_script: Path) -> list[dict[str, int | str]]:
    regions: list[dict[str, int | str]] = []
    for line in linker_script.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        match = _MEMORY_RE.match(line)
        if match is None:
            continue
        name, origin, length = match.groups()
        regions.append(
            {
                "name": name,
                "origin": int(origin, 0),
                "capacity": int(length, 0),
            }
        )
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


def _variant_defs(variant: Variant) -> list[str]:
    defs = [
        "-DHELIA_HARDWARE_BUILD=ON",
        "-DHELIA_BUILD_GENERATED_TESTS=OFF",
        "-DHELIA_BUILD_UNIVERSAL_SIZE_PROBE=ON",
        "-DHELIA_HARDWARE_BOARD=apollo510_evb",
        "-DTARGET_CPU=cortex-m55",
        f"-DARM_NN_ENABLE_F32={'ON' if variant.enable_f32 else 'OFF'}",
        f"-DARM_NN_ENABLE_F16={'ON' if variant.enable_f16 else 'OFF'}",
    ]
    return defs


def _cmake_configure(repo_root: Path, build_dir: Path, variant: Variant) -> None:
    toolchain = repo_root / "cmake" / "nsx" / "toolchains" / "arm-none-eabi-gcc.cmake"
    cmd = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        *(_variant_defs(variant)),
    ]
    _run(cmd, cwd=repo_root)


def _cmake_build(repo_root: Path, build_dir: Path) -> None:
    _run(["cmake", "--build", str(build_dir), "--target", "hct_universal_size_probe"], cwd=repo_root)


def _probe_binary(tool: str, args: Iterable[str]) -> str:
    return subprocess.run([tool, *args], capture_output=True, text=True, check=True).stdout


def build_variant(variant: Variant) -> Path:
    repo_root = _repo_root()
    phase0_root = repo_root / "artifacts" / "perf_stream" / "phase0" / variant.name
    build_dir = phase0_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _cmake_configure(repo_root, build_dir, variant)
    _cmake_build(repo_root, build_dir)

    out_dir = build_dir / "probe"
    elf = out_dir / "hct_universal_size_probe.elf"
    linker_script = nsx_ambiq_sdk_dir(repo_root) / "modules" / "nsx-core" / "src" / "apollo510" / "gcc" / "linker_script_sbl.ld"

    size_default = _probe_binary("arm-none-eabi-size", [str(elf)])
    size_sections = _probe_binary("arm-none-eabi-size", ["-A", str(elf)])
    nm_size_sort = _probe_binary("arm-none-eabi-nm", ["-S", "--size-sort", str(elf)])
    nm_symbols = _probe_binary("arm-none-eabi-nm", [str(elf)])
    objdump_headers = _probe_binary("arm-none-eabi-objdump", ["-h", str(elf)])

    sections = _parse_size_a(size_sections)
    memory_regions = _parse_memory_regions(linker_script)
    region_map = {str(row["name"]): int(row["capacity"]) for row in memory_regions}
    flash_image_bytes = sections.get(".text", 0) + sections.get(".itcm_text", 0) + sections.get(".data", 0)
    tcm_static_bytes = sections.get(".stack", 0) + sections.get(".data", 0) + sections.get(".bss", 0)
    heap_available_bytes = sections.get(".heap", 0)
    flash_capacity = region_map.get("MCU_MRAM", 0)
    tcm_capacity = region_map.get("MCU_TCM", 0)

    report = {
        "schema": "hct.memory_report",
        "schema_version": 1,
        "variant": variant.name,
        "feature_set": {
            "integer": True,
            "f32": variant.enable_f32,
            "f16": variant.enable_f16,
        },
        "artifacts": {
            "elf": str(elf.relative_to(repo_root)),
            "bin": str((out_dir / "hct_universal_size_probe.bin").relative_to(repo_root)),
            "map": str((out_dir / "hct_universal_size_probe.map").relative_to(repo_root)),
        },
        "memory_regions": memory_regions,
        "sections": sections,
        "usage": {
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
        },
        "size_summary": size_default,
        "retained_public_kernel_count": _retained_kernel_count(nm_symbols),
        "largest_symbols": _parse_top_symbols(nm_size_sort),
    }
    (phase0_root / "memory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (phase0_root / "size.txt").write_text(size_default, encoding="utf-8", newline="\n")
    (phase0_root / "size_A.txt").write_text(size_sections, encoding="utf-8", newline="\n")
    (phase0_root / "symbols_size_sort.txt").write_text(nm_size_sort, encoding="utf-8", newline="\n")
    (phase0_root / "symbols.txt").write_text(nm_symbols, encoding="utf-8", newline="\n")
    (phase0_root / "objdump_h.txt").write_text(objdump_headers, encoding="utf-8", newline="\n")
    return phase0_root


def main() -> int:
    for variant in VARIANTS:
        build_variant(variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
