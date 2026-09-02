"""Analyze the real benchmark-server firmware image."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .phase0 import _parse_memory_regions, _parse_size_a, _parse_top_symbols, _repo_root, _retained_kernel_count
from ..scripts.setup_dependencies import nsx_ambiq_sdk_dir


_SELECTED_ADAPTERS = (
    "arm_abs_s8",
    "arm_convolve_s8",
    "arm_add_s8",
    "arm_sub_s8",
    "arm_mul_s8",
    "arm_minimum_s8",
    "arm_maximum_s8",
)



def _probe_binary(tool: str, args: list[str]) -> str:
    return subprocess.run([tool, *args], capture_output=True, text=True, check=True).stdout



def generate_benchmark_server_memory_report(*, build_dir: Path | None = None, output_root: Path | None = None) -> Path:
    repo_root = _repo_root()
    build_root = build_dir or repo_root / "build" / "perf_stream" / "benchmark_server_gcc2"
    out_root = output_root or repo_root / "artifacts" / "perf_stream" / "benchmark_server"
    out_root.mkdir(parents=True, exist_ok=True)

    elf = build_root / "perf_stream" / "hct_benchmark_server.elf"
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
    symbols = set()
    for line in nm_symbols.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            symbols.add(parts[-1])
    retained = {name: name in symbols for name in _SELECTED_ADAPTERS}
    catalog = json.loads((repo_root / "cmake" / "perf_stream" / "kernel_catalog.json").read_text(encoding="utf-8"))

    report = {
        "schema": "hct.memory_report",
        "schema_version": 1,
        "artifact": "hct_benchmark_server",
        "target": {"board": "apollo510_evb", "cpu": "cortex-m55"},
        "artifacts": {
            "elf": str(elf.relative_to(repo_root)),
            "bin": str((build_root / "perf_stream" / "hct_benchmark_server.bin").relative_to(repo_root)),
            "map": str((build_root / "perf_stream" / "hct_benchmark_server.map").relative_to(repo_root)),
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
        "retained_public_kernel_count": _retained_kernel_count(nm_symbols),
        "verified_catalog_entries": retained,
        "kernel_catalog": catalog,
        "largest_symbols": _parse_top_symbols(nm_size_sort),
        "size_summary": size_default,
    }

    (out_root / "memory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (out_root / "size.txt").write_text(size_default, encoding="utf-8", newline="\n")
    (out_root / "size_A.txt").write_text(size_sections, encoding="utf-8", newline="\n")
    (out_root / "symbols.txt").write_text(nm_symbols, encoding="utf-8", newline="\n")
    (out_root / "symbols_size_sort.txt").write_text(nm_size_sort, encoding="utf-8", newline="\n")
    (out_root / "objdump_h.txt").write_text(objdump_headers, encoding="utf-8", newline="\n")
    (out_root / "kernel_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8", newline="\n")
    return out_root / "memory_report.json"


if __name__ == "__main__":
    path = generate_benchmark_server_memory_report()
    print(path)
