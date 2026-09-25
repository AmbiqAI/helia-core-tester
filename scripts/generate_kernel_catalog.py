#!/usr/bin/env python3
"""Generate the firmware kernel-id defines, the kernel catalog C data and `kernel_catalog.json`
from the single source of truth, `assets/kernel_registry.yaml` (F008).

Prior to this generator, `cmake/hardware/benchmark_server_catalog.c` was a hand-maintained
C array that had drifted from the registry: it only listed 7 kernels (instead of the
registry's 126) and even had kernel_id 6/7 (Maximum/Minimum) reversed relative to both the
registry and the session dispatcher's `HCT_KERNEL_ID_*` defines. This script makes the
registry the sole identity source: the `HCT_KERNEL_ID_*` block of
`cmake/hardware/benchmark_server_adapters.h`, the firmware's compiled catalog array and
`kernel_catalog.json` are all regenerated from it, so they can never independently drift.

When an ns-cmsis-nn checkout with a kernel contract export is reachable (``--cmsis-nn-root``,
``CMSIS_NN_ROOT`` or the nested Tests/helia-core-tester layout), every `cmsis_function` in the
registry must be declared in that export; a checkout without the export is reported, not fatal.

Usage:
    python3 scripts/generate_kernel_catalog.py            # regenerate the three outputs in place
    python3 scripts/generate_kernel_catalog.py --check     # verify they're already up to date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helia_core_tester.contract.ir import ContractError, load_contract_set  # noqa: E402
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root  # noqa: E402
from helia_core_tester.hardware.kernel_registry import KernelEntry, load_kernel_registry  # noqa: E402

REGISTRY_RELPATH = Path("assets/kernel_registry.yaml")
CATALOG_C_RELPATH = Path("cmake/hardware/benchmark_server_catalog.c")
CATALOG_JSON_RELPATH = Path("cmake/hardware/kernel_catalog.json")
ADAPTERS_H_RELPATH = Path("cmake/hardware/benchmark_server_adapters.h")

KERNEL_ID_BLOCK_BEGIN = "/* BEGIN GENERATED kernel ids -- python3 scripts/generate_kernel_catalog.py */"
KERNEL_ID_BLOCK_END = "/* END GENERATED kernel ids */"

# kernel_id 6/7 are the pair the audit finding (F008) specifically called out as reversed in
# the old hand-maintained catalog: id 6 must be the *maximum* kernel, id 7 the *minimum* one,
# matching both the registry and HCT_KERNEL_ID_MAXIMUM_S8=6 / HCT_KERNEL_ID_MINIMUM_S8=7 in
# benchmark_server_adapters.h. This is asserted by the `--check` drift test below.
_EXPECTED_MAX_ID = 6
_EXPECTED_MIN_ID = 7
_EXPECTED_MAX_OPERATOR = "Maximum"
_EXPECTED_MIN_OPERATOR = "Minimum"

_HEADER = '#include "benchmark_server_catalog.h"\n'


def _load_registry(project_root: Path) -> list[KernelEntry]:
    return sorted(load_kernel_registry(project_root), key=lambda entry: entry.kernel_id)


def _validate_registry(kernels: list[KernelEntry]) -> None:
    ids = [entry.kernel_id for entry in kernels]
    if ids != sorted(ids):
        raise ValueError("assets/kernel_registry.yaml kernel_ids are not strictly ordered ascending")
    missing_family = [entry.kernel_id for entry in kernels if not entry.family]
    if missing_family:
        raise ValueError(f"assets/kernel_registry.yaml rows without a family: kernel_id={missing_family}")

    by_id = {entry.kernel_id: entry for entry in kernels}
    max_entry = by_id.get(_EXPECTED_MAX_ID)
    min_entry = by_id.get(_EXPECTED_MIN_ID)
    if max_entry is None or max_entry.operator != _EXPECTED_MAX_OPERATOR:
        raise ValueError(
            f"kernel_id={_EXPECTED_MAX_ID} must be the {_EXPECTED_MAX_OPERATOR} kernel "
            f"(matches HCT_KERNEL_ID_MAXIMUM_S8 in benchmark_server_adapters.h), found "
            f"{max_entry.operator if max_entry else None!r}"
        )
    if min_entry is None or min_entry.operator != _EXPECTED_MIN_OPERATOR:
        raise ValueError(
            f"kernel_id={_EXPECTED_MIN_ID} must be the {_EXPECTED_MIN_OPERATOR} kernel "
            f"(matches HCT_KERNEL_ID_MINIMUM_S8 in benchmark_server_adapters.h), found "
            f"{min_entry.operator if min_entry else None!r}"
        )


def validate_against_contract(kernels: list[KernelEntry], cmsis_nn_root: Optional[Path]) -> str:
    """Every cmsis_function must be declared in the checkout's kernel contract export.
    Returns a one-line status for the caller to print; raises ValueError on a stale row."""
    contracts = load_contract_set(cmsis_nn_root)
    if not contracts.present:
        where = contracts.path if contracts.path is not None else "no ns-cmsis-nn checkout"
        return f"kernel contract absent ({where}); cmsis_function names not validated"
    missing = [entry for entry in kernels if contracts.find(entry.cmsis_function) is None]
    if missing:
        shown = ", ".join(f"kernel_id={entry.kernel_id} {entry.cmsis_function}" for entry in missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise ValueError(
            f"assets/kernel_registry.yaml names {len(missing)} cmsis_function(s) not declared in "
            f"{contracts.path}: {shown}{more}"
        )
    return f"kernel contract {contracts.path}: all {len(kernels)} cmsis_function names declared"


def render_kernel_id_block(kernels: list[KernelEntry]) -> str:
    lines = [
        KERNEL_ID_BLOCK_BEGIN,
        "/* Kernel IDs sent by the host in CASE_META. Source of truth: the c_define column of",
        " * assets/kernel_registry.yaml (helia_core_tester/hardware/kernel_registry.py on the",
        " * host side); do not edit by hand. */",
    ]
    lines.extend(f"#define {entry.c_define} {entry.kernel_id}u" for entry in kernels)
    lines.append(KERNEL_ID_BLOCK_END)
    return "\n".join(lines) + "\n"


def splice_kernel_id_block(header_text: str, block: str) -> str:
    """Replace the marked block of the adapters header, keeping everything around it."""
    if header_text.count(KERNEL_ID_BLOCK_BEGIN) != 1 or header_text.count(KERNEL_ID_BLOCK_END) != 1:
        raise ValueError(
            f"{ADAPTERS_H_RELPATH} must contain exactly one '{KERNEL_ID_BLOCK_BEGIN}' and one "
            f"'{KERNEL_ID_BLOCK_END}' marker"
        )
    begin = header_text.index(KERNEL_ID_BLOCK_BEGIN)
    end = header_text.index(KERNEL_ID_BLOCK_END) + len(KERNEL_ID_BLOCK_END)
    if end < begin:
        raise ValueError(f"{ADAPTERS_H_RELPATH}: the END marker precedes the BEGIN marker")
    if header_text[end : end + 1] == "\n":
        end += 1
    return header_text[:begin] + block + header_text[end:]


def _catalog_entry(kernel: KernelEntry) -> dict:
    """Build one canonical catalog entry (shared by the JSON and C emitters) from a
    registry kernel record. The registry itself doesn't carry catalog-only metadata
    (api_version, adapter_schema_version, stateless/repeated_invocation_safe/mutates_input,
    scratch_bytes) -- every bridged kernel today is a stateless, repeat-safe, non-mutating,
    scratch-free adapter (per-case scratch is negotiated separately via CASE_META, not
    advertised in the catalog), so those are fixed defaults rather than per-kernel fields.
    """
    return {
        "kernel_id": kernel.kernel_id,
        "canonical_name": kernel.cmsis_function,
        "operator_family": str(kernel.family),
        "api_version": 1,
        "supported_dtype": kernel.dtype,
        "adapter_schema_version": 1,
        "stateless": True,
        "repeated_invocation_safe": True,
        "mutates_input": False,
        "scratch_bytes": 0,
    }


def _canonical_json_bytes(entries: list[dict]) -> bytes:
    """Pretty-printed JSON written to kernel_catalog.json (human-diffable)."""
    return json.dumps(entries, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _catalog_hash(entries: list[dict]) -> bytes:
    """Canonical *compact* JSON encoding used only for the TARGET_INFO hash -- must match the
    host's re-serialization of kernel_catalog.json (`sort_keys=True,
    separators=(",", ":")`), independent of the pretty file format above."""
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).digest()


def _format_c_bool(value: bool) -> str:
    return "true" if value else "false"


def _render_catalog_c(entries: list[dict]) -> str:
    lines = [_HEADER]
    lines.append("")
    lines.append("/* The PMU capability is decided by the device header of the hardware build")
    lines.append(" * (__PMU_PRESENT / __PMU_NUM_EVENTCNT from apollo*.h via am_mcu_apollo.h); a host")
    lines.append(" * compile or a DWT-only core advertises no PMU and zero slots. */")
    lines.append("#ifdef HELIA_HARDWARE_BUILD")
    lines.append('#include "am_mcu_apollo.h"')
    lines.append("#endif")
    lines.append("#if defined(__PMU_PRESENT) && (__PMU_PRESENT == 1)")
    lines.append("#define HCT_PMU_CAPABILITY_FLAGS HCT_CAP_PMU_ARMV8M")
    lines.append("#define HCT_PMU_COUNTER_SLOTS ((uint8_t)__PMU_NUM_EVENTCNT)")
    lines.append("#else")
    lines.append("#define HCT_PMU_CAPABILITY_FLAGS 0u")
    lines.append("#define HCT_PMU_COUNTER_SLOTS 0u")
    lines.append("#endif")
    lines.append("")
    lines.append("#ifndef HCT_BENCHMARK_SERVER_BOARD_ID")
    lines.append('#define HCT_BENCHMARK_SERVER_BOARD_ID "apollo510_evb"')
    lines.append("#endif")
    lines.append("")
    lines.append("#ifndef HCT_BENCHMARK_SERVER_TARGET_CPU")
    lines.append('#define HCT_BENCHMARK_SERVER_TARGET_CPU "cortex-m55"')
    lines.append("#endif")
    lines.append("")
    lines.append("/* GENERATED FILE -- do not edit by hand.")
    lines.append(" * Regenerate with: python3 scripts/generate_kernel_catalog.py")
    lines.append(" * Source of truth: assets/kernel_registry.yaml")
    lines.append(" */")
    lines.append("static const hct_kernel_catalog_entry_t g_hct_kernel_catalog[] = {")
    for entry in entries:
        lines.append(
            "    {{{kernel_id}u, \"{name}\", \"{family}\", {api}u, \"{dtype}\", {schema}u, "
            "{stateless}, {repeat_safe}, {mutates}, {scratch}u}},".format(
                kernel_id=entry["kernel_id"],
                name=entry["canonical_name"],
                family=entry["operator_family"],
                api=entry["api_version"],
                dtype=entry["supported_dtype"],
                schema=entry["adapter_schema_version"],
                stateless=_format_c_bool(entry["stateless"]),
                repeat_safe=_format_c_bool(entry["repeated_invocation_safe"]),
                mutates=_format_c_bool(entry["mutates_input"]),
                scratch=entry["scratch_bytes"],
            )
        )
    lines.append("};")
    lines.append("")

    digest = _catalog_hash(entries)
    lines.append("static const uint8_t g_hct_kernel_catalog_hash[32] = {")
    # Wrap at 8 bytes per line to match the previous hand-written style.
    digest_list = list(digest)
    for i in range(0, len(digest_list), 8):
        row = digest_list[i : i + 8]
        lines.append("    " + ", ".join(f"0x{byte:02x}u" for byte in row) + ",")
    lines.append("};")
    lines.append("")

    lines.append("const hct_kernel_catalog_entry_t *hct_benchmark_server_catalog(size_t *count)")
    lines.append("{")
    lines.append("    if (count != NULL)")
    lines.append("    {")
    lines.append("        *count = sizeof(g_hct_kernel_catalog) / sizeof(g_hct_kernel_catalog[0]);")
    lines.append("    }")
    lines.append("    return g_hct_kernel_catalog;")
    lines.append("}")
    lines.append("")
    lines.append("const uint8_t *hct_benchmark_server_catalog_hash(void)")
    lines.append("{")
    lines.append("    return g_hct_kernel_catalog_hash;")
    lines.append("}")
    lines.append("")
    lines.append("const char *hct_benchmark_server_board_id(void)")
    lines.append("{")
    lines.append("    return HCT_BENCHMARK_SERVER_BOARD_ID;")
    lines.append("}")
    lines.append("")
    lines.append("const char *hct_benchmark_server_target_cpu(void)")
    lines.append("{")
    lines.append("    return HCT_BENCHMARK_SERVER_TARGET_CPU;")
    lines.append("}")
    lines.append("")
    lines.append("/* The firmware build defines HCT_BENCHMARK_SERVER_BUILD_ID_PATCHED and links")
    lines.append(" * hct_build_id.c, whose slot scripts/patch_build_id.py fills in after the link;")
    lines.append(" * this constant only serves host-side unit builds of the protocol code. */")
    lines.append("#ifndef HCT_BENCHMARK_SERVER_BUILD_ID_PATCHED")
    lines.append("const char *hct_benchmark_server_build_id(void)")
    lines.append("{")
    lines.append('    return "hct-benchmark-server-v0";')
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append("uint32_t hct_benchmark_server_capability_flags(void)")
    lines.append("{")
    lines.append("    return HCT_CAP_CASE_STREAMING")
    lines.append("         | HCT_CAP_CORRECTNESS")
    lines.append("         | HCT_CAP_PERFORMANCE")
    lines.append("         | HCT_CAP_RTT_TRANSPORT")
    lines.append("         | HCT_CAP_KERNEL_CATALOG")
    lines.append("         | HCT_CAP_ABS_S8")
    lines.append("         | HCT_PMU_CAPABILITY_FLAGS;")
    lines.append("}")
    lines.append("")
    lines.append("uint8_t hct_benchmark_server_pmu_counter_slots(void)")
    lines.append("{")
    lines.append("    return HCT_PMU_COUNTER_SLOTS;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def generate(*, check: bool, project_root: Path = PROJECT_ROOT, cmsis_nn_root: Optional[Path] = None) -> int:
    kernels = _load_registry(project_root)
    _validate_registry(kernels)
    print(validate_against_contract(kernels, cmsis_nn_root))
    entries = [_catalog_entry(k) for k in kernels]

    catalog_json_path = project_root / CATALOG_JSON_RELPATH
    catalog_c_path = project_root / CATALOG_C_RELPATH
    adapters_h_path = project_root / ADAPTERS_H_RELPATH
    json_bytes = _canonical_json_bytes(entries)
    c_text = _render_catalog_c(entries)
    if not adapters_h_path.exists():
        raise ValueError(f"{adapters_h_path} does not exist; the kernel-id block has nowhere to go")
    header_before = adapters_h_path.read_text(encoding="utf-8")
    header_after = splice_kernel_id_block(header_before, render_kernel_id_block(kernels))

    if check:
        errors = []
        if not catalog_json_path.exists() or catalog_json_path.read_bytes() != json_bytes:
            errors.append(f"{catalog_json_path} is stale relative to {REGISTRY_RELPATH}")
        if not catalog_c_path.exists() or catalog_c_path.read_text(encoding="utf-8") != c_text:
            errors.append(f"{catalog_c_path} is stale relative to {REGISTRY_RELPATH}")
        if header_after != header_before:
            errors.append(f"{adapters_h_path} kernel-id block is stale relative to {REGISTRY_RELPATH}")
        if errors:
            for error in errors:
                print(f"✗ {error}", file=sys.stderr)
            print("Run `python3 scripts/generate_kernel_catalog.py` to regenerate.", file=sys.stderr)
            return 1
        print(f"✓ Kernel catalog and kernel-id defines are up to date ({len(entries)} kernels).")
        return 0

    catalog_json_path.write_bytes(json_bytes)
    catalog_c_path.write_text(c_text, encoding="utf-8")
    if header_after != header_before:
        adapters_h_path.write_text(header_after, encoding="utf-8")
    print(
        f"✓ Regenerated kernel catalog ({len(entries)} kernels): {catalog_json_path}, {catalog_c_path}, "
        f"{adapters_h_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated files are up to date; do not write.")
    parser.add_argument(
        "--cmsis-nn-root",
        type=Path,
        default=None,
        help="ns-cmsis-nn checkout whose kernel contract export validates cmsis_function names "
        "(default: CMSIS_NN_ROOT or the nested Tests/helia-core-tester layout).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="helia-core-tester tree holding the registry and the generated outputs (default: this checkout).",
    )
    args = parser.parse_args()
    cmsis_nn_root = args.cmsis_nn_root if args.cmsis_nn_root is not None else resolve_cmsis_nn_root()
    try:
        return generate(check=args.check, project_root=args.project_root, cmsis_nn_root=cmsis_nn_root)
    except (ValueError, ContractError, OSError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
