#!/usr/bin/env python3
"""Generate the firmware kernel catalog C data and `kernel_catalog.json` from the single
source of truth, `assets/kernel_registry.yaml` (F008).

Prior to this generator, `cmake/perf_stream/benchmark_server_catalog.c` was a hand-maintained
C array that had drifted from the registry: it only listed 7 kernels (instead of the
registry's 126) and even had kernel_id 6/7 (Maximum/Minimum) reversed relative to both the
registry and the session dispatcher's `HCT_KERNEL_ID_*` defines. This script makes the
registry the catalog's sole identity source -- both the firmware's compiled catalog array and
`kernel_catalog.json` are regenerated from it, so they can never independently drift again.

Usage:
    python3 scripts/generate_kernel_catalog.py            # regenerate both files in place
    python3 scripts/generate_kernel_catalog.py --check     # verify they're already up to date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "assets" / "kernel_registry.yaml"
_CATALOG_C_PATH = _PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_catalog.c"
_CATALOG_JSON_PATH = _PROJECT_ROOT / "cmake" / "perf_stream" / "kernel_catalog.json"

# kernel_id 6/7 are the pair the audit finding (F008) specifically called out as reversed in
# the old hand-maintained catalog: id 6 must be the *maximum* kernel, id 7 the *minimum* one,
# matching both the registry and HCT_KERNEL_ID_MAXIMUM_S8=6 / HCT_KERNEL_ID_MINIMUM_S8=7 in
# benchmark_server_session.c. This is asserted by the `--check` drift test below.
_EXPECTED_MAX_ID = 6
_EXPECTED_MIN_ID = 7
_EXPECTED_MAX_OPERATOR = "Maximum"
_EXPECTED_MIN_OPERATOR = "Minimum"

_HEADER = '#include "benchmark_server_catalog.h"\n'


def _load_registry() -> list[dict]:
    data = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    kernels = data.get("kernels", [])
    return sorted(kernels, key=lambda entry: int(entry["kernel_id"]))


def _validate_registry(kernels: list[dict]) -> None:
    ids = [int(entry["kernel_id"]) for entry in kernels]
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"assets/kernel_registry.yaml has duplicate kernel_id(s): {duplicates}")
    if ids != sorted(ids):
        raise ValueError("assets/kernel_registry.yaml kernel_ids are not strictly ordered ascending")

    by_id = {int(entry["kernel_id"]): entry for entry in kernels}
    max_entry = by_id.get(_EXPECTED_MAX_ID)
    min_entry = by_id.get(_EXPECTED_MIN_ID)
    if max_entry is None or max_entry.get("operator") != _EXPECTED_MAX_OPERATOR:
        raise ValueError(
            f"kernel_id={_EXPECTED_MAX_ID} must be the {_EXPECTED_MAX_OPERATOR} kernel "
            f"(matches HCT_KERNEL_ID_MAXIMUM_S8 in benchmark_server_session.c), found "
            f"{max_entry.get('operator') if max_entry else None!r}"
        )
    if min_entry is None or min_entry.get("operator") != _EXPECTED_MIN_OPERATOR:
        raise ValueError(
            f"kernel_id={_EXPECTED_MIN_ID} must be the {_EXPECTED_MIN_OPERATOR} kernel "
            f"(matches HCT_KERNEL_ID_MINIMUM_S8 in benchmark_server_session.c), found "
            f"{min_entry.get('operator') if min_entry else None!r}"
        )


def _catalog_entry(kernel: dict) -> dict:
    """Build one canonical catalog entry (shared by the JSON and C emitters) from a
    registry kernel record. The registry itself doesn't carry catalog-only metadata
    (api_version, adapter_schema_version, stateless/repeated_invocation_safe/mutates_input,
    scratch_bytes) -- every bridged kernel today is a stateless, repeat-safe, non-mutating,
    scratch-free adapter (per-case scratch is negotiated separately via CASE_META, not
    advertised in the catalog), so those are fixed defaults rather than per-kernel fields.
    """
    return {
        "kernel_id": int(kernel["kernel_id"]),
        "canonical_name": str(kernel["cmsis_function"]),
        "operator_family": str(kernel["family"]),
        "api_version": 1,
        "supported_dtype": str(kernel["dtype"]),
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
    """Canonical *compact* JSON encoding used only for the HELLO hash -- must match the
    host's re-serialization of kernel_catalog.json (`sort_keys=True,
    separators=(",", ":")`), independent of the pretty file format above."""
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).digest()


def _format_c_bool(value: bool) -> str:
    return "true" if value else "false"


def _render_catalog_c(entries: list[dict]) -> str:
    lines = [_HEADER]
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
    hash_bytes = ", ".join(f"0x{byte:02x}u" for byte in digest)
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
    lines.append("const char *hct_benchmark_server_build_id(void)")
    lines.append("{")
    lines.append('    return "hct-benchmark-server-v0";')
    lines.append("}")
    lines.append("")
    lines.append("uint32_t hct_benchmark_server_capability_flags(void)")
    lines.append("{")
    lines.append("    return HCT_CAP_CASE_STREAMING")
    lines.append("         | HCT_CAP_CORRECTNESS")
    lines.append("         | HCT_CAP_PERFORMANCE")
    lines.append("         | HCT_CAP_RTT_TRANSPORT")
    lines.append("         | HCT_CAP_KERNEL_CATALOG")
    lines.append("         | HCT_CAP_ABS_S8;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def generate(*, check: bool) -> int:
    kernels = _load_registry()
    _validate_registry(kernels)
    entries = [_catalog_entry(k) for k in kernels]

    json_bytes = _canonical_json_bytes(entries)
    c_text = _render_catalog_c(entries)

    if check:
        errors = []
        if not _CATALOG_JSON_PATH.exists() or _CATALOG_JSON_PATH.read_bytes() != json_bytes:
            errors.append(f"{_CATALOG_JSON_PATH} is stale relative to assets/kernel_registry.yaml")
        if not _CATALOG_C_PATH.exists() or _CATALOG_C_PATH.read_text(encoding="utf-8") != c_text:
            errors.append(f"{_CATALOG_C_PATH} is stale relative to assets/kernel_registry.yaml")
        if errors:
            for error in errors:
                print(f"✗ {error}", file=sys.stderr)
            print("Run `python3 scripts/generate_kernel_catalog.py` to regenerate.", file=sys.stderr)
            return 1
        print(f"✓ Kernel catalog is up to date ({len(entries)} kernels).")
        return 0

    _CATALOG_JSON_PATH.write_bytes(json_bytes)
    _CATALOG_C_PATH.write_text(c_text, encoding="utf-8")
    print(f"✓ Regenerated kernel catalog ({len(entries)} kernels): {_CATALOG_JSON_PATH}, {_CATALOG_C_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated files are up to date; do not write.")
    args = parser.parse_args()
    try:
        return generate(check=args.check)
    except ValueError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
