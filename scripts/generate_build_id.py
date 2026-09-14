#!/usr/bin/env python3
"""Stamp a benchmark-server firmware build with a content-derived build id.

Run by CMake as a custom command whenever any object that goes into
`hct_benchmark_server.elf` changes (see the `hct_build_id` rules in
CMakeLists.txt). It hashes the linked object set -- every object of the server
target plus the cmsis-nn archive -- and emits:

- `hct_build_id.c`, defining `hct_benchmark_server_build_id()` so the firmware
  advertises the id in its HELLO frame, and
- `hct_build_id.txt`, the same string for the host, which compares it against
  HELLO before trusting a "firmware unchanged, skip the flash" decision and
  before streaming any case.

Hashing objects (rather than the ELF) sidesteps the circularity of a file that
is itself linked into the ELF; two builds with identical objects legitimately
share an id, because they produce identical firmware.

Usage:
    generate_build_id.py --output-c <path> --output-txt <path> -- <object|archive>...
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

BUILD_ID_PREFIX = "hct-"
# HELLO carries the id as a u16-length text field inside a 256-byte payload
# shared with the catalog hash, board id and CPU name; 60 characters keeps
# plenty of headroom.
BUILD_ID_HEX_CHARS = 56


def compute_build_id(inputs: list[Path]) -> str:
    """`hct-<sha256 over the sorted inputs' contents>`, truncated to 60 characters."""
    digest = hashlib.sha256()
    for path in sorted(inputs, key=lambda p: str(p)):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return BUILD_ID_PREFIX + digest.hexdigest()[:BUILD_ID_HEX_CHARS]


def render_build_id_c(build_id: str) -> str:
    return (
        "/* GENERATED at build time by scripts/generate_build_id.py -- do not edit.\n"
        " * The string is a content hash of the objects linked into this firmware;\n"
        " * the host reads the same value from hct_build_id.txt next to the build.\n"
        " */\n"
        '#include "benchmark_server_catalog.h"\n'
        "\n"
        "const char *hct_benchmark_server_build_id(void)\n"
        "{\n"
        f'    return "{build_id}";\n'
        "}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-c", required=True, type=Path)
    parser.add_argument("--output-txt", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path, help="object files / archives to hash")
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.inputs if not p.is_file()]
    if missing:
        print(f"generate_build_id.py: missing input(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    build_id = compute_build_id(args.inputs)
    args.output_c.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_c.write_text(render_build_id_c(build_id), encoding="utf-8")
    args.output_txt.write_text(build_id + "\n", encoding="utf-8")
    print(f"[hct] firmware build id {build_id} ({len(args.inputs)} inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
