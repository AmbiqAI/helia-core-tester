#!/usr/bin/env python3
"""Regenerate the auto-generated per-kernel adapter block inside
`cmake/perf_stream/benchmark_server_session.c` from the single source of truth in
`helia_core_tester/perf_stream/adapter_specs.py`.

Run this after editing `adapter_specs.py` (e.g. adding a new bridged kernel's firmware C
body, or changing an existing one). The script replaces everything between the
`GENERATED_BLOCK_BEGIN`/`GENERATED_BLOCK_END` marker comments in
`benchmark_server_session.c` with freshly rendered content and leaves the rest of the file
untouched.

Usage:
    python scripts/generate_perf_stream_adapters.py            # regenerate in place
    python scripts/generate_perf_stream_adapters.py --check    # exit 1 if regeneration
                                                                 # would change the file
                                                                 # (CI/pre-commit drift check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from helia_core_tester.perf_stream.adapter_specs import (  # noqa: E402
    GENERATED_BLOCK_BEGIN,
    GENERATED_BLOCK_END,
    render_generated_adapters_block,
)

SESSION_C_PATH = PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_session.c"


def splice(original_text: str, generated_block: str) -> str:
    begin_index = original_text.index(GENERATED_BLOCK_BEGIN)
    end_index = original_text.index(GENERATED_BLOCK_END) + len(GENERATED_BLOCK_END)
    return original_text[:begin_index] + generated_block.rstrip("\n") + original_text[end_index:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the file would change instead of writing it.")
    args = parser.parse_args()

    original_text = SESSION_C_PATH.read_text(encoding="utf-8")
    generated_block = render_generated_adapters_block()
    updated_text = splice(original_text, generated_block)

    if updated_text == original_text:
        print(f"{SESSION_C_PATH}: already up to date.")
        return 0

    if args.check:
        print(f"{SESSION_C_PATH}: OUT OF DATE -- rerun scripts/generate_perf_stream_adapters.py", file=sys.stderr)
        return 1

    SESSION_C_PATH.write_text(updated_text, encoding="utf-8")
    print(f"{SESSION_C_PATH}: regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
