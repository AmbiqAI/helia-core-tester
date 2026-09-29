#!/usr/bin/env python3
"""Regenerate `cmake/hardware/benchmark_server_adapters.gen.c` -- the per-kernel
firmware adapter bodies and their kernel-id dispatch -- from the single source of truth
in `helia_core_tester/hardware/adapter_specs.py`.

Run this after editing `adapter_specs.py` (e.g. adding a new bridged kernel's firmware C
body, or changing an existing one). The whole file is generated; the hand-written session
state machine in `benchmark_server_session.c` is untouched.

Usage:
    python scripts/generate_hardware_adapters.py            # regenerate in place
    python scripts/generate_hardware_adapters.py --check    # exit 1 if regeneration
                                                                 # would change the file
                                                                 # (CI/pre-commit drift check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from helia_core_tester.contract.ir import ContractError, load_contract_set  # noqa: E402
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root  # noqa: E402
from helia_core_tester.hardware.adapter_specs import render_generated_adapters_source  # noqa: E402

ADAPTERS_C_PATH = PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_adapters.gen.c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the file would change instead of writing it.")
    parser.add_argument(
        "--cmsis-nn-root",
        type=Path,
        default=None,
        help="ns-cmsis-nn checkout whose kernel contract export renders the templated adapter bodies "
        "(default: CMSIS_NN_ROOT or the nested Tests/helia-core-tester layout).",
    )
    args = parser.parse_args()
    cmsis_nn_root = args.cmsis_nn_root if args.cmsis_nn_root is not None else resolve_cmsis_nn_root()

    original_text = ADAPTERS_C_PATH.read_text(encoding="utf-8") if ADAPTERS_C_PATH.exists() else None
    try:
        updated_text = render_generated_adapters_source(load_contract_set(cmsis_nn_root))
    except ContractError as exc:
        print(f"{ADAPTERS_C_PATH}: cannot render -- {exc}", file=sys.stderr)
        return 1

    if updated_text == original_text:
        print(f"{ADAPTERS_C_PATH}: already up to date.")
        return 0

    if args.check:
        print(f"{ADAPTERS_C_PATH}: OUT OF DATE -- rerun scripts/generate_hardware_adapters.py", file=sys.stderr)
        return 1

    ADAPTERS_C_PATH.write_text(updated_text, encoding="utf-8")
    print(f"{ADAPTERS_C_PATH}: regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
