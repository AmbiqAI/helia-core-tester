#!/usr/bin/env python3
"""Derive the CMSIS-NN kernels helia-rt calls.

Writes assets/deployed_entry_points.json from a helia-rt checkout and an
ns-cmsis-nn checkout. Rerun when either pin moves:

    uv run python scripts/derive_deployed_entry_points.py \\
        --helia-rt ../helia-rt --cmsis-nn ../ns-cmsis-nn
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from helia_core_tester.hardware.adapter_specs import timed_kernel_calls
from helia_core_tester.hardware.entry_coverage import DEPLOYED_PATH

KERNELS_DIR = "tensorflow/lite/micro/kernels/helia"
REPO_ROOT = Path(__file__).resolve().parents[1]


def public_entry_points(include_dir: Path) -> list[str]:
    """Kernels the public headers declare."""
    headers = sorted(include_dir.glob("arm_nnfunctions*.h"))
    return timed_kernel_calls("\n".join(h.read_text(encoding="utf-8") for h in headers))


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True).stdout.strip()


def _require_clean(checkout: Path, path: str) -> None:
    """Reject uncommitted changes under path."""
    if _git(checkout, "status", "--porcelain", "--", path):
        raise SystemExit(f"Uncommitted changes in {checkout / path}")


def derive(helia_rt: Path, cmsis_nn: Path) -> dict:
    """Public kernels the helia-rt sources call."""
    # Recorded refs must match scanned files.
    _require_clean(helia_rt, KERNELS_DIR)
    _require_clean(cmsis_nn, "Include")
    kernels = helia_rt / KERNELS_DIR
    sources = sorted([*kernels.glob("*.cc"), *kernels.glob("*.h")])
    if not sources:
        raise SystemExit(f"No sources under {kernels}")
    calls = timed_kernel_calls("\n".join(p.read_text(encoding="utf-8") for p in sources))
    # Support helpers (arm_memcpy_s8) are not public.
    public = set(public_entry_points(cmsis_nn / "Include"))
    return {
        "source": {
            "repo": "AmbiqAI/helia-rt",
            "commit": _git(helia_rt, "rev-parse", "HEAD"),
            "path": KERNELS_DIR,
            "cmsis_nn": _git(cmsis_nn, "describe", "--tags", "--always"),
        },
        "entry_points": [name for name in calls if name in public],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--helia-rt", type=Path, required=True, help="helia-rt checkout")
    parser.add_argument("--cmsis-nn", type=Path, required=True, help="ns-cmsis-nn checkout at the pinned ref")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEPLOYED_PATH)
    args = parser.parse_args()
    data = derive(args.helia_rt, args.cmsis_nn)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(data['entry_points'])} entry points to {args.out}")


if __name__ == "__main__":
    main()
