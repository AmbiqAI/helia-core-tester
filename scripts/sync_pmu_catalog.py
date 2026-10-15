#!/usr/bin/env python3
"""Copy the PMU event catalog from the nsx-pmu-armv8m module.

Mirrors helia-profiler's tools/sync_pmu_catalog.py. The source defaults to the
module copy in a synced hardware app (`hardware build` syncs one).

Usage:
    python3 scripts/sync_pmu_catalog.py [path/to/armv8m_pmu_events.json]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helia_core_tester.hardware.pmu_catalog import synced_module_catalogs  # noqa: E402

DESTINATION = PROJECT_ROOT / "assets" / "pmu" / "armv8m_pmu_events.json"


def main(argv: list[str]) -> int:
    sources = [Path(argv[1]).resolve()] if len(argv) > 1 else list(synced_module_catalogs(PROJECT_ROOT))
    if not sources or not sources[0].is_file():
        print("No module catalog found. Pass its path.", file=sys.stderr)
        return 1
    shutil.copyfile(sources[0], DESTINATION)
    print(f"Synced PMU catalog from {sources[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
