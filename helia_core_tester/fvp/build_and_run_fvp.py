#!/usr/bin/env python3
"""Compatibility facade for FVP build/run orchestration.

This module intentionally preserves the historical import and execution entrypoint:

    python -m helia_core_tester.fvp.build_and_run_fvp

Implementation is delegated to focused internal modules under ``helia_core_tester.fvp``.
"""

from __future__ import annotations

import sys
from typing import List

from .env import _resolve_downloaded_fvp_executable
from .errors import FvpScriptError
from .orchestrator import run_main
from .runner import ProcessRecord, ProcessSupervisor, _resolve_run_jobs

__all__ = [
    "ProcessRecord",
    "ProcessSupervisor",
    "_resolve_downloaded_fvp_executable",
    "_resolve_run_jobs",
    "main",
]


def main(argv: List[str]) -> int:
    try:
        return run_main(argv)
    except FvpScriptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
