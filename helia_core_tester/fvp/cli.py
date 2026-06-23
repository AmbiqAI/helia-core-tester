"""CLI parser for FVP build/run orchestrator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_arg_parser(default_downloads_dir: Path, default_source_dir: Path) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build and run helia-core kernels unit tests on FVP Corstone-300 (Python).")
    ap.add_argument("-c", "--cpu", default="cortex-m55", help="Comma-separated cores, e.g. m0,m4,m55")
    ap.add_argument("--suite", choices=["int", "float"], default="int", help="Suite selector for artifact partitioning")
    ap.add_argument("-o", "--opt", default="-Ofast", help="Optimization level passed via CMSIS_OPTIMIZATION_LEVEL")
    ap.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3], default=0,
                   help="Output verbosity level (0=minimal, 1=progress, 2=commands, 3=debug)")
    ap.add_argument("-b", "--no-build", action="store_true", help="Skip build (only run)")
    ap.add_argument("-r", "--no-run", action="store_true", help="Skip run (only build)")
    ap.add_argument("-e", "--no-setup", action="store_true", help="Skip dependency setup")
    ap.add_argument("-a", "--use-arm-compiler", action="store_true", help="Use Arm Compiler (default: GCC)")
    ap.add_argument("-p", "--no-venv", action="store_true", help="(Kept for parity; no effect on CMake build)")
    ap.add_argument("-f", "--no-fvp-from-download", action="store_true", help="Do NOT use downloaded FVP; use FVP from PATH")
    ap.add_argument("-g", "--no-gcc-from-download", action="store_true", help="Do NOT use downloaded GCC; use system GCC")
    ap.add_argument("-u", "--ethos-path", type=Path, help="Override ethos-u-core-platform path")
    ap.add_argument("-C", "--cmsis5-path", type=Path, help="Override CMSIS_5 path")
    ap.add_argument("-D", "--cmake-def", action="append", default=[], help="Extra -DVAR=VAL for CMake (repeatable)")
    ap.add_argument("--coverage", action="store_true", help="Enable ns-cmsis-nn coverage instrumentation and gcda merge")
    ap.add_argument("--downloads-dir", type=Path, default=default_downloads_dir, help="Downloads directory (default: ./artifacts/downloads)")
    ap.add_argument("--source-dir", type=Path, default=default_source_dir, help="CMake source dir (UnitTest root)")
    ap.add_argument("--generator", help="CMake generator (e.g. Ninja)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="Parallel build jobs")
    ap.add_argument("--run-jobs", type=int, default=1, help="Parallel FVP test jobs (0 = auto/use all host cores)")
    ap.add_argument("--timeout-run", type=float, default=0.0, help="Per-test timeout in seconds (0 = none)")
    ap.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False, help="Stop on first failure")
    ap.add_argument("--fvp-arg", action="append", default=[], help="Extra args to pass to the FVP (repeatable)")
    ap.add_argument("--no-report", action="store_true", help="Disable comprehensive test reporting (enabled by default)")
    ap.add_argument("--report-formats", nargs="+", choices=["json", "html", "md", "junit"], default=["json"],
                   help="Report formats to generate (default: json)")
    ap.add_argument("--quiet", action="store_true", help="Quiet mode (no output)")
    return ap
