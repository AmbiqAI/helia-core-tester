"""
Merge multiple LCOV coverage.info files (e.g. per float-precision passes)
into a single coverage.info by summing per-file/per-line hit counts.

Used by CI to combine independent f32/f16 float coverage runs into one
canonical coverage.info before handing it to `helia_core_tester coverage-merge`,
which expects exactly one coverage.info per (suite, cpu) pair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_lcov(path: Path) -> Dict[str, Dict[int, int]]:
    """Parse an LCOV file into {source_file: {line_no: hit_count}}."""
    files: Dict[str, Dict[int, int]] = {}
    current_file: str | None = None

    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("SF:"):
            current_file = line[3:].strip()
            files.setdefault(current_file, {})
            continue
        if not current_file:
            continue
        if line.startswith("DA:"):
            payload = line[3:].split(",")
            if len(payload) >= 2:
                try:
                    line_no = int(payload[0])
                    hit_count = int(payload[1])
                except ValueError:
                    continue
                lines = files[current_file]
                lines[line_no] = lines.get(line_no, 0) + hit_count
            continue

    return files


def merge_lcov_files(inputs: List[Path]) -> Dict[str, Dict[int, int]]:
    merged: Dict[str, Dict[int, int]] = {}
    for input_path in inputs:
        if not input_path.exists():
            continue
        parsed = _parse_lcov(input_path)
        for source_file, lines in parsed.items():
            merged_lines = merged.setdefault(source_file, {})
            for line_no, hit_count in lines.items():
                merged_lines[line_no] = merged_lines.get(line_no, 0) + hit_count
    return merged


def _write_lcov(merged: Dict[str, Dict[int, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for source_file in sorted(merged.keys()):
        line_hits = merged[source_file]
        lines.append(f"SF:{source_file}")
        for line_no in sorted(line_hits.keys()):
            lines.append(f"DA:{line_no},{line_hits[line_no]}")
        lh = sum(1 for count in line_hits.values() if count > 0)
        lf = len(line_hits)
        lines.append(f"LH:{lh}")
        lines.append(f"LF:{lf}")
        lines.append("end_of_record")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple LCOV coverage.info files by summing per-line hit counts."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        dest="inputs",
        help="Path to an input coverage.info file. May be passed multiple times. Missing paths are skipped.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the merged coverage.info")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_paths = [Path(item) for item in args.inputs]
    existing = [path for path in input_paths if path.exists()]
    if not existing:
        print(f"error: none of the requested input coverage.info files exist: {input_paths}", file=sys.stderr)
        return 1

    merged = merge_lcov_files(existing)
    _write_lcov(merged, args.output)
    print(f"Merged {len(existing)}/{len(input_paths)} input(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
