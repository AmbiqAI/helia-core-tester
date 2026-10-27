#!/usr/bin/env python3
"""The contract-conformance CI job's helper (self-validate.yml).

    uv run python scripts/conformance_leg.py names
        Print the conformance cases (descriptors with `conformance: true`), comma-separated, for --name.

    uv run python scripts/conformance_leg.py check --expect present|absent --cpu CPU [--generated-tests-dir DIR]
        Assert what one generation of those cases observed, from its manifest:
        present: every case was generated or skipped for a capability the CPU lacks, at least one was generated,
                 and none was skipped for a missing planner;
        absent:  every case was skipped for a missing planner (status skipped_conformance, with a reason) and none
                 was generated, so the leg degrades explicitly instead of passing by omission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TESTER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTER_ROOT))

from helia_core_tester.generation.io.descriptors import load_all_descriptors  # noqa: E402

CONFORMANCE_SKIP = "skipped_conformance"
CAPABILITY_SKIP = "skipped_capability"


class LegError(Exception):
    """An observation contradicts the expected leg."""


def conformance_names() -> list[str]:
    names = [d["name"] for d in load_all_descriptors(str(TESTER_ROOT / "assets" / "descriptors"))
             if d.get("conformance")]
    if not names:
        raise LegError("no descriptor has conformance: true")
    return names


def check(expect: str, manifest_path: Path, names: list[str]) -> list[str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LegError(f"{manifest_path} cannot be read as the generation manifest ({error})") from None
    generated = {t.get("name") for t in manifest.get("tests", [])}
    skipped = {s.get("name"): s for s in manifest.get("skipped", [])}
    wanted = set(names)
    unaccounted = sorted(wanted - generated - set(skipped))
    if unaccounted:
        raise LegError(f"conformance cases neither generated nor skipped: {unaccounted}")
    if expect == "present":
        planner_skips = sorted(n for n in wanted if skipped.get(n, {}).get("status") == CONFORMANCE_SKIP)
        if planner_skips:
            raise LegError(f"skipped for a missing planner on a present leg: {planner_skips}")
        other = sorted(n for n in wanted & set(skipped) if skipped[n].get("status") != CAPABILITY_SKIP)
        if other:
            raise LegError(f"skipped for a reason other than a capability: {other}")
        ran = sorted(wanted & generated)
        if not ran:
            raise LegError("no conformance case was generated; an empty leg is not conformance")
        return [f"{len(ran)} generated, {len(wanted & set(skipped))} capability-skipped"]
    if wanted & generated:
        raise LegError(f"generated without a planner on an absent leg: {sorted(wanted & generated)}")
    bad = sorted(n for n in wanted if skipped[n].get("status") != CONFORMANCE_SKIP or not skipped[n].get("reason"))
    if bad:
        raise LegError(f"not skipped for a missing planner with a reason: {bad}")
    return [f"{len(wanted)} skipped for a missing planner"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("names")
    checker = sub.add_parser("check")
    checker.add_argument("--expect", choices=("present", "absent"), required=True)
    checker.add_argument("--cpu", required=True)
    checker.add_argument("--generated-tests-dir", type=Path, default=TESTER_ROOT / "artifacts" / "generated_tests")
    args = parser.parse_args(argv)
    try:
        names = conformance_names()
        if args.command == "names":
            print(",".join(names))
            return 0
        facts = check(args.expect, args.generated_tests_dir / "int" / args.cpu / "manifest.json", names)
    except LegError as error:
        print(f"✗ conformance leg: {error}", file=sys.stderr)
        return 1
    for fact in facts:
        print(f"✓ conformance ({args.expect}, {args.cpu}): {fact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
