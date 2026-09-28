#!/usr/bin/env python3
"""Assert what the self-validate contract leg observed.

The leg runs `helia_core_tester contract inventory`, the contract-dependent pytest
modules, `doctor`, and (on the absent leg) one Convolve generation, and records their
exit codes, logs, the inventory report and a JUnit file. This script turns those into a
verdict that cannot pass by omission: a `present` leg needs a real inventory and the
real-tree tests to have run and passed; an `absent` leg needs the inventory refused, the
tests skipped, doctor saying so, and the migrated template failing closed.

    python3 scripts/assert_contract_summary.py --expect present --inventory-exit 0 \\
        --report artifacts/reports/contracts/inventory.json --junit junit.xml \\
        --test test_real_tree_symbols_are_declared_or_known_drift --doctor-log doctor.log
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPORT_SCHEMA = "hct.contract_inventory/1"
INVENTORY_EXIT_PRESENT = 0
INVENTORY_EXIT_ABSENT = 2
DOCTOR_PRESENT_MARK = "public functions"
DOCTOR_ABSENT_MARK = "kernel contract: absent"
GENERATE_ABSENT_MARK = "needs an ns-cmsis-nn that carries the export"


class SummaryError(Exception):
    """One observed fact contradicts the expected leg."""


def _read_text(path: Path, what: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise SummaryError(f"{what} {path} cannot be read ({error})") from None


def check_report(path: Path) -> dict:
    text = _read_text(path, "inventory report")
    try:
        document = json.loads(text)
    except ValueError as error:
        raise SummaryError(f"inventory report {path} is not valid JSON ({error})") from None
    if not isinstance(document, dict) or document.get("schema") != REPORT_SCHEMA:
        raise SummaryError(f"inventory report {path} does not carry schema {REPORT_SCHEMA!r}")
    covered = document.get("covered")
    cases = document.get("cases_scanned")
    if not isinstance(covered, dict) or not isinstance(cases, int):
        raise SummaryError(f"inventory report {path} is missing covered/cases_scanned")
    if cases <= 0:
        raise SummaryError(f"inventory report {path} scanned no generated cases")
    if not covered:
        raise SummaryError(f"inventory report {path} lists no covered function; an empty inventory is not coverage")
    return document


def junit_outcomes(path: Path, tests: list[str]) -> dict[str, str]:
    """`passed` | `skipped` | `failed` per requested test name; a name with no testcase
    is an error (the test did not run at all)."""
    text = _read_text(path, "junit file")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise SummaryError(f"junit file {path} is not valid XML ({error})") from None
    outcomes: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = case.get("name", "")
        for wanted in tests:
            if name == wanted or name.startswith(wanted + "["):
                if case.find("skipped") is not None:
                    outcome = "skipped"
                elif case.find("failure") is not None or case.find("error") is not None:
                    outcome = "failed"
                else:
                    outcome = "passed"
                previous = outcomes.get(wanted)
                outcomes[wanted] = outcome if previous in (None, "passed") else previous
    missing = [name for name in tests if name not in outcomes]
    if missing:
        raise SummaryError(f"junit file {path} has no testcase for {missing}; the test never ran")
    return outcomes


def assert_leg(*, expect: str, inventory_exit: int, report: Path | None, junit: Path | None,
               tests: list[str], doctor_log: Path | None, generate_log: Path | None) -> list[str]:
    facts: list[str] = []
    if expect == "present":
        if inventory_exit != INVENTORY_EXIT_PRESENT:
            raise SummaryError(f"inventory exited {inventory_exit}, expected {INVENTORY_EXIT_PRESENT} on a present leg")
        if report is None:
            raise SummaryError("a present leg needs --report")
        document = check_report(report)
        facts.append(f"inventory: {document['cases_scanned']} cases, {len(document['covered'])} covered functions")
        if tests:
            if junit is None:
                raise SummaryError("a present leg with --test needs --junit")
            for name, outcome in junit_outcomes(junit, tests).items():
                if outcome != "passed":
                    raise SummaryError(f"{name} {outcome}; a present leg must run it to a pass")
                facts.append(f"{name}: passed")
        if doctor_log is not None:
            if DOCTOR_PRESENT_MARK not in _read_text(doctor_log, "doctor log"):
                raise SummaryError(f"doctor log {doctor_log} does not report the contract ({DOCTOR_PRESENT_MARK!r})")
            facts.append("doctor: contract present")
        return facts
    if inventory_exit != INVENTORY_EXIT_ABSENT:
        raise SummaryError(f"inventory exited {inventory_exit}, expected {INVENTORY_EXIT_ABSENT} on an absent leg")
    facts.append("inventory: refused (exit 2)")
    if tests:
        if junit is None:
            raise SummaryError("an absent leg with --test needs --junit")
        for name, outcome in junit_outcomes(junit, tests).items():
            if outcome != "skipped":
                raise SummaryError(f"{name} {outcome}; an absent leg must skip it, not {outcome}")
            facts.append(f"{name}: skipped")
    if doctor_log is not None:
        if DOCTOR_ABSENT_MARK not in _read_text(doctor_log, "doctor log"):
            raise SummaryError(f"doctor log {doctor_log} does not say {DOCTOR_ABSENT_MARK!r}")
        facts.append("doctor: contract absent")
    if generate_log is not None:
        if GENERATE_ABSENT_MARK not in _read_text(generate_log, "generate log"):
            raise SummaryError(f"generate log {generate_log} does not show the migrated template failing closed "
                               f"({GENERATE_ABSENT_MARK!r})")
        facts.append("generate: migrated template failed closed")
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect", choices=("present", "absent"), required=True)
    parser.add_argument("--inventory-exit", type=int, required=True, help="exit code of `contract inventory`")
    parser.add_argument("--report", type=Path, help="inventory.json written by `contract inventory`")
    parser.add_argument("--junit", type=Path, help="pytest --junitxml output of the contract-dependent tests")
    parser.add_argument("--test", action="append", default=[], help="test function that must have run (repeatable)")
    parser.add_argument("--doctor-log", type=Path, help="captured `doctor` output")
    parser.add_argument("--generate-log", type=Path, help="captured output of the absent-leg Convolve generation")
    args = parser.parse_args(argv)
    try:
        facts = assert_leg(expect=args.expect, inventory_exit=args.inventory_exit, report=args.report,
                           junit=args.junit, tests=args.test, doctor_log=args.doctor_log,
                           generate_log=args.generate_log)
    except SummaryError as error:
        print(f"✗ contract leg ({args.expect}): {error}", file=sys.stderr)
        return 1
    for fact in facts:
        print(f"✓ {fact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
