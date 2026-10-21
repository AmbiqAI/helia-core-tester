#!/usr/bin/env python3
"""Refresh the unit-test fallback contract (helia_core_tester/tests/fixtures/contract/bound_operators)
from an ns-cmsis-nn checkout that ships the export, the implementation manifest and ns_cmsis_nn_contracts.

The fixture is a slice of that checkout: the export filtered to BOUND_PREFIXES, the public headers it is
checked against, the reader package verbatim, and the manifest with its signature_sha256 rebound to the
filtered export (the only field that differs from the checkout's).

    uv run python scripts/refresh_contract_fixture.py <ns-cmsis-nn checkout>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

TESTER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTER_ROOT))

from helia_core_tester.contract.ir import CONTRACT_RELPATH, MANIFEST_RELPATH, PACKAGE_RELPATH  # noqa: E402
from helia_core_tester.tests.contract_fallback import BOUND_PREFIXES, FIXTURE_ROOT  # noqa: E402

HEADERS = ("arm_nnfunctions.h", "arm_nnfunctions_flt.h", "arm_nnsupportfunctions.h")


def fail(message: str) -> int:
    print(f"refresh_contract_fixture: {message}", file=sys.stderr)
    return 1


def manifest_symbols(record: dict) -> set[str]:
    symbols = {record["entry"]}
    if record["routed_by"] != "direct":
        symbols.add(record["routed_by"]["wrapper"])
    symbols.update(b["size_query"]["symbol"] for b in record["buffers"] if b["size_query"])
    symbols.update(step["symbol"] for step in record["prepare"])
    eligibility = record["eligibility"]
    if eligibility and eligibility["predicate"] and not eligibility["predicate_static"]:
        symbols.add(eligibility["predicate"])
    return symbols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args(argv)
    root = args.checkout.resolve()
    for relative in (CONTRACT_RELPATH, MANIFEST_RELPATH, PACKAGE_RELPATH / "__init__.py"):
        if not (root / relative).is_file():
            return fail(f"{root} has no {relative}")
    for header in HEADERS:
        if not (root / "Include" / header).is_file():
            return fail(f"{root} has no Include/{header}")

    try:
        export = json.loads((root / CONTRACT_RELPATH).read_text(encoding="utf-8"))
        manifest = json.loads((root / MANIFEST_RELPATH).read_text(encoding="utf-8"))
        functions = [record for record in export["functions"] if record["name"].startswith(BOUND_PREFIXES)]
        referenced = set().union(*(manifest_symbols(record) for record in manifest["implementations"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return fail(f"cannot read the contract files of {root} ({type(error).__name__}: {error})")
    if not functions:
        return fail(f"no function of {root / CONTRACT_RELPATH} matches BOUND_PREFIXES")
    outside = sorted(referenced - {record["name"] for record in functions})
    if outside:
        return fail(f"the manifest references {outside}, which BOUND_PREFIXES leaves out of the fixture; "
                    "add their prefixes to helia_core_tester/tests/contract_fallback.py")

    sliced = json.dumps({"schema": export["schema"], "functions": functions}, indent=2) + "\n"
    manifest["signature_sha256"] = hashlib.sha256(sliced.encode("utf-8")).hexdigest()

    (FIXTURE_ROOT / CONTRACT_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (FIXTURE_ROOT / CONTRACT_RELPATH).write_text(sliced, encoding="utf-8")
    (FIXTURE_ROOT / MANIFEST_RELPATH).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for header in HEADERS:
        shutil.copyfile(root / "Include" / header, FIXTURE_ROOT / "Include" / header)
    package = FIXTURE_ROOT / PACKAGE_RELPATH
    shutil.rmtree(package, ignore_errors=True)
    package.mkdir(parents=True)
    for source in sorted((root / PACKAGE_RELPATH).glob("*.py")):
        shutil.copyfile(source, package / source.name)
    print(f"{FIXTURE_ROOT}: {len(functions)} functions, {len(manifest['implementations'])} implementations, "
          f"reader from {root / PACKAGE_RELPATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
