"""Temporary ns-cmsis-nn checkouts for tests that read a kernel contract.

A checkout with a signature export needs the ns_cmsis_nn_contracts package that reads it
(contract/ir.py); tests install the copy vendored in the fallback fixture, which
test_contract_fallback keeps equal to the real checkout's.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from helia_core_tester.contract.ir import CONTRACT_RELPATH, MANIFEST_RELPATH, PACKAGE_RELPATH
from helia_core_tester.tests.contract_fallback import FIXTURE_ROOT

READER = FIXTURE_ROOT / PACKAGE_RELPATH


def install_reader(root: Path) -> Path:
    shutil.copytree(READER, Path(root) / PACKAGE_RELPATH, dirs_exist_ok=True)
    return Path(root)


def function(name: str, params: Sequence[tuple[str, str, str]] = (), *, returns: str = "arm_cmsis_nn_status",
             header: str = "Include/arm_nnfunctions.h", guards: Sequence[str] = ()) -> dict:
    """An export record; ``params`` are (type, name, direction)."""
    return {"name": name, "header": header, "line": 1, "guards": list(guards), "returns": returns,
            "params": [{"direction": d, "name": n, "type": t} for t, n, d in params]}


def implementation(impl_id: str, entry: str, *, requires: Iterable[str] = (), since: str = "7.0.0",
                   op: str = "depthwise_conv", wrapper: Optional[str] = None, route: Optional[str] = None,
                   declines: Sequence[str] = ()) -> dict:
    """A manifest record the ns_cmsis_nn_contracts loader accepts."""
    routed_by = "direct" if wrapper is None else {"wrapper": wrapper, "route": route, "requires": [], "when": "test"}
    return {"id": impl_id, "family": "test", "op": op, "quantization": "per_tensor", "dtypes": {}, "layout": {},
            "entry": entry, "operands": None, "routed_by": routed_by, "requires": sorted(requires),
            "varies_with": [], "eligibility": None, "buffers": [], "prepare": [],
            "declines": [{"cause": cause, "when": "test"} for cause in declines], "since": since,
            "numerics": {"reference": None, "claim": "bit_exact", "holds_with": []}, "note": None}


def _declaration(record: Mapping) -> str:
    params = ", ".join(f"{p['type']} {p['name']}" for p in record["params"]) or "void"
    return f"{record['returns']} {record['name']}({params});\n"


def write_checkout(root: Path, functions: Sequence[Mapping], *, implementations: Optional[Sequence[Mapping]] = None,
                   library_version: str = "7.47.0", reader: bool = True) -> Path:
    """Headers declaring ``functions``, their export, the reader and, when ``implementations`` is
    given, a manifest bound to that export."""
    root = Path(root)
    headers: dict[str, str] = {}
    for record in functions:
        headers[record["header"]] = headers.get(record["header"], "") + _declaration(record)
    for relative, text in headers.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    (root / "Include").mkdir(parents=True, exist_ok=True)
    export = json.dumps({"schema": "ns-cmsis-nn/kernel-contracts/1", "functions": list(functions)}, indent=2) + "\n"
    (root / CONTRACT_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (root / CONTRACT_RELPATH).write_text(export)
    if implementations is not None:
        write_manifest(root, implementations, library_version=library_version)
    if reader:
        install_reader(root)
    return root


def write_manifest(root: Path, implementations: Sequence[Mapping], *, library_version: str = "7.47.0") -> Path:
    export = (Path(root) / CONTRACT_RELPATH).read_bytes()
    document = {"schema": "ns-cmsis-nn/kernel-implementations/1", "library_version": library_version,
                "signature_schema": "ns-cmsis-nn/kernel-contracts/1",
                "signature_sha256": hashlib.sha256(export).hexdigest(), "planner_abi": None,
                "implementations": list(implementations)}
    path = Path(root) / MANIFEST_RELPATH
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path
