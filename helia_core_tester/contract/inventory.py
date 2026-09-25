"""Which public ns-cmsis-nn functions the generated cases actually call.

The inventory joins two sources: the kernel contract export (every public function the
checkout declares, see contract/ir.py) and the generated test tree (which symbols each
case calls, read from the case's ``*.sidecar.json`` when it has one and from the ``arm_*(``
call tokens in its C sources otherwise). The result is the per-symbol coverage split that
ns-cmsis-nn#400 keeps by hand today: covered, uncovered, and the symbols cases call that
the contract does not know, which is a stale export or a private helper leaking into a
harness and is reported rather than dropped.

Nothing here is a skip: a checkout without a contract, or a tree with no generated cases,
is an explicit non-zero outcome, because an empty inventory reads exactly like full
coverage of nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from helia_core_tester.contract.ir import ContractSet, FunctionDecl

REPORT_SCHEMA = "hct.contract_inventory/1"
REPORT_RELPATH = Path("contracts") / "inventory.json"

_C_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_CALL_RE = re.compile(r"\b(arm_[a-z0-9_]+)\s*\(")
# Preprocessor lines and the kernel's own prototype in a harness are not calls.
_DIRECTIVE_RE = re.compile(r"^[ \t]*#[^\n]*$", re.MULTILINE)


class InventoryError(ValueError):
    """The inventory cannot be produced (no contract, no cases, unreadable tree)."""


@dataclass(frozen=True)
class CaseCalls:
    case_dir: Path
    symbols: tuple[str, ...]
    source: str  # "sidecar" | "source" | "both"


def discover_case_dirs(generated_tests_dir: Path) -> list[Path]:
    """Every generated case directory under the tree: a directory holding descriptor.yaml
    (the shape test_ops.py emits and the hardware bridge discovers)."""
    root = Path(generated_tests_dir)
    if not root.is_dir():
        raise InventoryError(f"{root}: generated-tests directory not found")
    return sorted(path.parent for path in root.rglob("descriptor.yaml") if path.is_file())


def _called_in_source(text: str) -> set[str]:
    text = _DIRECTIVE_RE.sub("", _C_COMMENT_RE.sub(" ", text))
    return set(_CALL_RE.findall(text))


def case_calls(case_dir: Path) -> CaseCalls:
    """Symbols a case calls: the sidecar's kernel_fn plus every arm_*( token in its C
    sources (the sidecar names the kernel, the sources also show sizers and helpers)."""
    symbols: set[str] = set()
    sources = []
    for sidecar in sorted(case_dir.glob("*.sidecar.json")):
        try:
            kernel_fn = json.loads(sidecar.read_text(encoding="utf-8")).get("kernel_fn")
        except (OSError, ValueError) as error:
            raise InventoryError(f"{sidecar}: unreadable sidecar ({error})") from None
        if isinstance(kernel_fn, str) and kernel_fn:
            symbols.add(kernel_fn)
            sources.append("sidecar")
    for c_file in sorted(case_dir.glob("*.c")):
        try:
            found = _called_in_source(c_file.read_text(encoding="utf-8", errors="replace"))
        except OSError as error:
            raise InventoryError(f"{c_file}: unreadable source ({error})") from None
        if found:
            symbols.update(found)
            sources.append("source")
    origin = "both" if {"sidecar", "source"} <= set(sources) else (sources[0] if sources else "none")
    return CaseCalls(case_dir=case_dir, symbols=tuple(sorted(symbols)), source=origin)


@dataclass(frozen=True)
class Inventory:
    contract_path: Optional[Path]
    generated_tests_dir: Path
    cases_scanned: int
    covered: Mapping[str, int]          # symbol -> number of cases calling it
    uncovered: tuple[FunctionDecl, ...]  # public functions no case calls
    unknown: Mapping[str, int]          # called symbols the contract does not know -> cases
    kinds: Mapping[str, int] = field(default_factory=dict)

    def uncovered_by_kind(self, kind: str) -> list[str]:
        return [decl.name for decl in self.uncovered if decl.kind == kind]

    def to_document(self) -> dict:
        return {
            "schema": REPORT_SCHEMA,
            "contract": str(self.contract_path),
            "generated_tests_dir": str(self.generated_tests_dir),
            "cases_scanned": self.cases_scanned,
            "public_functions": dict(self.kinds),
            "covered": {name: self.covered[name] for name in sorted(self.covered)},
            "uncovered": {
                kind: self.uncovered_by_kind(kind) for kind in ("kernel", "sizer", "support")
            },
            "unknown_symbols": {name: self.unknown[name] for name in sorted(self.unknown)},
        }


def build_inventory(contracts: ContractSet, generated_tests_dir: Path) -> Inventory:
    if not contracts.present:
        raise InventoryError(
            f"no kernel contract: the checkout {contracts.root} has no "
            f"Tests/KernelContracts/kernel_contracts.json (needs an ns-cmsis-nn with the export)"
        )
    case_dirs = discover_case_dirs(generated_tests_dir)
    if not case_dirs:
        raise InventoryError(f"{generated_tests_dir}: no generated cases (no descriptor.yaml found); "
                             "run `helia_core_tester generate` first")
    covered: dict[str, int] = {}
    unknown: dict[str, int] = {}
    for case_dir in case_dirs:
        for symbol in case_calls(case_dir).symbols:
            target = covered if symbol in contracts.functions else unknown
            target[symbol] = target.get(symbol, 0) + 1
    uncovered = tuple(decl for name, decl in contracts.functions.items() if name not in covered)
    kinds: dict[str, int] = {}
    for decl in contracts.functions.values():
        kinds[decl.kind] = kinds.get(decl.kind, 0) + 1
    return Inventory(
        contract_path=contracts.path,
        generated_tests_dir=Path(generated_tests_dir),
        cases_scanned=len(case_dirs),
        covered=covered,
        uncovered=uncovered,
        unknown=unknown,
        kinds=kinds,
    )


def write_report(inventory: Inventory, reports_root: Path) -> Path:
    path = Path(reports_root) / REPORT_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory.to_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summary_lines(inventory: Inventory) -> Iterable[str]:
    yield f"contract: {inventory.contract_path}"
    yield f"generated tests: {inventory.generated_tests_dir} ({inventory.cases_scanned} cases)"
    for kind in ("kernel", "sizer", "support"):
        total = inventory.kinds.get(kind, 0)
        missing = len(inventory.uncovered_by_kind(kind))
        yield f"{kind:8s} {total - missing:4d} covered / {total:4d} public ({missing} uncovered)"
    if inventory.unknown:
        yield f"unknown symbols called by cases but absent from the contract: {len(inventory.unknown)}"
        for name in sorted(inventory.unknown):
            yield f"  ? {name} ({inventory.unknown[name]} cases)"
