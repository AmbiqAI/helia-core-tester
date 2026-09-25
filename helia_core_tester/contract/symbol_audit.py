"""Audit the kernel symbols the tester hard-codes against the kernel contract export.

Op modules and generation/kernel_dispatch.py name ns-cmsis-nn functions as string
literals (`'kernel_fn': 'arm_abs_s8'`) or as stems and prefixes completed at runtime
(`f"{wrapper_prefix}_{kernel_suffix}"`, `f"{base}_mve"`). Descriptors may also name
symbols in `required_kernel_symbols`, which is the tester's declared way to say "this
symbol may be absent from a given checkout" (one tester pin serves several in-flight
kernel branches).

The audit classifies every literal against the loaded contract:

- present:    the literal is a public function the checkout declares;
- completed:  a stem the module completes at runtime: an f-string fragment
              (`f"arm_arg{kind}_{dtype}"`) that prefixes a declared symbol, or a whole
              literal that ends in `_` or prefixes a declared symbol at a `_` boundary;
- gated:      absent from the contract, but some descriptor lists it in
              required_kernel_symbols, so generation skips it with a manifest entry;
- undeclared: absent from the contract and gated by nothing. A harness rendered from it
              calls a function no public header declares -- the class of defect the
              audit exists to catch, and the only class it reports.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from helia_core_tester.contract.ir import ContractSet

SYMBOL_RE = re.compile(r"^arm_[a-z0-9_]+$")

PRESENT = "present"
COMPLETED = "completed"
GATED = "gated"
UNDECLARED = "undeclared"


@dataclass(frozen=True)
class Literal:
    symbol: str
    files: tuple[str, ...]
    fragment: bool = False  # a piece of an f-string, partial by construction


@dataclass(frozen=True)
class SymbolAudit:
    classification: Mapping[str, str]        # literal -> PRESENT | COMPLETED | GATED | UNDECLARED
    literals: Mapping[str, Literal]
    gated_symbols: frozenset[str]

    def of(self, kind: str) -> list[str]:
        return sorted(name for name, k in self.classification.items() if k == kind)

    @property
    def undeclared(self) -> list[Literal]:
        return [self.literals[name] for name in self.of(UNDECLARED)]


def collect_literals(paths: Iterable[Path]) -> dict[str, Literal]:
    """Every string constant of arm_* shape in the given Python files, with the files
    that carry it. Parsed with ast, so docstrings and comments about symbols in prose
    do not count unless they are whole-string constants (a docstring never is)."""
    files_by_symbol: dict[str, set[str]] = {}
    fragments: set[str] = set()
    for path in sorted(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise ValueError(f"{path}: cannot be parsed for symbol literals ({error})") from None
        joined_parts = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                        for part in node.values if isinstance(part, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and SYMBOL_RE.match(node.value):
                files_by_symbol.setdefault(node.value, set()).add(path.name)
                if id(node) in joined_parts:
                    fragments.add(node.value)
    return {symbol: Literal(symbol, tuple(sorted(files)), symbol in fragments)
            for symbol, files in files_by_symbol.items()}


def gated_symbols_from_descriptors(descriptors: Iterable[Mapping]) -> frozenset[str]:
    gated: set[str] = set()
    for descriptor in descriptors:
        raw = descriptor.get("required_kernel_symbols") or []
        if isinstance(raw, str):
            raw = [raw]
        gated.update(str(symbol).strip() for symbol in raw if str(symbol).strip())
    return frozenset(gated)


def classify(literals: Mapping[str, Literal], contracts: ContractSet, gated: frozenset[str]) -> SymbolAudit:
    declared = set(contracts.functions)
    classification: dict[str, str] = {}
    for symbol, literal in literals.items():
        if symbol in declared:
            classification[symbol] = PRESENT
        elif literal.fragment and any(name.startswith(symbol) for name in declared):
            classification[symbol] = COMPLETED
        elif symbol.endswith("_") and any(name.startswith(symbol) for name in declared):
            classification[symbol] = COMPLETED
        elif any(name.startswith(symbol + "_") for name in declared):
            classification[symbol] = COMPLETED
        elif symbol in gated:
            classification[symbol] = GATED
        else:
            classification[symbol] = UNDECLARED
    return SymbolAudit(classification=classification, literals=dict(literals), gated_symbols=gated)


def audit_symbol_literals(contracts: ContractSet, python_files: Iterable[Path],
                          descriptors: Iterable[Mapping]) -> SymbolAudit:
    if not contracts.present:
        raise ValueError("no kernel contract to audit against (checkout has no "
                         "Tests/KernelContracts/kernel_contracts.json)")
    return classify(collect_literals(python_files), contracts, gated_symbols_from_descriptors(descriptors))


def symbol_source_files(tester_root: Path) -> list[Path]:
    """The Python files that may hard-code ns-cmsis-nn symbols: every op module and the
    dispatch table."""
    generation = Path(tester_root) / "helia_core_tester" / "generation"
    return sorted(generation.glob("ops/**/*.py")) + [generation / "kernel_dispatch.py"]
