"""Which cases run kernel code a candidate changed.

Code layout moves every kernel's cycles a little, so a per-case gate
on untouched kernels fails correct candidates. This module reads the
kernel library's object files, before link, so layout cannot reach it.
With -ffunction-sections and -fdata-sections each function and data
object owns one section; one section is one node. A node's digest
covers its type, flags, size, bytes and every relocation (offset,
type, target node). Its references are the relocation targets: calls,
tables, literals. Locals are named `unit:name` and anonymous sections
`unit:section`; local labels (.LC0) resolve to their section's node,
so renumbering stays unseen. Inlined helper edits change the caller's
bytes.

A case is touched when a node reachable from its inner symbol (else
its timed symbol), in either build, changed or came or went, or the
timed symbol's own node changed. A case whose symbol neither build
defines counts as touched.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Iterable, Optional

from .candidate_scan import _kernel_units, elf_sections

SCHEMA = "hct.code_graph"
SCHEMA_VERSION = 1
_SYMTAB, _PROGBITS, _NOBITS, _REL, _RELA = 2, 1, 8, 9, 4
_LOCAL = 0
# Symbol types that name a node.
_NAMED = (1, 2)
_SHN_UNDEF, _SHN_LORESERVE = 0, 0xFF00


def _symbols(table: list, strings: bytes) -> list[tuple[str, int, int, int]]:
    """(name, bind, type, section) per symbol."""
    symtab = next((data for _, kind, _, data, _ in table if kind == _SYMTAB), None)
    if symtab is None:
        return []
    out = []
    for at in range(0, len(symtab) - 15, 16):
        name_at, _, _, info, _, shndx = struct.unpack_from("<IIIBBH", symtab, at)
        end = strings.index(b"\0", name_at)
        out.append((strings[name_at:end].decode(errors="replace"), info >> 4, info & 0xF, shndx))
    return out


def _node_names(unit: str, table: list, symbols: list) -> dict[int, str]:
    """Section index to node name."""
    names: dict[int, str] = {}
    # Globals name a node before locals.
    for name, bind, kind, shndx in sorted(symbols, key=lambda s: s[1] == _LOCAL):
        if kind in _NAMED and 0 < shndx < _SHN_LORESERVE and shndx not in names:
            names[shndx] = name if bind != _LOCAL else f"{unit}:{name}"
    for index, (section, kind, flags, _, _) in enumerate(table):
        if "A" in flags and kind in (_PROGBITS, _NOBITS) and index not in names:
            names[index] = f"{unit}:{section}"
    return names


def object_nodes(unit: str, obj: Path) -> dict[str, dict]:
    """Nodes {name: {digest, refs}} of one object."""
    table = elf_sections(obj)
    strtab = next((bytes(data) for name, kind, _, data, _ in table if name == ".strtab"), b"")
    symbols = _symbols(table, strtab)
    names = _node_names(unit, table, symbols)

    def target(index: int) -> str:
        name, _, _, shndx = symbols[index]
        if shndx == _SHN_UNDEF or shndx >= _SHN_LORESERVE:
            return name
        # Defined here: its section's node.
        return names.get(shndx, f"{unit}:#{shndx}")

    relocs: dict[int, list[tuple[int, int, str, int]]] = {}
    for _, kind, _, data, info in table:
        if kind not in (_REL, _RELA):
            continue
        size = 8 if kind == _REL else 12
        for at in range(0, len(data) - size + 1, size):
            offset, rinfo = struct.unpack_from("<II", data, at)
            addend = struct.unpack_from("<i", data, at + 8)[0] if kind == _RELA else 0
            relocs.setdefault(info, []).append((offset, rinfo & 0xFF, target(rinfo >> 8), addend))
    nodes: dict[str, dict] = {}
    for index, name in names.items():
        _, kind, flags, data, _ = table[index]
        digest = hashlib.sha256(f"{kind}:{flags}:{len(data)}:".encode())
        if kind == _PROGBITS:
            digest.update(data)
        refs = set()
        for offset, rtype, ref, addend in sorted(relocs.get(index, [])):
            digest.update(f"|{offset}:{rtype}:{ref}:{addend}".encode())
            refs.add(ref)
        refs.discard(name)
        node = nodes.setdefault(name, {"digest": "", "refs": []})
        # Duplicate names (weak, COMDAT) fold together.
        node["digest"] = hashlib.sha256((node["digest"] + digest.hexdigest()).encode()).hexdigest()
        node["refs"] = sorted(set(node["refs"]) | refs)
    # Other symbols sharing a section alias it.
    for name, bind, kind, shndx in symbols:
        alias = name if bind != _LOCAL else f"{unit}:{name}"
        if kind in _NAMED and shndx in names and alias != names[shndx] and alias not in nodes:
            nodes[alias] = {"digest": nodes[names[shndx]]["digest"], "refs": [names[shndx]]}
    return nodes


def code_graph(build_dir: Path) -> dict:
    """Digests and references of every kernel node."""
    nodes: dict[str, dict] = {}
    units = _kernel_units(build_dir)
    if not units:
        raise ValueError("no kernel objects found")
    for unit, obj, _ in sorted(units):
        for name, node in object_nodes(unit, obj).items():
            if name in nodes:
                node = {"digest": hashlib.sha256((nodes[name]["digest"] + node["digest"]).encode()).hexdigest(),
                        "refs": sorted(set(nodes[name]["refs"]) | set(node["refs"]))}
            nodes[name] = node
    return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "nodes": nodes}


def read_graph(data: object) -> Optional[dict[str, dict]]:
    """Nodes of a stored graph; None if unusable."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("schema_version") != SCHEMA_VERSION:
        return None
    nodes = data.get("nodes")
    return nodes if isinstance(nodes, dict) else None


def _reach(roots: Iterable[str], graphs: tuple[dict, dict]) -> set[str]:
    seen, todo = set(), list(roots)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for nodes in graphs:
            todo.extend((nodes.get(name) or {}).get("refs", ()))
    return seen


def changed_nodes(base: dict[str, dict], cand: dict[str, dict]) -> set[str]:
    """Nodes whose digest differs or that one side lacks."""
    return {name for name in base.keys() | cand.keys() if (base.get(name) or {}).get("digest") != (cand.get(name) or {}).get("digest")}


def is_touched(timed: str, inner: Optional[str], base: dict, cand: dict, changed: set[str]) -> bool:
    """Case code changed; unknown roots count."""
    root = inner or timed
    if root not in base and root not in cand:
        return True
    return timed in changed or bool(_reach([root], (base, cand)) & changed)
