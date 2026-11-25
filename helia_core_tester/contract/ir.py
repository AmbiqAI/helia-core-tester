"""The tester's view of a library's kernel contracts.

A library ships a signature export and, optionally, an implementation manifest (for ns-cmsis-nn,
``Tests/KernelContracts/kernel_contracts.json`` and ``kernel_implementations.json``). hct reads
and checks them itself against its own schemas (``contract/documents.py``), so reading a
contract runs none of the library's Python. The library's host planner, which is the library's
own code, is loaded separately (``contract/planner.py``).

Fail-closed by design. A checkout without the export is an explicit ``absent`` status that
callers count. An export that is present and fails a check is a ContractError, never a skip: a
contract that silently lost a kernel would be consumed as if that kernel did not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from helia_core_tester.contract.documents import (
    SUPPORTED_IMPLEMENTATION_SCHEMAS,
    SUPPORTED_SIGNATURE_SCHEMAS,
    ContractError,
    ContractSchemaError,
    load_implementation_document,
    load_signature_document,
)
from helia_core_tester.generation.kernel_dispatch import _cpu_buffer_api

__all__ = ["ContractError", "ContractSchemaError", "SUPPORTED_SIGNATURE_SCHEMAS", "SUPPORTED_IMPLEMENTATION_SCHEMAS"]

CONTRACT_RELPATH = Path("Tests") / "KernelContracts" / "kernel_contracts.json"
MANIFEST_RELPATH = Path("Tests") / "KernelContracts" / "kernel_implementations.json"
# ns-cmsis-nn's own Python package; hct loads it only for the host planner (contract/planner.py).
PACKAGE_RELPATH = Path("python") / "ns_cmsis_nn_contracts"
STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"

_FUNCTION_POINTER_RE = re.compile(r"^.+\(\s*\*\s*\w*\s*\)\s*\(.*\)$")


@dataclass(frozen=True)
class ParamDecl:
    name: str
    c_type: str
    direction: str
    extent: str = ""

    @property
    def is_pointer(self) -> bool:
        return "*" in self.c_type or bool(self.extent)

    @property
    def is_function_pointer(self) -> bool:
        return bool(_FUNCTION_POINTER_RE.match(self.c_type))


@dataclass(frozen=True)
class FunctionDecl:
    name: str
    header: str
    line: int
    guards: tuple[str, ...]
    returns: str
    params: tuple[ParamDecl, ...]

    @property
    def is_sizer(self) -> bool:
        return re.search(r"_get_\w*size(?:_mve|_dsp)?$", self.name) is not None

    @property
    def is_support(self) -> bool:
        return self.name.startswith("arm_nn_")

    @property
    def kind(self) -> str:
        if self.is_sizer:
            return "sizer"
        if self.is_support:
            return "support"
        return "kernel"

    def param(self, name: str) -> ParamDecl:
        for param in self.params:
            if param.name == name:
                return param
        raise KeyError(f"{self.name} has no parameter {name!r}")


@dataclass(frozen=True)
class BufferDecl:
    """A buffer of a manifest record: the parameter it binds to (None for one the call never
    sees), its size query, and whether it lives for the call, for the prepared data or only
    while the prepare steps run."""

    name: str
    param: Optional[str]
    size_query: Optional[str]
    lifetime: str
    optional: bool = False


@dataclass(frozen=True)
class PrepareDecl:
    """A preparation step of a manifest record: the public function, its arguments as references
    (an operand, `operand.field`, `buffer:<name>`, or None for NULL) and the buffer it fills."""

    symbol: str
    args: Mapping[str, Optional[str]]
    fills: Optional[str]


@dataclass(frozen=True)
class NumericsDecl:
    """What a manifest record claims about its results: the reference they are judged against
    (None when the record names none), `bit_exact` or `bounded`, and the conditions it holds under."""

    reference: Optional[str]
    claim: str
    holds_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImplementationDecl:
    """One implementation-manifest record, as far as generation reads it."""

    id: str
    entry: str
    wrapper: Optional[str]
    requires: frozenset[str]
    since: str
    declines_on_build: bool = False
    buffers: tuple[BufferDecl, ...] = ()
    prepare: tuple[PrepareDecl, ...] = ()
    numerics: Optional[NumericsDecl] = None

    def buffer(self, name: str) -> BufferDecl:
        for buffer in self.buffers:
            if buffer.name == name:
                return buffer
        raise KeyError(f"{self.id} has no buffer {name!r}")


@dataclass(frozen=True)
class ContractSet:
    status: str
    root: Optional[Path]
    path: Optional[Path]
    functions: Mapping[str, FunctionDecl] = field(default_factory=dict)
    implementations: Mapping[str, ImplementationDecl] = field(default_factory=dict)
    library_version: Optional[str] = None
    manifest_path: Optional[Path] = None
    library: Optional[str] = None
    schema: Optional[str] = None

    @property
    def present(self) -> bool:
        return self.status == STATUS_PRESENT

    def find(self, symbol: str) -> Optional[FunctionDecl]:
        return self.functions.get(symbol)

    def require(self, symbol: str) -> FunctionDecl:
        decl = self.find(symbol)
        if decl is None:
            where = f"{self.path}" if self.present else "no kernel contract (checkout has no " \
                f"{CONTRACT_RELPATH}; needs an ns-cmsis-nn with the export)"
            raise ContractError(f"{symbol}: not in the kernel contract ({where})")
        return decl

    def kernels(self) -> list[FunctionDecl]:
        return [decl for decl in self.functions.values() if decl.kind == "kernel"]

    def sizer_for(self, symbol: str, cpu: str) -> Optional[str]:
        """The buffer-size query for ``symbol`` on ``cpu`` by ns-cmsis-nn naming: the
        ``_mve``/``_dsp`` variant when the checkout declares it, else the plain form,
        else None."""
        base = f"{symbol}_get_buffer_size"
        for candidate in (_cpu_buffer_api(base, cpu), base):
            if candidate in self.functions:
                return candidate
        return None

    def implementations_of(self, symbol: str) -> list[ImplementationDecl]:
        """Manifest records that ``symbol`` itself implements, in id order. A record that only
        names ``symbol`` as the wrapper routing to it (the depthwise wrapper's to_conv route)
        describes a route, not a direct call, and is left out."""
        return [impl for _, impl in sorted(self.implementations.items())
                if impl.entry == symbol and impl.wrapper != symbol]


def _fingerprint(paths: list[Path]) -> Optional[tuple]:
    """Path, size and mtime of each file, or None when one cannot be stat'ed."""
    entries = []
    try:
        for path in paths:
            stat = path.stat()
            entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    except OSError:
        return None
    return tuple(entries)


def _adapt_function(raw: Mapping[str, Any]) -> FunctionDecl:
    return FunctionDecl(name=raw["name"], header=raw["header"], line=raw["line"], guards=tuple(raw["guards"]),
                        returns=raw["returns"],
                        params=tuple(ParamDecl(name=p["name"], c_type=p["type"], direction=p["direction"],
                                               extent=p.get("extent", "")) for p in raw["params"]))


def _adapt_implementation(raw: Mapping[str, Any]) -> ImplementationDecl:
    routed = raw["routed_by"]
    numerics = raw["numerics"]
    return ImplementationDecl(
        id=raw["id"], entry=raw["entry"], wrapper=None if routed == "direct" else routed["wrapper"],
        requires=frozenset(raw["requires"]), since=raw["since"],
        declines_on_build=any(decline["cause"] == "build" for decline in raw["declines"]),
        buffers=tuple(BufferDecl(name=b["name"], param=b["param"],
                                 size_query=None if b["size_query"] is None else b["size_query"]["symbol"],
                                 lifetime=b["lifetime"], optional=b["optional"]) for b in raw["buffers"]),
        prepare=tuple(PrepareDecl(symbol=step["symbol"], args=dict(step["args"]), fills=step["fills"])
                      for step in raw["prepare"]),
        numerics=NumericsDecl(reference=numerics["reference"], claim=numerics["claim"],
                              holds_with=tuple(numerics["holds_with"])),
    )


# One loaded set per (export, manifest) on disk, with the fingerprint of the headers it was checked against.
_CACHE: dict[tuple, tuple[Optional[tuple], ContractSet]] = {}


def clear_contract_cache() -> None:
    _CACHE.clear()


def _headers_of(contracts: ContractSet) -> list[Path]:
    return sorted({contracts.root / decl.header for decl in contracts.functions.values()})


def load_contract_set(cmsis_nn_root: Optional[Path], *, signatures: Optional[Path] = None,
                      implementations: Optional[Path] = None) -> ContractSet:
    """Load a checkout's kernel contract, or an ``absent`` set when it has none.

    `signatures` and `implementations` name the documents when they are not at ns-cmsis-nn's
    default places under the checkout; a named export that does not exist is an error, not
    ``absent``. A present set is cached for the life of the process and reloaded when the export,
    the manifest or any header it names changes on disk (size or mtime), so generating a suite
    pays the header cross-check once instead of per case."""
    if cmsis_nn_root is None:
        return ContractSet(status=STATUS_ABSENT, root=None, path=None)
    root = Path(cmsis_nn_root)
    path = Path(signatures) if signatures is not None else root / CONTRACT_RELPATH
    manifest = Path(implementations) if implementations is not None else root / MANIFEST_RELPATH
    if not path.exists():
        if signatures is not None:
            raise ContractError(f"{path}: the signature export the library names does not exist")
        return ContractSet(status=STATUS_ABSENT, root=root, path=path)
    if implementations is not None and not manifest.exists():
        raise ContractError(f"{manifest}: the implementation manifest the library names does not exist")
    manifest_key = _fingerprint([manifest]) if manifest.exists() else ("no manifest", str(manifest))
    document_key = _fingerprint([path])
    key = None if manifest_key is None or document_key is None else (str(root),) + document_key + manifest_key
    cached = _CACHE.get(key) if key is not None else None
    if cached is not None and cached[0] is not None and cached[0] == _fingerprint(_headers_of(cached[1])):
        return cached[1]
    contracts = _load_contract_set(root, path, manifest)
    if key is not None:
        _CACHE.clear()
        _CACHE[key] = (_fingerprint(_headers_of(contracts)), contracts)
    return contracts


def _load_contract_set(root: Path, path: Path, manifest: Path) -> ContractSet:
    signature_document = load_signature_document(path, root)
    implementation_document = load_implementation_document(manifest, signature_document) \
        if manifest.exists() else None
    functions = {record["name"]: _adapt_function(record) for record in signature_document.functions}
    records = {} if implementation_document is None else {
        record["id"]: _adapt_implementation(record) for record in implementation_document.implementations}
    return ContractSet(
        status=STATUS_PRESENT, root=root, path=path, functions=functions, implementations=records,
        library_version=None if implementation_document is None else implementation_document.library_version,
        manifest_path=None if implementation_document is None else manifest,
        library=signature_document.library, schema=signature_document.schema,
    )
