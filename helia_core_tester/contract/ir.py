"""The tester's view of an ns-cmsis-nn checkout's kernel contracts.

ns-cmsis-nn ships the signature export ``Tests/KernelContracts/kernel_contracts.json``, the
implementation manifest ``Tests/KernelContracts/kernel_implementations.json`` and their one
reader, the ``ns_cmsis_nn_contracts`` package under ``python/`` (RFC AmbiqAI/ns-cmsis-nn#829).
This module imports that package from the same checkout it reads (the one CMSIS_NN_ROOT
names), so the reader always matches the files it validates, and adapts what it returns to
the tester-facing ContractSet. Validation lives in the package; the tester checks only what
it reads beyond the package's loader (a manifest record's decline causes).

The reader is taken from the checkout rather than pinned as a dependency: ns-cmsis-nn runs
this tester as a submodule against its own tree, so a pin would need a tester release before
an ns-cmsis-nn change to the reader could be tested, and the host planner (a later consumer)
binds to the C sources of the checkout it is built from.

Fail-closed by design. A checkout without the export is an explicit ``absent`` status that
callers count. An export with no package to read it, a package that cannot be imported or
reads a schema this tester does not, and every error the package raises are ContractErrors,
never a skip: a contract that silently lost a kernel would be consumed as if that kernel did
not exist.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Mapping, Optional

from helia_core_tester import __version__
from helia_core_tester.core.errors import GenerationError
from helia_core_tester.generation.kernel_dispatch import _cpu_buffer_api

SUPPORTED_CONTRACT_SCHEMAS = frozenset({"ns-cmsis-nn/kernel-contracts/1"})
SUPPORTED_MANIFEST_SCHEMAS = frozenset({"ns-cmsis-nn/kernel-implementations/1"})
CONTRACT_RELPATH = Path("Tests") / "KernelContracts" / "kernel_contracts.json"
MANIFEST_RELPATH = Path("Tests") / "KernelContracts" / "kernel_implementations.json"
PACKAGE_RELPATH = Path("python") / "ns_cmsis_nn_contracts"
STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"

_PACKAGE_API = ("ContractError", "ContractSchemaError", "load_signatures", "load_implementations")
_FUNCTION_POINTER_RE = re.compile(r"^.+\(\s*\*\s*\w*\s*\)\s*\(.*\)$")


class ContractError(GenerationError):
    """The contract export exists but cannot be trusted."""


class ContractSchemaError(ContractError):
    """The export declares a schema this tester does not read."""


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
class ImplementationDecl:
    """One implementation-manifest record, as far as generation gating reads it."""

    id: str
    entry: str
    wrapper: Optional[str]
    requires: frozenset[str]
    since: str
    declines_on_build: bool = False


@dataclass(frozen=True)
class ContractSet:
    status: str
    root: Optional[Path]
    path: Optional[Path]
    functions: Mapping[str, FunctionDecl] = field(default_factory=dict)
    implementations: Mapping[str, ImplementationDecl] = field(default_factory=dict)
    library_version: Optional[str] = None
    manifest_path: Optional[Path] = None

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


def _package_files(package_dir: Path) -> list[Path]:
    return sorted(package_dir.glob("*.py"))


_PACKAGES: dict[tuple, ModuleType] = {}


def _forget_module(name: str) -> None:
    for loaded in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
        del sys.modules[loaded]


def load_contracts_package(cmsis_nn_root: Optional[Path]) -> Optional[ModuleType]:
    """The checkout's own ``ns_cmsis_nn_contracts``, or None when the checkout ships none.

    Each checkout's package is imported under a private module name derived from its files,
    so several checkouts (and an edited one) can be read in one process without one reader
    standing in for another."""
    if cmsis_nn_root is None:
        return None
    package_dir = Path(cmsis_nn_root) / PACKAGE_RELPATH
    init = package_dir / "__init__.py"
    if not init.is_file():
        return None
    key = _fingerprint(_package_files(package_dir))
    if key is None:
        raise ContractError(f"{package_dir}: the reader's files cannot be read")
    cached = _PACKAGES.get(key)
    if cached is not None:
        return cached
    name = "_hct_ns_cmsis_nn_contracts_" + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        raise ContractError(f"{package_dir}: not importable as a package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        _forget_module(name)
        raise ContractError(f"{package_dir}: cannot be imported ({type(error).__name__}: {error})") from None
    missing = [attribute for attribute in _PACKAGE_API if not hasattr(module, attribute)]
    if missing:
        _forget_module(name)
        raise ContractError(f"{package_dir}: does not export {missing}; it is not an ns_cmsis_nn_contracts this "
                            f"tester reads")
    _PACKAGES[key] = module
    return module


def _schema_of(package: ModuleType, submodule: str) -> Optional[str]:
    return getattr(getattr(package, submodule, None), "SCHEMA", None)


def _check_reader_schemas(package: ModuleType, package_dir: Path) -> frozenset[str]:
    """The reader's decline causes, once both schemas it reads are ones this tester reads."""
    for submodule, supported in (("signatures", SUPPORTED_CONTRACT_SCHEMAS),
                                 ("vocabulary", SUPPORTED_MANIFEST_SCHEMAS)):
        schema = _schema_of(package, submodule)
        if schema not in supported:
            raise ContractSchemaError(
                f"{package_dir}: reads schema {schema!r}, which helia-core-tester {__version__} does not "
                f"(supported: {sorted(supported)}); update the tester pin or the checkout")
    causes = getattr(package.vocabulary, "DECLINE_CAUSES", None)
    if not isinstance(causes, (set, frozenset)) or "build" not in causes:
        raise ContractError(f"{package_dir}: vocabulary.DECLINE_CAUSES {causes!r} has no 'build' cause")
    return frozenset(causes)


def _adapt_function(raw) -> FunctionDecl:
    return FunctionDecl(name=raw.name, header=raw.header, line=raw.line, guards=tuple(raw.guards),
                        returns=raw.returns,
                        params=tuple(ParamDecl(name=p.name, c_type=p.c_type, direction=p.direction,
                                               extent=p.extent) for p in raw.params))


def _adapt_implementation(raw, path: Path, causes: frozenset[str]) -> ImplementationDecl:
    declines = raw.record.get("declines")
    if not isinstance(declines, list) or not all(isinstance(d, Mapping) and d.get("cause") in causes
                                                 for d in declines):
        raise ContractError(f"{path}: {raw.id}: declines {declines!r} is not a list of {{cause, when}} with a cause "
                            f"in {sorted(causes)}")
    return ImplementationDecl(id=raw.id, entry=raw.entry, wrapper=raw.wrapper,
                              requires=frozenset(raw.requires), since=raw.since,
                              declines_on_build=any(d["cause"] == "build" for d in declines))


_CACHE: dict[tuple, ContractSet] = {}
_HEADER_GLOB = "arm_nn*.h"


def _cache_key(root: Path, path: Path) -> Optional[tuple]:
    """Identity of everything a load reads: the export, the manifest, the reader package and the
    public headers the export is checked against, each by path, size and mtime. None when any
    of them cannot be stat'ed, so the load runs uncached and reports the failure itself."""
    manifest = root / MANIFEST_RELPATH
    files = [path] + _package_files(root / PACKAGE_RELPATH) + sorted((root / "Include").glob(_HEADER_GLOB))
    key = _fingerprint(files)
    if key is None:
        return None
    manifest_key = _fingerprint([manifest]) if manifest.exists() else ("no manifest",)
    return None if manifest_key is None else key + manifest_key


def clear_contract_cache() -> None:
    _CACHE.clear()


def load_contract_set(cmsis_nn_root: Optional[Path]) -> ContractSet:
    """Load the checkout's kernel contract, or an ``absent`` set when it has none.

    A present set is cached per checkout for the life of the process and reloaded when the
    export, the manifest, the reader or any public header changes on disk (size or mtime), so
    generating a suite pays the header cross-check once instead of per case."""
    if cmsis_nn_root is None:
        return ContractSet(status=STATUS_ABSENT, root=None, path=None)
    root = Path(cmsis_nn_root)
    path = root / CONTRACT_RELPATH
    if not path.exists():
        return ContractSet(status=STATUS_ABSENT, root=root, path=path)
    key = _cache_key(root, path)
    cached = _CACHE.get(key) if key is not None else None
    if cached is not None:
        return cached
    contracts = _load_contract_set(root, path)
    if key is not None:
        _CACHE.clear()
        _CACHE[key] = contracts
    return contracts


def _load_contract_set(root: Path, path: Path) -> ContractSet:
    package = load_contracts_package(root)
    package_dir = root / PACKAGE_RELPATH
    if package is None:
        raise ContractError(
            f"{path}: the checkout has the signature export but no {PACKAGE_RELPATH} to read it; it needs an "
            "ns-cmsis-nn that ships ns_cmsis_nn_contracts (AmbiqAI/ns-cmsis-nn#889 or later)")
    decline_causes = _check_reader_schemas(package, package_dir)
    manifest = root / MANIFEST_RELPATH
    try:
        signatures = package.load_signatures(root)
        if not signatures.present:
            raise ContractError(f"{path}: exists, but {package_dir} reports it absent")
        implementations = package.load_implementations(root, signatures) if manifest.exists() else None
        functions = {name: _adapt_function(decl) for name, decl in signatures.functions.items()}
        records = {} if implementations is None else {
            impl_id: _adapt_implementation(impl, manifest, decline_causes)
            for impl_id, impl in implementations.implementations.items()}
    except ContractError:
        raise
    except package.ContractSchemaError as error:
        raise ContractSchemaError(f"{error} (helia-core-tester {__version__} reads "
                                  f"{sorted(SUPPORTED_CONTRACT_SCHEMAS | SUPPORTED_MANIFEST_SCHEMAS)})") from None
    except package.ContractError as error:
        raise ContractError(str(error)) from None
    except Exception as error:
        raise ContractError(f"{package_dir} failed reading {root}: {type(error).__name__}: {error}") from None
    return ContractSet(
        status=STATUS_PRESENT, root=root, path=path, functions=functions, implementations=records,
        library_version=None if implementations is None else implementations.library_version,
        manifest_path=None if implementations is None else manifest,
    )
