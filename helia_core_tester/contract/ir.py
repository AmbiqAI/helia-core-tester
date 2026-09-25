"""Load the ns-cmsis-nn kernel contract export into a frozen, validated IR.

ns-cmsis-nn commits ``Tests/KernelContracts/kernel_contracts.json`` (schema
``ns-cmsis-nn/kernel-contracts/1``): one record per public function with its return
type, the ``#if`` conditions it sits under and every parameter's C type, array extent
and doc-resolved direction. This module is the only place the tester reads that file.

Fail-closed by design. A checkout without the file is an explicit ``absent`` status
that callers count; a file that is present but unreadable, of an unsupported schema,
internally inconsistent, or that names a symbol the checkout's own header does not
declare is a ContractError, never a skip -- a contract that silently lost a kernel
would be consumed as if that kernel did not exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from helia_core_tester import __version__
from helia_core_tester.core.errors import GenerationError
from helia_core_tester.generation.kernel_dispatch import _cpu_buffer_api
from helia_core_tester.generation.utils.temp_sizer_probe import undeclared_in_header

SUPPORTED_CONTRACT_SCHEMAS = frozenset({"ns-cmsis-nn/kernel-contracts/1"})
CONTRACT_RELPATH = Path("Tests") / "KernelContracts" / "kernel_contracts.json"
DIRECTIONS = frozenset({"in", "out", "in,out"})
STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"

_SCALAR_TYPES = frozenset({
    "void", "bool", "char", "int", "size_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "float", "double", "float32_t", "float16_t",
})
_TYPEDEF_RE = re.compile(r"^(cmsis_nn_|arm_nn_|arm_cmsis_nn_)\w+$")
_TYPE_RE = re.compile(r"^(?:const )?(?P<base>\w+)(?: const)?(?: \*(?: const)?)*$")
_FUNCTION_POINTER_RE = re.compile(r"^.+\(\s*\*\s*\w*\s*\)\s*\(.*\)$")
_PUBLIC_HEADER_RE = re.compile(r"^Include/arm_nn[^/]*functions[^/]*\.h$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
_EXTENT_RE = re.compile(r"^(\[[^\]]*\])+$")


class ContractError(GenerationError):
    """The contract export exists but cannot be trusted."""


class ContractSchemaError(ContractError):
    """The export declares a schema this tester does not read."""


def _type_is_known(c_type: str) -> bool:
    if _FUNCTION_POINTER_RE.match(c_type):
        return True
    match = _TYPE_RE.match(c_type)
    if match is None:
        return False
    base = match.group("base")
    return base in _SCALAR_TYPES or bool(_TYPEDEF_RE.match(base))


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
class ContractSet:
    status: str
    root: Optional[Path]
    path: Optional[Path]
    functions: Mapping[str, FunctionDecl] = field(default_factory=dict)

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


def _fail(path: Path, message: str) -> ContractError:
    return ContractError(f"{path}: {message}")


def _parse_param(path: Path, function: str, raw: object) -> ParamDecl:
    if not isinstance(raw, dict):
        raise _fail(path, f"{function}: parameter record is not an object: {raw!r}")
    unknown = set(raw) - {"name", "type", "direction", "extent"}
    if unknown:
        raise _fail(path, f"{function}: parameter has unknown keys {sorted(unknown)}")
    name, c_type, direction = raw.get("name"), raw.get("type"), raw.get("direction")
    extent = raw.get("extent", "")
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise _fail(path, f"{function}: parameter name {name!r} is not an identifier")
    if not isinstance(c_type, str) or not c_type:
        raise _fail(path, f"{function}.{name}: type missing")
    if not _type_is_known(c_type):
        raise _fail(path, f"{function}.{name}: type {c_type!r} is outside the known vocabulary "
                          "(scalar C types, cmsis_nn_*/arm_nn_* typedefs, pointers to them, "
                          "function pointers); extend ir.py if ns-cmsis-nn added a type")
    if direction not in DIRECTIONS:
        raise _fail(path, f"{function}.{name}: direction {direction!r} not in {sorted(DIRECTIONS)}")
    if not isinstance(extent, str) or (extent and not _EXTENT_RE.match(extent)):
        raise _fail(path, f"{function}.{name}: extent {extent!r} is not an array extent")
    return ParamDecl(name=name, c_type=c_type, direction=direction, extent=extent)


def _parse_function(path: Path, raw: object) -> FunctionDecl:
    if not isinstance(raw, dict):
        raise _fail(path, f"function record is not an object: {raw!r}")
    missing = {"name", "header", "line", "guards", "returns", "params"} - set(raw)
    if missing:
        raise _fail(path, f"{raw.get('name', '<unnamed>')}: record lacks {sorted(missing)}")
    name = raw["name"]
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise _fail(path, f"function name {name!r} is not an identifier")
    header = raw["header"]
    if not isinstance(header, str) or not _PUBLIC_HEADER_RE.match(header):
        raise _fail(path, f"{name}: header {header!r} is not a public functions header")
    line = raw["line"]
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        raise _fail(path, f"{name}: line {line!r} is not a positive integer")
    guards = raw["guards"]
    if not isinstance(guards, list) or not all(isinstance(g, str) and g for g in guards):
        raise _fail(path, f"{name}: guards {guards!r} is not a list of conditions")
    returns = raw["returns"]
    if not isinstance(returns, str) or not returns or not _type_is_known(returns):
        raise _fail(path, f"{name}: return type {returns!r} is missing or outside the known vocabulary")
    params_raw = raw["params"]
    if not isinstance(params_raw, list):
        raise _fail(path, f"{name}: params is not a list")
    params = tuple(_parse_param(path, name, item) for item in params_raw)
    names = [param.name for param in params]
    if len(set(names)) != len(names):
        raise _fail(path, f"{name}: duplicate parameter names {names}")
    return FunctionDecl(name=name, header=header, line=line, guards=tuple(guards),
                        returns=returns, params=params)


def _cross_check_headers(root: Path, path: Path, functions: Mapping[str, FunctionDecl]) -> None:
    """Every exported symbol must be declared in the header the export names; the
    export was generated from these headers, so a miss means the checkout and its
    export disagree (a stale export, or the wrong checkout)."""
    by_header: dict[str, list[str]] = {}
    for decl in functions.values():
        by_header.setdefault(decl.header, []).append(decl.name)
    for header, symbols in sorted(by_header.items()):
        header_path = root / header
        if not header_path.is_file():
            raise _fail(path, f"names {header}, which the checkout {root} does not have")
        try:
            missing = undeclared_in_header(header_path, symbols)
        except OSError as error:
            raise _fail(path, f"{header} cannot be read ({error})") from None
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise _fail(path, f"{len(missing)} symbol(s) not declared in {header}: {shown}; "
                              "the export is stale for this checkout "
                              "(python3 scripts/check_kernel_contract.py export)")


def load_contract_set(cmsis_nn_root: Optional[Path]) -> ContractSet:
    """Load the checkout's kernel contract, or an ``absent`` set when it has none."""
    if cmsis_nn_root is None:
        return ContractSet(status=STATUS_ABSENT, root=None, path=None)
    root = Path(cmsis_nn_root)
    path = root / CONTRACT_RELPATH
    if not path.exists():
        return ContractSet(status=STATUS_ABSENT, root=root, path=path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise _fail(path, f"cannot be read ({error})") from None
    except ValueError as error:
        raise _fail(path, f"is not valid JSON ({error})") from None
    if not isinstance(document, dict):
        raise _fail(path, "top level is not an object")
    schema = document.get("schema")
    if schema not in SUPPORTED_CONTRACT_SCHEMAS:
        raise ContractSchemaError(
            f"{path}: schema {schema!r} is not readable by helia-core-tester {__version__} "
            f"(supported: {sorted(SUPPORTED_CONTRACT_SCHEMAS)}); update the tester pin or "
            "the export"
        )
    raw_functions = document.get("functions")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise _fail(path, "has no functions; an export never writes an empty contract")
    functions: dict[str, FunctionDecl] = {}
    for raw in raw_functions:
        decl = _parse_function(path, raw)
        if decl.name in functions:
            raise _fail(path, f"{decl.name} appears twice")
        functions[decl.name] = decl
    _cross_check_headers(root, path, functions)
    return ContractSet(status=STATUS_PRESENT, root=root, path=path, functions=functions)
