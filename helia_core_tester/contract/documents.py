"""The kernel contract documents: the signature export and the implementation manifest.

hct owns their schemas, ``hct/kernel-contracts/1`` and ``hct/kernel-implementations/1``
(``core/schemas/``), and reads them itself; no library's Python is imported to read them. The
ns-cmsis-nn ids ``ns-cmsis-nn/kernel-contracts/1`` and ``ns-cmsis-nn/kernel-implementations/1``
are read as the same shape, with ns-cmsis-nn's vocabulary checked on top (``legacy_nn``).

A document is checked in two passes: its JSON schema (shape, types, closed vocabularies hct
itself interprets), then what a schema cannot say (unique names, declarations in the named
headers, references between records, the manifest's binding to the export it was made from).
Every failure is a ContractError naming the file; nothing is skipped or defaulted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import jsonschema

from helia_core_tester import __version__
from helia_core_tester.contract import legacy_nn
from helia_core_tester.core.errors import GenerationError

SIGNATURES_SCHEMA = "hct/kernel-contracts/1"
IMPLEMENTATIONS_SCHEMA = "hct/kernel-implementations/1"
SUPPORTED_SIGNATURE_SCHEMAS = frozenset({SIGNATURES_SCHEMA, legacy_nn.SIGNATURES_SCHEMA})
SUPPORTED_IMPLEMENTATION_SCHEMAS = frozenset({IMPLEMENTATIONS_SCHEMA, legacy_nn.IMPLEMENTATIONS_SCHEMA})
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "core" / "schemas"
_SCHEMA_FILES = {"signatures": "kernel-contracts-1.json", "implementations": "kernel-implementations-1.json"}

_TYPE_RE = re.compile(r"^(?:const )?(?P<base>[A-Za-z_]\w*)(?: const)?(?: \*(?: const)?)*$")
_FUNCTION_POINTER_RE = re.compile(r"^.+\(\s*\*\s*\w*\s*\)\s*\(.*\)$")
_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


class ContractError(GenerationError):
    """A contract document exists but cannot be trusted."""


class ContractSchemaError(ContractError):
    """A contract document declares a schema this hct does not read."""


@dataclass(frozen=True)
class SignatureDocument:
    path: Path
    schema: str
    library: Optional[str]
    functions: tuple[Mapping[str, Any], ...]
    sha256: str


@dataclass(frozen=True)
class ImplementationDocument:
    path: Path
    schema: str
    library: Optional[str]
    library_version: str
    planner_abi: Optional[int]
    implementations: tuple[Mapping[str, Any], ...]


def _fail(path: Path, message: str) -> ContractError:
    return ContractError(f"{path}: {message}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    document: dict = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate key {key!r}")
        document[key] = value
    return document


def _read_json(path: Path) -> tuple[Any, bytes]:
    try:
        size = path.stat().st_size
        if size > MAX_DOCUMENT_BYTES:
            raise _fail(path, f"{size} bytes; a contract document is at most {MAX_DOCUMENT_BYTES}")
        raw = path.read_bytes()
    except OSError as error:
        raise _fail(path, f"cannot be read ({error.strerror or error})") from None
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys), raw
    except UnicodeDecodeError as error:
        raise _fail(path, f"is not UTF-8 ({error.reason})") from None
    except ValueError as error:
        raise _fail(path, f"is not valid JSON ({error})") from None


@lru_cache(maxsize=None)
def _validator(kind: str) -> jsonschema.Draft202012Validator:
    schema = json.loads((_SCHEMA_DIR / _SCHEMA_FILES[kind]).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def _check_schema_id(path: Path, document: Any, supported: frozenset[str]) -> str:
    if not isinstance(document, dict):
        raise _fail(path, f"top level must be an object, not {type(document).__name__}")
    schema = document.get("schema")
    if schema not in supported:
        raise ContractSchemaError(f"{path}: schema {schema!r} is not one helia-core-tester {__version__} reads "
                                  f"(supported: {sorted(supported)}); update the hct pin or the export")
    if schema in legacy_nn.SCHEMAS and "library" in document:
        raise _fail(path, f"{schema} documents take no library field; a document that names its library is "
                          f"{'hct/' + schema.split('/', 1)[1]}")
    return schema


def _check_shape(path: Path, kind: str, document: Any) -> None:
    errors = list(_validator(kind).iter_errors(document))
    if not errors:
        return
    first = jsonschema.exceptions.best_match(errors)
    where = "/".join(str(part) for part in first.absolute_path) or "(top level)"
    more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    raise _fail(path, f"{where}: {first.message}{more}")


def type_is_well_formed(c_type: str) -> bool:
    """A C type an export may name: `[const] name [const] [* [const]]...`, or a function pointer."""
    return bool(_FUNCTION_POINTER_RE.match(c_type) or _TYPE_RE.match(c_type))


def type_base(c_type: str) -> Optional[str]:
    match = _TYPE_RE.match(c_type)
    return None if match is None else match.group("base")


def _check_functions(path: Path, root: Path, functions: list[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for record in functions:
        name = record["name"]
        if name in seen:
            raise _fail(path, f"{name} appears twice")
        seen.add(name)
        if not type_is_well_formed(record["returns"]):
            raise _fail(path, f"{name}: return type {record['returns']!r} is not a C type hct reads")
        names = [param["name"] for param in record["params"]]
        if len(set(names)) != len(names):
            raise _fail(path, f"{name}: duplicate parameter names {names}")
        for param in record["params"]:
            if not type_is_well_formed(param["type"]):
                raise _fail(path, f"{name}.{param['name']}: type {param['type']!r} is not a C type hct reads")
    _cross_check_headers(path, root, functions)


def _cross_check_headers(path: Path, root: Path, functions: list[Mapping[str, Any]]) -> None:
    """Every symbol must be declared in the header the export names, or the export is stale."""
    resolved_root = root.resolve()
    by_header: dict[str, list[str]] = {}
    for record in functions:
        by_header.setdefault(record["header"], []).append(record["name"])
    for header, symbols in sorted(by_header.items()):
        target = (root / header).resolve()
        if resolved_root not in target.parents:
            raise _fail(path, f"names {header}, which is outside {root}")
        try:
            text = _COMMENT_RE.sub(" ", target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as error:
            raise _fail(path, f"names {header}, which cannot be read in {root} ({error})") from None
        missing = [s for s in symbols if not re.search(rf"\b{re.escape(s)}\s*\(", text)]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise _fail(path, f"{len(missing)} symbol(s) not declared in {header}: {shown}; the export is stale "
                              "for this checkout")


def load_signature_document(path: Path, root: Path) -> SignatureDocument:
    """Read and check a signature export whose headers are relative to `root`."""
    path = Path(path)
    document, raw = _read_json(path)
    schema = _check_schema_id(path, document, SUPPORTED_SIGNATURE_SCHEMAS)
    _check_shape(path, "signatures", document)
    functions = document["functions"]
    if schema == legacy_nn.SIGNATURES_SCHEMA:
        legacy_nn.check_signatures(path, functions, _fail)
    _check_functions(path, Path(root), functions)
    return SignatureDocument(path=path, schema=schema, library=document.get("library", legacy_nn.LIBRARY if
                                                                             schema == legacy_nn.SIGNATURES_SCHEMA
                                                                             else None),
                             functions=tuple(functions), sha256=hashlib.sha256(raw).hexdigest())


def _symbol(path: Path, where: str, known: Mapping[str, Mapping[str, Any]], symbol: Optional[str]) -> None:
    if symbol not in known:
        raise _fail(path, f"{where}: {symbol!r} is not a public function in the signature export")


def _check_implementation(path: Path, record: Mapping[str, Any], functions: Mapping[str, Mapping[str, Any]]) -> None:
    where = record["id"]
    entry = record["entry"]
    _symbol(path, f"{where}.entry", functions, entry)
    if record["routed_by"] != "direct":
        _symbol(path, f"{where}.routed_by.wrapper", functions, record["routed_by"]["wrapper"])
    entry_params = {param["name"] for param in functions[entry]["params"]}
    names = []
    for index, buffer in enumerate(record["buffers"]):
        here = f"{where}.buffers[{index}]"
        if buffer["param"] is not None and buffer["param"] not in entry_params:
            raise _fail(path, f"{here}: {entry} has no parameter {buffer['param']!r}")
        if buffer["size_query"] is not None:
            _symbol(path, f"{here}.size_query", functions, buffer["size_query"]["symbol"])
        names.append(buffer["name"])
    if len(set(names)) != len(names):
        raise _fail(path, f"{where}: duplicate buffer names {names}")
    for index, step in enumerate(record["prepare"]):
        here = f"{where}.prepare[{index}]"
        _symbol(path, here, functions, step["symbol"])
        if step["fills"] is not None and step["fills"] not in names:
            raise _fail(path, f"{here}: fills {step['fills']!r}, which is not one of its buffers {names}")
    eligibility = record["eligibility"]
    if eligibility is not None and eligibility["predicate"] is not None and not eligibility["predicate_static"]:
        _symbol(path, f"{where}.eligibility", functions, eligibility["predicate"])


def load_implementation_document(path: Path, signatures: SignatureDocument) -> ImplementationDocument:
    """Read and check an implementation manifest against the signature export it was made from."""
    path = Path(path)
    document, _raw = _read_json(path)
    schema = _check_schema_id(path, document, SUPPORTED_IMPLEMENTATION_SCHEMAS)
    _check_shape(path, "implementations", document)
    if document["signature_schema"] != signatures.schema:
        raise _fail(path, f"was made against a {document['signature_schema']} export, but {signatures.path} is "
                          f"{signatures.schema}")
    library = document.get("library", legacy_nn.LIBRARY if schema == legacy_nn.IMPLEMENTATIONS_SCHEMA else None)
    if library != signatures.library:
        raise _fail(path, f"is for library {library!r}, but {signatures.path} is for {signatures.library!r}")
    if document["signature_sha256"] != signatures.sha256:
        raise _fail(path, f"was generated against a signature export with sha256 {document['signature_sha256']}, "
                          f"not {signatures.path} ({signatures.sha256}); the manifest is stale")
    records = document["implementations"]
    if schema == legacy_nn.IMPLEMENTATIONS_SCHEMA:
        legacy_nn.check_implementations(path, records, _fail)
    functions = {record["name"]: record for record in signatures.functions}
    ids: set[str] = set()
    routes: set[tuple[str, str]] = set()
    for record in records:
        if record["id"] in ids:
            raise _fail(path, f"implementation {record['id']} appears twice")
        ids.add(record["id"])
        _check_implementation(path, record, functions)
        if record["routed_by"] != "direct":
            route = (record["routed_by"]["wrapper"], record["routed_by"]["route"])
            if route in routes:
                raise _fail(path, f"two implementations claim the wrapper route {route[0]} {route[1]}")
            routes.add(route)
    return ImplementationDocument(path=path, schema=schema, library=library,
                                  library_version=document["library_version"], planner_abi=document["planner_abi"],
                                  implementations=tuple(records))
