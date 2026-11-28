"""Load a pack's committed descriptors (``hct/descriptor/1``) into the cases generation runs.

The descriptors directory holds ``<Family>/<stem>.yaml`` files of one or more documents. An
ns-cmsis-nn pack may instead hold hct's own operator descriptors, which load exactly as the built-in
ones do; the two formats are not mixed in one pack.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import jsonschema
import yaml

from helia_core_tester.pack import DESCRIPTOR_SCHEMA, PackError

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "core" / "schemas" / "descriptor-1.json"
MAX_DESCRIPTOR_BYTES = 1024 * 1024
PARITY_KIND = "pack"

# Contract element types a pack case may use, as hct's descriptor dtypes.
ELEMENT_DTYPES: Dict[str, str] = {
    "q7_t": "S8", "int8_t": "S8",
    "q15_t": "S16", "int16_t": "S16",
    "q31_t": "S32", "int32_t": "S32",
    "uint16_t": "U16",
    "float32_t": "FP32", "float": "FP32",
    "float16_t": "FP16",
}
FLOAT_ELEMENTS = frozenset({"float32_t", "float", "float16_t"})
# By-value element types a pack case passes.
INTEGER_SCALARS = frozenset({
    "int8_t", "q7_t", "uint8_t", "int16_t", "q15_t", "uint16_t", "int32_t", "q31_t", "uint32_t",
})
FLOAT_SCALARS = frozenset({"float32_t", "float", "float16_t"})
_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,95}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def element_type(c_type: str) -> str:
    """`const q15_t *` -> `q15_t`."""
    return " ".join(token for token in c_type.replace("*", " ").split() if token not in ("const", "volatile"))


def is_pointer(c_type: str) -> bool:
    return "*" in c_type


def signature_problem(function: Any) -> Optional[str]:
    """Why a pack case cannot call `function` with what it supports, or None."""
    for param in function.params:
        element = element_type(param.c_type)
        if is_pointer(param.c_type):
            if param.direction not in ("in", "out"):
                return (f"{function.name}: {param.name} is an {param.direction} pointer, which pack cases do not "
                        "support yet")
            if element not in ELEMENT_DTYPES:
                return (f"{function.name}: {param.name} is {param.c_type}, which pack cases do not support yet "
                        f"(they take {', '.join(sorted(ELEMENT_DTYPES))})")
        elif element not in INTEGER_SCALARS | FLOAT_SCALARS:
            return f"{function.name}: {param.name} is {param.c_type}, which pack scalars do not support yet"
    return None


def _read_documents(path: Path) -> List[Any]:
    from helia_core_tester.core.library import _UniqueKeyLoader

    try:
        size = path.stat().st_size
        if size > MAX_DESCRIPTOR_BYTES:
            raise PackError(f"{path}: {size} bytes; a descriptor file is at most {MAX_DESCRIPTOR_BYTES}")
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PackError(f"{path}: not UTF-8 text: {exc.reason}") from None
    except OSError as exc:
        raise PackError(f"{path}: cannot read: {exc.strerror}") from None
    try:
        return [doc for doc in yaml.load_all(text, Loader=_UniqueKeyLoader) if doc is not None]  # noqa: S506
    except yaml.YAMLError as exc:
        raise PackError(f"{path}: not valid YAML: {exc}") from None
    except RecursionError:
        raise PackError(f"{path}: nested too deeply to read") from None


def descriptor_files(directory: Path) -> List[Path]:
    """Every descriptor file, sorted; each must be <Family>/<stem>.yaml."""
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix in (".yaml", ".yml"))
    for path in files:
        if path.suffix != ".yaml":
            raise PackError(f"{path}: a descriptor file is named <stem>.yaml, not .yml")
        parts = path.relative_to(directory).parts
        if len(parts) != 2:
            raise PackError(f"{path}: a descriptor file must be <Family>/<stem>.yaml under {directory}")
    return files


def _schema_problem(doc: Any) -> Optional[str]:
    errors = sorted(_validator().iter_errors(doc), key=lambda error: list(error.absolute_path))
    if not errors:
        return None
    first = errors[0]
    where = "/".join(str(part) for part in first.absolute_path) or "(top level)"
    more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"{where}: {first.message}{more}"


def _comparison(doc: Mapping[str, Any], output_dtype: str, where: str) -> Dict[str, Any]:
    from helia_core_tester.generation.io.dtypes import default_comparison_for_dtype

    comparison = dict(default_comparison_for_dtype(output_dtype))
    raw = doc.get("comparison") or {}
    if comparison["mode"] == "exact_int":
        if "atol" in raw or "rtol" in raw:
            raise PackError(f"{where}: comparison atol/rtol apply to float outputs; {output_dtype} compares exactly")
        if "tolerance" in raw:
            if not raw.get("reason"):
                raise PackError(f"{where}: an integer tolerance needs a comparison reason")
            comparison = {"mode": "tolerant_int", "tolerance": int(raw["tolerance"])}
        return comparison
    if "tolerance" in raw:
        raise PackError(f"{where}: comparison tolerance applies to integer outputs; {output_dtype} is float")
    for key in ("atol", "rtol"):
        if key in raw:
            value = float(raw[key])
            if not math.isfinite(value) or value < 0:
                raise PackError(f"{where}: comparison {key} {raw[key]!r} must be a finite number >= 0")
            if value > comparison[key]:
                raise PackError(f"{where}: comparison {key} {value} is wider than the {output_dtype} default "
                                f"{comparison[key]}; a pack may only tighten it")
            comparison[key] = value
    return comparison


def _case(doc: Mapping[str, Any], path: Path, directory: Path, contracts: Any, registry: Any,
          status_codes: Iterable[str]) -> Dict[str, Any]:
    name = str(doc["name"])
    where = f"{path}: {name!r}"
    entry = str(doc["entry"])
    for key, pattern in (("name", _NAME_RE), ("entry", _IDENTIFIER_RE)):
        if not pattern.fullmatch(str(doc[key])):
            raise PackError(f"{where}: {key} {doc[key]!r} must match {pattern.pattern}")
    function = contracts.functions.get(entry) if contracts is not None else None
    if function is None:
        raise PackError(f"{where}: entry {entry} is not in the library's contract "
                        f"({getattr(contracts, 'path', None) or 'none'})")
    generator = registry.generator_for(entry, doc.get("generator"), where)
    outputs = [p for p in function.params if p.direction == "out" and is_pointer(p.c_type)]
    inputs = [p for p in function.params if p.direction == "in" and is_pointer(p.c_type)]
    if len(outputs) != 1:
        raise PackError(f"{where}: {entry} has {len(outputs)} output pointers; a pack case has exactly one")
    problem = signature_problem(function)
    if problem:
        raise PackError(f"{where}: {problem}")
    output_element = element_type(outputs[0].c_type)
    output_dtype = ELEMENT_DTYPES[output_element]
    input_element = element_type(inputs[0].c_type) if inputs else output_element
    input_dtype = ELEMENT_DTYPES[input_element]
    expected = doc.get("expected_status")
    codes = set(status_codes)
    if expected is not None and expected not in codes:
        raise PackError(f"{where}: expected_status {expected} is not one of the profile's codes "
                        f"({', '.join(sorted(codes))})")
    family = path.parent.name
    suite = "float" if output_element in FLOAT_ELEMENTS else "int"
    case: Dict[str, Any] = {
        "name": name,
        "operator": generator.key,
        "entry": entry,
        "params": dict(doc.get("params") or {}),
        "suite": suite,
        "resolved_tensor_dtypes": {"input": input_dtype, "output": output_dtype},
        "resolved_comparison": _comparison(doc, output_dtype, where),
    }
    for key in ("description", "expected_status", "required_capabilities", "degenerate_golden_reason"):
        if key in doc:
            case[key] = doc[key]
    case.update({
        "_pack": True,
        "_family": family,
        "_parity_kind": PARITY_KIND,
        "_source_family": family,
        "_source_stem": path.stem,
        "_source_relpath": path.relative_to(directory).as_posix(),
        "_descriptor_suite": suite,
    })
    return case


def load_pack_descriptors(pack: Any, contracts: Any, status_codes: Iterable[str]) -> List[Dict[str, Any]]:
    """Every case of the pack, checked against its schema, the contract and the registry; every
    problem across the pack is reported together."""
    directory = pack.spec.descriptors
    status_codes = tuple(status_codes)
    cases: List[Dict[str, Any]] = []
    seen: Dict[str, Path] = {}
    failures: List[str] = []
    for path in descriptor_files(directory):
        try:
            documents = _read_documents(path)
        except PackError as exc:
            failures.append(str(exc))
            continue
        for index, doc in enumerate(documents):
            if not isinstance(doc, Mapping) or doc.get("schema") != DESCRIPTOR_SCHEMA:
                failures.append(f"{path}: document {index + 1} is not {DESCRIPTOR_SCHEMA}")
                continue
            problem = _schema_problem(doc)
            if problem:
                failures.append(f"{path}: document {index + 1}: {problem}")
                continue
            if doc["name"] in seen:
                failures.append(f"{path}: case {doc['name']} is also defined in {seen[doc['name']]}")
                continue
            seen[doc["name"]] = path
            try:
                cases.append(_case(doc, path, directory, contracts, pack.registry, status_codes))
            except PackError as exc:
                failures.append(str(exc))
    if failures:
        raise PackError(f"{len(failures)} problem(s) in the pack's descriptors ({directory}):\n"
                        + "\n".join(f"  - {f}" for f in failures))
    return cases


def operator_format(directory: Path) -> bool:
    """True when the directory holds hct's operator descriptors rather than hct/descriptor/1: it has
    documents and none of them names a schema."""
    documents = 0
    for path in descriptor_files(directory):
        for doc in _read_documents(path):
            documents += 1
            if isinstance(doc, Mapping) and "schema" in doc:
                return False
    return documents > 0


def load_case_descriptors(load_operator_descriptors) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    """The cases of the library generation runs against and its pack (None for hct's built-in one).

    `load_operator_descriptors(directory)` loads hct's operator descriptors: the built-in ones, or an
    ns-cmsis-nn pack's."""
    from helia_core_tester.contract import legacy_nn
    from helia_core_tester.contract.render import load_current_contracts
    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.core.harness import current_harness, current_library
    from helia_core_tester.pack.loader import pack_of

    profile = current_library()
    if profile is None or profile.pack is None:
        if profile is not None and profile.library != legacy_nn.LIBRARY:
            raise PackError(f"{profile.describe()}: its profile names no pack, so hct has no tests for it")
        return load_operator_descriptors(str(find_descriptors_dir())), None
    pack = pack_of(profile)
    if operator_format(pack.spec.descriptors):
        if profile.library != legacy_nn.LIBRARY:
            raise PackError(f"{pack.spec.descriptors}: hct's operator descriptors are ns-cmsis-nn's; a "
                            f"{profile.library} pack holds {DESCRIPTOR_SCHEMA} descriptors")
        return load_operator_descriptors(str(pack.spec.descriptors)), pack
    return load_pack_descriptors(pack, load_current_contracts(),
                                  [name for name, _ in current_harness().status.codes]), pack
