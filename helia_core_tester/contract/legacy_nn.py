"""ns-cmsis-nn's own contract ids and vocabulary, read until ns-cmsis-nn exports ``hct/*`` documents.

The ``ns-cmsis-nn/kernel-contracts/1`` and ``ns-cmsis-nn/kernel-implementations/1`` documents have
the ``hct/*`` shape without a ``library`` field. Their vocabulary is closed by ns-cmsis-nn's
reader (``python/ns_cmsis_nn_contracts``), so hct checks the same closed sets here: a document
those ids name is held to what ns-cmsis-nn's own reader accepts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping

LIBRARY = "ns-cmsis-nn"
SIGNATURES_SCHEMA = "ns-cmsis-nn/kernel-contracts/1"
IMPLEMENTATIONS_SCHEMA = "ns-cmsis-nn/kernel-implementations/1"
SCHEMAS = frozenset({SIGNATURES_SCHEMA, IMPLEMENTATIONS_SCHEMA})

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
_BASE_FEATURES = ("mvei", "dsp", "mvef", "mve_f16", "f32_api", "f16_api", "autovectorize", "single_rounding")
FEATURES = frozenset(_BASE_FEATURES[:6]) | {"!" + feature for feature in _BASE_FEATURES}
OPS = frozenset({"fully_connected", "depthwise_conv", "avg_pool", "max_pool"})

Fail = Callable[[Path, str], Exception]


def type_is_known(c_type: str) -> bool:
    if _FUNCTION_POINTER_RE.match(c_type):
        return True
    match = _TYPE_RE.match(c_type)
    if match is None:
        return False
    base = match.group("base")
    return base in _SCALAR_TYPES or bool(_TYPEDEF_RE.match(base))


def check_signatures(path: Path, functions: list[Mapping[str, Any]], fail: Fail) -> None:
    for record in functions:
        name = record["name"]
        if not _PUBLIC_HEADER_RE.match(record["header"]):
            raise fail(path, f"{name}: header {record['header']!r} is not a public functions header")
        if not type_is_known(record["returns"]):
            raise fail(path, f"{name}: return type {record['returns']!r} is outside the known vocabulary")
        for param in record["params"]:
            if not type_is_known(param["type"]):
                raise fail(path, f"{name}.{param['name']}: type {param['type']!r} is outside the known vocabulary")


def _features(path: Path, where: str, tokens: list[str], fail: Fail) -> None:
    unknown = sorted(set(tokens) - FEATURES)
    if unknown:
        raise fail(path, f"{where}: unknown feature token(s) {unknown}")


def check_implementations(path: Path, records: list[Mapping[str, Any]], fail: Fail) -> None:
    for record in records:
        where = record["id"]
        if record["op"] not in OPS:
            raise fail(path, f"{where}: op {record['op']!r} is not one of {sorted(OPS)}")
        _features(path, where, record["requires"], fail)
        if record["routed_by"] != "direct":
            _features(path, f"{where}.routed_by", record["routed_by"]["requires"], fail)
