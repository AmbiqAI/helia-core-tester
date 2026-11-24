"""Freeze recorded outputs into a generated case's golden arrays.

A golden is the array the harness validates against, named in the capture by its C
identifier. Its definition must appear exactly once among the case's .c/.h files as
`<qualifiers> <ctype> <symbol>[...] = { ... };`, with an element type that matches the
recorded dtype and as many elements as were recorded; anything else is a FreezeError,
and so is a rewritten initializer that does not parse back to the recorded values.

`freeze_tree` applies a record report to a generated tree. Only blocking cases that
recorded cleanly are frozen; advisory cases (generated for a capability the host build
lacks) and cases record mode could not compare keep their generated golden, and every
case gets a `<name>.golden.json` saying where its golden came from. A case whose record
run failed fails the freeze: its golden cannot come from the library.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from helia_core_tester.generation.reference import host_kernels
from helia_core_tester.record import runner
from helia_core_tester.record.capture import CaptureError, TensorRecord, read_capture

SIDECAR_SCHEMA = "hct/golden-source/1"
REPORT_SCHEMA = "hct/golden-freeze/1"
REPORT_NAME = "golden_freeze.json"
SOURCE_RECORD = "record"
SOURCE_GENERATED = "hct_ref"

C_TYPES: Dict[str, str] = {
    "int8_t": "int8", "q7_t": "int8", "signed char": "int8",
    "uint8_t": "uint8", "unsigned char": "uint8",
    "int16_t": "int16", "q15_t": "int16",
    "uint16_t": "uint16",
    "int32_t": "int32", "q31_t": "int32",
    "uint32_t": "uint32",
    "int64_t": "int64", "q63_t": "int64",
    "uint64_t": "uint64",
    "float16_t": "float16", "__fp16": "float16", "_Float16": "float16",
    "float": "float32", "float32_t": "float32",
    "double": "float64", "float64_t": "float64",
    "bool": "bool", "_Bool": "bool",
}

# Outcomes that leave the generated golden in place, with the reason recorded.
KEEP_OUTCOMES = frozenset({runner.NOT_RECORDABLE, runner.REFUSED_F16, runner.NOT_APPLICABLE})

_SYMBOL = r"[A-Za-z_]\w*"
_FIRST_ELEMENT = re.compile(r"&\s*(" + _SYMBOL + r")\s*\[\s*0\s*\]")
_MEMBER = re.compile(r"(" + _SYMBOL + r")\s*\.\s*" + _SYMBOL)
_FLOAT16_CAST = re.compile(r"^\(\s*(?:float16_t|__fp16|_Float16)\s*\)\s*")


class FreezeError(RuntimeError):
    """A recorded output cannot be written into its golden, or a case's record run failed."""


@dataclass(frozen=True)
class GoldenSite:
    path: Path
    ctype: str
    start: int
    end: int
    values: Tuple[str, ...]


def _definition_pattern(symbol: str) -> re.Pattern:
    return re.compile(
        r"(?P<head>(?:\b(?:static|const|volatile|extern)\s+)*"
        r"(?P<ctype>(?:unsigned\s+|signed\s+)?" + _SYMBOL + r")\s+(?:const\s+)?"
        + re.escape(symbol)
        + r"\s*\[[^\]\n]*\]\s*(?:__attribute__\s*\(\([^;{]*\)\)\s*)?=\s*)\{(?P<body>[^{}]*)\}\s*;"
    )


def _split_values(body: str) -> Tuple[str, ...]:
    return tuple(token.strip() for token in body.split(",") if token.strip())


def golden_symbol(expression: str) -> str:
    """The array a recorded golden expression names: `&x[0]` is `x`."""
    match = _FIRST_ELEMENT.fullmatch(expression.strip())
    return match.group(1) if match else expression.strip()


def _case_sources(case_dir: Path) -> List[Path]:
    return sorted(list(case_dir.glob("*.c")) + list(case_dir.rglob("*.h")))


def is_runtime_buffer(case_dir: Path, expression: str) -> bool:
    """True when the golden is a writable object the case fills at run time (`x` or `x.member`
    defined without an initializer and not const), so the case compares two of its own runs
    and has no generated golden to freeze."""
    expression = expression.strip()
    member = _MEMBER.fullmatch(expression)
    name = member.group(1) if member else expression
    if not re.fullmatch(_SYMBOL, name):
        return False
    array = re.compile(r"(?m)^\s*(?![^;\n]*\b(?:const|extern)\b)(?:[\w ]+?)\s+" + re.escape(name) + r"\s*\[[^\]\n]*\]\s*;")
    struct = re.compile(r"(?m)^\s*\}\s*" + re.escape(name) + r"\s*;")
    struct_head = re.compile(r"(?m)^\s*(?:static\s+)?struct\b(?![^{;]*\bconst\b)[^{;]*\{")
    for path in _case_sources(case_dir):
        text = path.read_text(encoding="utf-8")
        if not member and array.search(text):
            return True
        if member:
            for end in struct.finditer(text):
                head = list(struct_head.finditer(text, 0, end.start()))
                if head and "=" not in text[head[-1].start() : end.end()].split("}")[-1]:
                    return True
    return False


def find_golden(case_dir: Path, symbol: str) -> GoldenSite:
    if not re.fullmatch(_SYMBOL, symbol):
        raise FreezeError(f"{case_dir.name}: golden {symbol!r} is not a plain C identifier")
    pattern = _definition_pattern(symbol)
    sites: List[GoldenSite] = []
    for path in _case_sources(case_dir):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            ctype = " ".join(match.group("ctype").split())
            sites.append(GoldenSite(path, ctype, match.start("body"), match.end("body"), _split_values(match.group("body"))))
    if not sites:
        raise FreezeError(f"{case_dir.name}: no initialised definition of golden {symbol!r} in the case's files")
    if len(sites) > 1:
        where = ", ".join(sorted({s.path.name for s in sites}))
        raise FreezeError(f"{case_dir.name}: golden {symbol!r} is defined {len(sites)} times ({where})")
    return sites[0]


def parse_literal(token: str, dtype: str):
    """One C initializer element as the generator writes it."""
    text = _FLOAT16_CAST.sub("", token.strip())
    if dtype == "bool":
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        raise FreezeError(f"{token!r} is not a bool literal")
    if dtype.startswith("float"):
        sign = -1.0 if text.startswith("-") else 1.0
        bare = text.lstrip("+-")
        if bare == "NAN":
            return math.copysign(math.nan, sign)
        if bare == "INFINITY":
            return math.copysign(math.inf, sign)
        try:
            return float(re.sub(r"[fFlL]$", "", text))
        except ValueError as exc:
            raise FreezeError(f"{token!r} is not a float literal") from exc
    try:
        return int(re.sub(r"[uUlL]+$", "", text), 0)
    except ValueError as exc:
        raise FreezeError(f"{token!r} is not an integer literal") from exc


def format_values(values: np.ndarray, dtype: str) -> str:
    from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

    array = np.asarray(values).reshape(-1)
    return TemplateContextBuilder.format_array_as_c_literal(array)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def _round_trips(parsed: np.ndarray, actual: np.ndarray) -> bool:
    """Same bits, except that a NaN only keeps its sign: NAN carries no payload."""
    if parsed.shape != actual.shape or parsed.dtype != actual.dtype:
        return False
    if actual.dtype.kind != "f":
        return _same_bits(parsed, actual)
    nan = np.isnan(actual)
    if not np.array_equal(nan, np.isnan(parsed)) or not np.array_equal(np.signbit(actual[nan]), np.signbit(parsed[nan])):
        return False
    return _same_bits(parsed[~nan], actual[~nan])


def _as_dtype(values: Sequence, dtype: str) -> np.ndarray:
    np_dtype = np.bool_ if dtype == "bool" else np.dtype(dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(values, dtype=np_dtype)


def rewrite_golden(case_dir: Path, record: TensorRecord) -> Optional[Path]:
    """Write `record.actual` into the golden it names. Returns the file changed, or None when
    the golden already holds those bits."""
    site = find_golden(case_dir, golden_symbol(record.symbol))
    expected_type = C_TYPES.get(site.ctype)
    if expected_type is None:
        raise FreezeError(f"{case_dir.name}: golden {record.symbol} has element type {site.ctype!r}, which hct cannot map")
    if expected_type != record.dtype:
        raise FreezeError(f"{case_dir.name}: golden {record.symbol} is {site.ctype} but {record.dtype} was recorded")
    count = int(record.actual.size)
    if count == 0:
        return None
    if len(site.values) != count:
        raise FreezeError(
            f"{case_dir.name}: golden {record.symbol} holds {len(site.values)} element(s), {count} were recorded"
        )
    if _same_bits(record.actual, record.golden):
        return None
    actual = record.actual.reshape(-1)
    body = format_values(actual, record.dtype)
    parsed = _as_dtype([parse_literal(t, record.dtype) for t in _split_values(body)], record.dtype)
    if not _round_trips(parsed, actual):
        raise FreezeError(f"{case_dir.name}: golden {record.symbol} does not round-trip through its C literal")
    text = site.path.read_text(encoding="utf-8")
    site.path.write_text(text[: site.start] + "\n" + body + "\n" + text[site.end :], encoding="utf-8")
    return site.path


@dataclass
class FrozenCase:
    family: str
    name: str
    source: str
    reason: str = ""
    rewritten: List[str] = field(default_factory=list)
    runtime: List[str] = field(default_factory=list)
    verdict: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "name": self.name,
            "source": self.source,
            "reason": self.reason,
            "rewritten": self.rewritten,
            "runtime_goldens": self.runtime,
            "verdict_against_generated": self.verdict,
        }


@dataclass
class FreezeReport:
    record: runner.RecordReport
    cases: List[FrozenCase] = field(default_factory=list)
    schema: str = REPORT_SCHEMA

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for case in self.cases:
            key = case.source if case.source == SOURCE_GENERATED else f"{case.source}:{'rewritten' if case.rewritten else 'unchanged'}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def to_json(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "library": {
                "tree": self.record.tree,
                "tree_identity": self.record.tree_identity,
                "library_key": self.record.library_key,
                "compiler": self.record.compiler,
                "float16_type": self.record.float16_type,
            },
            "counts": self.counts(),
            "cases": [case.to_json() for case in self.cases],
        }


def _sidecar(case_dir: Path, frozen: FrozenCase, record: runner.RecordReport, tensors: List[TensorRecord]) -> None:
    payload: Dict[str, object] = {
        "schema": SIDECAR_SCHEMA,
        "source": frozen.source,
        "reason": frozen.reason,
        "library_tree_identity": record.tree_identity,
        "float16_type": record.float16_type,
        "runtime_goldens": frozen.runtime,
        "goldens": {
            golden_symbol(t.symbol): {
                "dtype": t.dtype,
                "count": int(t.actual.size),
                "sha256": hashlib.sha256(t.actual.tobytes()).hexdigest(),
            }
            for t in tensors
            if t.symbol not in frozen.runtime
        }
        if frozen.source == SOURCE_RECORD
        else {},
    }
    (case_dir / f"{frozen.name}.golden.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_tree(record: runner.RecordReport) -> FreezeReport:
    """Write each cleanly recorded blocking case's outputs into its goldens, then restamp it."""
    from helia_core_tester.generation.reuse import read_stamp, write_stamp

    failed = [c for c in record.cases if c.outcome not in KEEP_OUTCOMES | {runner.RECORDED}
              and c.case_class != host_kernels.STATUS_ADVISORY]
    if failed:
        listed = "; ".join(f"{c.family}/{c.name}: {c.outcome} {c.detail}".rstrip() for c in failed[:10])
        raise FreezeError(f"{len(failed)} case(s) could not be recorded, so their goldens cannot come from the library: {listed}")
    report = FreezeReport(record=record)
    for case in record.cases:
        case_dir = Path(case.case_dir)
        (case_dir / f"{case.name}.golden.json").unlink(missing_ok=True)
        tensors: List[TensorRecord] = []
        if case.case_class == host_kernels.STATUS_ADVISORY:
            frozen = FrozenCase(case.family, case.name, SOURCE_GENERATED,
                                "generated for a capability the host build lacks", verdict=case.verdict)
        elif case.outcome != runner.RECORDED:
            frozen = FrozenCase(case.family, case.name, SOURCE_GENERATED, f"{case.outcome}: {case.detail}".rstrip(": "))
        else:
            if not case.capture_path:
                raise FreezeError(f"{case.family}/{case.name}: recorded, but its capture was not kept (run_record needs a work_dir)")
            try:
                tensors = read_capture(Path(case.capture_path)).tensors
            except CaptureError as exc:
                raise FreezeError(f"{case.family}/{case.name}: {exc}") from exc
            frozen = FrozenCase(case.family, case.name, SOURCE_RECORD, verdict=case.verdict)
            for tensor in tensors:
                if is_runtime_buffer(case_dir, tensor.symbol):
                    frozen.runtime.append(tensor.symbol)
                elif rewrite_golden(case_dir, tensor) is not None:
                    frozen.rewritten.append(golden_symbol(tensor.symbol))
        if case.outcome == runner.NOT_APPLICABLE:
            report.cases.append(frozen)
            continue
        _sidecar(case_dir, frozen, record, tensors)
        stamp = read_stamp(case_dir)
        if stamp is not None and stamp.get("stamp"):
            write_stamp(case_dir, str(stamp["stamp"]))
        report.cases.append(frozen)
    report.cases.sort(key=lambda c: (c.family, c.name))
    return report


def invalidate_unfrozen(cases_root: Path) -> List[Path]:
    """Drop the reuse stamp of every case without a golden sidecar, so a freeze that
    stopped part way cannot leave hct_ref goldens standing under a record stamp."""
    from helia_core_tester.generation.reuse import STAMP_FILENAME

    dropped: List[Path] = []
    for stamp in sorted(Path(cases_root).rglob(STAMP_FILENAME)):
        case_dir = stamp.parent
        if not (case_dir / f"{case_dir.name}.golden.json").is_file():
            stamp.unlink()
            dropped.append(case_dir)
    return dropped


def write_report(report: FreezeReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    return path
