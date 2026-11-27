"""Record mode on the host: every generated case compiled with HCT_RECORD against the
library's host reference build, run, and its capture compared with the case's golden.

The kernels must come from the library's own host reference build (for ns-cmsis-nn,
NS_CMSIS_NN_HOST_REFERENCE): a tree without one is refused rather than scanned, since
the recorded values are meant to become goldens. f16 outputs are refused unless the
build's float16_t is `_Float16` (Linux x86-64), the one host f16 goldens come from.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from helia_core_tester.generation.reference import host_kernels
from helia_core_tester.mutation import host_build
from helia_core_tester.record import parity
from helia_core_tester.record.capture import CaptureError, read_capture

REPORT_SCHEMA = "hct/record-parity/1"
RECORD_SOURCE = host_kernels.tester_root() / "src" / "test_runtime" / "hct_record.c"
RECORD_CFLAGS = ("-DHCT_RECORD",)
CAPTURE_ENV = "HCT_RECORD_PATH"
F16_HOST_TYPE = "_Float16"
# hct_record.c exits with this when it cannot write a complete capture.
RECORD_EXIT = 3

# Case outcomes. Only RECORDED cases carry a parity verdict.
RECORDED = "recorded"
NOT_RECORDABLE = "not_recordable"
REFUSED_F16 = "refused_f16_host"
STATUS_MISMATCH = "status_mismatch"
HARNESS_FAILED = "harness_failed"
NO_RECORDS = "no_records"
RECORD_ERROR = "record_error"
CAPTURE_ERROR = "capture_error"
CRASHED = "crashed"
COMPILE_FAILED = "compile_failed"
TIMEOUT = "timeout"
NO_SOURCE = "no_source"
NOT_APPLICABLE = "not_applicable"
# Outcomes that say nothing about the golden either way.
SKIPPED_OUTCOMES = frozenset({NOT_APPLICABLE, NOT_RECORDABLE, REFUSED_F16})


class RecordError(RuntimeError):
    """Record mode could not run at all (no host reference build, bad roots)."""


@dataclass
class CaseRecord:
    family: str
    name: str
    case_dir: str
    case_class: str
    outcome: str
    verdict: Optional[str] = None
    detail: str = ""
    tensors: List[Dict[str, object]] = field(default_factory=list)
    statuses: List[Dict[str, object]] = field(default_factory=list)
    unrecordable: List[str] = field(default_factory=list)
    capture_path: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "name": self.name,
            "case_dir": self.case_dir,
            "case_class": self.case_class,
            "outcome": self.outcome,
            "verdict": self.verdict,
            "detail": self.detail,
            "tensors": self.tensors,
            "statuses": self.statuses,
            "unrecordable": self.unrecordable,
        }


@dataclass
class RecordReport:
    tree: str
    tree_identity: Dict[str, str]
    library_key: str
    compiler: str
    float16_type: str
    cases_roots: List[str]
    cases: List[CaseRecord] = field(default_factory=list)
    schema: str = REPORT_SCHEMA

    def families(self) -> Dict[str, Dict[str, int]]:
        summary: Dict[str, Counter] = defaultdict(Counter)
        for case in self.cases:
            summary[case.family][case.verdict or case.outcome] += 1
        return {family: dict(sorted(counts.items())) for family, counts in sorted(summary.items())}

    def totals(self) -> Dict[str, int]:
        return dict(sorted(Counter(case.verdict or case.outcome for case in self.cases).items()))

    def problems(self) -> List[CaseRecord]:
        """Blocking cases that did not record identical. Skipped cases have a stated reason, and
        advisory ones were generated for a capability the host build lacks, as in host-check."""
        return [
            case
            for case in self.cases
            if case.verdict != parity.IDENTICAL
            and case.outcome not in SKIPPED_OUTCOMES
            and case.case_class != host_kernels.STATUS_ADVISORY
        ]

    @property
    def ok(self) -> bool:
        return bool(self.cases) and not self.problems()

    def to_json(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "tree": self.tree,
            "tree_identity": self.tree_identity,
            "library_key": self.library_key,
            "compiler": self.compiler,
            "float16_type": self.float16_type,
            "cases_roots": self.cases_roots,
            "ok": self.ok,
            "totals": self.totals(),
            "families": self.families(),
            "cases": [case.to_json() for case in self.cases],
        }


def judge(case: CaseRecord, result: host_build.CaseResult, capture_path: Path, float16_type: str) -> CaseRecord:
    """Turn one host run and its capture into an outcome and, when recorded, a parity verdict."""
    if result.kind == host_build.KIND_COMPILE_FAILED:
        case.outcome, case.detail = COMPILE_FAILED, result.detail
        return case
    if result.kind == host_build.KIND_TIMEOUT:
        case.outcome, case.detail = TIMEOUT, result.detail
        return case
    if result.kind == host_build.KIND_NO_SOURCE:
        case.outcome, case.detail = NO_SOURCE, result.detail
        return case
    if result.exit_code == RECORD_EXIT:
        case.outcome, case.detail = RECORD_ERROR, result.stderr.strip() or result.detail
        return case
    if result.exit_code not in (0, 1):
        case.outcome, case.detail = CRASHED, f"exit {result.exit_code}: {result.detail}"
        return case
    try:
        capture = read_capture(capture_path)
    except CaptureError as exc:
        case.outcome, case.detail = CAPTURE_ERROR, str(exc)
        return case
    case.statuses = [{"label": s.label, "status": s.status, "expected": s.expected} for s in capture.statuses]
    case.unrecordable = list(capture.unrecordable)
    compared = [parity.compare(record) for record in capture.tensors]
    case.tensors = [p.to_json() for p in compared]
    wrong_status = [s for s in capture.statuses if s.status != s.expected]
    if wrong_status:
        first = wrong_status[0]
        case.outcome = STATUS_MISMATCH
        case.detail = f"{first.label}: returned {first.status}, the case expects {first.expected}"
        return case
    if result.exit_code == 1:
        case.outcome, case.detail = HARNESS_FAILED, result.detail
        return case
    if capture.unrecordable:
        case.outcome = NOT_RECORDABLE
        case.detail = "validated per element, not as an array: " + ", ".join(capture.unrecordable)
        return case
    if not capture.tensors and not capture.statuses:
        case.outcome, case.detail = NO_RECORDS, "the harness validated nothing record mode captures"
        return case
    half = sorted({str(t["symbol"]) for t in case.tensors if t["dtype"] == "float16"})
    if half and float16_type != F16_HOST_TYPE:
        case.outcome = REFUSED_F16
        case.detail = f"f16 goldens are recorded with {F16_HOST_TYPE} only; this build has {float16_type}: {', '.join(half)}"
        return case
    case.outcome = RECORDED
    case.verdict = parity.worst(p.verdict for p in compared)
    return case


def run_record(
    cases_roots: Sequence[Path],
    tree: Path,
    jobs: Optional[int] = None,
    cc: Optional[str] = None,
    cache_root: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    timeout_s: int = 60,
) -> RecordReport:
    listed, _seed = host_kernels.int_case_dirs(cases_roots)
    library = host_kernels.ensure_host_kernels(tree, "m0", cache_root=cache_root, cc=cc, jobs=jobs)
    if library.provider != host_kernels.PROVIDER_HOST_REFERENCE:
        raise RecordError(
            f"{library.tree} has no host reference build; record mode takes goldens only from the "
            "library's own scalar build, which its profile's host_reference block names "
            "(ns-cmsis-nn: NS_CMSIS_NN_HOST_REFERENCE)"
        )
    report = RecordReport(
        tree=str(library.tree),
        tree_identity=host_kernels.kernel_tree_identity(library.tree),
        library_key=library.key,
        compiler=library.compiler,
        float16_type=library.float16_type,
        cases_roots=[str(Path(r)) for r in cases_roots],
    )
    classes = {case: host_kernels.classify_case(case, cpu, "m0") for case, cpu in listed}
    runnable = [case for case, cls in classes.items() if cls.status != host_kernels.STATUS_NOT_APPLICABLE]
    for case, cls in sorted(classes.items()):
        if cls.status == host_kernels.STATUS_NOT_APPLICABLE:
            report.cases.append(CaseRecord(case.parent.name, case.name, str(case), cls.status, NOT_APPLICABLE, detail=cls.reason))
    if not runnable:
        return report
    own_work = work_dir is None
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="hct-record-"))
    bin_dir = root / "bin"
    capture_dir = root / "captures"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)

    def one(index_case):
        index, case_dir = index_case
        capture_path = capture_dir / f"{index:05d}_{case_dir.name}.hctr"
        if capture_path.exists():
            capture_path.unlink()
        case_bin = bin_dir / f"{index:05d}"
        case_bin.mkdir(parents=True, exist_ok=True)
        result = host_build.build_and_run_case(
            case_dir,
            library.tree,
            library.library,
            library.runtime_obj,
            host_kernels.tester_root(),
            case_bin,
            cc=library.compiler,
            timeout_s=timeout_s,
            mode="m0",
            kernel_flags=library.kernel_flags,
            link_inputs=library.link_inputs,
            extra_cflags=RECORD_CFLAGS,
            extra_sources=(RECORD_SOURCE,),
            run_env={**os.environ, CAPTURE_ENV: str(capture_path)},
        )
        cls = classes[case_dir]
        record = CaseRecord(case_dir.parent.name, case_dir.name, str(case_dir), cls.status, RECORDED)
        if not own_work:
            record.capture_path = str(capture_path)
        return judge(record, result, capture_path, library.float16_type)

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=jobs or os.cpu_count() or 4) as pool:
            report.cases += list(pool.map(one, enumerate(sorted(runnable))))
    finally:
        if own_work:
            shutil.rmtree(root, ignore_errors=True)
    report.cases.sort(key=lambda c: (c.family, c.name))
    return report


def write_report(report: RecordReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    return path
