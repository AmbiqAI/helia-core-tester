"""FVP-pass gating for the hardware perf-stream bridge (Phase 2 of the
generation/bridge unification plan).

FVP already builds and executes the *real* CMSIS-NN kernel against each
generated case's golden data via the standalone C harness
(HELIA_VALIDATE_OUTPUTS(...) in the generated .c file) -- making its most
recent test_report_*.json the most trustworthy correctness oracle already
present in the system for a given generated artifact. Before bridging a
generated test case onto real hardware, this module lets the bridge check
"did this exact case pass under FVP?" and skip (with a clear reason) rather
than silently ship a hardware result for a case FVP itself doesn't believe is
correct.

This is deliberately advisory-only right now: if no FVP report is found (e.g.
because FVP cannot run in this environment, or the artifacts were newly
regenerated and no FVP pass has been recorded yet), gating is a no-op rather
than a hard failure -- callers that want a strict gate should call
`require_fvp_pass()` explicitly and handle the "no report available" case.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from helia_core_tester.generation.artifact_identity import generated_case_artifact_sha256


class FvpReportUnavailableError(Exception):
    """Raised when a caller requires an FVP report but none can be found."""


class FvpCaseFailedGateError(Exception):
    """Raised when a case's most recently recorded FVP status is not PASS."""


class FvpCaseStaleGateError(FvpCaseFailedGateError):
    """Raised when a recorded result is not for the current generated artifacts."""



@dataclass(frozen=True)
class FvpCaseStatus:
    name: str
    status: str
    report_path: Path
    artifact_sha256: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _tests_report_dir(project_root: Path, cpu: str, suite: str) -> Path:
    return project_root / "artifacts" / "reports" / "tests" / suite / cpu


_REPORT_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})\.json$")


def _report_sort_key(path: Path) -> tuple[str, float]:
    """Sort by the filename's embedded timestamp, falling back to mtime."""
    match = _REPORT_TIMESTAMP_RE.search(path.name)
    timestamp = match.group(1) if match else ""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (timestamp, mtime)


def find_latest_fvp_report(project_root: Path, *, cpu: str = "cortex-m55", suite: str = "int") -> Optional[Path]:
    """Return the most recent test_report_<cpu>_<timestamp>.json for (cpu, suite),
    or None if no FVP run has ever been recorded for it.
    """
    report_dir = _tests_report_dir(project_root, cpu, suite)
    if not report_dir.is_dir():
        return None
    candidates = sorted(report_dir.glob(f"test_report_{cpu}_*.json"), key=_report_sort_key)
    return candidates[-1] if candidates else None


_REPORT_CACHE: dict[Path, dict] = {}


def _load_report(report_path: Path) -> dict:
    cached = _REPORT_CACHE.get(report_path)
    if cached is not None:
        return cached
    data = json.loads(report_path.read_text(encoding="utf-8"))
    _REPORT_CACHE[report_path] = data
    return data


def lookup_fvp_case_status(
    project_root: Path,
    case_name: str,
    *,
    cpu: str = "cortex-m55",
    suite: str = "int",
) -> Optional[FvpCaseStatus]:
    """Return the most recently recorded FVP status for `case_name`, or None if
    no FVP report is available at all, or the report doesn't mention this case
    (e.g. it's a newly added case generated after the last FVP run).
    """
    report_path = find_latest_fvp_report(project_root, cpu=cpu, suite=suite)
    if report_path is None:
        return None
    report = _load_report(report_path)
    descriptor_results = report.get("descriptor_results", {})
    entry = descriptor_results.get(case_name)
    if entry is None:
        return None
    status = str(entry.get("test_result", {}).get("status", ""))
    digest = entry.get("artifact_sha256")
    return FvpCaseStatus(
        name=case_name,
        status=status,
        report_path=report_path,
        artifact_sha256=str(digest) if digest is not None else None,
    )


def require_fvp_pass(
    project_root: Path,
    case_name: str,
    *,
    cpu: str = "cortex-m55",
    suite: str = "int",
    allow_missing_report: bool = True,
    case_dir: Path | None = None,
) -> None:
    """Raise if `case_name` has a recorded, non-PASS FVP status.

    If no FVP report exists yet for (cpu, suite), or the report doesn't
    mention this case, this is a no-op when `allow_missing_report` is True
    (the common case in environments where FVP cannot be run, e.g. this
    sandbox) -- otherwise raises FvpReportUnavailableError so callers that
    want a strict "must have a recorded FVP pass" gate can opt into that.
    """
    fvp_status = lookup_fvp_case_status(project_root, case_name, cpu=cpu, suite=suite)
    if fvp_status is None:
        if allow_missing_report:
            return
        raise FvpReportUnavailableError(
            f"{case_name}: no recorded FVP test result found under "
            f"artifacts/reports/tests/{suite}/{cpu}/test_report_{cpu}_*.json -- "
            f"run the FVP suite before bridging this case onto hardware, or "
            f"pass allow_missing_report=True to skip this gate."
        )
    if not fvp_status.passed:
        raise FvpCaseFailedGateError(
            f"{case_name}: recorded FVP status is {fvp_status.status!r} (not PASS) in "
            f"{fvp_status.report_path} -- refusing to bridge a case FVP itself does not "
            f"believe is correct onto real hardware."
        )
    if case_dir is not None:
        current_digest = generated_case_artifact_sha256(case_dir)
        if fvp_status.artifact_sha256 is None:
            raise FvpCaseStaleGateError(
                f"{case_name}: recorded PASS in {fvp_status.report_path} has no artifact_sha256; "
                "rerun the FVP suite before bridging this case."
            )
        if fvp_status.artifact_sha256 != current_digest:
            raise FvpCaseStaleGateError(
                f"{case_name}: recorded PASS artifact {fvp_status.artifact_sha256} does not match "
                f"current generated artifact {current_digest}; rerun the FVP suite."
            )
