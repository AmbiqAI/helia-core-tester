"""Write performance-stream result bundles from session results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

from .measurement import compute_counter_medians
from .session import SessionResult

_CASE_SUMMARY_BASE_FIELDS = [
    "case_id",
    "kernel_id",
    "comparison_passed",
    "mismatch_count",
    "sample_count",
    "median_cycles",
    "mad_cycles",
    "p90_cycles",
    "p99_cycles",
    "fvp_status",
]
_CASE_SUMMARY_FLAG_FIELDS = ["overflow_detected", "valid_for_regression"]


def _split_protocol_trace_entry(entry: str) -> tuple[int | None, str, str]:
    """Parse one protocol_trace entry.

    A single-session run records "direction:message_type" (see HostSession._trace).
    session_runner's batched runner prefixes each entry with "batchN:" before
    merging traces across sessions, giving "batchN:direction:message_type" --
    splitting on the first colon alone would misparse that as
    direction="batchN", message_type="direction:message_type".
    """
    prefix, sep, rest = entry.partition(":")
    if sep and prefix.startswith("batch") and prefix[len("batch"):].isdigit():
        direction, message_type = rest.split(":", 1)
        return int(prefix[len("batch"):]), direction, message_type
    direction, message_type = entry.split(":", 1)
    return None, direction, message_type



def write_timing(bundle_root: Path, timing: dict) -> Path:
    """Merge wall-clock `timing` into an existing bundle's session_summary.json.

    The bundle is written by the batched runner before the pipeline knows the stage
    totals, so the pipeline adds them afterwards instead of threading a callback
    through every layer."""
    path = bundle_root / "session_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary["timing"] = timing
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return path


def write_result_bundle(
    result: SessionResult,
    *,
    session_id: str,
    output_root: Path,
    memory_report: dict,
    kernel_catalog: list[dict],
    target_info: dict | None = None,
    host_log_text: str = "session completed\n",
    target_log_text: str = "no physical target log captured\n",
    timing: dict | None = None,
) -> Path:
    bundle_root = output_root / "artifacts" / "reports" / "performance_stream" / session_id
    (bundle_root / "correctness").mkdir(parents=True, exist_ok=True)
    (bundle_root / "outputs").mkdir(parents=True, exist_ok=True)
    (bundle_root / "logs").mkdir(parents=True, exist_ok=True)

    session_manifest = {
        "schema": "hct.performance_stream.session_manifest",
        "schema_version": 1,
        "session_id": session_id,
        "case_count": len(result.cases),
        "target": target_info or {"board": "apollo510_evb", "cpu": "cortex-m55", "transport": "fake-target"},
        "artifacts": {
            "memory_report": "memory_report.json",
            "kernel_catalog": "kernel_catalog.json",
            "cases": "cases.json",
            "case_summary": "case_summary.csv",
            "raw_samples": "raw_samples.csv",
            "protocol_trace": "protocol_trace.jsonl",
            "junit": "junit.xml",
        },
    }
    (bundle_root / "session_manifest.json").write_text(json.dumps(session_manifest, indent=2), encoding="utf-8", newline="\n")

    case_rows = []
    case_summary_rows = []
    raw_sample_rows = []
    # Counter names in first-seen order (ARM_PMU_CPU_CYCLES leads every sample) and
    # the PMU passes run, for the case_summary.csv columns and session_summary.json.
    counter_names: list[str] = []
    pass_names: list[str] = []
    passed = 0
    for case in result.cases:
        passed += 1 if case.comparison.passed else 0
        counter_medians = compute_counter_medians(case.normalized_samples)
        for name in counter_medians:
            if name not in counter_names:
                counter_names.append(name)
        for sample in case.samples:
            if sample.pass_name not in pass_names:
                pass_names.append(sample.pass_name)
        case_rows.append(
            {
                "case_id": case.case_bundle.case_id,
                "kernel_id": case.case_bundle.kernel_id,
                "comparison_passed": case.comparison.passed,
                "mismatch_count": case.comparison.mismatch_count,
                "sample_count": len(case.samples),
                "median_cycles": case.statistics.median_cycles,
                "p90_cycles": case.statistics.p90_cycles,
                "p99_cycles": case.statistics.p99_cycles,
                "mad_cycles": case.statistics.mad_cycles,
                "fvp_status": case.case_bundle.fvp_status,
                "unsupported_counters": list(case.statistics.unsupported_counters),
                # Median per-invocation value of every supported counter across samples.
                "counters": counter_medians,
                "overflow_detected": case.statistics.overflow_detected,
                "valid_for_regression": case.statistics.valid_for_regression,
            }
        )
        summary_row = {
            "case_id": case.case_bundle.case_id,
            "kernel_id": case.case_bundle.kernel_id,
            "comparison_passed": str(case.comparison.passed).lower(),
            "mismatch_count": case.comparison.mismatch_count,
            "sample_count": case.statistics.sample_count,
            "median_cycles": case.statistics.median_cycles,
            "mad_cycles": case.statistics.mad_cycles,
            "p90_cycles": case.statistics.p90_cycles,
            "p99_cycles": case.statistics.p99_cycles,
            "fvp_status": case.case_bundle.fvp_status,
        }
        summary_row.update(counter_medians)
        summary_row["overflow_detected"] = str(case.statistics.overflow_detected).lower()
        summary_row["valid_for_regression"] = str(case.statistics.valid_for_regression).lower()
        case_summary_rows.append(summary_row)
        (bundle_root / "outputs" / f"{case.case_bundle.case_id}.bin").write_bytes(case.output_bytes)
        (bundle_root / "correctness" / f"{case.case_bundle.case_id}.json").write_text(
            json.dumps(
                {
                    "case_id": case.case_bundle.case_id,
                    "passed": case.comparison.passed,
                    "mismatch_count": case.comparison.mismatch_count,
                    "comparison": case.case_bundle.comparison,
                },
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )
        for sample, normalized in zip(case.samples, case.normalized_samples, strict=True):
            for counter in sample.counters:
                raw_sample_rows.append(
                    {
                        "case_id": case.case_bundle.case_id,
                        "sample_index": sample.sample_index,
                        "pass_name": sample.pass_name,
                        "iterations": sample.iterations,
                        "cycles": sample.cycles,
                        "cycles_per_invocation": normalized.cycles_per_invocation,
                        "counter_name": counter.name,
                        "event_id": counter.event_id,
                        "counter_value": counter.value,
                        "overflow": int(counter.overflow),
                        "supported": int(counter.supported),
                    }
                )

    (bundle_root / "cases.json").write_text(json.dumps(case_rows, indent=2), encoding="utf-8", newline="\n")
    session_summary = {
        "session_id": session_id,
        "case_count": len(result.cases),
        "passed_cases": passed,
        "failed_cases": len(result.cases) - passed,
        "session_complete_cases": result.session_complete_cases,
        "batch_count": int(getattr(result, "batch_count", 1)),
        "counters": counter_names,
        "passes": pass_names,
        "cases_with_overflow": [row["case_id"] for row in case_rows if row["overflow_detected"]],
    }
    if timing is not None:
        session_summary["timing"] = timing
    (bundle_root / "session_summary.json").write_text(json.dumps(session_summary, indent=2), encoding="utf-8", newline="\n")
    (bundle_root / "memory_report.json").write_text(json.dumps(memory_report, indent=2), encoding="utf-8", newline="\n")
    (bundle_root / "kernel_catalog.json").write_text(json.dumps(kernel_catalog, indent=2), encoding="utf-8", newline="\n")

    with (bundle_root / "case_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        # One column per counter name seen in any case (a case that did not run a
        # counter leaves that cell empty), then the overflow/validity flags.
        case_summary_fieldnames = _CASE_SUMMARY_BASE_FIELDS + counter_names + _CASE_SUMMARY_FLAG_FIELDS
        writer = csv.DictWriter(handle, fieldnames=case_summary_fieldnames, restval="")
        writer.writeheader()
        writer.writerows(case_summary_rows)
    with (bundle_root / "raw_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        raw_sample_fieldnames = list(raw_sample_rows[0].keys()) if raw_sample_rows else [
            "case_id",
            "sample_index",
            "pass_name",
            "iterations",
            "cycles",
            "cycles_per_invocation",
            "counter_name",
            "event_id",
            "counter_value",
            "overflow",
            "supported",
        ]
        writer = csv.DictWriter(handle, fieldnames=raw_sample_fieldnames)
        writer.writeheader()
        writer.writerows(raw_sample_rows)

    with (bundle_root / "protocol_trace.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for index, entry in enumerate(result.protocol_trace):
            batch_index, direction, message_type = _split_protocol_trace_entry(entry)
            record = {"index": index, "direction": direction, "message_type": message_type}
            if batch_index is not None:
                record["batch"] = batch_index
            handle.write(json.dumps(record) + "\n")

    testsuite = Element("testsuite", name="performance_stream", tests=str(len(result.cases)), failures=str(len(result.cases) - passed))
    for case in result.cases:
        testcase = SubElement(testsuite, "testcase", name=case.case_bundle.case_id, classname="perf_stream")
        if not case.comparison.passed:
            failure = SubElement(testcase, "failure", message="correctness mismatch")
            failure.text = f"mismatch_count={case.comparison.mismatch_count}"
    ElementTree(testsuite).write(bundle_root / "junit.xml", encoding="utf-8", xml_declaration=True)
    (bundle_root / "logs" / "host.log").write_text(host_log_text, encoding="utf-8", newline="\n")
    (bundle_root / "logs" / "target.log").write_text(target_log_text, encoding="utf-8", newline="\n")
    return bundle_root
