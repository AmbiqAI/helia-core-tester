"""Write performance-stream result bundles from session results."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, ElementTree

from .case_bundle import input_digest
from .comparison import finite_or_none
from .harness_lock import HARNESS_FIELD, HARNESS_INPUTS
from .measurement import compute_counter_medians, counter_names_for_passes
from .session import SessionResult
from .wire import boot_record, placement_record
from .work_count import case_work, per_unit
from .wrapper_route import build_gate, inner_symbol, inner_variant
from .pathutil import write_text_lf

CASE_SUMMARY_BASE_FIELDS = [
    "case_id",
    "kernel_id",
    "comparison_passed",
    "mismatch_count",
    "max_abs_diff",
    "diff_count",
    "sample_count",
    "median_cycles",
    "mad_cycles",
    "p90_cycles",
    "p99_cycles",
    "fvp_status",
    "timed_symbol",
    "inner_symbol",
    "inner_variant",
]
# Work counts and prepare cost.
CASE_SUMMARY_WORK_FIELDS = ["macs", "ops", "cycles_per_mac", "cycles_per_op", "prepare_cycles"]
CASE_SUMMARY_FLAG_FIELDS = ["overflow_detected", "valid_for_regression", "timing_status", "hidden"]


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



def _text(value: Any) -> str | None:
    """A non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def build_provenance(build_dir: Path | None) -> tuple[dict, Path | None]:
    """What the last build used, plus nsx.lock."""
    # Saved records, not flags; missing means null.
    from . import nsx_cli
    from .firmware_build import built_record, nsx_app_dir
    from .nsx_app import CMSIS_NN_MODULE, saved_options

    kernels: dict[str, Any] = dict.fromkeys(
        ("ref", "commit", "root", "root_head", "root_dirty", "tree_hash", "base_ref", "base_commit")
    )
    provenance: dict[str, Any] = {
        "options": None, "kernels": kernels, "neuralspotx_version": None, "nsx_lock_sha256": None,
        "modules": None, "toolchain": None,
    }
    if build_dir is None:
        return provenance, None
    app_dir = nsx_app_dir(build_dir)
    built = built_record(app_dir)
    built_lock = _text(built.get("lock"))
    kernels["tree_hash"] = _text(built.get("kernels"))
    provenance["nsx_lock_sha256"] = built_lock
    provenance["neuralspotx_version"] = _text(built.get("nsx_version"))
    toolchain = built.get("toolchain")
    if isinstance(toolchain, dict):
        provenance["toolchain"] = {"name": _text(toolchain.get("name")), "version": _text(toolchain.get("version"))}

    options = saved_options(app_dir)
    if options is not None:
        provenance["options"] = json.loads(options.to_json())
        kernels["base_ref"] = options.cmsis_nn_ref
        if options.cmsis_nn_root is None:
            kernels["ref"] = options.cmsis_nn_ref
        else:
            kernels["base_commit"] = _text(built.get("base_commit"))
            kernels["root"] = str(options.cmsis_nn_root)
            dirty = built.get("root_dirty")
            kernels["root_head"] = _text(built.get("root_head"))
            kernels["root_dirty"] = dirty if isinstance(dirty, bool) else None

    # Trust nsx.lock only if it built.
    if built_lock is None or nsx_cli.lock_digest(app_dir) != built_lock:
        return provenance, None
    modules = nsx_cli.locked_modules(app_dir)
    provenance["modules"] = modules
    kernels["commit"] = next((m["commit"] for m in modules or () if m["name"] == CMSIS_NN_MODULE), None)
    if kernels["root"] is None:
        kernels["base_commit"] = kernels["commit"]
    return provenance, app_dir / "nsx.lock"


def harness_section(build_dir: Path | None) -> tuple[str | None, dict]:
    """Harness digest over the build's inputs."""
    from .boards import repo_root
    from .firmware_build import built_record, nsx_app_dir
    from .harness_lock import harness_record

    firmware = built_record(nsx_app_dir(build_dir)).get("harness") if build_dir is not None else None
    return harness_record(firmware if isinstance(firmware, dict) else None, repo_root())


def _rejection_record(case) -> dict | None:
    """The case's rejection, as bundle JSON."""
    rejection = case.rejection
    if rejection is None:
        return None
    return {"kernel_status": rejection.kernel_status, "stage": rejection.stage, "reason": rejection.reason}


def _work_fields(case) -> dict[str, Any]:
    """Work counts, per-unit cycles, prepare cycles."""
    work = case_work(case.case_bundle)
    median = case.statistics.median_cycles if case.samples else None
    return {
        **work,
        "cycles_per_mac": per_unit(median, work["macs"]),
        "cycles_per_op": per_unit(median, work["ops"]),
        "prepare_cycles": case.prepare_cycles,
    }


def write_timing(bundle_root: Path, timing: dict) -> Path:
    """Merge wall-clock `timing` into an existing bundle's session_summary.json.

    The bundle is written by the batched runner before the pipeline knows the stage
    totals, so the pipeline adds them afterwards instead of threading a callback
    through every layer."""
    return merge_summary(bundle_root, "timing", timing)


def merge_summary(bundle_root: Path, key: str, value: Any) -> Path:
    """Set one key in session_summary.json."""
    path = bundle_root / "session_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary[key] = value
    write_text_lf(path, json.dumps(summary, indent=2))
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
    build_dir: Path | None = None,
    timing_floor: dict | None = None,
    compare: dict | None = None,
) -> Path:
    for case in result.cases:
        if len(case.samples) != len(case.normalized_samples):
            raise ValueError("Sample and normalized sample counts must match")
    bundle_root = output_root / "artifacts" / "reports" / "hardware" / session_id
    (bundle_root / "correctness").mkdir(parents=True, exist_ok=True)
    (bundle_root / "outputs").mkdir(parents=True, exist_ok=True)
    (bundle_root / "logs").mkdir(parents=True, exist_ok=True)

    session_manifest = {
        "schema": "hct.hardware.session_manifest",
        "schema_version": 1,
        "session_id": session_id,
        "case_count": len(result.cases),
        "target": {
            **(target_info or {"board": "apollo510_evb", "cpu": "cortex-m55", "transport": "fake-target"}),
            # Runs pair only within one placement.
            "placement": placement_record(result.target_info),
        },
        "artifacts": {
            "memory_report": "memory_report.json",
            "kernel_catalog": "kernel_catalog.json",
            "cases": "cases.json",
            "case_summary": "case_summary.csv",
            "raw_samples": "raw_samples.csv",
            "protocol_trace": "protocol_trace.jsonl",
            "junit": "junit.xml",
        },
        # Board-reported TARGET_INFO build id.
        "firmware_build_id": result.build_id,
        "boot": boot_record(result.target_info),
        # Empty-call cycles; null when unmeasured.
        "timing_floor": timing_floor,
        # How outputs were judged.
        "compare": {"strict": False, "golden_from": None, "golden_session_id": None, **(compare or {})},
    }
    session_manifest["build"], lock_file = build_provenance(build_dir)
    session_manifest[HARNESS_FIELD], session_manifest[HARNESS_INPUTS] = harness_section(build_dir)
    if lock_file is not None:
        shutil.copyfile(lock_file, bundle_root / "nsx.lock")
        session_manifest["artifacts"]["nsx_lock"] = "nsx.lock"
    else:
        # Drop a reused session's stale copy.
        (bundle_root / "nsx.lock").unlink(missing_ok=True)
    write_text_lf(bundle_root / "session_manifest.json", json.dumps(session_manifest, indent=2))

    case_rows = []
    case_summary_rows = []
    raw_sample_rows = []
    # Counter columns for case_summary.csv and session_summary.json: every counter the
    # selected passes asked for (ARM_PMU_CPU_CYCLES first, then plan order), plus any
    # counter a sample reported that the plan did not list (unknown-id placeholders,
    # legacy results without counter_passes). A counter the target marked unsupported
    # in every sample keeps its column with an empty median cell, so the schema is
    # fixed by the selection rather than by target support.
    counter_names: list[str] = counter_names_for_passes(result.counter_passes) if result.counter_passes else []
    pass_names: list[str] = [counter_pass.name for counter_pass in result.counter_passes]
    # The kernel each sample timed.
    timed_symbols = {int(entry["kernel_id"]): str(entry.get("canonical_name", "")) for entry in kernel_catalog}
    # Route rules follow the built kernels.
    gate_1xn = build_gate(build_dir)
    passed = 0
    for case in result.cases:
        passed += 1 if case.comparison.passed else 0
        timed_symbol = timed_symbols.get(case.case_bundle.kernel_id, "")
        # The kernel the wrapper routes to.
        inner = inner_symbol(timed_symbol, case.case_bundle.manifest, gate_1xn)
        # Planar or channelwise inside opt.
        variant = inner_variant(inner, case.case_bundle.manifest)
        rejection = _rejection_record(case)
        digests = case_digests(case.case_bundle)
        counter_medians = compute_counter_medians(case.normalized_samples)
        work_fields = _work_fields(case)
        for sample in case.samples:
            if sample.pass_name not in pass_names:
                pass_names.append(sample.pass_name)
            for counter in sample.counters:
                if counter.name not in counter_names:
                    counter_names.append(counter.name)
        for name in counter_medians:
            if name not in counter_names:
                counter_names.append(name)
        # Scorer splits hidden from public.
        hidden = bool(case.case_bundle.manifest.get("hidden"))
        case_rows.append(
            {
                "case_id": case.case_bundle.case_id,
                "kernel_id": case.case_bundle.kernel_id,
                "comparison_passed": case.comparison.passed,
                "mismatch_count": case.comparison.mismatch_count,
                "max_abs_diff": finite_or_none(case.comparison.max_abs_diff),
                "diff_count": case.comparison.diff_count,
                "sample_count": len(case.samples),
                "median_cycles": case.statistics.median_cycles,
                "p90_cycles": case.statistics.p90_cycles,
                "p99_cycles": case.statistics.p99_cycles,
                "mad_cycles": case.statistics.mad_cycles,
                "fvp_status": case.case_bundle.fvp_status,
                "timed_symbol": timed_symbol,
                "inner_symbol": inner,
                "inner_variant": variant,
                **work_fields,
                "shapes": {blob.role: list(blob.dimensions) for blob in case.case_bundle.blobs},
                "unsupported_counters": list(case.statistics.unsupported_counters),
                # Median per-invocation value of every supported counter across samples.
                "counters": counter_medians,
                "overflow_detected": case.statistics.overflow_detected,
                "valid_for_regression": case.statistics.valid_for_regression,
                "timing_status": case.statistics.timing_status,
                "rejection": rejection,
                "hidden": hidden,
                **digests,
            }
        )
        summary_row = {
            "case_id": case.case_bundle.case_id,
            "kernel_id": case.case_bundle.kernel_id,
            "comparison_passed": str(case.comparison.passed).lower(),
            "mismatch_count": case.comparison.mismatch_count,
            "max_abs_diff": finite_or_none(case.comparison.max_abs_diff),
            "diff_count": case.comparison.diff_count,
            "sample_count": case.statistics.sample_count,
            "median_cycles": case.statistics.median_cycles,
            "mad_cycles": case.statistics.mad_cycles,
            "p90_cycles": case.statistics.p90_cycles,
            "p99_cycles": case.statistics.p99_cycles,
            "fvp_status": case.case_bundle.fvp_status,
            "timed_symbol": timed_symbol,
            "inner_symbol": inner,
            "inner_variant": variant,
            **work_fields,
        }
        summary_row.update(counter_medians)
        summary_row["overflow_detected"] = str(case.statistics.overflow_detected).lower()
        summary_row["valid_for_regression"] = str(case.statistics.valid_for_regression).lower()
        summary_row["timing_status"] = case.statistics.timing_status
        summary_row["hidden"] = str(hidden).lower()
        case_summary_rows.append(summary_row)
        (bundle_root / "outputs" / f"{case.case_bundle.case_id}.bin").write_bytes(case.output_bytes)
        write_text_lf(
            bundle_root / "correctness" / f"{case.case_bundle.case_id}.json",
            json.dumps(
                {
                    "case_id": case.case_bundle.case_id,
                    "passed": case.comparison.passed,
                    "mismatch_count": case.comparison.mismatch_count,
                    "max_abs_diff": finite_or_none(case.comparison.max_abs_diff),
                    "diff_count": case.comparison.diff_count,
                    "comparison": case.case_bundle.comparison,
                    "rejection": rejection,
                    **digests,
                },
                indent=2,
            ),
        )
        for sample, normalized in zip(case.samples, case.normalized_samples):
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

    write_text_lf(bundle_root / "cases.json", json.dumps(case_rows, indent=2))
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
        "rejected_cases": [row["case_id"] for row in case_rows if row["rejection"]],
        "timing_status_counts": dict(Counter(row["timing_status"] for row in case_rows)),
    }
    if timing is not None:
        session_summary["timing"] = timing
    write_text_lf(bundle_root / "session_summary.json", json.dumps(session_summary, indent=2))
    write_text_lf(bundle_root / "memory_report.json", json.dumps(memory_report, indent=2))
    write_text_lf(bundle_root / "kernel_catalog.json", json.dumps(kernel_catalog, indent=2))

    with (bundle_root / "case_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        # One column per selected/reported counter name (a case with no supported
        # value for a counter leaves that cell empty), then the overflow/validity flags.
        case_summary_fieldnames = CASE_SUMMARY_BASE_FIELDS + CASE_SUMMARY_WORK_FIELDS + counter_names + CASE_SUMMARY_FLAG_FIELDS
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

    testsuite = Element("testsuite", name="hardware", tests=str(len(result.cases)), failures=str(len(result.cases) - passed))
    for case in result.cases:
        testcase = SubElement(testsuite, "testcase", name=case.case_bundle.case_id, classname="hardware")
        if case.rejection is not None:
            failure = SubElement(testcase, "failure", message="kernel rejected case")
            failure.text = case.rejection.reason
        elif not case.comparison.passed:
            failure = SubElement(testcase, "failure", message="correctness mismatch")
            failure.text = f"mismatch_count={case.comparison.mismatch_count}"
    ElementTree(testsuite).write(bundle_root / "junit.xml", encoding="utf-8", xml_declaration=True)
    write_text_lf(bundle_root / "logs" / "host.log", host_log_text)
    write_text_lf(bundle_root / "logs" / "target.log", target_log_text)
    return bundle_root


def case_digests(bundle) -> dict:
    """Input and compared-output digests."""
    # Status-only cases compare no output.
    compared = None if bundle.expected_status_code is not None else bundle.expected_output.sha256
    return {"input_digest": input_digest(bundle), "expected_output_sha256": compared}

