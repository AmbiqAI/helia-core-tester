from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from helia_core_tester.perf_stream.benchmark_firmware_report import generate_benchmark_server_memory_report
from helia_core_tester.perf_stream.case_bundle import build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from helia_core_tester.perf_stream.fake_target import FakeTargetTransport
from helia_core_tester.perf_stream.measurement import counter_passes_for_selection
from helia_core_tester.perf_stream.result_bundle import write_result_bundle, write_timing
from helia_core_tester.perf_stream.session import HostSession, SessionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# generate_benchmark_server_memory_report() shells out to arm-none-eabi-size/nm/objdump
# against an already-built firmware ELF; both are unavailable in a pure-Python
# environment (e.g. the pytest.yml CI job, which intentionally skips the ARM toolchain).
pytestmark = pytest.mark.skipif(
    shutil.which("arm-none-eabi-size") is None,
    reason="ARM GCC toolchain (arm-none-eabi-*) not installed",
)



def test_result_bundle_writer_emits_spec_artifacts(tmp_path: Path) -> None:
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_bundle").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_bundle").manifest_path)
    passes = counter_passes_for_selection({"cpu": "default", "memory": "default", "mve": "default"})
    result = HostSession(FakeTargetTransport(max_frame_payload=15, read_chunk_size=9), counter_passes=passes).run_many([abs_bundle, conv_bundle])
    memory_report_path = generate_benchmark_server_memory_report()
    memory_report = json.loads(memory_report_path.read_text())
    kernel_catalog = json.loads((PROJECT_ROOT / "cmake" / "perf_stream" / "kernel_catalog.json").read_text())

    bundle_root = write_result_bundle(result, session_id="fake-session-001", output_root=tmp_path, memory_report=memory_report, kernel_catalog=kernel_catalog)

    expected = {
        "session_manifest.json",
        "session_summary.json",
        "memory_report.json",
        "kernel_catalog.json",
        "cases.json",
        "case_summary.csv",
        "raw_samples.csv",
        "protocol_trace.jsonl",
        "junit.xml",
    }
    assert expected.issubset({path.name for path in bundle_root.iterdir()})
    assert (bundle_root / "correctness" / "abs_bundle.json").exists()
    assert (bundle_root / "correctness" / "conv_bundle.json").exists()
    assert (bundle_root / "outputs" / "abs_bundle.bin").exists()
    assert (bundle_root / "outputs" / "conv_bundle.bin").exists()

    # case_summary.csv: one column per counter name (median per invocation), then the
    # overflow/validity flags; cases.json mirrors them; session_summary lists what ran.
    import csv

    with (bundle_root / "case_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    header = list(rows[0].keys())
    assert header[:10] == ["case_id", "kernel_id", "comparison_passed", "mismatch_count", "sample_count", "median_cycles", "mad_cycles", "p90_cycles", "p99_cycles", "fvp_status"]
    assert header[10] == "ARM_PMU_CPU_CYCLES"
    assert header[-2:] == ["overflow_detected", "valid_for_regression"]
    assert "ARM_PMU_INST_RETIRED" in header and "ARM_PMU_MEM_ACCESS" in header
    abs_row, conv_row = rows
    assert float(abs_row["ARM_PMU_CPU_CYCLES"]) > 0
    assert float(abs_row["ARM_PMU_INST_RETIRED"]) > 0
    assert abs_row["ARM_PMU_MEM_ACCESS"] == ""  # abs fake adapter only supports the cpu group
    assert float(conv_row["ARM_PMU_MEM_ACCESS"]) > 0
    assert abs_row["overflow_detected"] == "false" and abs_row["valid_for_regression"] == "true"

    cases = json.loads((bundle_root / "cases.json").read_text())
    assert cases[0]["counters"]["ARM_PMU_CPU_CYCLES"] == float(abs_row["ARM_PMU_CPU_CYCLES"])
    assert cases[0]["overflow_detected"] is False and cases[0]["valid_for_regression"] is True

    summary = json.loads((bundle_root / "session_summary.json").read_text())
    assert summary["counters"][0] == "ARM_PMU_CPU_CYCLES"
    assert set(summary["counters"]) == {name for name in header[10:-2]}
    assert summary["passes"] == ["cpu_0", "memory_0", "mve_0"]
    assert summary["batch_count"] == 1
    assert summary["cases_with_overflow"] == []
    assert "timing" not in summary

    write_timing(bundle_root, {"stream_s": 1.5, "cases": {"abs_bundle": 0.7}})
    summary = json.loads((bundle_root / "session_summary.json").read_text())
    assert summary["timing"] == {"stream_s": 1.5, "cases": {"abs_bundle": 0.7}}
    assert summary["passes"] == ["cpu_0", "memory_0", "mve_0"]


def test_result_bundle_writer_handles_empty_session(tmp_path: Path) -> None:
    # Regression test: a session with zero cases (e.g. an empty plan, or one
    # where session_complete short-circuits before any case runs) must not
    # raise IndexError when writing case_summary.csv / raw_samples.csv.
    result = SessionResult(cases=(), protocol_trace=(), session_complete_cases=0)
    memory_report_path = generate_benchmark_server_memory_report()
    memory_report = json.loads(memory_report_path.read_text())
    kernel_catalog = json.loads((PROJECT_ROOT / "cmake" / "perf_stream" / "kernel_catalog.json").read_text())

    bundle_root = write_result_bundle(result, session_id="empty-session-001", output_root=tmp_path, memory_report=memory_report, kernel_catalog=kernel_catalog)

    case_summary_text = (bundle_root / "case_summary.csv").read_text(encoding="utf-8")
    raw_samples_text = (bundle_root / "raw_samples.csv").read_text(encoding="utf-8")
    assert case_summary_text.splitlines() == ["case_id,kernel_id,comparison_passed,mismatch_count,sample_count,median_cycles,mad_cycles,p90_cycles,p99_cycles,fvp_status,overflow_detected,valid_for_regression"]
    assert raw_samples_text.splitlines() == ["case_id,sample_index,pass_name,iterations,cycles,cycles_per_invocation,counter_name,event_id,counter_value,overflow,supported"]
    assert (bundle_root / "junit.xml").exists()
