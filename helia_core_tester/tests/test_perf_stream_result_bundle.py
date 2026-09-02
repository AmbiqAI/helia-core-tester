from __future__ import annotations

import json
from pathlib import Path

from helia_core_tester.perf_stream.benchmark_firmware_report import generate_benchmark_server_memory_report
from helia_core_tester.perf_stream.case_bundle import build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from helia_core_tester.perf_stream.fake_target import FakeTargetTransport
from helia_core_tester.perf_stream.result_bundle import write_result_bundle
from helia_core_tester.perf_stream.session import HostSession, SessionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]



def test_result_bundle_writer_emits_spec_artifacts(tmp_path: Path) -> None:
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_bundle").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_bundle").manifest_path)
    result = HostSession(FakeTargetTransport(max_frame_payload=15, read_chunk_size=9), requested_counter_groups=("cpu", "memory", "mve")).run_many([abs_bundle, conv_bundle])
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
    assert case_summary_text.splitlines() == ["case_id,kernel_id,comparison_passed,mismatch_count,sample_count,median_cycles,mad_cycles,p90_cycles,p99_cycles,fvp_status"]
    assert raw_samples_text.splitlines() == ["case_id,sample_index,pass_name,iterations,cycles,cycles_per_invocation,counter_name,event_id,counter_value,overflow,supported"]
    assert (bundle_root / "junit.xml").exists()
