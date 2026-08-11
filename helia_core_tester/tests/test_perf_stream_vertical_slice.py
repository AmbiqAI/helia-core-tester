from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.perf_stream.case_bundle import (
    build_abs_s8_case_bundle,
    build_convolve_s8_case_bundle,
    blob_numpy,
    load_case_bundle,
)
from helia_core_tester.perf_stream.fake_target import FakeAbsS8Adapter, FakeKernelAdapter, FakeTargetTransport
from helia_core_tester.perf_stream.measurement import compute_sample_statistics, normalize_samples, plan_counter_passes, resolve_counter_selection
from helia_core_tester.perf_stream.session import HostSession, run_fake_abs_vertical_slice, run_fake_convolve_vertical_slice

PROJECT_ROOT = Path(__file__).resolve().parents[2]



def test_case_bundle_writes_binary_artifacts(tmp_path: Path) -> None:
    bundle = build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path)
    loaded = load_case_bundle(bundle.manifest_path)

    assert loaded.manifest_path.name == "case_manifest.json"
    assert loaded.input_blob.path.read_bytes() == bundle.input_blob.path.read_bytes()
    assert loaded.expected_output.path.read_bytes() == bundle.expected_output.path.read_bytes()
    assert loaded.manifest["correctness_comparison"]["mode"] == "exact_int"



def test_convolve_case_bundle_writes_multiple_binary_blobs(tmp_path: Path) -> None:
    bundle = build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path)
    loaded = load_case_bundle(bundle.manifest_path)

    assert [blob.role for blob in loaded.blobs] == ["input_0", "weights", "bias", "multiplier", "shift", "expected_output"]
    assert loaded.blob_by_role("weights").path.read_bytes() == bundle.blob_by_role("weights").path.read_bytes()
    assert loaded.manifest["scratch_buffer"]["bytes"] == 64
    scalars = loaded.manifest["serialized_scalar_parameters"]
    assert scalars["padding"] == "VALID"
    # Ground-truth output dims/padding must be sent explicitly so firmware doesn't have to
    # re-derive them from a SAME/VALID formula (see benchmark_server_session.c history).
    assert scalars["pad_h"] == 0
    assert scalars["pad_w"] == 0
    assert scalars["output_h"] > 0
    assert scalars["output_w"] > 0
    assert scalars["output_c"] > 0
    assert scalars["dilation_h"] == 1
    assert scalars["dilation_w"] == 1



def test_fake_abs_vertical_slice_end_to_end(tmp_path: Path) -> None:
    result = run_fake_abs_vertical_slice(PROJECT_ROOT, output_root=tmp_path)

    assert result.comparison.passed is True
    assert result.comparison.mismatch_count == 0
    assert result.session_complete_cases == 1
    assert len(result.samples) == 3
    assert result.samples[0].iterations == 4
    assert result.samples[0].cycles < result.samples[1].cycles < result.samples[2].cycles
    assert result.cases[0].statistics.median_cycles > 0

    trace = result.protocol_trace
    assert trace[0] == "RX:HELLO"
    assert "TX:CASE_META" in trace
    assert trace.count("RX:REQUEST_BLOB") >= 2
    assert "RX:CASE_READY" in trace
    assert "TX:RUN_CORRECTNESS" in trace
    assert "RX:OUTPUT_BEGIN" in trace
    assert trace.count("RX:OUTPUT_CHUNK") >= 2
    assert "TX:CORRECTNESS_ACK" in trace
    assert "TX:RUN_PERFORMANCE" in trace
    assert trace.count("RX:SAMPLE_RESULT") == 3
    assert trace[-1] == "RX:SESSION_COMPLETE"



def test_fake_convolve_vertical_slice_end_to_end(tmp_path: Path) -> None:
    result = run_fake_convolve_vertical_slice(PROJECT_ROOT, output_root=tmp_path)
    case = result.cases[0]

    assert case.comparison.passed is True
    assert case.comparison.mismatch_count == 0
    assert case.case_bundle.manifest["scratch_buffer"]["bytes"] == 64
    assert len(case.samples) == 15
    assert {sample.pass_name for sample in case.samples} == {"cpu_0", "memory_0", "mve_0"}
    assert case.statistics.sample_count == 15
    assert case.statistics.unsupported_counters == ("ARM_PMU_MVE_INST_RETIRED",)
    assert case.statistics.median_cycles > 0
    assert result.protocol_trace.count("RX:REQUEST_BLOB") >= 6



def test_multi_case_session_rewinds_arena(tmp_path: Path) -> None:
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_case_a").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_case_b").manifest_path)
    transport = FakeTargetTransport(max_frame_payload=15, read_chunk_size=9)

    result = HostSession(transport, requested_counter_groups=("cpu",)).run_many([abs_bundle, conv_bundle])

    assert result.session_complete_cases == 2
    assert transport.completed_case_count == 2
    assert transport.rewind_count == 2
    assert transport.arena_used_bytes == 0
    assert all(case.comparison.passed for case in result.cases)


def test_persistent_fake_target_multi_operator_session_without_reflash(tmp_path: Path) -> None:
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_persistent").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_persistent").manifest_path)
    transport = FakeTargetTransport(max_frame_payload=9, read_chunk_size=7)

    result = HostSession(transport, requested_counter_groups=("cpu", "memory", "mve")).run_many([abs_bundle, conv_bundle])

    assert transport.flash_count == 1
    assert result.session_complete_cases == 2
    assert transport.completed_case_count == 2
    assert transport.rewind_count == 2
    assert transport.arena_used_bytes == 0
    assert [case.case_bundle.case_id for case in result.cases] == ["abs_persistent", "conv_persistent"]
    assert result.protocol_trace.count("RX:REQUEST_CASE") == 2
    assert result.protocol_trace.count("RX:REQUEST_BLOB") > 10
    assert result.protocol_trace.index("TX:RUN_CORRECTNESS") < result.protocol_trace.index("TX:RUN_PERFORMANCE")
    assert result.protocol_trace[-1] == "RX:SESSION_COMPLETE"
    assert result.protocol_trace.count("RX:SAMPLE_RESULT") == 18
    assert result.cases[0].output_bytes == blob_numpy(abs_bundle.expected_output).tobytes(order="C")
    assert result.cases[1].output_bytes == blob_numpy(conv_bundle.expected_output).tobytes(order="C")
    assert result.cases[0].comparison.passed is True and result.cases[1].comparison.passed is True


def test_case_too_large_fails(tmp_path: Path) -> None:
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path).manifest_path)
    transport = FakeTargetTransport(runtime_arena_capacity=32)

    with pytest.raises(RuntimeError, match="exceeds arena capacity"):
        HostSession(transport).run(conv_bundle)


def test_correctness_failure_still_completes_session_instead_of_deadlocking(tmp_path: Path) -> None:
    """Regression test: a case that fails correctness must not hang the session.

    Historically the host only sent RUN_PERFORMANCE when the correctness
    comparison passed, but the target (both the fake target and the real
    firmware) always advances to WAIT_RUN_PERFORMANCE after CORRECTNESS_ACK
    regardless of the pass/fail byte. That mismatch deadlocked both sides
    whenever a real case legitimately failed correctness on hardware.
    """
    bundle_summary = build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path)
    # Corrupt the expected_output blob on disk so the host's comparison will
    # deliberately mismatch the fake target's (correct) computed output.
    expected_output_path = bundle_summary.manifest_path.parent / "blobs" / "expected_output.bin"
    corrupted = np.frombuffer(expected_output_path.read_bytes(), dtype=np.int8).copy()
    corrupted[0] = np.int8(corrupted[0] + 1)
    expected_output_path.write_bytes(corrupted.tobytes())

    bundle = load_case_bundle(bundle_summary.manifest_path)
    transport = FakeTargetTransport()

    result = HostSession(transport).run(bundle)

    assert result.comparison.passed is False
    assert result.comparison.mismatch_count > 0
    assert result.session_complete_cases == 1
    assert len(result.samples) > 0
    assert result.cases[0].statistics.median_cycles > 0
    assert "TX:RUN_PERFORMANCE" in result.protocol_trace
    assert result.protocol_trace[-1] == "RX:SESSION_COMPLETE"


def test_deliberate_performance_regression_is_detected() -> None:
    class SlowerAbsAdapter(FakeAbsS8Adapter):
        base_cycles_per_iteration = FakeAbsS8Adapter.base_cycles_per_iteration + 9

    input_blob = {"input_0": np.array([-12, -1, 0, 7, -99, 5, -8, 3, -4, 11, -2, 100], dtype=np.int8)}
    counter_passes = plan_counter_passes(resolve_counter_selection({"cpu": "default"}))
    baseline_iterations, baseline_samples = FakeAbsS8Adapter().measure(
        input_blob, {}, warmups=0, samples=3, iterations=4, counter_passes=counter_passes
    )
    candidate_iterations, candidate_samples = SlowerAbsAdapter().measure(
        input_blob, {}, warmups=0, samples=3, iterations=4, counter_passes=counter_passes
    )
    baseline_stats = compute_sample_statistics(normalize_samples(baseline_samples))
    candidate_stats = compute_sample_statistics(normalize_samples(candidate_samples))

    assert baseline_iterations == candidate_iterations == 4
    assert candidate_stats.median_cycles > baseline_stats.median_cycles
