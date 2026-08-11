"""Regression test for hardware_run.py's session-size batching.

Guards against the real hardware bug hit in practice: `run_apollo510_generated_test_session`
used to send every discovered/bridged case in a single LOAD_PLAN. The firmware's
HCT_SERVER_MAX_CASES (see cmake/perf_stream/benchmark_server_session.h) is only 4,
and handle_load_plan() in benchmark_server_session.c silently drops (no reply frame)
a plan naming more cases than that -- which manifested on real Apollo510 hardware as
the host hanging with "Transport stalled without a complete frame." for any
`perf-stream run-generated` invocation discovering more than 4 bridgeable cases
(the default --family ConvolutionFunctions with no --limit discovers 200+).

This test does not touch real hardware/J-Link; it monkeypatches the single-session
runner and result-bundle writer to verify the batching/merging logic in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helia_core_tester.perf_stream import hardware_run
from helia_core_tester.perf_stream.session import SessionResult


class _DummyCaseBundle:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id


def test_max_cases_per_session_matches_firmware_constant() -> None:
    # Keep this in lockstep with HCT_SERVER_MAX_CASES in
    # cmake/perf_stream/benchmark_server_session.h.
    assert hardware_run.MAX_CASES_PER_SESSION == 4


def test_run_single_session_rejects_oversized_plan_instead_of_hanging(tmp_path: Path) -> None:
    bundles = [_DummyCaseBundle(f"case_{i}") for i in range(hardware_run.MAX_CASES_PER_SESSION + 1)]
    try:
        hardware_run._run_single_session(
            tmp_path,
            bundles,  # type: ignore[arg-type]
            serial_no=1,
            chip_name="AP510NFA-CBR",
            speed_khz=4000,
            requested_counter_groups=("cpu",),
            build_dir=tmp_path,
        )
        assert False, "expected ValueError for an oversized single-session plan"
    except ValueError as exc:
        assert "HCT_SERVER_MAX_CASES" in str(exc)


def test_run_case_bundles_in_batches_splits_and_merges(tmp_path: Path, monkeypatch) -> None:
    total_cases = 10  # more than MAX_CASES_PER_SESSION (4) -> 3 batches: 4, 4, 2
    bundles = [_DummyCaseBundle(f"case_{i}") for i in range(total_cases)]

    calls: list[list[Any]] = []

    def _fake_run_single_session(project_root, case_bundles, *, serial_no, chip_name, speed_khz, requested_counter_groups, build_dir):
        assert len(case_bundles) <= hardware_run.MAX_CASES_PER_SESSION
        calls.append(list(case_bundles))
        # One fake "case result" per bundle in this batch, tagged with its case_id.
        fake_result = SessionResult(
            cases=tuple(f"result-for-{b.case_id}" for b in case_bundles),  # type: ignore[arg-type]
            protocol_trace=(f"TX:HELLO_ACK-{case_bundles[0].case_id}",),
            session_complete_cases=len(case_bundles),
        )
        return fake_result, 0xDEADBEEF

    written_results = {}

    def _fake_write_result_bundle(result, *, session_id, output_root, memory_report, kernel_catalog, target_info, host_log_text, target_log_text):
        written_results["result"] = result
        written_results["session_id"] = session_id
        return output_root / "artifacts" / "reports" / "performance_stream" / session_id

    monkeypatch.setattr(hardware_run, "_run_single_session", _fake_run_single_session)
    monkeypatch.setattr(hardware_run, "write_result_bundle", _fake_write_result_bundle)
    monkeypatch.setattr(hardware_run, "generate_benchmark_server_memory_report", lambda build_dir=None: tmp_path / "memory_report.json")
    (tmp_path / "memory_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cmake" / "perf_stream").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cmake" / "perf_stream" / "kernel_catalog.json").write_text("[]", encoding="utf-8")

    merged_result, bundle_root = hardware_run._run_case_bundles_in_batches(
        tmp_path,
        bundles,  # type: ignore[arg-type]
        serial_no=1160002276,
        chip_name="AP510NFA-CBR",
        speed_khz=4000,
        requested_counter_groups=("cpu", "memory", "mve"),
        session_id="test-batching-session",
        build_dir=tmp_path,
        session_id_prefix="apollo510-generated-tests",
    )

    # Batched into ceil(10/4) = 3 sessions of sizes 4, 4, 2 -- never exceeding the
    # firmware's HCT_SERVER_MAX_CASES.
    assert [len(call) for call in calls] == [4, 4, 2]
    assert [b.case_id for b in calls[0]] == [f"case_{i}" for i in range(0, 4)]
    assert [b.case_id for b in calls[1]] == [f"case_{i}" for i in range(4, 8)]
    assert [b.case_id for b in calls[2]] == [f"case_{i}" for i in range(8, 10)]

    # All per-batch case results are merged into one SessionResult, in order.
    assert merged_result.cases == tuple(f"result-for-case_{i}" for i in range(total_cases))
    assert merged_result.session_complete_cases == total_cases
    assert len(merged_result.protocol_trace) == 3
    assert all(entry.startswith("batch") for entry in merged_result.protocol_trace)

    # Exactly one result bundle written for the whole (merged) session, not one per batch.
    assert written_results["session_id"] == "test-batching-session"
    assert written_results["result"] is merged_result
    assert bundle_root == tmp_path / "artifacts" / "reports" / "performance_stream" / "test-batching-session"
