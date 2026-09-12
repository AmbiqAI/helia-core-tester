"""Regression test for hardware_run.py's session-size batching.

Guards against the real hardware bug hit in practice: `run_apollo510_generated_test_session`
used to send every discovered/bridged case in a single LOAD_PLAN. The firmware's
HCT_SERVER_MAX_CASES (see cmake/perf_stream/benchmark_server_session.h) bounds the
cases per plan, and the plan also has to fit the firmware's 2 KiB receive buffer
(case ids can be 96 characters and every PMU pass adds an entry) -- a plan over
either limit is rejected by the target, which used to show up on real Apollo510
hardware as the host hanging with "Transport stalled without a complete frame."

This test does not touch real hardware/J-Link; it monkeypatches the single-session
runner and result-bundle writer to verify the batching/merging logic in isolation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from helia_core_tester.perf_stream import hardware_run
from helia_core_tester.perf_stream.boards import resolve_board
from helia_core_tester.perf_stream.measurement import counter_passes_for_selection
from helia_core_tester.perf_stream.session import SessionResult, load_plan_size

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _DummyCaseBundle:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id


def test_max_cases_per_session_matches_firmware_constant() -> None:
    # Keep this in lockstep with HCT_SERVER_MAX_CASES / HCT_SERVER_RX_BUFFER_BYTES in
    # cmake/perf_stream/benchmark_server_session.h.
    header = (PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_session.h").read_text()
    assert hardware_run.MAX_CASES_PER_SESSION == 32
    assert re.search(r"#define HCT_SERVER_MAX_CASES 32u", header)
    assert hardware_run.FIRMWARE_RX_BUFFER_BYTES == 2048
    assert re.search(r"#define HCT_SERVER_RX_BUFFER_BYTES 2048u", header)
    assert hardware_run.MAX_LOAD_PLAN_PAYLOAD_BYTES == 2048 - 32


def test_batches_are_split_by_case_count_and_encoded_plan_size() -> None:
    passes = counter_passes_for_selection({"cpu": "default", "memory": "default", "mve": "default"})
    short = [_DummyCaseBundle(f"case_{i}") for i in range(70)]
    assert [len(b) for b in hardware_run.split_case_bundles_into_batches(short, passes)] == [32, 32, 6]

    # 96-character case ids (HCT_SERVER_MAX_CASE_ID) cannot all fit 32 to a plan: each
    # costs 102 bytes on the wire, so the 2016-byte rx bound caps a batch well below 32.
    long_ids = [_DummyCaseBundle(f"{i:04d}_" + "x" * 91) for i in range(40)]
    batches = hardware_run.split_case_bundles_into_batches(long_ids, passes)
    assert all(len(b) < 32 for b in batches)
    assert sum(len(b) for b in batches) == 40
    assert [b.case_id for batch in batches for b in batch] == [b.case_id for b in long_ids]
    for batch in batches:
        assert load_plan_size([b.case_id for b in batch], passes) <= hardware_run.MAX_LOAD_PLAN_PAYLOAD_BYTES
    # Adding one more case to any batch would have overflowed the plan.
    for batch, following in zip(batches, batches[1:]):
        ids = [b.case_id for b in batch] + [following[0].case_id]
        assert load_plan_size(ids, passes) > hardware_run.MAX_LOAD_PLAN_PAYLOAD_BYTES

    # mve:all is nine passes; the plan header grows but every batch still fits.
    many_passes = counter_passes_for_selection({"mve": "all", "cpu": "default"})
    assert len(many_passes) == 10
    for batch in hardware_run.split_case_bundles_into_batches(long_ids, many_passes):
        assert load_plan_size([b.case_id for b in batch], many_passes) <= hardware_run.MAX_LOAD_PLAN_PAYLOAD_BYTES

    with pytest.raises(ValueError, match="alone needs"):
        hardware_run.split_case_bundles_into_batches([_DummyCaseBundle("x" * 96)], passes, max_plan_bytes=100)


def test_run_single_session_rejects_oversized_plan_instead_of_hanging(tmp_path: Path) -> None:
    bundles = [_DummyCaseBundle(f"case_{i}") for i in range(hardware_run.MAX_CASES_PER_SESSION + 1)]
    try:
        hardware_run._run_single_session(
            tmp_path,
            bundles,  # type: ignore[arg-type]
            serial_no=1,
            chip_name="AP510NFA-CBR",
            speed_khz=4000,
            counter_passes=counter_passes_for_selection({"cpu": "default"}),
            build_dir=tmp_path,
        )
        assert False, "expected ValueError for an oversized single-session plan"
    except ValueError as exc:
        assert "HCT_SERVER_MAX_CASES" in str(exc)


def test_run_case_bundles_in_batches_splits_and_merges(tmp_path: Path, monkeypatch) -> None:
    total_cases = 70  # more than MAX_CASES_PER_SESSION (32) -> 3 batches: 32, 32, 6
    bundles = [_DummyCaseBundle(f"case_{i}") for i in range(total_cases)]

    calls: list[list[Any]] = []

    def _fake_run_single_session(project_root, case_bundles, *, serial_no, chip_name, speed_khz, counter_passes, build_dir, on_case_complete=None):
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
        written_results["target_info"] = target_info
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
        counter_passes=counter_passes_for_selection({"cpu": "default", "memory": "default", "mve": "default"}),
        session_id="test-batching-session",
        build_dir=tmp_path,
        board=resolve_board("apollo510_evb"),
    )

    # Batched into ceil(70/32) = 3 sessions of sizes 32, 32, 6 -- never exceeding the
    # firmware's HCT_SERVER_MAX_CASES.
    assert [len(call) for call in calls] == [32, 32, 6]
    assert [b.case_id for b in calls[0]] == [f"case_{i}" for i in range(0, 32)]
    assert [b.case_id for b in calls[1]] == [f"case_{i}" for i in range(32, 64)]
    assert [b.case_id for b in calls[2]] == [f"case_{i}" for i in range(64, 70)]

    # All per-batch case results are merged into one SessionResult, in order.
    assert merged_result.cases == tuple(f"result-for-case_{i}" for i in range(total_cases))
    assert merged_result.session_complete_cases == total_cases
    assert merged_result.batch_count == 3
    assert len(merged_result.protocol_trace) == 3
    assert all(entry.startswith("batch") for entry in merged_result.protocol_trace)

    # Exactly one result bundle written for the whole (merged) session, not one per batch.
    assert written_results["session_id"] == "test-batching-session"
    # target_info is derived from the board row, not hard-coded.
    assert written_results["target_info"]["board"] == "apollo510_evb"
    assert written_results["target_info"]["cpu"] == "cortex-m55"
    assert written_results["target_info"]["transport"] == "jlink-rtt"
    assert written_results["result"] is merged_result
    assert bundle_root == tmp_path / "artifacts" / "reports" / "performance_stream" / "test-batching-session"
