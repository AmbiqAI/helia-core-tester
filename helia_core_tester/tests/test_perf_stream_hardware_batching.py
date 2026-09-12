"""Batching of case bundles into SESSION_PLAN-sized sessions (session_runner.py).

The firmware bounds the cases per plan (HCT_SERVER_MAX_CASES), the PMU passes per
plan (HCT_SERVER_MAX_PASSES) and the plan payload (its 2 KiB receive buffer; case
ids can be 96 characters and every PMU pass adds an entry). A plan over any of them
is rejected by the target, which used to show up on real hardware as the host
hanging with "Transport stalled without a complete frame." The host learns all
three limits from TARGET_INFO and cuts every batch from them -- nothing here is a
mirrored constant.

These tests do not touch hardware/J-Link: they drive the batching from a fake
TARGET_INFO and stub the RTT session factory and result-bundle writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from helia_core_tester.perf_stream import session_runner
from helia_core_tester.perf_stream.boards import resolve_board
from helia_core_tester.perf_stream.case_bundle import build_abs_s8_case_bundle, load_case_bundle
from helia_core_tester.perf_stream.fake_target import FakeTargetTransport
from helia_core_tester.perf_stream.measurement import MAX_COUNTERS_PER_PASS, counter_passes_for_selection
from helia_core_tester.perf_stream.session import HostSession, SessionResult, TargetLimits
from helia_core_tester.perf_stream.wire import CAP_PMU_ARMV8M, TargetInfo, session_plan_size

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASSES = counter_passes_for_selection({"cpu": "default", "memory": "default", "mve": "default"})


class _DummyCaseBundle:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id


def _target_info(**overrides: Any) -> TargetInfo:
    """A TARGET_INFO like the real apollo510_evb firmware's (32 cases, 16 passes, 2016-byte
    plans, 8 PMU slots) unless overridden."""
    fields = dict(
        build_id="fake", catalog_hash=bytes(32), max_frame_payload=256, runtime_arena_capacity=114688,
        transfer_mode=1, output_mode=1, board_id="apollo510_evb", target_cpu="cortex-m55", transport_kind=1,
        capability_flags=CAP_PMU_ARMV8M, pmu_counter_slots=8, max_rx_payload=2048 - 32,
        max_cases_per_session=32, max_passes=16,
    )
    fields.update(overrides)
    return TargetInfo(**fields)


class _FakeSession:
    """Stands in for HostSession: handshake() announces `info`, run_many() records the batch."""

    def __init__(self, info: TargetInfo, calls: list[list[Any]], *, fail: Exception | None = None) -> None:
        self._info = info
        self._calls = calls
        self._fail = fail
        self.target_info: TargetInfo | None = None

    def handshake(self) -> TargetInfo:
        if self._fail is not None:
            raise self._fail
        self.target_info = self._info
        return self._info

    @property
    def limits(self) -> TargetLimits:
        return TargetLimits.from_target_info(self._info)

    def run_many(self, case_bundles, *, on_case_complete=None) -> SessionResult:
        assert len(case_bundles) <= self.limits.max_cases
        self._calls.append(list(case_bundles))
        return SessionResult(
            cases=tuple(f"result-for-{b.case_id}" for b in case_bundles),  # type: ignore[arg-type]
            protocol_trace=(f"TX:TARGET_INFO_ACK-{case_bundles[0].case_id}",),
            session_complete_cases=len(case_bundles),
        )


class _FakeTransport:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_limits_are_derived_from_target_info() -> None:
    limits = TargetLimits.from_target_info(_target_info())
    assert (limits.max_cases, limits.max_plan_bytes, limits.max_passes, limits.pmu_counter_slots) == (32, 2016, 16, 8)
    assert limits.has_pmu and limits.max_counters_per_pass == MAX_COUNTERS_PER_PASS == 4
    # The chained-pair planning rule is bounded by the advertised slot count.
    assert TargetLimits.from_target_info(_target_info(pmu_counter_slots=4)).max_counters_per_pass == 2
    assert TargetLimits.from_target_info(_target_info(capability_flags=0, pmu_counter_slots=0)).max_counters_per_pass == 0
    assert TargetLimits.from_target_info(_target_info(max_cases_per_session=7, max_rx_payload=500)).max_cases == 7


def test_batches_are_split_by_case_count_and_encoded_plan_size() -> None:
    limits = TargetLimits.from_target_info(_target_info())
    short = [_DummyCaseBundle(f"case_{i}") for i in range(70)]
    assert [len(b) for b in session_runner.split_case_bundles_into_batches(short, DEFAULT_PASSES, limits)] == [32, 32, 6]
    # A target that takes fewer cases per plan gets smaller batches -- no host constant involved.
    small = TargetLimits.from_target_info(_target_info(max_cases_per_session=10))
    assert [len(b) for b in session_runner.split_case_bundles_into_batches(short, DEFAULT_PASSES, small)] == [10] * 7

    # 96-character case ids (HCT_SERVER_MAX_CASE_ID) cannot all fit 32 to a plan: each
    # costs 102 bytes on the wire, so the 2016-byte rx bound caps a batch well below 32.
    long_ids = [_DummyCaseBundle(f"{i:04d}_" + "x" * 91) for i in range(40)]
    batches = session_runner.split_case_bundles_into_batches(long_ids, DEFAULT_PASSES, limits)
    assert all(len(b) < 32 for b in batches)
    assert sum(len(b) for b in batches) == 40
    assert [b.case_id for batch in batches for b in batch] == [b.case_id for b in long_ids]
    for batch in batches:
        assert session_plan_size([b.case_id for b in batch], DEFAULT_PASSES) <= limits.max_plan_bytes
    # Adding one more case to any batch would have overflowed the plan.
    for batch, following in zip(batches, batches[1:]):
        ids = [b.case_id for b in batch] + [following[0].case_id]
        assert session_plan_size(ids, DEFAULT_PASSES) > limits.max_plan_bytes

    # mve:all is nine passes; the plan header grows but every batch still fits.
    many_passes = counter_passes_for_selection({"mve": "all", "cpu": "default"})
    assert len(many_passes) == 10
    for batch in session_runner.split_case_bundles_into_batches(long_ids, many_passes, limits):
        assert session_plan_size([b.case_id for b in batch], many_passes) <= limits.max_plan_bytes

    tiny = TargetLimits.from_target_info(_target_info(max_rx_payload=100))
    with pytest.raises(ValueError, match="alone needs"):
        session_runner.take_batch([_DummyCaseBundle("x" * 96)], DEFAULT_PASSES, tiny)


def test_session_refuses_more_cases_than_the_target_takes(tmp_path: Path) -> None:
    bundles = [
        load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id=f"abs_{i}").manifest_path)
        for i in range(2)
    ]
    session = HostSession(FakeTargetTransport(max_cases_per_session=1), counter_passes=counter_passes_for_selection({"cpu": "default"}))
    with pytest.raises(RuntimeError, match="takes at most 1 per SESSION_PLAN"):
        session.run_many(bundles)
    assert "TX:SESSION_PLAN" not in session._trace
    # Likewise a pass list longer than max_passes is refused at the handshake.
    session = HostSession(FakeTargetTransport(max_passes=2), counter_passes=DEFAULT_PASSES)
    with pytest.raises(RuntimeError, match="3 PMU passes requested.*at most 2"):
        session.run_many(bundles[:1])


def test_run_case_bundles_batches_from_each_sessions_target_info(tmp_path: Path, monkeypatch) -> None:
    bundles = [_DummyCaseBundle(f"case_{i}") for i in range(70)]
    calls: list[list[Any]] = []
    transports: list[_FakeTransport] = []
    announced = {"info": _target_info()}

    def _open(board, serial_no, *, build_dir, counter_passes):
        assert (board.id, serial_no, build_dir) == ("apollo510_evb", 1160002276, tmp_path)
        transport = _FakeTransport()
        transports.append(transport)
        return _FakeSession(announced["info"], calls), transport, 0xDEADBEEF

    written: dict[str, Any] = {}

    def _fake_write_result_bundle(result, *, session_id, output_root, memory_report, kernel_catalog, target_info, host_log_text, target_log_text):
        written.update(result=result, session_id=session_id, target_info=target_info, host_log=host_log_text)
        return output_root / "artifacts" / "reports" / "performance_stream" / session_id

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    monkeypatch.setattr(session_runner, "write_result_bundle", _fake_write_result_bundle)
    monkeypatch.setattr(session_runner, "generate_memory_report", lambda board, project_root=None, build_dir=None: tmp_path / "memory_report.json")
    (tmp_path / "memory_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cmake" / "perf_stream").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cmake" / "perf_stream" / "kernel_catalog.json").write_text("[]", encoding="utf-8")

    merged, bundle_root = session_runner.run_case_bundles(
        tmp_path,
        bundles,  # type: ignore[arg-type]
        board=resolve_board("apollo510_evb"),
        serial_no=1160002276,
        counter_passes=DEFAULT_PASSES,
        session_id="test-batching-session",
        build_dir=tmp_path,
    )

    # The target advertised 32 cases per plan: ceil(70/32) = 3 sessions of 32, 32, 6,
    # each over its own transport, closed afterwards.
    assert [len(call) for call in calls] == [32, 32, 6]
    assert [b.case_id for b in calls[0]] == [f"case_{i}" for i in range(0, 32)]
    assert [b.case_id for b in calls[2]] == [f"case_{i}" for i in range(64, 70)]
    assert [t.closed for t in transports] == [1, 1, 1]

    # All per-batch case results are merged into one SessionResult, in order.
    assert merged.cases == tuple(f"result-for-case_{i}" for i in range(70))
    assert merged.session_complete_cases == 70
    assert merged.batch_count == 3
    assert merged.target_info == announced["info"]
    assert all(entry.startswith("batch") for entry in merged.protocol_trace) and len(merged.protocol_trace) == 3

    # Exactly one result bundle for the whole (merged) session, with board-derived target info.
    assert written["session_id"] == "test-batching-session"
    assert written["target_info"]["board"] == "apollo510_evb"
    assert written["target_info"]["cpu"] == "cortex-m55"
    assert written["target_info"]["transport"] == "jlink-rtt"
    assert "max_cases_per_session=32 max_session_plan_bytes=2016" in written["host_log"]
    assert written["result"] is merged
    assert bundle_root == tmp_path / "artifacts" / "reports" / "performance_stream" / "test-batching-session"

    # A different target announces different limits and the same run batches differently.
    calls.clear()
    announced["info"] = _target_info(max_cases_per_session=25)
    session_runner.run_case_bundles(
        tmp_path, bundles, board=resolve_board("apollo510_evb"), serial_no=1160002276,  # type: ignore[arg-type]
        counter_passes=DEFAULT_PASSES, session_id="s", build_dir=tmp_path,
    )
    assert [len(call) for call in calls] == [25, 25, 20]


def test_run_case_bundles_names_the_batch_when_a_session_fails(tmp_path: Path, monkeypatch) -> None:
    def _open(board, serial_no, *, build_dir, counter_passes):
        return _FakeSession(_target_info(), [], fail=RuntimeError("Transport stalled")), _FakeTransport(), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    with pytest.raises(RuntimeError, match=r"Transport stalled \(batch 0, candidate case_ids=\['case_0', 'case_1'\]\)"):
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle("case_0"), _DummyCaseBundle("case_1")],  # type: ignore[arg-type]
            board=resolve_board("apollo510_evb"), serial_no=1, counter_passes=DEFAULT_PASSES, build_dir=tmp_path,
        )
