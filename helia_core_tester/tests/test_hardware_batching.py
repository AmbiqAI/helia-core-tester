"""Batching of case bundles into SESSION_PLAN-sized sessions (session_runner.py).

The firmware bounds the cases per plan (HCT_SERVER_MAX_CASES), the PMU passes per
plan (HCT_SERVER_MAX_PASSES) and the plan payload (its 2 KiB receive buffer; case
ids can be 95 bytes and every PMU pass adds an entry). A plan over any of them
is rejected by the target, which used to show up on real hardware as the host
hanging with "Transport stalled without a complete frame." The host learns all
three limits from TARGET_INFO and cuts every batch from them. The host constants
(measurement.MAX_PASSES_PER_PLAN / MAX_CASES_PER_PLAN, session.MAX_CASE_ID_BYTES)
mirror the firmware header only for the checks that run before the probe is opened
and for the fake target, and are lockstep-tested against the header here.

These tests do not touch hardware/J-Link: they drive the batching from a fake
TARGET_INFO and stub the RTT session factory and result-bundle writer.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from helia_core_tester.hardware import measurement, session, session_runner
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.case_bundle import build_abs_s8_case_bundle, load_case_bundle
from helia_core_tester.hardware.fake_target import FakeTargetTransport
from helia_core_tester.hardware.hctp import HEADER_SIZE
from helia_core_tester.hardware.measurement import (
    MAX_COUNTERS_PER_PASS,
    CounterPass,
    OutboxOverflowError,
    RawCounterValue,
    RawSample,
    case_tail_bytes,
    check_outbox_fits,
    counter_passes_for_selection,
    sample_frame_bytes,
)
from helia_core_tester.hardware.pmu_catalog import CPU_CYCLES_EVENT_ID, counter_by_name
from helia_core_tester.hardware.session import HostSession, SessionResult, TargetLimits
from helia_core_tester.hardware.wire import (
    CAP_PMU_ARMV8M,
    CaseComplete,
    RequestCase,
    TargetInfo,
    encode_case_complete,
    encode_request_case,
    encode_sample_result,
    session_plan_size,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASSES = counter_passes_for_selection({"cpu": "default", "memory": "default", "mve": "default"})


class _DummyCaseBundle:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id


def _target_info(**overrides: Any) -> TargetInfo:
    """A TARGET_INFO like the real apollo510_evb firmware's (32 cases, 32 passes, 2016-byte
    plans, 8 PMU slots) unless overridden."""
    fields = dict(
        build_id="fake", catalog_hash=bytes(32), max_frame_payload=256, runtime_arena_capacity=114688,
        transfer_mode=1, output_mode=1, board_id="apollo510_evb", target_cpu="cortex-m55", transport_kind=1,
        capability_flags=CAP_PMU_ARMV8M, pmu_counter_slots=8, max_rx_payload=2048 - 32,
        max_cases_per_session=32, max_passes=32,
    )
    fields.update(overrides)
    return TargetInfo(**fields)


def test_host_constants_match_the_firmware_header() -> None:
    # Keep the pre-probe checks and the fake target in lockstep with HCT_SERVER_MAX_CASES /
    # HCT_SERVER_MAX_PASSES / HCT_SERVER_MAX_CASE_ID / HCT_SERVER_RX_BUFFER_BYTES in
    # cmake/hardware/benchmark_server_session.h.
    header = (PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_session.h").read_text()
    assert measurement.MAX_CASES_PER_PLAN == 32
    assert re.search(r"#define HCT_SERVER_MAX_CASES 32u", header)
    assert measurement.MAX_PASSES_PER_PLAN == 32
    assert re.search(r"#define HCT_SERVER_MAX_PASSES 32u", header)
    # char[96] storage and cursor_text() needs the NUL, so 95 payload bytes.
    assert session.MAX_CASE_ID_BYTES == 96 - 1
    assert re.search(r"#define HCT_SERVER_MAX_CASE_ID 96u", header)
    assert re.search(r"#define HCT_SERVER_RX_BUFFER_BYTES 2048u", header)
    assert measurement.MAX_OUTBOX_BYTES == 32768
    assert re.search(r"#define HCT_SERVER_MAX_OUTBOX_BYTES 32768u", header)
    # The fake target advertises the same limits by default.
    info = FakeTargetTransport()
    assert (info._max_cases_per_session, info._max_passes) == (measurement.MAX_CASES_PER_PLAN, measurement.MAX_PASSES_PER_PLAN)


def test_run_case_bundles_refuses_more_passes_than_the_firmware_runs_before_opening_the_probe(tmp_path: Path, monkeypatch) -> None:
    # One pass over HCT_SERVER_MAX_PASSES: the runner must refuse before symbol
    # lookup / J-Link, naming the passes and the limit.
    single = counter_by_name("ARM_PMU_INST_RETIRED")
    passes = tuple(CounterPass("cpu", i, (single,)) for i in range(measurement.MAX_PASSES_PER_PLAN + 1))
    monkeypatch.setattr(session_runner, "open_rtt_session", lambda *a, **k: pytest.fail("probe opened"))
    with pytest.raises(ValueError, match=r"33 PMU passes planned \(cpu_0, .*cpu_32\) but the firmware runs at most 32 per SESSION_PLAN"):
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle("case_0")],  # type: ignore[arg-type]
            board=resolve_board("apollo510_evb"), serial_no=1, counter_passes=passes, build_dir=tmp_path,
        )


def test_outbox_check_sizes_real_sample_frames_and_refuses_overflow() -> None:
    # The sizing matches the encoder, cycles entry included.
    passes = counter_passes_for_selection({"cpu": "all", "memory": "all", "mve": "all"})
    for counter_pass in passes:
        counters = [RawCounterValue("", CPU_CYCLES_EVENT_ID, 0)] + [RawCounterValue("", c.event_id, 0) for c in counter_pass.counters]
        sample = RawSample(sample_index=0, iterations=1, cycles=1, counters=tuple(counters), pass_name=counter_pass.name)
        assert sample_frame_bytes(counter_pass) == HEADER_SIZE + len(encode_sample_result(sample))
    assert max(sample_frame_bytes(counter_pass) for counter_pass in passes) == 127

    # The tail matches the encoders too.
    case_id = "c" * session.MAX_CASE_ID_BYTES
    complete = encode_case_complete(CaseComplete(case_id=case_id, workspace_used_bytes=0))
    assert case_tail_bytes(case_id) == 2 * HEADER_SIZE + len(complete) + len(encode_request_case(RequestCase(0)))

    # Full catalog: 14 samples fit, 15 overflow.
    check_outbox_fits(passes, 14, case_id)
    with pytest.raises(OutboxOverflowError, match=rf"Case '{case_id}' results need \d+ B \(15 samples x 18 passes\); outbox holds 32768 B"):
        check_outbox_fits(passes, 15, case_id)
    # An empty plan still costs one cpu_0 frame.
    check_outbox_fits((), 480, "")
    with pytest.raises(OutboxOverflowError, match=r"481 samples x 1 passes"):
        check_outbox_fits((), 481, "")


class _FakeSession:
    """Stands in for HostSession: handshake() announces `info`, run_many() records the batch."""

    def __init__(self, info: TargetInfo, calls: list[list[Any]], *, fail: Exception | None = None) -> None:
        self._info = info
        self._calls = calls
        self._fail = fail
        self.target_info: TargetInfo | None = None
        self.expected_build_id: str | None = None

    def handshake(self, *, expected_build_id: str | None = None, expected_clock_hz: int | None = None) -> TargetInfo:
        if self._fail is not None:
            raise self._fail
        self.expected_build_id = expected_build_id
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
    assert (limits.max_cases, limits.max_plan_bytes, limits.max_passes, limits.pmu_counter_slots) == (32, 2016, 32, 8)
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

    # 95-byte case ids (the longest the firmware's char[HCT_SERVER_MAX_CASE_ID] takes
    # with its NUL) cannot all fit 32 to a plan: each costs 101 bytes on the wire, so
    # the 2016-byte rx bound caps a batch well below 32.
    long_ids = [_DummyCaseBundle(f"{i:04d}_" + "x" * 90) for i in range(40)]
    assert all(len(b.case_id.encode("utf-8")) == session.MAX_CASE_ID_BYTES for b in long_ids)
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

    # The full catalog is 18 passes; the plan header grows but every batch still fits.
    many_passes = counter_passes_for_selection({"cpu": "all", "memory": "all", "mve": "all"})
    assert len(many_passes) == 18
    for batch in session_runner.split_case_bundles_into_batches(long_ids, many_passes, limits):
        assert session_plan_size([b.case_id for b in batch], many_passes) <= limits.max_plan_bytes

    tiny = TargetLimits.from_target_info(_target_info(max_rx_payload=100))
    with pytest.raises(ValueError, match="alone needs"):
        session_runner.take_batch([_DummyCaseBundle("x" * 95)], DEFAULT_PASSES, tiny)


def test_case_ids_over_the_firmware_storage_are_refused_before_any_session(tmp_path: Path, monkeypatch) -> None:
    limits = TargetLimits.from_target_info(_target_info())
    passes = counter_passes_for_selection({"cpu": "default"})
    # Exactly 95 bytes fits (char[96] with the NUL); 96 does not, and the batcher says
    # so rather than the firmware truncating the plan.
    ok = _DummyCaseBundle("y" * 95)
    assert session_runner.split_case_bundles_into_batches([ok], passes, limits) == [[ok]]
    with pytest.raises(ValueError, match=r"Case id 'y{96}' is 96 bytes; the firmware stores at most 95 \(HCT_SERVER_MAX_CASE_ID 96"):
        session_runner.take_batch([ok, _DummyCaseBundle("y" * 96)], passes, limits)
    # The limit is in bytes, not characters: 48 two-byte characters fit, 48 plus one more does not.
    assert session_runner.take_batch([_DummyCaseBundle("\u00e9" * 47 + "z")], passes, limits)
    with pytest.raises(ValueError, match="is 96 bytes"):
        session_runner.take_batch([_DummyCaseBundle("\u00e9" * 48)], passes, limits)
    # The runner checks every id before the J-Link session is opened.
    monkeypatch.setattr(session_runner, "open_rtt_session", lambda *a, **k: pytest.fail("probe opened"))
    with pytest.raises(ValueError, match="is 96 bytes"):
        session_runner.run_case_bundles(
            tmp_path, [ok, _DummyCaseBundle("y" * 96)],  # type: ignore[arg-type]
            board=resolve_board("apollo510_evb"), serial_no=1, counter_passes=passes, build_dir=tmp_path,
        )


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
    with pytest.raises(RuntimeError, match=r"3 PMU passes planned \(cpu_0, memory_0, mve_0\) but target 'fake_board' runs at most 2 per SESSION_PLAN"):
        session.run_many(bundles[:1])


# None: --allow-unverified-firmware.
@pytest.mark.parametrize("expected", ["fake", None])
def test_run_case_bundles_batches_from_each_sessions_target_info(tmp_path: Path, monkeypatch, expected) -> None:
    bundles = [_DummyCaseBundle(f"case_{i}") for i in range(70)]
    calls: list[list[Any]] = []
    transports: list[_FakeTransport] = []
    announced = {"info": _target_info()}
    sessions: list[_FakeSession] = []

    def _open(board, serial_no, *, build_dir, counter_passes):
        assert (board.id, serial_no, build_dir) == ("apollo510_evb", 1160002276, tmp_path)
        transport = _FakeTransport()
        transports.append(transport)
        session = _FakeSession(announced["info"], calls)
        sessions.append(session)
        return session, transport, 0xDEADBEEF

    written: dict[str, Any] = {}

    def _fake_write_result_bundle(result, *, session_id, output_root, memory_report, kernel_catalog, target_info, host_log_text, target_log_text, build_dir, compare):
        written.update(result=result, session_id=session_id, target_info=target_info, host_log=host_log_text, build_dir=build_dir)
        return output_root / "artifacts" / "reports" / "hardware" / session_id

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    monkeypatch.setattr(session_runner, "write_result_bundle", _fake_write_result_bundle)
    report_roots: list[Path] = []

    def _fake_memory_report(board, *, project_root, build_dir, output_root):
        report_roots.append(output_root)
        return tmp_path / "memory_report.json"

    monkeypatch.setattr(session_runner, "generate_memory_report", _fake_memory_report)
    (tmp_path / "memory_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cmake" / "hardware").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cmake" / "hardware" / "kernel_catalog.json").write_text("[]", encoding="utf-8")

    merged, bundle_root = session_runner.run_case_bundles(
        tmp_path,
        bundles,  # type: ignore[arg-type]
        board=resolve_board("apollo510_evb"),
        serial_no=1160002276,
        counter_passes=DEFAULT_PASSES,
        session_id="test-batching-session",
        build_dir=tmp_path,
        expected_build_id=expected,
    )

    # The target advertised 32 cases per plan: ceil(70/32) = 3 sessions of 32, 32, 6,
    # each over its own transport, closed afterwards.
    assert [len(call) for call in calls] == [32, 32, 6]
    # Concurrent boards keep separate reports.
    assert report_roots == [tmp_path / "artifacts" / "hardware" / "benchmark_server" / "apollo510_evb"]
    assert [b.case_id for b in calls[0]] == [f"case_{i}" for i in range(0, 32)]
    assert [b.case_id for b in calls[2]] == [f"case_{i}" for i in range(64, 70)]
    assert [t.closed for t in transports] == [1, 1, 1]

    # All per-batch case results are merged into one SessionResult, in order.
    assert merged.cases == tuple(f"result-for-case_{i}" for i in range(70))
    assert merged.session_complete_cases == 70
    assert merged.batch_count == 3
    assert merged.target_info == announced["info"]
    # The bundle writer seeds its counter columns from the passes the plan asked for.
    assert merged.counter_passes == DEFAULT_PASSES
    # The build dir's id is checked at every session's handshake and reported once.
    assert [s.expected_build_id for s in sessions] == [expected] * 3
    assert merged.build_id == "fake"
    assert all(entry.startswith("batch") for entry in merged.protocol_trace) and len(merged.protocol_trace) == 3

    # Exactly one result bundle for the whole (merged) session, with board-derived target info.
    assert written["session_id"] == "test-batching-session"
    assert written["target_info"]["board"] == "apollo510_evb"
    assert written["target_info"]["cpu"] == "cortex-m55"
    assert written["target_info"]["transport"] == "jlink-rtt"
    assert "max_cases_per_session=32 max_session_plan_bytes=2016" in written["host_log"]
    assert "firmware_build_id=fake" in written["host_log"]
    assert written["result"] is merged
    # Provenance only for verified firmware.
    assert written["build_dir"] == (tmp_path if expected else None)
    assert bundle_root == tmp_path / "artifacts" / "reports" / "hardware" / "test-batching-session"

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


class _HaltableTransport(_FakeTransport):
    """Reports a target parked in its fault handler."""

    def __init__(self, state: dict[str, int] | Exception) -> None:
        super().__init__()
        self._state = state

    def target_state(self) -> dict[str, int]:
        if isinstance(self._state, Exception):
            raise self._state
        return self._state


_SYMBOLS = [
    (0x00410798, "T", "main"), (0x004967E0, "W", "BusFault_Handler"), (0x004967E0, "W", "HardFault_Handler"),
    (0x20000020, "b", "g_pui32Stack"),
]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            dict(pc=0x004967E0, lr=0xFFFFFFF9, sp=0x20003F00, cfsr=0x00100000, hfsr=0x40000000, rtt_size=8192, rtt_write=10, rtt_read=10),
            "Target faulted; decode CFSR/HFSR. PC=0x004967e0 (BusFault_Handler/HardFault_Handler), LR=0xfffffff9, "
            "SP=0x20003f00, CFSR=0x00100000 HFSR=0x40000000, RTT up write=10 read=10 size=8192.",
        ),
        (
            dict(pc=0x004107A1, lr=0x00410799, sp=0x20003F00, cfsr=0, hfsr=0, rtt_size=8192, rtt_write=99, rtt_read=100),
            "Target blocked on a full RTT buffer. PC=0x004107a1 (main+0x8), LR=0x00410799, SP=0x20003f00, "
            "CFSR=0x00000000 HFSR=0x00000000, RTT up write=99 read=100 size=8192.",
        ),
        (
            dict(pc=0x004107A1, lr=0x00410799, sp=0x20003F00, cfsr=0, hfsr=0, rtt_size=8192, rtt_write=7, rtt_read=7),
            "No fault; target sent all queued RTT bytes. PC=0x004107a1 (main+0x8), LR=0x00410799, SP=0x20003f00, "
            "CFSR=0x00000000 HFSR=0x00000000, RTT up write=7 read=7 size=8192.",
        ),
        (
            dict(pc=0x004107A1, lr=0x00410799, sp=0x20003F00, cfsr=0, hfsr=0),
            "Target running; no fault latched. PC=0x004107a1 (main+0x8), LR=0x00410799, SP=0x20003f00, "
            "CFSR=0x00000000 HFSR=0x00000000.",
        ),
    ],
    ids=["faulted", "rtt-full", "drained", "running"],
)
def test_stall_names_the_target_state(tmp_path: Path, monkeypatch, state, expected) -> None:
    def _open(board, serial_no, *, build_dir, counter_passes):
        return _FakeSession(_target_info(), [], fail=session.TransportStall("Transport stalled")), _HaltableTransport(state), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    monkeypatch.setattr(session_runner, "elf_symbols", lambda elf: _SYMBOLS)
    with pytest.raises(RuntimeError) as raised:
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle("case_0")],  # type: ignore[arg-type]
            board=resolve_board("apollo510_evb"), serial_no=1, counter_passes=DEFAULT_PASSES, build_dir=tmp_path,
        )
    assert str(raised.value) == f"Transport stalled {expected} (batch 0, candidate case_ids=['case_0'])"


def test_stall_survives_an_unreadable_target(tmp_path: Path, monkeypatch) -> None:
    def _open(board, serial_no, *, build_dir, counter_passes):
        stall = session.TransportStall("Transport stalled")
        return _FakeSession(_target_info(), [], fail=stall), _HaltableTransport(OSError("probe gone")), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    with pytest.raises(RuntimeError, match=r"^Transport stalled Target state unreadable: probe gone\. \(batch 0, "):
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle("case_0")],  # type: ignore[arg-type]
            board=resolve_board("apollo510_evb"), serial_no=1, counter_passes=DEFAULT_PASSES, build_dir=tmp_path,
        )


def test_consistency_check_covers_boot_health() -> None:
    first = _target_info(boot_status=0, core_clock_hz=250_000_000)
    with pytest.raises(RuntimeError, match=r"core_clock_hz: 250000000 -> 96000000"):
        session_runner.check_target_info_consistent(first, replace(first, core_clock_hz=96_000_000), batch_index=1)


def test_consistency_check_covers_fpscr() -> None:
    first = _target_info(fpscr_boot=0x03040000, fpscr=0x00040000)
    with pytest.raises(RuntimeError, match=r"fpscr: 262144 -> 50593792"):
        session_runner.check_target_info_consistent(first, replace(first, fpscr=0x03040000), batch_index=1)


def test_boot_failure_skips_batch_context(tmp_path: Path, monkeypatch) -> None:
    def _open(board, serial_no, *, build_dir, counter_passes):
        return HostSession(FakeTargetTransport(boot_status=7, core_clock_hz=96_000_000)), _FakeTransport(), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    with pytest.raises(RuntimeError, match=r"^Board init failed: nsx_system_init status 7, core 96 MHz\.$"):
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle("case_0")],  # type: ignore[arg-type]
            board=resolve_board("apollo510_evb"), serial_no=1, counter_passes=DEFAULT_PASSES, build_dir=tmp_path,
        )


def test_runner_checks_board_row_clock(tmp_path: Path, monkeypatch) -> None:
    def _open(board, serial_no, *, build_dir, counter_passes):
        return HostSession(FakeTargetTransport(core_clock_hz=250_000_000)), _FakeTransport(), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    with pytest.raises(RuntimeError, match=r"^Board core clock 250 MHz, expected 48 MHz\.$"):
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle("case_0")],  # type: ignore[arg-type]
            board=resolve_board("apollo3p_evb"), serial_no=1, counter_passes=DEFAULT_PASSES, build_dir=tmp_path,
        )


def test_run_case_bundles_refuses_to_merge_sessions_from_different_firmware(tmp_path: Path, monkeypatch) -> None:
    # A board reflashed mid-run (or a second host on the probe) announces a different
    # TARGET_INFO on a later batch; the runner must fail fast instead of merging results
    # from two firmware builds into one bundle.
    bundles = [_DummyCaseBundle(f"case_{i}") for i in range(40)]
    calls: list[list[Any]] = []
    infos = iter([_target_info(build_id="hct-first"), _target_info(build_id="hct-second", max_passes=8)])

    def _open(board, serial_no, *, build_dir, counter_passes):
        return _FakeSession(next(infos), calls), _FakeTransport(), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    with pytest.raises(RuntimeError, match=r"TARGET_INFO of batch 1 differs from the first session's.*build_id: 'hct-first' -> 'hct-second'.*max_passes: 32 -> 8"):
        session_runner.run_case_bundles(
            tmp_path, bundles, board=resolve_board("apollo510_evb"), serial_no=1160002276,  # type: ignore[arg-type]
            counter_passes=DEFAULT_PASSES, session_id="s", build_dir=tmp_path,
        )
    assert [len(call) for call in calls] == [32]  # the first batch ran; the second never did


def test_run_case_bundles_wraps_a_case_that_cannot_fit_the_advertised_plan_size(tmp_path: Path, monkeypatch) -> None:
    # take_batch() raises ValueError when one case alone exceeds max_rx_payload; the runner
    # must surface it as the RuntimeError the CLI turns into a one-line hardware error.
    calls: list[list[Any]] = []
    tiny = _target_info(max_rx_payload=40)

    def _open(board, serial_no, *, build_dir, counter_passes):
        return _FakeSession(tiny, calls), _FakeTransport(), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    with pytest.raises(RuntimeError, match=r"alone needs a \d+-byte SESSION_PLAN.*\(batch 0, candidate case_ids="):
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle("a_case_id_longer_than_the_tiny_limit")], board=resolve_board("apollo510_evb"),  # type: ignore[arg-type]
            serial_no=1160002276, counter_passes=DEFAULT_PASSES, session_id="s", build_dir=tmp_path,
        )
    assert calls == []


def test_run_case_bundles_refuses_a_later_session_with_different_capabilities(tmp_path: Path, monkeypatch) -> None:
    # A cycles-only plan never trips the PMU validation, so a later session that
    # advertises different capability_flags (PMU gone, or appeared) must still be refused
    # rather than merged under the first session's target metadata.
    calls: list[list[Any]] = []
    first = _target_info()
    infos = iter([first, _target_info(capability_flags=first.capability_flags ^ 0x40)])

    def _open(board, serial_no, *, build_dir, counter_passes):
        return _FakeSession(next(infos), calls), _FakeTransport(), 0

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    cycles_only = (CounterPass("cpu", 0, (), chained=True),)
    with pytest.raises(RuntimeError, match=r"TARGET_INFO of batch 1 differs.*capability_flags: "):
        session_runner.run_case_bundles(
            tmp_path, [_DummyCaseBundle(f"case_{i}") for i in range(40)], board=resolve_board("apollo510_evb"),  # type: ignore[arg-type]
            serial_no=1160002276, counter_passes=cycles_only, session_id="s", build_dir=tmp_path,
        )
    assert [len(call) for call in calls] == [32]


def test_non_positive_target_limits_are_refused_before_batching() -> None:
    # take_batch()'s contract is "at most max_cases"; a target advertising 0 must be
    # refused outright rather than yield a one-case batch the next layer rejects.
    with pytest.raises(ValueError, match=r"non-positive session limits: \{'max_cases_per_session': 0\}"):
        TargetLimits.from_target_info(_target_info(max_cases_per_session=0))
    with pytest.raises(ValueError, match=r"'max_rx_payload': 0"):
        TargetLimits.from_target_info(_target_info(max_rx_payload=0))
    # max_passes too: firmware turns a zero-pass plan into one cpu_0 pass, so a target
    # advertising 0 could never honour its own limit.
    with pytest.raises(ValueError, match=r"'max_passes': 0"):
        TargetLimits.from_target_info(_target_info(max_passes=0))
    zero = TargetLimits(max_cases=0, max_plan_bytes=2016, max_passes=16, pmu_counter_slots=8, has_pmu=True)
    with pytest.raises(ValueError, match=r"max_cases=0, max_plan_bytes=2016: both must be positive"):
        session_runner.take_batch([_DummyCaseBundle("case_0")], DEFAULT_PASSES, zero)  # type: ignore[list-item]


def test_duplicate_case_ids_are_refused_before_the_probe_opens(tmp_path: Path, monkeypatch) -> None:
    opened: list[int] = []

    def _open(board, serial_no, *, build_dir, counter_passes):
        opened.append(1)
        raise AssertionError("must not open a session")

    monkeypatch.setattr(session_runner, "open_rtt_session", _open)
    bundles = [_DummyCaseBundle("case_a"), _DummyCaseBundle("case_b"), _DummyCaseBundle("case_a")]
    with pytest.raises(ValueError, match=r"Duplicate case id\(s\) in one run: \['case_a'\]"):
        session_runner.run_case_bundles(
            tmp_path, bundles, board=resolve_board("apollo510_evb"), serial_no=1160002276,  # type: ignore[arg-type]
            counter_passes=DEFAULT_PASSES, session_id="s", build_dir=tmp_path,
        )
    assert opened == []
