from __future__ import annotations

import json
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
from helia_core_tester.perf_stream.measurement import (
    MAX_CASES_PER_PLAN,
    MAX_PASSES_PER_PLAN,
    CounterPass,
    TooManyPassesError,
    check_pass_count,
    compute_sample_statistics,
    counter_names_for_passes,
    counter_passes_for_selection,
    normalize_samples,
    plan_counter_passes,
    resolve_counter_selection,
)
from helia_core_tester.perf_stream.pmu_catalog import CPU_CYCLES_EVENT_ID, counter_by_name
from helia_core_tester.perf_stream.session import (
    MAX_CASE_ID_BYTES,
    HostSession,
    run_fake_abs_vertical_slice,
    run_fake_convolve_vertical_slice,
    session_plan_for_bundles,
)
from helia_core_tester.perf_stream.wire import PlannedCase, SessionPlan, encode_session_plan, session_plan_size

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

    # TARGET_INFO: the fake advertises the Armv8.1-M PMU with 8 slots, its rx bound and limits.
    assert result.target_info is not None and result.target_info.has_pmu
    assert result.target_info.pmu_counter_slots == 8
    assert result.target_info.max_rx_payload == 2048 - 32
    assert result.target_info.max_cases_per_session == 32
    assert result.target_info.max_passes == 16
    # Every sample leads with ARM_PMU_CPU_CYCLES from CCNTR, close to the DWT cycles,
    # and the remaining entries are the pass's counters named from the catalog (the
    # target sends empty names).
    for sample in result.samples:
        first = sample.counters[0]
        assert first.name == "ARM_PMU_CPU_CYCLES" and first.event_id == CPU_CYCLES_EVENT_ID
        assert 0 <= first.value - sample.cycles < 16
        assert [c.name for c in sample.counters[1:]] == ["ARM_PMU_INST_RETIRED", "ARM_PMU_STALL_FRONTEND", "ARM_PMU_STALL_BACKEND"]
        assert all(not c.overflow and c.supported for c in sample.counters)

    trace = result.protocol_trace
    assert trace[0] == "RX:TARGET_INFO"
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
    # The fake convolve adapter has no mve group, so the whole default mve pass is unsupported.
    assert case.statistics.unsupported_counters == (
        "ARM_PMU_MVE_INST_RETIRED", "ARM_PMU_MVE_INT_MAC_RETIRED", "ARM_PMU_MVE_LDST_RETIRED", "ARM_PMU_MVE_STALL",
    )
    assert case.statistics.overflow_detected is False
    assert case.statistics.median_cycles > 0
    assert result.protocol_trace.count("RX:REQUEST_BLOB") >= 6



def test_multi_case_session_rewinds_arena(tmp_path: Path) -> None:
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_case_a").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_case_b").manifest_path)
    transport = FakeTargetTransport(max_frame_payload=15, read_chunk_size=9)

    result = HostSession(transport, counter_passes=counter_passes_for_selection({"cpu": "default"})).run_many([abs_bundle, conv_bundle])

    assert result.session_complete_cases == 2
    assert transport.completed_case_count == 2
    assert transport.rewind_count == 2
    assert transport.arena_used_bytes == 0
    assert all(case.comparison.passed for case in result.cases)
    assert transport.case_workspace_history == (
        abs_bundle.workspace_bytes_required,
        conv_bundle.workspace_bytes_required,
    )
    assert abs_bundle.workspace_bytes_required != conv_bundle.workspace_bytes_required


def test_persistent_fake_target_multi_operator_session_without_reflash(tmp_path: Path) -> None:
    abs_bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_persistent").manifest_path)
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_persistent").manifest_path)
    transport = FakeTargetTransport(max_frame_payload=9, read_chunk_size=7)

    passes = counter_passes_for_selection({"cpu": "default", "memory": "default", "mve": "default"})
    result = HostSession(transport, counter_passes=passes).run_many([abs_bundle, conv_bundle])

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


def test_mve_all_plans_nine_passes_and_every_pass_reports_ccntr(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_mve_all").manifest_path)
    passes = counter_passes_for_selection({"mve": "all", "cpu": "default"})
    assert [p.name for p in passes] == [f"mve_{i}" for i in range(9)] + ["cpu_0"]
    assert [len(p.counters) for p in passes] == [4] * 8 + [2, 3]

    result = HostSession(FakeTargetTransport(), counter_passes=passes).run(bundle)

    case = result.cases[0]
    assert len(case.samples) == 3 * 10
    assert [s.pass_name for s in case.samples][::3] == [p.name for p in passes]
    assert all(s.counters[0].name == "ARM_PMU_CPU_CYCLES" for s in case.samples)
    # 34 mve names + CPU_CYCLES + the 3 cpu defaults; mve is unsupported on the abs fake.
    reported = {c.name for s in case.samples for c in s.counters}
    assert len(reported) == 34 + 1 + 3
    assert len(case.statistics.unsupported_counters) == 34


def test_every_group_all_exceeds_the_firmware_pass_limit_and_fails_before_session_plan(tmp_path: Path) -> None:
    # Every value the --pmu-counters help advertises at once: 21 cpu + 15 memory + 34 mve
    # events plan 5 + 4 + 9 = 18 passes, two over HCT_SERVER_MAX_PASSES. The firmware
    # would answer the SESSION_PLAN with an ERROR frame after flash/TARGET_INFO/catalog; the host
    # must refuse first (the parser does from MAX_PASSES_PER_PLAN; the session from the
    # target-advertised max_passes), naming the passes and the limit, and never send a plan.
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_all_groups").manifest_path)
    passes = counter_passes_for_selection({"cpu": "all", "memory": "all", "mve": "all"})
    assert len(passes) == 18 and MAX_PASSES_PER_PLAN == 16

    session = HostSession(FakeTargetTransport(), counter_passes=passes)
    with pytest.raises(RuntimeError, match=r"18 PMU passes planned \(cpu_0, cpu_1, cpu_2, cpu_3, cpu_4, memory_0, .*mve_8\) but target 'fake_board' runs at most 16 per SESSION_PLAN \(TARGET_INFO max_passes\)"):
        session.run_many([bundle])
    assert "TX:SESSION_PLAN" not in session._trace and "TX:TARGET_INFO_ACK" not in session._trace

    # Exactly the limit is accepted (16 x 4 chained counters would need 64 slots, so
    # build it from single-counter passes) and the fake target runs every pass.
    single = counter_by_name("ARM_PMU_INST_RETIRED")
    sixteen = tuple(CounterPass("cpu", i, (single,)) for i in range(MAX_PASSES_PER_PLAN))
    check_pass_count(sixteen)
    result = HostSession(FakeTargetTransport(), counter_passes=sixteen).run(bundle)
    assert len({s.pass_name for s in result.samples}) == MAX_PASSES_PER_PLAN

    # The fake target mirrors parse_pmu_passes(): a 17-pass plan is rejected on decode.
    seventeen = sixteen + (CounterPass("cpu", 16, (single,)),)
    with pytest.raises(TooManyPassesError, match="17 PMU passes planned"):
        check_pass_count(seventeen)
    with pytest.raises(ValueError, match="17 PMU passes; fake target accepts at most 16"):
        FakeTargetTransport()._admit_session_plan(encode_session_plan(session_plan_for_bundles([bundle], seventeen)))


def test_case_id_at_the_firmware_limit_runs_and_one_byte_over_fails_before_session_plan(tmp_path: Path) -> None:
    assert MAX_CASE_ID_BYTES == 95
    passes = counter_passes_for_selection({"cpu": "default"})
    longest = "abs_" + "l" * (MAX_CASE_ID_BYTES - 4)
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id=longest).manifest_path)
    result = HostSession(FakeTargetTransport(), counter_passes=passes).run(bundle)
    assert result.cases[0].case_bundle.case_id == longest and result.cases[0].comparison.passed

    over = longest + "l"
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id=over).manifest_path)
    session = HostSession(FakeTargetTransport(), counter_passes=passes)
    with pytest.raises(RuntimeError, match=r"is 96 bytes; the firmware stores at most 95 \(HCT_SERVER_MAX_CASE_ID 96 including the NUL"):
        session.run(bundle)
    assert "TX:SESSION_PLAN" not in session._trace
    # The fake target rejects the same plan the way the firmware's cursor_text() does.
    with pytest.raises(ValueError, match="does not fit the fake target's 96-byte case-id storage"):
        FakeTargetTransport()._admit_session_plan(encode_session_plan(session_plan_for_bundles([bundle], passes)))


def test_unchained_pass_overflows_sixteen_bit_counter_and_invalidates_case(tmp_path: Path) -> None:
    # 40000 elements x 3 cycles x 4 iterations = 480000 events: well over a single
    # 16-bit slot, comfortably inside a chained 32-bit pair.
    bundle = load_case_bundle(
        build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_overflow", input_shape=(40000,)).manifest_path
    )
    chained = HostSession(
        FakeTargetTransport(runtime_arena_capacity=114688), counter_passes=counter_passes_for_selection({"cpu": "default"})
    ).run(bundle)
    assert chained.cases[0].statistics.overflow_detected is False
    assert chained.cases[0].statistics.valid_for_regression is True

    unchained = HostSession(
        FakeTargetTransport(runtime_arena_capacity=114688),
        counter_passes=counter_passes_for_selection({"cpu": "default"}, chained=False),
    ).run(bundle)
    case = unchained.cases[0]
    assert all(c.overflow for s in case.samples for c in s.counters if c.name == "ARM_PMU_INST_RETIRED")
    assert all(c.value <= 0xFFFF for s in case.samples for c in s.counters if c.name != "ARM_PMU_CPU_CYCLES")
    assert case.statistics.overflow_detected is True
    assert case.statistics.valid_for_regression is False
    # The DWT cycle statistics are unaffected by an event-counter overflow.
    assert case.statistics.median_cycles == chained.cases[0].statistics.median_cycles


def test_dwt_only_target_refuses_event_counter_passes_but_times_cycles(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_dwt").manifest_path)

    session = HostSession(FakeTargetTransport(pmu_present=False), counter_passes=counter_passes_for_selection({"cpu": "default"}))
    with pytest.raises(RuntimeError, match="has no Armv8.1-M PMU.*cpu_0"):
        session.run(bundle)
    assert "TX:TARGET_INFO_ACK" not in session._trace

    # A cycles-only selection plans one empty pass, which a DWT-only target can run.
    passes = counter_passes_for_selection({"cpu": ["ARM_PMU_CPU_CYCLES"]})
    assert passes == (CounterPass("cpu", 0, ()),)
    result = HostSession(FakeTargetTransport(pmu_present=False), counter_passes=passes).run(bundle)
    assert [c.name for c in result.samples[0].counters] == ["ARM_PMU_CPU_CYCLES"]
    assert result.cases[0].statistics.median_cycles > 0


def test_host_refuses_passes_needing_more_slots_than_advertised(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_slots").manifest_path)
    passes = counter_passes_for_selection({"memory": "default"})  # 4 chained counters = 8 slots
    with pytest.raises(RuntimeError, match="needs 8 event-counter slots.*advertises only 4"):
        HostSession(FakeTargetTransport(pmu_counter_slots=4), counter_passes=passes).run(bundle)
    # Unchained, the same four counters fit four slots.
    unchained = counter_passes_for_selection({"memory": "default"}, chained=False)
    HostSession(FakeTargetTransport(pmu_counter_slots=4), counter_passes=unchained).run(bundle)


def test_session_plan_over_target_rx_buffer_is_refused_before_sending(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_plan_size").manifest_path)
    passes = counter_passes_for_selection({"mve": "all"})
    size = session_plan_size([bundle.case_id], passes)
    session = HostSession(FakeTargetTransport(max_rx_payload=size - 1), counter_passes=passes)
    with pytest.raises(RuntimeError, match=f"encodes to {size} bytes.*only takes {size - 1}-byte"):
        session.run(bundle)
    assert "TX:SESSION_PLAN" not in session._trace
    HostSession(FakeTargetTransport(max_rx_payload=size), counter_passes=passes).run(bundle)


def test_unknown_event_ids_are_reported_with_placeholder_names(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_unknown").manifest_path)
    exotic = CounterPass("cpu", 0, (counter_by_name("ARM_PMU_INST_RETIRED"), type(counter_by_name("ARM_PMU_INST_RETIRED"))("vendor", 0x0C00, "cpu")))
    result = HostSession(FakeTargetTransport(), counter_passes=(exotic,)).run(bundle)
    assert [c.name for c in result.samples[0].counters] == ["ARM_PMU_CPU_CYCLES", "ARM_PMU_INST_RETIRED", "event_0x0c00"]
    # A PMU-present target counts whatever it was programmed with: the fake reports the
    # unknown id supported, exactly like firmware, rather than zeroing it.
    unknown = [c for c in result.samples[0].counters if c.name == "event_0x0c00"][0]
    assert unknown.supported is True
    # The bundle schema is seeded by event id too, so the caller's "vendor" label never
    # becomes a dead column next to the populated placeholder.
    assert counter_names_for_passes((exotic,)) == ["ARM_PMU_CPU_CYCLES", "ARM_PMU_INST_RETIRED", "event_0x0c00"]


def test_fake_target_rejects_session_plans_the_firmware_would_reject() -> None:
    # handle_session_plan() admits 1..HCT_SERVER_MAX_CASES cases; the fake mirrors it so a
    # host that bypasses the batch splitter cannot pass on the fake and fail on hardware.
    fake = FakeTargetTransport()
    for case_count in (0, MAX_CASES_PER_PLAN + 1):
        plan = SessionPlan(1, 1, 1, 1, 1, (), tuple(PlannedCase(f"case_{i}", 1) for i in range(case_count)))
        with pytest.raises(ValueError, match=rf"{case_count} cases; fake target accepts 1\.\.{MAX_CASES_PER_PLAN}"):
            fake._admit_session_plan(encode_session_plan(plan))


def test_case_too_large_fails(tmp_path: Path) -> None:
    conv_bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path).manifest_path)
    transport = FakeTargetTransport(runtime_arena_capacity=32)

    with pytest.raises(RuntimeError, match="requires .* workspace bytes.*advertises"):
        HostSession(transport).run(conv_bundle)


def test_case_one_byte_over_advertised_workspace_fails_before_plan(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path).manifest_path)
    transport = FakeTargetTransport(runtime_arena_capacity=bundle.workspace_bytes_required - 1)
    session = HostSession(transport)

    with pytest.raises(RuntimeError, match=rf"requires {bundle.workspace_bytes_required} workspace bytes"):
        session.run(bundle)

    assert "TX:SESSION_PLAN" not in session._trace


def test_large_correctness_output_exceeding_old_outbox_streams_in_order(tmp_path: Path) -> None:
    bundle = load_case_bundle(
        build_abs_s8_case_bundle(
            PROJECT_ROOT,
            output_root=tmp_path,
            case_id="abs_large_output",
            input_shape=(40000,),
        ).manifest_path
    )
    assert bundle.expected_output.byte_length > 32768
    transport = FakeTargetTransport(runtime_arena_capacity=114688, max_frame_payload=256)

    result = HostSession(transport).run(bundle)

    assert result.cases[0].comparison.passed
    assert result.cases[0].output_bytes == blob_numpy(bundle.expected_output).tobytes(order="C")
    assert result.protocol_trace.count("RX:OUTPUT_CHUNK") > 100


def test_firmware_output_stream_is_pumped_after_outbox_drains() -> None:
    source = (PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_session.c").read_text()
    queue_body = source.split("static hctp_status_t queue_correctness_output", 1)[1].split(
        "static hctp_status_t pump_correctness_output", 1
    )[0]
    assert "while (cursor < session->output_length)" not in queue_body
    assert "output_stream_active = 1u" in queue_body
    assert "outbox_length == 0u && session->output_stream_active != 0u" in source


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


def test_exact_status_mode_ignores_output_arrays_and_uses_kernel_status(tmp_path: Path) -> None:
    bundle_summary = build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path)
    manifest_path = bundle_summary.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["correctness_comparison"] = {
        "mode": "exact_status",
        "expected_status": 0,
        "expected_status_name": "ARM_CMSIS_NN_SUCCESS",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    expected_output_path = manifest_path.parent / "blobs" / "expected_output.bin"
    expected_output_path.write_bytes(b"\x7f")

    bundle = load_case_bundle(manifest_path)
    result = HostSession(FakeTargetTransport()).run(bundle)

    assert result.comparison.passed is True
    assert result.comparison.mode == "exact_status"
    assert len(result.output_bytes) > 1
    assert "RX:CORRECTNESS_RESULT" in result.protocol_trace


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


def test_host_refuses_more_than_four_counters_per_pass_even_unchained(tmp_path: Path) -> None:
    # Firmware bounds counter_count at HCT_SERVER_MAX_COUNTERS_PER_PASS (4) whether or not
    # the pass chains slot pairs; five unchained counters fit eight slots but must still be
    # refused at the handshake instead of sent as a SESSION_PLAN the firmware rejects.
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_five").manifest_path)
    names = ["ARM_PMU_INST_RETIRED", "ARM_PMU_STALL_FRONTEND", "ARM_PMU_STALL_BACKEND", "ARM_PMU_MEM_ACCESS", "ARM_PMU_BUS_ACCESS"]
    five = CounterPass("cpu", 0, tuple(counter_by_name(n) for n in names), chained=False)
    assert five.slots_required == 5
    session = HostSession(FakeTargetTransport(pmu_counter_slots=8), counter_passes=(five,))
    with pytest.raises(RuntimeError, match=r"names 5 counters; the firmware runs at most 4 per pass"):
        session.run(bundle)
    assert "TX:SESSION_PLAN" not in session._trace
