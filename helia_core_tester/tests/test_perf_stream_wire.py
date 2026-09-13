"""Round-trip tests for the HCTP payload codec (helia_core_tester/perf_stream/wire.py).

One test per payload: encode -> decode reproduces the value, and the byte layout
matches the format the firmware writes/reads (see the docstrings in wire.py and the
C harnesses that prove the same bytes against the real firmware).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from helia_core_tester.perf_stream.hctp import ByteReader
from helia_core_tester.perf_stream.measurement import CounterPass, RawCounterValue, RawSample, counter_passes_for_selection
from helia_core_tester.perf_stream.pmu_catalog import CPU_CYCLES_EVENT_ID, counter_by_name
from helia_core_tester.perf_stream import wire


def _target_info(**overrides) -> wire.TargetInfo:
    fields = dict(
        build_id="hct-benchmark-server-v0",
        catalog_hash=bytes(range(32)),
        max_frame_payload=256,
        runtime_arena_capacity=114688,
        transfer_mode=1,
        output_mode=1,
        board_id="apollo510_evb",
        target_cpu="cortex-m55",
        transport_kind=1,
        capability_flags=wire.CAP_CASE_STREAMING | wire.CAP_PMU_ARMV8M,
        pmu_counter_slots=8,
        max_rx_payload=2016,
        max_cases_per_session=32,
        max_passes=16,
    )
    fields.update(overrides)
    return wire.TargetInfo(**fields)


def test_target_info_round_trip_and_layout() -> None:
    info = _target_info()
    payload = wire.encode_target_info(info)
    assert wire.decode_target_info(payload) == info
    assert info.has_pmu
    assert not wire.decode_target_info(wire.encode_target_info(_target_info(capability_flags=0, pmu_counter_slots=0))).has_pmu
    # The v3 limits trail max_rx_payload: u16 max_cases_per_session, u8 max_passes.
    reader = ByteReader(payload)
    reader.text(); reader.fixed(32); reader.u32(); reader.u32(); reader.u8(); reader.u8()
    reader.text(); reader.text(); reader.u8(); reader.u32(); reader.u8()
    assert reader.u32() == 2016
    assert reader.u16() == 32
    assert reader.u8() == 16
    assert reader.remaining() == b""
    with pytest.raises(ValueError, match="catalog_hash"):
        wire.encode_target_info(_target_info(catalog_hash=b"short"))


def test_kernel_catalog_round_trip_and_hash() -> None:
    entries = (
        wire.CatalogEntry(1, "arm_abs_s8", "BasicMathFunctions", 1, "S8", 1, True, True, False, 0),
        wire.CatalogEntry(2, "arm_convolve_s8", "ConvolutionFunctions", 1, "S8", 1, True, True, False, 64),
    )
    assert wire.decode_kernel_catalog(wire.encode_kernel_catalog(entries)) == entries
    assert wire.decode_kernel_catalog(wire.encode_kernel_catalog(())) == ()
    # The hash is the SHA-256 of the canonical compact JSON, regardless of entry order.
    rows = [
        {
            "kernel_id": e.kernel_id, "canonical_name": e.canonical_name, "operator_family": e.operator_family,
            "api_version": e.api_version, "supported_dtype": e.supported_dtype,
            "adapter_schema_version": e.adapter_schema_version, "stateless": e.stateless,
            "repeated_invocation_safe": e.repeated_invocation_safe, "mutates_input": e.mutates_input,
            "scratch_bytes": e.scratch_bytes,
        }
        for e in entries
    ]
    expected = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).digest()
    assert wire.kernel_catalog_hash(entries) == expected
    assert wire.kernel_catalog_hash(entries[::-1]) == expected


def test_session_plan_round_trip_and_size() -> None:
    passes = counter_passes_for_selection({"cpu": "default", "mve": "all"})
    plan = wire.SessionPlan(
        warmups=2, samples=5, iterations_per_sample=4, min_cycles=1024, max_iterations=256,
        passes=passes,
        cases=(wire.PlannedCase("abs_default_s8_hw_generated", 1), wire.PlannedCase("x" * 96, 173)),
    )
    payload = wire.encode_session_plan(plan)
    assert wire.decode_session_plan(payload) == plan
    assert len(payload) == wire.session_plan_size([c.case_id for c in plan.cases], passes)
    # Passes decode back to CounterPass values with catalog descriptors; unknown event
    # ids get the same placeholder name the sample decoder uses.
    exotic = wire.SessionPlan(1, 1, 1, 1, 1, (CounterPass("cpu", 0, (wire.CounterDescriptor("vendor", 0x0C00, "cpu"),), False),), (wire.PlannedCase("c", 1),))
    decoded = wire.decode_session_plan(wire.encode_session_plan(exotic))
    assert decoded.passes[0].counters[0].name == "event_0x0c00"
    assert decoded.passes[0].chained is False and decoded.passes[0].name == "cpu_0"
    empty = wire.SessionPlan(1, 1, 1, 1, 1, (), (wire.PlannedCase("c", 1),))
    assert wire.decode_session_plan(wire.encode_session_plan(empty)) == empty


def test_request_case_round_trip() -> None:
    assert wire.decode_request_case(wire.encode_request_case(wire.RequestCase(7))) == wire.RequestCase(7)


def test_case_meta_round_trip_and_layout() -> None:
    meta = wire.CaseMeta(
        case_id="convolve_case_03_s8_hw_generated",
        kernel_id=2,
        schema_version=1,
        comparison_mode=wire.COMPARISON_MODE_CODES["tolerant_int"],
        tolerance=-1,
        atol_q16=65536,
        rtol_q16=3277,
        scalar_parameters=(("stride_h", 2), ("padding", 1), ("output_capacity_bytes", 64), ("input_offset", -128)),
        blobs=(
            wire.BlobDescriptor(1, "input_0", "S8", (1, 4, 4, 3), 48, 16, 0xDEADBEEF, False),
            wire.BlobDescriptor(2, "weights", "S8", (2, 2, 3, 4), 48, 4, 0x12345678, True),
            wire.BlobDescriptor(3, "bias", "S32", (4,), 16, 4, 1, False),
        ),
        scratch_bytes=512,
    )
    payload = wire.encode_case_meta(meta)
    assert wire.decode_case_meta(payload) == meta
    assert meta.blobs[0].rank == 4
    # Every blob carries exactly six u32 dims on the wire (zero-padded past its rank).
    reader = ByteReader(payload)
    reader.text(); reader.u32(); reader.u16(); reader.u8(); reader.i32(); reader.u32(); reader.u32()
    for _ in range(reader.u8()):
        reader.text(); reader.i32()
    assert reader.u16() == 3
    assert reader.u32() == 1 and reader.text() == "input_0" and reader.text() == "S8" and reader.u8() == 4
    assert [reader.u32() for _ in range(6)] == [1, 4, 4, 3, 0, 0]
    with pytest.raises(ValueError, match="rank"):
        wire.encode_case_meta(wire.CaseMeta("c", 1, 1, 1, 0, 0, 0, (), (wire.BlobDescriptor(1, "r", "S8", (1,) * 7, 1, 1, 0),), 0))


def test_blob_request_chunk_and_case_ready_round_trip() -> None:
    request = wire.RequestBlob(blob_id=3, offset=128, max_length=64)
    assert wire.decode_request_blob(wire.encode_request_blob(request)) == request
    chunk = wire.BlobChunk(blob_id=3, offset=128, data=bytes(range(64)))
    assert wire.decode_blob_chunk(wire.encode_blob_chunk(chunk)) == chunk
    assert wire.decode_blob_chunk(wire.encode_blob_chunk(wire.BlobChunk(1, 0, b""))) == wire.BlobChunk(1, 0, b"")
    ready = wire.CaseReady(blob_id=3, bytes_received=192)
    assert wire.decode_case_ready(wire.encode_case_ready(ready)) == ready


def test_correctness_result_and_ack_round_trip() -> None:
    result = wire.CorrectnessResult(status=-1)
    assert wire.decode_correctness_result(wire.encode_correctness_result(result)) == result
    for passed in (True, False):
        ack = wire.CorrectnessAck(passed=passed)
        assert wire.decode_correctness_ack(wire.encode_correctness_ack(ack)) == ack
    assert wire.encode_correctness_ack(wire.CorrectnessAck(True)) == b"\x01"


def test_output_stream_round_trip() -> None:
    begin = wire.OutputBegin(length=300)
    assert wire.decode_output_begin(wire.encode_output_begin(begin)) == begin
    assert wire.encode_output_begin(begin)[:4] == b"\x00\x00\x00\x00"
    chunk = wire.OutputChunk(offset=224, data=b"\x01\x02\x03")
    assert wire.decode_output_chunk(wire.encode_output_chunk(chunk)) == chunk
    with pytest.raises(ValueError, match="trailing"):
        wire.decode_output_chunk(wire.encode_output_chunk(chunk) + b"\x00")
    data = bytes(range(256)) * 2
    end = wire.OutputEnd(length=len(data), checksum=wire.output_checksum(data))
    assert wire.decode_output_end(wire.encode_output_end(end)) == end
    assert wire.output_checksum(data) == sum(data)


def test_sample_result_round_trip_resolves_names_from_catalog() -> None:
    inst = counter_by_name("ARM_PMU_INST_RETIRED")
    sample = RawSample(
        sample_index=2,
        iterations=4,
        cycles=0x1_0000_0000 + 5,
        counters=(
            RawCounterValue("ARM_PMU_CPU_CYCLES", CPU_CYCLES_EVENT_ID, 123456, False, True),
            RawCounterValue(inst.name, inst.event_id, 0xFFFF_FFFF, True, True),
            RawCounterValue("event_0x0c00", 0x0C00, 0, False, False),
        ),
        pass_name="cpu_0",
    )
    payload = wire.encode_sample_result(sample)
    assert wire.decode_sample_result(payload) == sample
    # Names go out empty, exactly like the firmware sends them.
    reader = ByteReader(payload)
    reader.u16(); reader.u32(); reader.u64(); reader.text()
    assert reader.u8() == 3
    assert reader.text() == "" and reader.u16() == CPU_CYCLES_EVENT_ID


def test_case_and_session_complete_round_trip() -> None:
    complete = wire.CaseComplete(case_id="abs_default_s8_hw_generated", workspace_used_bytes=4096)
    assert wire.decode_case_complete(wire.encode_case_complete(complete)) == complete
    partial = wire.CaseComplete("c", 0, correctness_ran=True, performance_ran=False)
    assert wire.decode_case_complete(wire.encode_case_complete(partial)) == partial
    session = wire.SessionComplete(case_count=32)
    assert wire.decode_session_complete(wire.encode_session_complete(session)) == session


def test_error_round_trip() -> None:
    error = wire.ErrorPayload("message_type=4 status=-1")
    assert wire.decode_error(wire.encode_error(error)) == error
