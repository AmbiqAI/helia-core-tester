"""HCTP payload codec: one encode/decode pair per message payload.

Every payload the host and the target exchange is described here exactly once, on
top of the `ByteWriter`/`ByteReader` primitives in hctp.py. The host session
(session.py) encodes what it sends and decodes what it receives through these; the
fake target (fake_target.py) uses the same pairs in the other direction, so the two
never drift apart. The C firmware is the other real peer -- the host harness tests
(test_perf_stream_c_wire_compat.py, test_perf_stream_firmware_messages.py,
test_perf_stream_firmware_session.py) prove its bytes agree with this module.

Integers are little-endian; `text` is `u16 length + UTF-8 bytes`; `raw` is
`u32 length + bytes`. Payload-free messages (TARGET_INFO_ACK, RUN_CORRECTNESS,
RUN_PERFORMANCE) have no codec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

from .hctp import ByteReader, ByteWriter
from .measurement import CounterPass, RawCounterValue, RawSample
from .pmu_catalog import CounterDescriptor, counter_by_event_id, counter_name_for_event_id

# TARGET_INFO capability_flags bits -- must match the HCT_CAP_* enum in
# cmake/perf_stream/benchmark_server_catalog.h.
CAP_CASE_STREAMING = 1 << 0
CAP_CORRECTNESS = 1 << 1
CAP_PERFORMANCE = 1 << 2
CAP_RTT_TRANSPORT = 1 << 3
CAP_KERNEL_CATALOG = 1 << 4
CAP_ABS_S8 = 1 << 5
# Set only when the firmware was built for a core with the Armv8.1-M PMU
# (__PMU_PRESENT == 1); absent on DWT-only targets such as Cortex-M4.
CAP_PMU_ARMV8M = 1 << 6

CATALOG_HASH_SIZE = 32
BLOB_MAX_RANK = 6

# CASE_META comparison_mode codes -- must match HCT_COMPARISON_MODE_* in the firmware.
COMPARISON_MODE_CODES = {"exact_int": 1, "tolerant_int": 2, "float": 3, "bool": 4, "exact_status": 5}
COMPARISON_MODE_NAMES = {code: name for name, code in COMPARISON_MODE_CODES.items()}


# --- TARGET_INFO ---------------------------------------------------------------------


@dataclass(frozen=True)
class TargetInfo:
    """What the target announces about itself before any plan is loaded."""

    build_id: str
    catalog_hash: bytes
    max_frame_payload: int
    runtime_arena_capacity: int
    transfer_mode: int
    output_mode: int
    board_id: str
    target_cpu: str
    transport_kind: int
    capability_flags: int
    # Number of 16-bit PMU event-counter slots (8 on Cortex-M55, 0 without a PMU)
    # and the largest frame payload the target's receive buffer can hold.
    pmu_counter_slots: int
    max_rx_payload: int
    # The firmware's fixed session limits (HCT_SERVER_MAX_CASES / HCT_SERVER_MAX_PASSES);
    # the host batches cases and refuses over-long pass lists from these.
    max_cases_per_session: int
    max_passes: int

    @property
    def has_pmu(self) -> bool:
        return bool(self.capability_flags & CAP_PMU_ARMV8M)


def encode_target_info(info: TargetInfo) -> bytes:
    if len(info.catalog_hash) != CATALOG_HASH_SIZE:
        raise ValueError(f"catalog_hash must be {CATALOG_HASH_SIZE} bytes, got {len(info.catalog_hash)}.")
    writer = ByteWriter()
    writer.text(info.build_id)
    writer.fixed(info.catalog_hash)
    writer.u32(info.max_frame_payload)
    writer.u32(info.runtime_arena_capacity)
    writer.u8(info.transfer_mode)
    writer.u8(info.output_mode)
    writer.text(info.board_id)
    writer.text(info.target_cpu)
    writer.u8(info.transport_kind)
    writer.u32(info.capability_flags)
    writer.u8(info.pmu_counter_slots)
    writer.u32(info.max_rx_payload)
    writer.u16(info.max_cases_per_session)
    writer.u8(info.max_passes)
    return writer.finish()


def decode_target_info(payload: bytes) -> TargetInfo:
    reader = ByteReader(payload)
    return TargetInfo(
        build_id=reader.text(),
        catalog_hash=reader.fixed(CATALOG_HASH_SIZE),
        max_frame_payload=reader.u32(),
        runtime_arena_capacity=reader.u32(),
        transfer_mode=reader.u8(),
        output_mode=reader.u8(),
        board_id=reader.text(),
        target_cpu=reader.text(),
        transport_kind=reader.u8(),
        capability_flags=reader.u32(),
        pmu_counter_slots=reader.u8(),
        max_rx_payload=reader.u32(),
        max_cases_per_session=reader.u16(),
        max_passes=reader.u8(),
    )


# --- KERNEL_CATALOG ------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    kernel_id: int
    canonical_name: str
    operator_family: str
    api_version: int
    supported_dtype: str
    adapter_schema_version: int
    stateless: bool
    repeated_invocation_safe: bool
    mutates_input: bool
    scratch_bytes: int


def encode_kernel_catalog(entries: Sequence[CatalogEntry]) -> bytes:
    """One KERNEL_CATALOG page: `u16 count` then the entries. A large catalog is split
    over several pages by the sender (HCTP_FLAG_MORE on every non-final one)."""
    writer = ByteWriter()
    writer.u16(len(entries))
    for entry in entries:
        writer.u32(entry.kernel_id)
        writer.text(entry.canonical_name)
        writer.text(entry.operator_family)
        writer.u16(entry.api_version)
        writer.text(entry.supported_dtype)
        writer.u16(entry.adapter_schema_version)
        writer.u8(1 if entry.stateless else 0)
        writer.u8(1 if entry.repeated_invocation_safe else 0)
        writer.u8(1 if entry.mutates_input else 0)
        writer.u32(entry.scratch_bytes)
    return writer.finish()


def decode_kernel_catalog(payload: bytes) -> tuple[CatalogEntry, ...]:
    reader = ByteReader(payload)
    count = reader.u16()
    return tuple(
        CatalogEntry(
            kernel_id=reader.u32(),
            canonical_name=reader.text(),
            operator_family=reader.text(),
            api_version=reader.u16(),
            supported_dtype=reader.text(),
            adapter_schema_version=reader.u16(),
            stateless=bool(reader.u8()),
            repeated_invocation_safe=bool(reader.u8()),
            mutates_input=bool(reader.u8()),
            scratch_bytes=reader.u32(),
        )
        for _ in range(count)
    )


def kernel_catalog_hash(entries: Sequence[CatalogEntry]) -> bytes:
    """SHA-256 of the catalog's canonical compact JSON (entries sorted by kernel_id,
    keys sorted) -- the value TARGET_INFO carries in `catalog_hash`. Must match
    scripts/generate_kernel_catalog.py, which bakes the same hash into the firmware."""
    rows = [
        {
            "kernel_id": entry.kernel_id,
            "canonical_name": entry.canonical_name,
            "operator_family": entry.operator_family,
            "api_version": entry.api_version,
            "supported_dtype": entry.supported_dtype,
            "adapter_schema_version": entry.adapter_schema_version,
            "stateless": entry.stateless,
            "repeated_invocation_safe": entry.repeated_invocation_safe,
            "mutates_input": entry.mutates_input,
            "scratch_bytes": entry.scratch_bytes,
        }
        for entry in sorted(entries, key=lambda entry: entry.kernel_id)
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).digest()


# --- SESSION_PLAN --------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedCase:
    case_id: str
    kernel_id: int


@dataclass(frozen=True)
class SessionPlan:
    """One session's worth of cases and the timing/PMU plan they all share."""

    warmups: int
    samples: int
    iterations_per_sample: int
    min_cycles: int
    max_iterations: int
    passes: tuple[CounterPass, ...]
    cases: tuple[PlannedCase, ...]
    transfer_mode: int = 1


def encode_session_plan(plan: SessionPlan) -> bytes:
    """`u16 case_count, u8 transfer_mode, u16 warmups, u16 samples,
    u32 iterations_per_sample, u32 min_cycles, u32 max_iterations, u8 pass_count`, per
    pass `text pass_name, u8 chained, u8 counter_count, u16 event_id[counter_count]`,
    then per case `text case_id, u32 kernel_id`."""
    writer = ByteWriter()
    writer.u16(len(plan.cases))
    writer.u8(plan.transfer_mode)
    writer.u16(plan.warmups)
    writer.u16(plan.samples)
    writer.u32(plan.iterations_per_sample)
    writer.u32(plan.min_cycles)
    writer.u32(plan.max_iterations)
    writer.u8(len(plan.passes))
    for counter_pass in plan.passes:
        writer.text(counter_pass.name)
        writer.u8(1 if counter_pass.chained else 0)
        writer.u8(len(counter_pass.counters))
        for counter in counter_pass.counters:
            writer.u16(counter.event_id)
    for case in plan.cases:
        writer.text(case.case_id)
        writer.u32(case.kernel_id)
    return writer.finish()


def decode_session_plan(payload: bytes) -> SessionPlan:
    reader = ByteReader(payload)
    case_count = reader.u16()
    transfer_mode = reader.u8()
    warmups = reader.u16()
    samples = reader.u16()
    iterations_per_sample = reader.u32()
    min_cycles = reader.u32()
    max_iterations = reader.u32()
    passes = []
    for _ in range(reader.u8()):
        pass_name = reader.text()
        chained = bool(reader.u8())
        counters = tuple(_counter_for_event_id(reader.u16()) for _ in range(reader.u8()))
        group, _, index = pass_name.rpartition("_")
        passes.append(CounterPass(group=group or pass_name, pass_index=int(index or 0), counters=counters, chained=chained))
    cases = tuple(PlannedCase(case_id=reader.text(), kernel_id=reader.u32()) for _ in range(case_count))
    return SessionPlan(
        warmups=warmups,
        samples=samples,
        iterations_per_sample=iterations_per_sample,
        min_cycles=min_cycles,
        max_iterations=max_iterations,
        passes=tuple(passes),
        cases=cases,
        transfer_mode=transfer_mode,
    )


def session_plan_size(case_ids: Sequence[str], counter_passes: Sequence[CounterPass]) -> int:
    """Encoded SESSION_PLAN size for these case ids and passes, without needing bundles
    -- the batch splitter uses it to keep every plan within the target's receive buffer."""
    size = 2 + 1 + 2 + 2 + 4 + 4 + 4 + 1
    for counter_pass in counter_passes:
        size += 2 + len(counter_pass.name.encode("utf-8")) + 1 + 1 + 2 * len(counter_pass.counters)
    for case_id in case_ids:
        size += 2 + len(case_id.encode("utf-8")) + 4
    return size


def _counter_for_event_id(event_id: int) -> CounterDescriptor:
    """The catalog descriptor for an event id, or a placeholder for ids the catalog
    does not know (the firmware programs and reports whatever it was asked for)."""
    return counter_by_event_id(event_id) or CounterDescriptor(counter_name_for_event_id(event_id), event_id, "unknown")


# --- REQUEST_CASE / CASE_META ----------------------------------------------------------


@dataclass(frozen=True)
class RequestCase:
    case_index: int


def encode_request_case(request: RequestCase) -> bytes:
    writer = ByteWriter()
    writer.u16(request.case_index)
    return writer.finish()


def decode_request_case(payload: bytes) -> RequestCase:
    return RequestCase(case_index=ByteReader(payload).u16())


@dataclass(frozen=True)
class BlobDescriptor:
    blob_id: int
    role: str
    dtype: str
    dimensions: tuple[int, ...]
    byte_length: int
    alignment: int
    crc32: int
    mutable_data: bool = False

    @property
    def rank(self) -> int:
        return len(self.dimensions)


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    kernel_id: int
    schema_version: int
    comparison_mode: int
    tolerance: int
    atol_q16: int
    rtol_q16: int
    # Ordered (name, int32) pairs; every scalar crosses the wire as i32.
    scalar_parameters: tuple[tuple[str, int], ...]
    blobs: tuple[BlobDescriptor, ...]
    scratch_bytes: int


def encode_case_meta(meta: CaseMeta) -> bytes:
    """`text case_id, u32 kernel_id, u16 schema_version, u8 comparison_mode,
    i32 tolerance, u32 atol_q16, u32 rtol_q16, u8 scalar_count`, per scalar
    `text name, i32 value`, `u16 blob_count`, per blob `u32 blob_id, text role,
    text dtype, u8 rank, u32 dims[6], u32 byte_length, u32 alignment, u32 crc32,
    u8 mutable_data`, then `u32 scratch_bytes`."""
    writer = ByteWriter()
    writer.text(meta.case_id)
    writer.u32(meta.kernel_id)
    writer.u16(meta.schema_version)
    writer.u8(meta.comparison_mode)
    writer.i32(meta.tolerance)
    writer.u32(meta.atol_q16)
    writer.u32(meta.rtol_q16)
    writer.u8(len(meta.scalar_parameters))
    for name, value in meta.scalar_parameters:
        writer.text(name)
        writer.i32(value)
    writer.u16(len(meta.blobs))
    for blob in meta.blobs:
        if blob.rank > BLOB_MAX_RANK:
            raise ValueError(f"Blob {blob.blob_id} has rank {blob.rank}; the wire carries at most {BLOB_MAX_RANK} dims.")
        writer.u32(blob.blob_id)
        writer.text(blob.role)
        writer.text(blob.dtype)
        writer.u8(blob.rank)
        for dim in list(blob.dimensions) + [0] * (BLOB_MAX_RANK - blob.rank):
            writer.u32(dim)
        writer.u32(blob.byte_length)
        writer.u32(blob.alignment)
        writer.u32(blob.crc32)
        writer.u8(1 if blob.mutable_data else 0)
    writer.u32(meta.scratch_bytes)
    return writer.finish()


def decode_case_meta(payload: bytes) -> CaseMeta:
    reader = ByteReader(payload)
    case_id = reader.text()
    kernel_id = reader.u32()
    schema_version = reader.u16()
    comparison_mode = reader.u8()
    tolerance = reader.i32()
    atol_q16 = reader.u32()
    rtol_q16 = reader.u32()
    scalar_parameters = tuple((reader.text(), reader.i32()) for _ in range(reader.u8()))
    blobs = []
    for _ in range(reader.u16()):
        blob_id = reader.u32()
        role = reader.text()
        dtype = reader.text()
        rank = reader.u8()
        dims = tuple(reader.u32() for _ in range(BLOB_MAX_RANK))[:rank]
        blobs.append(
            BlobDescriptor(
                blob_id=blob_id,
                role=role,
                dtype=dtype,
                dimensions=dims,
                byte_length=reader.u32(),
                alignment=reader.u32(),
                crc32=reader.u32(),
                mutable_data=bool(reader.u8()),
            )
        )
    return CaseMeta(
        case_id=case_id,
        kernel_id=kernel_id,
        schema_version=schema_version,
        comparison_mode=comparison_mode,
        tolerance=tolerance,
        atol_q16=atol_q16,
        rtol_q16=rtol_q16,
        scalar_parameters=scalar_parameters,
        blobs=tuple(blobs),
        scratch_bytes=reader.u32(),
    )


# --- REQUEST_BLOB / BLOB_CHUNK / CASE_READY ------------------------------------------


@dataclass(frozen=True)
class RequestBlob:
    blob_id: int
    offset: int
    max_length: int


def encode_request_blob(request: RequestBlob) -> bytes:
    writer = ByteWriter()
    writer.u32(request.blob_id)
    writer.u32(request.offset)
    writer.u16(request.max_length)
    return writer.finish()


def decode_request_blob(payload: bytes) -> RequestBlob:
    reader = ByteReader(payload)
    return RequestBlob(blob_id=reader.u32(), offset=reader.u32(), max_length=reader.u16())


@dataclass(frozen=True)
class BlobChunk:
    blob_id: int
    offset: int
    data: bytes


def encode_blob_chunk(chunk: BlobChunk) -> bytes:
    writer = ByteWriter()
    writer.u32(chunk.blob_id)
    writer.u32(chunk.offset)
    writer.raw(chunk.data)
    return writer.finish()


def decode_blob_chunk(payload: bytes) -> BlobChunk:
    reader = ByteReader(payload)
    return BlobChunk(blob_id=reader.u32(), offset=reader.u32(), data=reader.raw())


@dataclass(frozen=True)
class CaseReady:
    blob_id: int
    bytes_received: int


def encode_case_ready(ready: CaseReady) -> bytes:
    writer = ByteWriter()
    writer.u32(ready.blob_id)
    writer.u32(ready.bytes_received)
    return writer.finish()


def decode_case_ready(payload: bytes) -> CaseReady:
    reader = ByteReader(payload)
    return CaseReady(blob_id=reader.u32(), bytes_received=reader.u32())


# --- correctness: CORRECTNESS_RESULT / OUTPUT_* / CORRECTNESS_ACK ----------------------


@dataclass(frozen=True)
class CorrectnessResult:
    status: int  # arm_cmsis_nn_status of the correctness invocation


def encode_correctness_result(result: CorrectnessResult) -> bytes:
    writer = ByteWriter()
    writer.i32(result.status)
    return writer.finish()


def decode_correctness_result(payload: bytes) -> CorrectnessResult:
    return CorrectnessResult(status=ByteReader(payload).i32())


@dataclass(frozen=True)
class OutputBegin:
    length: int
    offset: int = 0  # always 0 today; the output is streamed from its start


def encode_output_begin(begin: OutputBegin) -> bytes:
    writer = ByteWriter()
    writer.u32(begin.offset)
    writer.u32(begin.length)
    return writer.finish()


def decode_output_begin(payload: bytes) -> OutputBegin:
    reader = ByteReader(payload)
    offset = reader.u32()
    return OutputBegin(length=reader.u32(), offset=offset)


@dataclass(frozen=True)
class OutputChunk:
    offset: int
    data: bytes


def encode_output_chunk(chunk: OutputChunk) -> bytes:
    writer = ByteWriter()
    writer.u32(chunk.offset)
    writer.u32(len(chunk.data))
    writer.fixed(chunk.data)
    return writer.finish()


def decode_output_chunk(payload: bytes) -> OutputChunk:
    reader = ByteReader(payload)
    offset = reader.u32()
    data = reader.fixed(reader.u32())
    if reader.remaining():
        raise ValueError(f"OUTPUT_CHUNK at offset {offset} carries {len(reader.remaining())} trailing byte(s).")
    return OutputChunk(offset=offset, data=data)


@dataclass(frozen=True)
class OutputEnd:
    length: int
    checksum: int  # byte sum of the whole output, mod 2**32


def output_checksum(data: bytes) -> int:
    return sum(data) & 0xFFFFFFFF


def encode_output_end(end: OutputEnd) -> bytes:
    writer = ByteWriter()
    writer.u32(end.length)
    writer.u32(end.checksum)
    return writer.finish()


def decode_output_end(payload: bytes) -> OutputEnd:
    reader = ByteReader(payload)
    return OutputEnd(length=reader.u32(), checksum=reader.u32())


@dataclass(frozen=True)
class CorrectnessAck:
    passed: bool  # informational: the target runs performance either way


def encode_correctness_ack(ack: CorrectnessAck) -> bytes:
    writer = ByteWriter()
    writer.u8(1 if ack.passed else 0)
    return writer.finish()


def decode_correctness_ack(payload: bytes) -> CorrectnessAck:
    return CorrectnessAck(passed=bool(ByteReader(payload).u8()))


# --- SAMPLE_RESULT / CASE_COMPLETE / SESSION_COMPLETE / ERROR --------------------------


def encode_sample_result(sample: RawSample) -> bytes:
    """`u16 sample_index, u32 iterations, u64 cycles (DWT), text pass_name,
    u8 counter_count`, per counter `text name, u16 event_id, u64 value, u8 overflow,
    u8 supported`. Names go out empty, as the firmware sends them; the decoder
    resolves them from the PMU catalog by event id."""
    writer = ByteWriter()
    writer.u16(sample.sample_index)
    writer.u32(sample.iterations)
    writer.u64(sample.cycles)
    writer.text(sample.pass_name)
    writer.u8(len(sample.counters))
    for counter in sample.counters:
        writer.text("")
        writer.u16(counter.event_id)
        writer.u64(counter.value)
        writer.u8(1 if counter.overflow else 0)
        writer.u8(1 if counter.supported else 0)
    return writer.finish()


def decode_sample_result(payload: bytes) -> RawSample:
    reader = ByteReader(payload)
    sample_index = reader.u16()
    iterations = reader.u32()
    cycles = reader.u64()
    pass_name = reader.text()
    counters = []
    for _ in range(reader.u8()):
        name = reader.text()
        event_id = reader.u16()
        counters.append(
            RawCounterValue(
                name=name or counter_name_for_event_id(event_id),
                event_id=event_id,
                value=reader.u64(),
                overflow=bool(reader.u8()),
                supported=bool(reader.u8()),
            )
        )
    return RawSample(sample_index=sample_index, iterations=iterations, cycles=cycles, counters=tuple(counters), pass_name=pass_name)


@dataclass(frozen=True)
class CaseComplete:
    case_id: str
    workspace_used_bytes: int
    correctness_ran: bool = True
    performance_ran: bool = True


def encode_case_complete(complete: CaseComplete) -> bytes:
    writer = ByteWriter()
    writer.text(complete.case_id)
    writer.u8(1 if complete.correctness_ran else 0)
    writer.u8(1 if complete.performance_ran else 0)
    writer.u32(complete.workspace_used_bytes)
    return writer.finish()


def decode_case_complete(payload: bytes) -> CaseComplete:
    reader = ByteReader(payload)
    case_id = reader.text()
    correctness_ran = bool(reader.u8())
    performance_ran = bool(reader.u8())
    return CaseComplete(
        case_id=case_id,
        workspace_used_bytes=reader.u32(),
        correctness_ran=correctness_ran,
        performance_ran=performance_ran,
    )


@dataclass(frozen=True)
class SessionComplete:
    case_count: int


def encode_session_complete(complete: SessionComplete) -> bytes:
    writer = ByteWriter()
    writer.u16(complete.case_count)
    return writer.finish()


def decode_session_complete(payload: bytes) -> SessionComplete:
    return SessionComplete(case_count=ByteReader(payload).u16())


@dataclass(frozen=True)
class ErrorPayload:
    message: str


def encode_error(error: ErrorPayload) -> bytes:
    writer = ByteWriter()
    writer.text(error.message)
    return writer.finish()


def decode_error(payload: bytes) -> ErrorPayload:
    return ErrorPayload(message=ByteReader(payload).text())
