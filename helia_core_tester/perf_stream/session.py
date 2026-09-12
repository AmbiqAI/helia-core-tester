"""Host-side streaming session runner for the fake vertical slices."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .case_bundle import CaseBundle, BlobInfo, blob_numpy, build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from .comparison import ComparisonResult, compare_output, compare_status
from .fake_target import FakeTargetTransport
from .firmware_messages import CatalogEntry, TargetInfo, decode_kernel_catalog, decode_target_info
from .hctp import HCTP_FLAG_MORE, ByteReader, ByteWriter, Frame, FrameDecoder, MessageType, SessionFrameValidator, encode_frame
from .measurement import (
    CounterPass,
    NormalizedSample,
    RawCounterValue,
    RawSample,
    SampleStatistics,
    compute_sample_statistics,
    counter_passes_for_selection,
    normalize_samples,
)
from .pmu_catalog import counter_name_for_event_id
from .transport import Transport


_COMPARISON_MODE_TO_CODE = {"exact_int": 1, "tolerant_int": 2, "float": 3, "bool": 4, "exact_status": 5}


@dataclass(frozen=True)
class SampleResult:
    sample_index: int
    iterations: int
    cycles: int
    pass_name: str
    counters: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CaseRunResult:
    case_bundle: CaseBundle
    comparison: ComparisonResult
    output_bytes: bytes
    samples: tuple[SampleResult, ...]
    normalized_samples: tuple[NormalizedSample, ...]
    statistics: SampleStatistics


@dataclass(frozen=True)
class SessionResult:
    cases: tuple[CaseRunResult, ...]
    protocol_trace: tuple[str, ...]
    session_complete_cases: int
    # Number of RTT sessions (SESSION_PLANs) the cases were spread over; >1 only for
    # hardware_run's batched runner, which merges several sessions into one result.
    batch_count: int = 1
    target_info: TargetInfo | None = None

    @property
    def case_bundle(self) -> CaseBundle:
        return self.cases[0].case_bundle

    @property
    def comparison(self) -> ComparisonResult:
        return self.cases[0].comparison

    @property
    def output_bytes(self) -> bytes:
        return self.cases[0].output_bytes

    @property
    def samples(self) -> tuple[SampleResult, ...]:
        return self.cases[0].samples


def default_counter_passes() -> tuple[CounterPass, ...]:
    """One `cpu_0` pass at the cpu group's default selection."""
    return counter_passes_for_selection({"cpu": "default"})


class HostSession:
    def __init__(self, transport: Transport, *, counter_passes: Sequence[CounterPass] | None = None) -> None:
        self._transport = transport
        self._decoder = FrameDecoder(max_payload=4096)
        self._session_id: int | None = None
        self._incoming_validator: SessionFrameValidator | None = None
        self._outgoing_sequence_id = 0
        self._trace: list[str] = []
        self._frames: list[Frame] = []
        self._counter_passes: tuple[CounterPass, ...] = (
            tuple(counter_passes) if counter_passes is not None else default_counter_passes()
        )
        self._last_sent_message_type: str | None = None
        self._target_info: TargetInfo | None = None

    @property
    def counter_passes(self) -> tuple[CounterPass, ...]:
        return self._counter_passes

    def run(self, case_bundle: CaseBundle) -> SessionResult:
        return self.run_many([case_bundle])

    def run_many(
        self,
        case_bundles: list[CaseBundle],
        *,
        on_case_complete: Callable[[CaseRunResult], None] | None = None,
    ) -> SessionResult:
        """Run every case in case_bundles over one SESSION_PLAN.

        If on_case_complete is given, it is invoked with each case's CaseRunResult
        immediately after it finishes (i.e. as soon as its CASE_COMPLETE frame is
        decoded), before waiting on the next case -- callers can use this for live
        per-case progress output instead of waiting for the whole batch/session to
        finish before seeing anything.
        """
        target_info_frame = self._recv_one(MessageType.TARGET_INFO)
        target_info = self._decode_target_info(target_info_frame.payload)
        self._target_info = target_info
        self._session_id = target_info_frame.header.session_id
        self._incoming_validator = SessionFrameValidator(session_id=self._session_id, next_sequence_id=1)
        self._check_counter_passes(target_info)
        self._send(MessageType.TARGET_INFO_ACK, b"")

        catalog = self._recv_catalog(target_info.catalog_hash)
        known_kernel_ids = {entry.kernel_id for entry in catalog}
        for bundle in case_bundles:
            if bundle.kernel_id not in known_kernel_ids:
                raise RuntimeError(
                    f"Case {bundle.case_id!r} references kernel_id {bundle.kernel_id}, "
                    "which is not present in the target's advertised catalog."
                )
            required = bundle.workspace_bytes_required
            available = int(target_info.runtime_arena_capacity)
            if required > available:
                raise RuntimeError(
                    f"Case {bundle.case_id!r} requires {required} workspace bytes, "
                    f"but the target advertises only {available} bytes."
                )

        plan = self._encode_plan(case_bundles)
        if len(plan) > target_info.max_rx_payload:
            raise RuntimeError(
                f"SESSION_PLAN for {len(case_bundles)} case(s) and {len(self._counter_passes)} PMU pass(es) "
                f"encodes to {len(plan)} bytes, but the target's receive buffer only takes "
                f"{target_info.max_rx_payload}-byte payloads (TARGET_INFO max_rx_payload). Split the batch."
            )
        self._send(MessageType.SESSION_PLAN, plan)

        case_map = {bundle.case_id: bundle for bundle in case_bundles}
        results: dict[str, CaseRunResult] = {}
        current_case_id: str | None = None
        actual_output_bytes = bytearray()
        samples: list[SampleResult] = []
        comparison_result: ComparisonResult | None = None
        reported_status: int | None = None
        session_complete_cases = 0

        while True:
            frame = self._recv_any()
            if frame.header.message_type == MessageType.KERNEL_CATALOG:
                continue
            if frame.header.message_type == MessageType.REQUEST_CASE:
                case_index = ByteReader(frame.payload).u16()
                bundle = case_bundles[case_index]
                current_case_id = bundle.case_id
                samples = []
                comparison_result = None
                actual_output_bytes = bytearray()
                reported_status = None
                self._send(MessageType.CASE_META, self._encode_case_meta(bundle))
            elif frame.header.message_type == MessageType.REQUEST_BLOB:
                if current_case_id is None:
                    raise RuntimeError("Target requested a blob before selecting a case.")
                self._handle_blob_request(frame.payload, case_map[current_case_id])
            elif frame.header.message_type == MessageType.CASE_READY:
                self._send(MessageType.RUN_CORRECTNESS, b"")
            elif frame.header.message_type == MessageType.CORRECTNESS_RESULT:
                reported_status = ByteReader(frame.payload).i32()
            elif frame.header.message_type == MessageType.OUTPUT_BEGIN:
                actual_output_bytes = bytearray()
            elif frame.header.message_type == MessageType.OUTPUT_CHUNK:
                reader = ByteReader(frame.payload)
                chunk_offset = reader.u32()
                chunk_length = reader.u32()
                chunk = reader.fixed(chunk_length)
                trailing_bytes = reader.remaining()
                if chunk_offset != len(actual_output_bytes) or trailing_bytes:
                    raise RuntimeError(
                        f"Invalid OUTPUT_CHUNK: offset={chunk_offset}, length={chunk_length}, "
                        f"received={len(chunk)}, trailing={len(trailing_bytes)}, "
                        f"expected_offset={len(actual_output_bytes)}."
                    )
                actual_output_bytes.extend(chunk)
            elif frame.header.message_type == MessageType.OUTPUT_END:
                if current_case_id is None:
                    raise RuntimeError("Received OUTPUT_END without an active case.")
                output_end = ByteReader(frame.payload)
                declared_length = output_end.u32()
                declared_checksum = output_end.u32()
                actual_checksum = sum(actual_output_bytes) & 0xFFFFFFFF
                if declared_length != len(actual_output_bytes) or declared_checksum != actual_checksum:
                    raise RuntimeError(
                        f"Invalid OUTPUT_END for {current_case_id!r}: declared length/checksum "
                        f"{declared_length}/{declared_checksum}, received "
                        f"{len(actual_output_bytes)}/{actual_checksum}."
                    )
                bundle = case_map[current_case_id]
                if bundle.comparison["mode"] == "exact_status":
                    if reported_status is None:
                        raise RuntimeError("Received OUTPUT_END before CORRECTNESS_RESULT status payload.")
                    comparison_result = compare_status(reported_status, bundle.comparison)
                else:
                    expected_output = blob_numpy(bundle.expected_output)
                    expected_size = expected_output.size
                    itemsize = expected_output.dtype.itemsize
                    actual_size = len(actual_output_bytes) // itemsize if itemsize else 0
                    if actual_size != expected_size or len(actual_output_bytes) % itemsize != 0:
                        # Defensive: an earlier case's kernel writing out-of-bounds into a
                        # shared session/output buffer (observed after a run of failing
                        # DepthwiseConv float cases) can leave a *later*, otherwise-correct
                        # case's OUTPUT_CHUNK stream holding more/fewer bytes than its own
                        # expected_output size implies. np.reshape() would raise here and
                        # crash the entire suite run over one corrupted case -- report it as
                        # a failed comparison instead so the rest of the suite still runs and
                        # the failure (and its diagnostic) is visible in the results table.
                        print(
                            f"[perf-stream] WARNING: case {current_case_id!r} output size mismatch -- "
                            f"received {len(actual_output_bytes)} bytes ({actual_size} elements of "
                            f"dtype {expected_output.dtype}), expected {expected_size} elements "
                            f"(shape {expected_output.shape}). Likely firmware/session state "
                            "corruption from a preceding case; reporting as a failed comparison "
                            "rather than aborting the run.",
                            file=sys.stderr,
                        )
                        comparison_result = ComparisonResult(
                            passed=False,
                            mismatch_count=abs(actual_size - expected_size),
                            max_abs_diff=float("nan"),
                            mode=str(bundle.comparison.get("mode", "unknown")),
                        )
                    else:
                        actual = np.frombuffer(bytes(actual_output_bytes), dtype=expected_output.dtype).reshape(
                            expected_output.shape
                        )
                        comparison_result = compare_output(actual, expected_output, bundle.comparison)
                writer = ByteWriter()
                writer.u8(1 if comparison_result.passed else 0)
                self._send(MessageType.CORRECTNESS_ACK, writer.finish())
                # The target always advances to WAIT_RUN_PERFORMANCE after CORRECTNESS_ACK
                # regardless of the pass/fail byte (it's informational only, for reporting).
                # RUN_PERFORMANCE must always be sent next, or the session deadlocks: the
                # host would wait forever for a reply while the target waits forever for
                # RUN_PERFORMANCE. Failing cases still get a full CaseRunResult (with
                # comparison.passed=False) so the CLI can report FAIL + cycle stats per case.
                self._send(MessageType.RUN_PERFORMANCE, b"")
            elif frame.header.message_type == MessageType.SAMPLE_RESULT:
                samples.append(self._decode_sample(frame.payload))
            elif frame.header.message_type == MessageType.CASE_COMPLETE:
                if current_case_id is None or comparison_result is None:
                    raise RuntimeError("CASE_COMPLETE arrived before correctness finished.")
                raw_samples = tuple(samples)
                normalized_samples = tuple(normalize_samples(self._to_raw_samples(raw_samples)))
                case_result = CaseRunResult(
                    case_bundle=case_map[current_case_id],
                    comparison=comparison_result,
                    output_bytes=bytes(actual_output_bytes),
                    samples=raw_samples,
                    normalized_samples=normalized_samples,
                    statistics=compute_sample_statistics(normalized_samples),
                )
                results[current_case_id] = case_result
                if on_case_complete is not None:
                    on_case_complete(case_result)
            elif frame.header.message_type == MessageType.SESSION_COMPLETE:
                session_complete_cases = ByteReader(frame.payload).u16()
                break
            elif frame.header.message_type == MessageType.ERROR:
                error_text = ByteReader(frame.payload).text()
                case_context = f" (while running case_id={current_case_id!r})" if current_case_id is not None else ""
                raise RuntimeError(f"{error_text}{case_context}")
            else:
                raise ValueError(f"Unhandled frame type: {frame.header.message_type}")

        ordered = tuple(results[bundle.case_id] for bundle in case_bundles)
        return SessionResult(
            cases=ordered,
            protocol_trace=tuple(self._trace),
            session_complete_cases=session_complete_cases,
            target_info=self._target_info,
        )

    def _check_counter_passes(self, target_info: TargetInfo) -> None:
        """Refuse PMU passes the target cannot run, before any plan is sent.

        A DWT-only target (no HCT_CAP_PMU_ARMV8M) only ever reports the cycle counter,
        so any pass asking for event counters would come back `supported=0` -- fail
        loudly instead. With a PMU, a chained counter takes two of the advertised
        16-bit slots, an unchained one takes one.
        """
        needs_events = [counter_pass for counter_pass in self._counter_passes if counter_pass.counters]
        if not needs_events:
            return
        if not target_info.has_pmu:
            names = ", ".join(counter_pass.name for counter_pass in needs_events)
            raise RuntimeError(
                f"Target {target_info.board_id!r} ({target_info.target_cpu}) has no Armv8.1-M PMU "
                f"(capability_flags=0x{target_info.capability_flags:08x}); it can only report "
                f"ARM_PMU_CPU_CYCLES from DWT. Drop the event-counter passes ({names}) or run on a PMU board."
            )
        for counter_pass in needs_events:
            if counter_pass.slots_required > target_info.pmu_counter_slots:
                raise RuntimeError(
                    f"PMU pass {counter_pass.name!r} needs {counter_pass.slots_required} event-counter "
                    f"slots ({len(counter_pass.counters)} {'chained' if counter_pass.chained else 'unchained'} "
                    f"counter(s)) but the target advertises only {target_info.pmu_counter_slots}."
                )

    def _to_raw_samples(self, samples: tuple[SampleResult, ...]) -> list[RawSample]:
        raw: list[RawSample] = []
        for sample in samples:
            raw.append(
                RawSample(
                    sample_index=sample.sample_index,
                    iterations=sample.iterations,
                    cycles=sample.cycles,
                    counters=tuple(
                        RawCounterValue(
                            name=str(counter["name"]),
                            event_id=int(counter["event_id"]),
                            value=int(counter["value"]),
                            overflow=bool(counter["overflow"]),
                            supported=bool(counter["supported"]),
                        )
                        for counter in sample.counters
                    ),
                    pass_name=sample.pass_name,
                )
            )
        return raw

    def _decode_target_info(self, payload: bytes) -> TargetInfo:
        """Every TARGET_INFO field: build_id, catalog_hash, max_frame_payload,
        runtime_arena_capacity, transfer_mode, output_mode, board_id, target_cpu,
        transport_kind, capability_flags, pmu_counter_slots, max_rx_payload,
        max_cases_per_session, max_passes."""
        return decode_target_info(payload)

    def _recv_catalog(self, expected_hash: bytes) -> tuple[CatalogEntry, ...]:
        """Accumulate one or more paginated KERNEL_CATALOG chunks (each chunk carries
        HCTP_FLAG_MORE until the final one) into the full kernel catalog, rejecting
        duplicate/missing kernel ids and verifying the assembled catalog's canonical-JSON
        SHA-256 matches the TARGET_INFO frame's advertised catalog_hash before returning."""
        entries_by_id: dict[int, CatalogEntry] = {}
        while True:
            frame = self._recv_one(MessageType.KERNEL_CATALOG)
            for entry in decode_kernel_catalog(frame.payload):
                if entry.kernel_id in entries_by_id:
                    raise RuntimeError(f"Duplicate kernel_id {entry.kernel_id} in paginated catalog.")
                entries_by_id[entry.kernel_id] = entry
            if (frame.header.flags & HCTP_FLAG_MORE) == 0:
                break

        entries = tuple(entries_by_id[kernel_id] for kernel_id in sorted(entries_by_id))
        canonical = json.dumps(
            [
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
                for entry in entries
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        actual_hash = hashlib.sha256(canonical).digest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Assembled kernel catalog hash {actual_hash.hex()} does not match TARGET_INFO "
                f"catalog_hash {expected_hash.hex()}."
            )
        return entries

    def _encode_plan(self, case_bundles: list[CaseBundle]) -> bytes:
        return encode_session_plan(case_bundles, self._counter_passes)

    def _encode_case_meta(self, case_bundle: CaseBundle) -> bytes:
        comparison = case_bundle.comparison
        scalar_parameters = dict(case_bundle.manifest.get("serialized_scalar_parameters", {}))
        scalar_parameters["output_capacity_bytes"] = case_bundle.expected_output.byte_length
        writer = ByteWriter()
        writer.text(case_bundle.case_id)
        writer.u32(case_bundle.kernel_id)
        writer.u16(int(case_bundle.manifest["adapter_metadata_schema"]))
        writer.u8(_COMPARISON_MODE_TO_CODE[str(comparison["mode"])])
        writer.i32(int(comparison.get("tolerance", 0)))
        writer.u32(int(round(float(comparison.get("atol", 0.0)) * 65536)))
        writer.u32(int(round(float(comparison.get("rtol", 0.0)) * 65536)))
        writer.u8(len(scalar_parameters))
        for key, value in scalar_parameters.items():
            writer.text(str(key))
            writer.i32(_encode_scalar(value))
        writer.u16(len(case_bundle.streamable_blobs))
        for blob in case_bundle.streamable_blobs:
            self._encode_blob_descriptor(writer, blob)
        writer.u32(int(case_bundle.manifest.get("scratch_buffer", {}).get("bytes", 0)))
        return writer.finish()

    def _encode_blob_descriptor(self, writer: ByteWriter, blob: BlobInfo) -> None:
        writer.u32(int(blob.blob_id))
        writer.text(str(blob.role))
        writer.text(str(blob.dtype))
        writer.u8(int(blob.rank))
        dims = list(blob.dimensions) + [0] * (6 - int(blob.rank))
        for dim in dims[:6]:
            writer.u32(int(dim))
        writer.u32(int(blob.byte_length))
        writer.u32(int(blob.required_alignment))
        writer.u32(int(blob.expected_crc32))
        writer.u8(1 if getattr(blob, "mutable_data", False) else 0)

    def _decode_sample(self, payload: bytes) -> SampleResult:
        """SAMPLE_RESULT: u16 sample_index, u32 iterations, u64 cycles (DWT), text pass_name,
        u8 counter_count, then per counter (text name, u16 event_id, u64 value, u8 overflow,
        u8 supported). Firmware sends the name empty; it is resolved from the catalog by
        event id here. The first entry is always ARM_PMU_CPU_CYCLES from the PMU CCNTR."""
        reader = ByteReader(payload)
        sample_index = reader.u16()
        iterations = reader.u32()
        cycles = reader.u64()
        pass_name = reader.text()
        counter_count = reader.u8()
        counters = []
        for _ in range(counter_count):
            name = reader.text()
            event_id = reader.u16()
            counters.append(
                {
                    "name": name or counter_name_for_event_id(event_id),
                    "event_id": event_id,
                    "value": reader.u64(),
                    "overflow": reader.u8(),
                    "supported": reader.u8(),
                }
            )
        return SampleResult(sample_index=sample_index, iterations=iterations, cycles=cycles, pass_name=pass_name, counters=tuple(counters))

    def _handle_blob_request(self, payload: bytes, case_bundle: CaseBundle) -> None:
        reader = ByteReader(payload)
        blob_id = reader.u32()
        offset = reader.u32()
        max_length = reader.u16()
        blob = next(blob for blob in case_bundle.streamable_blobs if blob.blob_id == blob_id)
        chunk = blob.path.read_bytes()[offset : offset + max_length]
        writer = ByteWriter()
        writer.u32(blob_id)
        writer.u32(offset)
        writer.raw(chunk)
        self._send(MessageType.BLOB_CHUNK, writer.finish())

    def _recv_any(self) -> Frame:
        while not self._frames:
            chunk = self._transport.read()
            if not chunk:
                raise RuntimeError(
                    "Transport stalled without a complete frame. "
                    f"Last message sent to target: {self._last_sent_message_type or '<none>'}. "
                    f"{len(self._trace)} frame(s) exchanged so far; last few: {self._trace[-6:]}."
                )
            self._frames.extend(self._decoder.feed(chunk))
        frame = self._frames.pop(0)
        if self._incoming_validator is not None:
            self._incoming_validator.accept(frame)
        self._trace.append(f"RX:{frame.header.message_type.name}")
        return frame

    def _recv_one(self, message_type: MessageType) -> Frame:
        frame = self._recv_any()
        if frame.header.message_type == MessageType.ERROR:
            raise RuntimeError(ByteReader(frame.payload).text())
        if frame.header.message_type != message_type:
            raise RuntimeError(f"Expected {message_type.name}, got {frame.header.message_type.name}")
        return frame

    def _send(self, message_type: MessageType, payload: bytes) -> None:
        if self._session_id is None:
            raise RuntimeError("Session has not been established yet.")
        frame = encode_frame(message_type, payload, session_id=self._session_id, sequence_id=self._outgoing_sequence_id)
        self._outgoing_sequence_id += 1
        self._trace.append(f"TX:{message_type.name}")
        self._last_sent_message_type = message_type.name
        self._transport.write(frame)



def _encode_scalar(value: Any) -> int:
    if isinstance(value, str):
        return {"VALID": 0, "SAME": 1}[value]
    return int(value)


def encode_session_plan(case_bundles: Sequence[CaseBundle], counter_passes: Sequence[CounterPass]) -> bytes:
    """SESSION_PLAN: u16 case_count, u8 transfer_mode(=1), u16 warmups, u16 samples,
    u32 iterations_per_sample, u32 min_cycles, u32 max_iterations, u8 pass_count,
    per pass (text pass_name, u8 chained, u8 counter_count, u16 event_id[counter_count]),
    then per case (text case_id, u32 kernel_id). The timing block is taken from the
    first bundle -- every generated case carries the same fixed timing plan."""
    first = case_bundles[0]
    writer = ByteWriter()
    writer.u16(len(case_bundles))
    writer.u8(1)
    writer.u16(int(first.manifest["timing"]["warmups"]))
    writer.u16(int(first.manifest["timing"]["samples"]))
    writer.u32(int(first.manifest["timing"]["iterations_per_sample"]))
    writer.u32(int(first.manifest["timing"].get("min_cycles", 1024)))
    writer.u32(int(first.manifest["timing"].get("max_iterations", 256)))
    writer.u8(len(counter_passes))
    for counter_pass in counter_passes:
        writer.text(counter_pass.name)
        writer.u8(1 if counter_pass.chained else 0)
        writer.u8(len(counter_pass.counters))
        for counter in counter_pass.counters:
            writer.u16(counter.event_id)
    for case_bundle in case_bundles:
        writer.text(case_bundle.case_id)
        writer.u32(case_bundle.kernel_id)
    return writer.finish()


def session_plan_size(case_ids: Sequence[str], counter_passes: Sequence[CounterPass]) -> int:
    """Encoded SESSION_PLAN size for these case ids and passes, without needing bundles --
    hardware_run uses it to keep every batch's plan within the target's receive buffer."""
    size = 2 + 1 + 2 + 2 + 4 + 4 + 4 + 1
    for counter_pass in counter_passes:
        size += 2 + len(counter_pass.name.encode("utf-8")) + 1 + 1 + 2 * len(counter_pass.counters)
    for case_id in case_ids:
        size += 2 + len(case_id.encode("utf-8")) + 4
    return size



def run_fake_abs_vertical_slice(project_root: Path, *, output_root: Path | None = None) -> SessionResult:
    case_bundle = build_abs_s8_case_bundle(project_root, output_root=output_root)
    reloaded = load_case_bundle(case_bundle.manifest_path)
    transport = FakeTargetTransport(max_frame_payload=17, read_chunk_size=13)
    return HostSession(transport).run(reloaded)



def run_fake_convolve_vertical_slice(project_root: Path, *, output_root: Path | None = None) -> SessionResult:
    case_bundle = build_convolve_s8_case_bundle(project_root, output_root=output_root)
    reloaded = load_case_bundle(case_bundle.manifest_path)
    transport = FakeTargetTransport(max_frame_payload=19, read_chunk_size=11)
    passes = counter_passes_for_selection({"cpu": "default", "memory": "default", "mve": "default"})
    return HostSession(transport, counter_passes=passes).run(reloaded)
