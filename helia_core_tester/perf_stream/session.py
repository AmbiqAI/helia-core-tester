"""Host side of one HCTP session: handshake, plan, per-case streaming, results."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .case_bundle import CaseBundle, blob_numpy, build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from .comparison import ComparisonResult, compare_output, compare_status
from .fake_target import FakeTargetTransport
from .hctp import HCTP_FLAG_MORE, Frame, FrameDecoder, MessageType, SessionFrameValidator, encode_frame
from .measurement import (
    MAX_COUNTERS_PER_PASS,
    CounterPass,
    NormalizedSample,
    RawSample,
    SampleStatistics,
    compute_sample_statistics,
    counter_passes_for_selection,
    normalize_samples,
)
from .transport import Transport
from .wire import (
    COMPARISON_MODE_CODES,
    BlobChunk,
    BlobDescriptor,
    CaseMeta,
    CatalogEntry,
    CorrectnessAck,
    PlannedCase,
    SessionPlan,
    TargetInfo,
    decode_case_complete,
    decode_correctness_result,
    decode_error,
    decode_kernel_catalog,
    decode_output_chunk,
    decode_output_end,
    decode_request_blob,
    decode_request_case,
    decode_sample_result,
    decode_session_complete,
    decode_target_info,
    encode_blob_chunk,
    encode_case_meta,
    encode_correctness_ack,
    encode_session_plan,
    kernel_catalog_hash,
    output_checksum,
)


@dataclass(frozen=True)
class CaseRunResult:
    case_bundle: CaseBundle
    comparison: ComparisonResult
    output_bytes: bytes
    samples: tuple[RawSample, ...]
    normalized_samples: tuple[NormalizedSample, ...]
    statistics: SampleStatistics


@dataclass(frozen=True)
class SessionResult:
    cases: tuple[CaseRunResult, ...]
    protocol_trace: tuple[str, ...]
    session_complete_cases: int
    # Number of RTT sessions (SESSION_PLANs) the cases were spread over; >1 only for
    # the batched runner in session_runner.py, which merges several into one result.
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
    def samples(self) -> tuple[RawSample, ...]:
        return self.cases[0].samples


@dataclass(frozen=True)
class TargetLimits:
    """The session-sizing limits a target advertises in TARGET_INFO, as the host uses them.

    `max_counters_per_pass` is the chained-pair planning rule (MAX_COUNTERS_PER_PASS)
    checked against the advertised PMU width: a chained counter takes two 16-bit slots.
    """

    max_cases: int
    max_plan_bytes: int
    max_passes: int
    pmu_counter_slots: int
    has_pmu: bool

    @classmethod
    def from_target_info(cls, info: TargetInfo) -> TargetLimits:
        return cls(
            max_cases=int(info.max_cases_per_session),
            max_plan_bytes=int(info.max_rx_payload),
            max_passes=int(info.max_passes),
            pmu_counter_slots=int(info.pmu_counter_slots),
            has_pmu=info.has_pmu,
        )

    @property
    def max_counters_per_pass(self) -> int:
        return min(MAX_COUNTERS_PER_PASS, self.pmu_counter_slots // 2) if self.has_pmu else 0


def check_counter_passes(counter_passes: Sequence[CounterPass], info: TargetInfo) -> None:
    """Refuse PMU passes the target cannot run, before any plan is sent.

    The pass list must fit the target's `max_passes`. A DWT-only target (no
    HCT_CAP_PMU_ARMV8M) only ever reports the cycle counter, so any pass asking for
    event counters would come back `supported=0` -- fail loudly instead. With a PMU, a
    chained counter takes two of the advertised 16-bit slots (the host plans at most
    MAX_COUNTERS_PER_PASS chained counters per pass, which must fit `slots / 2`), an
    unchained one takes one.
    """
    limits = TargetLimits.from_target_info(info)
    if len(counter_passes) > limits.max_passes:
        raise RuntimeError(
            f"{len(counter_passes)} PMU passes requested but target {info.board_id!r} takes at most "
            f"{limits.max_passes} per session (TARGET_INFO max_passes). Request fewer counters."
        )
    needs_events = [counter_pass for counter_pass in counter_passes if counter_pass.counters]
    if not needs_events:
        return
    if not info.has_pmu:
        names = ", ".join(counter_pass.name for counter_pass in needs_events)
        raise RuntimeError(
            f"Target {info.board_id!r} ({info.target_cpu}) has no Armv8.1-M PMU "
            f"(capability_flags=0x{info.capability_flags:08x}); it can only report "
            f"ARM_PMU_CPU_CYCLES from DWT. Drop the event-counter passes ({names}) or run on a PMU board."
        )
    for counter_pass in needs_events:
        if counter_pass.slots_required > info.pmu_counter_slots:
            raise RuntimeError(
                f"PMU pass {counter_pass.name!r} needs {counter_pass.slots_required} event-counter "
                f"slots ({len(counter_pass.counters)} {'chained' if counter_pass.chained else 'unchained'} "
                f"counter(s)) but the target advertises only {info.pmu_counter_slots}."
            )
        if counter_pass.chained and len(counter_pass.counters) > limits.max_counters_per_pass:
            raise RuntimeError(
                f"PMU pass {counter_pass.name!r} chains {len(counter_pass.counters)} counters, over the "
                f"{limits.max_counters_per_pass} chained pair(s) the target's {info.pmu_counter_slots} "
                "event-counter slots allow per pass."
            )


def default_counter_passes() -> tuple[CounterPass, ...]:
    """One `cpu_0` pass at the cpu group's default selection."""
    return counter_passes_for_selection({"cpu": "default"})


def session_plan_for_bundles(case_bundles: Sequence[CaseBundle], counter_passes: Sequence[CounterPass]) -> SessionPlan:
    """The SESSION_PLAN for these bundles. The timing block is taken from the first
    bundle -- every generated case carries the same fixed timing plan."""
    timing = case_bundles[0].manifest["timing"]
    return SessionPlan(
        warmups=int(timing["warmups"]),
        samples=int(timing["samples"]),
        iterations_per_sample=int(timing["iterations_per_sample"]),
        min_cycles=int(timing.get("min_cycles", 1024)),
        max_iterations=int(timing.get("max_iterations", 256)),
        passes=tuple(counter_passes),
        cases=tuple(PlannedCase(case_id=bundle.case_id, kernel_id=bundle.kernel_id) for bundle in case_bundles),
    )


def case_meta_for_bundle(case_bundle: CaseBundle) -> CaseMeta:
    comparison = case_bundle.comparison
    scalar_parameters = dict(case_bundle.manifest.get("serialized_scalar_parameters", {}))
    scalar_parameters["output_capacity_bytes"] = case_bundle.expected_output.byte_length
    return CaseMeta(
        case_id=case_bundle.case_id,
        kernel_id=case_bundle.kernel_id,
        schema_version=int(case_bundle.manifest["adapter_metadata_schema"]),
        comparison_mode=COMPARISON_MODE_CODES[str(comparison["mode"])],
        tolerance=int(comparison.get("tolerance", 0)),
        atol_q16=int(round(float(comparison.get("atol", 0.0)) * 65536)),
        rtol_q16=int(round(float(comparison.get("rtol", 0.0)) * 65536)),
        scalar_parameters=tuple((str(key), _encode_scalar(value)) for key, value in scalar_parameters.items()),
        blobs=tuple(
            BlobDescriptor(
                blob_id=int(blob.blob_id),
                role=str(blob.role),
                dtype=str(blob.dtype),
                dimensions=tuple(int(dim) for dim in blob.dimensions)[: int(blob.rank)],
                byte_length=int(blob.byte_length),
                alignment=int(blob.required_alignment),
                crc32=int(blob.expected_crc32),
                mutable_data=bool(getattr(blob, "mutable_data", False)),
            )
            for blob in case_bundle.streamable_blobs
        ),
        scratch_bytes=int(case_bundle.manifest.get("scratch_buffer", {}).get("bytes", 0)),
    )


def _encode_scalar(value: Any) -> int:
    if isinstance(value, str):
        return {"VALID": 0, "SAME": 1}[value]
    return int(value)


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
        self._catalog: tuple[CatalogEntry, ...] = ()

    @property
    def counter_passes(self) -> tuple[CounterPass, ...]:
        return self._counter_passes

    @property
    def target_info(self) -> TargetInfo | None:
        """What the target announced, once `handshake()` has run."""
        return self._target_info

    @property
    def limits(self) -> TargetLimits:
        if self._target_info is None:
            raise RuntimeError("Session has not completed its handshake yet.")
        return TargetLimits.from_target_info(self._target_info)

    def run(self, case_bundle: CaseBundle) -> SessionResult:
        return self.run_many([case_bundle])

    def handshake(self) -> TargetInfo:
        """TARGET_INFO -> TARGET_INFO_ACK -> KERNEL_CATALOG: learn the target's limits and
        catalog, refusing PMU passes it cannot run before any plan is sent."""
        target_info_frame = self._recv_one(MessageType.TARGET_INFO)
        target_info = decode_target_info(target_info_frame.payload)
        self._target_info = target_info
        self._session_id = target_info_frame.header.session_id
        self._incoming_validator = SessionFrameValidator(session_id=self._session_id, next_sequence_id=1)
        check_counter_passes(self._counter_passes, target_info)
        self._send(MessageType.TARGET_INFO_ACK, b"")
        self._catalog = self._recv_catalog(target_info.catalog_hash)
        return target_info

    def run_many(
        self,
        case_bundles: list[CaseBundle],
        *,
        on_case_complete: Callable[[CaseRunResult], None] | None = None,
    ) -> SessionResult:
        """Run every case in case_bundles over one SESSION_PLAN (after `handshake()`,
        which is performed here if the caller has not already done so).

        If on_case_complete is given, it is invoked with each case's CaseRunResult
        immediately after it finishes (i.e. as soon as its CASE_COMPLETE frame is
        decoded), before waiting on the next case -- callers can use this for live
        per-case progress output instead of waiting for the whole batch/session to
        finish before seeing anything.
        """
        target_info = self._target_info if self._target_info is not None else self.handshake()

        known_kernel_ids = {entry.kernel_id for entry in self._catalog}
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

        limits = self.limits
        if len(case_bundles) > limits.max_cases:
            raise RuntimeError(
                f"Cannot run {len(case_bundles)} cases in a single session: target {target_info.board_id!r} "
                f"takes at most {limits.max_cases} per SESSION_PLAN (TARGET_INFO max_cases_per_session). "
                "Split into batches first."
            )
        plan = encode_session_plan(session_plan_for_bundles(case_bundles, self._counter_passes))
        if len(plan) > limits.max_plan_bytes:
            raise RuntimeError(
                f"SESSION_PLAN for {len(case_bundles)} case(s) and {len(self._counter_passes)} PMU pass(es) "
                f"encodes to {len(plan)} bytes, but the target's receive buffer only takes "
                f"{limits.max_plan_bytes}-byte payloads (TARGET_INFO max_rx_payload). Split the batch."
            )
        self._send(MessageType.SESSION_PLAN, plan)

        case_map = {bundle.case_id: bundle for bundle in case_bundles}
        results: dict[str, CaseRunResult] = {}
        current_case_id: str | None = None
        actual_output_bytes = bytearray()
        samples: list[RawSample] = []
        comparison_result: ComparisonResult | None = None
        reported_status: int | None = None
        session_complete_cases = 0

        while True:
            frame = self._recv_any()
            message_type = frame.header.message_type
            if message_type == MessageType.KERNEL_CATALOG:
                continue
            if message_type == MessageType.REQUEST_CASE:
                bundle = case_bundles[decode_request_case(frame.payload).case_index]
                current_case_id = bundle.case_id
                samples = []
                comparison_result = None
                actual_output_bytes = bytearray()
                reported_status = None
                self._send(MessageType.CASE_META, encode_case_meta(case_meta_for_bundle(bundle)))
            elif message_type == MessageType.REQUEST_BLOB:
                if current_case_id is None:
                    raise RuntimeError("Target requested a blob before selecting a case.")
                self._handle_blob_request(frame.payload, case_map[current_case_id])
            elif message_type == MessageType.CASE_READY:
                self._send(MessageType.RUN_CORRECTNESS, b"")
            elif message_type == MessageType.CORRECTNESS_RESULT:
                reported_status = decode_correctness_result(frame.payload).status
            elif message_type == MessageType.OUTPUT_BEGIN:
                actual_output_bytes = bytearray()
            elif message_type == MessageType.OUTPUT_CHUNK:
                try:
                    chunk = decode_output_chunk(frame.payload)
                except ValueError as exc:
                    raise RuntimeError(f"Invalid OUTPUT_CHUNK: {exc}") from exc
                if chunk.offset != len(actual_output_bytes):
                    raise RuntimeError(
                        f"Invalid OUTPUT_CHUNK: offset={chunk.offset}, length={len(chunk.data)}, "
                        f"expected_offset={len(actual_output_bytes)}."
                    )
                actual_output_bytes.extend(chunk.data)
            elif message_type == MessageType.OUTPUT_END:
                if current_case_id is None:
                    raise RuntimeError("Received OUTPUT_END without an active case.")
                output_end = decode_output_end(frame.payload)
                actual_checksum = output_checksum(bytes(actual_output_bytes))
                if output_end.length != len(actual_output_bytes) or output_end.checksum != actual_checksum:
                    raise RuntimeError(
                        f"Invalid OUTPUT_END for {current_case_id!r}: declared length/checksum "
                        f"{output_end.length}/{output_end.checksum}, received "
                        f"{len(actual_output_bytes)}/{actual_checksum}."
                    )
                bundle = case_map[current_case_id]
                if bundle.comparison["mode"] == "exact_status":
                    if reported_status is None:
                        raise RuntimeError("Received OUTPUT_END before CORRECTNESS_RESULT status payload.")
                    comparison_result = compare_status(reported_status, bundle.comparison)
                else:
                    comparison_result = _compare_output_bytes(current_case_id, bytes(actual_output_bytes), bundle)
                self._send(MessageType.CORRECTNESS_ACK, encode_correctness_ack(CorrectnessAck(passed=comparison_result.passed)))
                # The target always advances to WAIT_RUN_PERFORMANCE after CORRECTNESS_ACK
                # regardless of the pass/fail byte (it's informational only, for reporting).
                # RUN_PERFORMANCE must always be sent next, or the session deadlocks: the
                # host would wait forever for a reply while the target waits forever for
                # RUN_PERFORMANCE. Failing cases still get a full CaseRunResult (with
                # comparison.passed=False) so the CLI can report FAIL + cycle stats per case.
                self._send(MessageType.RUN_PERFORMANCE, b"")
            elif message_type == MessageType.SAMPLE_RESULT:
                samples.append(decode_sample_result(frame.payload))
            elif message_type == MessageType.CASE_COMPLETE:
                if current_case_id is None or comparison_result is None:
                    raise RuntimeError("CASE_COMPLETE arrived before correctness finished.")
                decode_case_complete(frame.payload)
                raw_samples = tuple(samples)
                normalized_samples = tuple(normalize_samples(raw_samples))
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
            elif message_type == MessageType.SESSION_COMPLETE:
                session_complete_cases = decode_session_complete(frame.payload).case_count
                break
            elif message_type == MessageType.ERROR:
                error_text = decode_error(frame.payload).message
                case_context = f" (while running case_id={current_case_id!r})" if current_case_id is not None else ""
                raise RuntimeError(f"{error_text}{case_context}")
            else:
                raise ValueError(f"Unhandled frame type: {message_type}")

        ordered = tuple(results[bundle.case_id] for bundle in case_bundles)
        return SessionResult(
            cases=ordered,
            protocol_trace=tuple(self._trace),
            session_complete_cases=session_complete_cases,
            target_info=self._target_info,
        )

    def _recv_catalog(self, expected_hash: bytes) -> tuple[CatalogEntry, ...]:
        """Accumulate one or more paginated KERNEL_CATALOG chunks (each chunk carries
        HCTP_FLAG_MORE until the final one) into the full kernel catalog, rejecting
        duplicate kernel ids and verifying the assembled catalog's canonical-JSON SHA-256
        matches the TARGET_INFO frame's advertised catalog_hash before returning."""
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
        actual_hash = kernel_catalog_hash(entries)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Assembled kernel catalog hash {actual_hash.hex()} does not match TARGET_INFO "
                f"catalog_hash {expected_hash.hex()}."
            )
        return entries

    def _handle_blob_request(self, payload: bytes, case_bundle: CaseBundle) -> None:
        request = decode_request_blob(payload)
        blob = next(blob for blob in case_bundle.streamable_blobs if blob.blob_id == request.blob_id)
        chunk = blob.path.read_bytes()[request.offset : request.offset + request.max_length]
        self._send(MessageType.BLOB_CHUNK, encode_blob_chunk(BlobChunk(blob_id=request.blob_id, offset=request.offset, data=chunk)))

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
            raise RuntimeError(decode_error(frame.payload).message)
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


def _compare_output_bytes(case_id: str, actual_output_bytes: bytes, bundle: CaseBundle) -> ComparisonResult:
    expected_output = blob_numpy(bundle.expected_output)
    expected_size = expected_output.size
    itemsize = expected_output.dtype.itemsize
    actual_size = len(actual_output_bytes) // itemsize if itemsize else 0
    if actual_size != expected_size or len(actual_output_bytes) % itemsize != 0:
        # Defensive: an earlier case's kernel writing out-of-bounds into a shared
        # session/output buffer (observed after a run of failing DepthwiseConv float
        # cases) can leave a *later*, otherwise-correct case's OUTPUT_CHUNK stream
        # holding more/fewer bytes than its own expected_output size implies.
        # np.reshape() would raise here and crash the entire suite run over one
        # corrupted case -- report it as a failed comparison instead so the rest of the
        # suite still runs and the failure (and its diagnostic) is visible in the results.
        print(
            f"[perf-stream] WARNING: case {case_id!r} output size mismatch -- "
            f"received {len(actual_output_bytes)} bytes ({actual_size} elements of "
            f"dtype {expected_output.dtype}), expected {expected_size} elements "
            f"(shape {expected_output.shape}). Likely firmware/session state "
            "corruption from a preceding case; reporting as a failed comparison "
            "rather than aborting the run.",
            file=sys.stderr,
        )
        return ComparisonResult(
            passed=False,
            mismatch_count=abs(actual_size - expected_size),
            max_abs_diff=float("nan"),
            mode=str(bundle.comparison.get("mode", "unknown")),
        )
    actual = np.frombuffer(actual_output_bytes, dtype=expected_output.dtype).reshape(expected_output.shape)
    return compare_output(actual, expected_output, bundle.comparison)


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
