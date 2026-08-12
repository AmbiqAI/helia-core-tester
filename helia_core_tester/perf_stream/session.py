"""Host-side streaming session runner for the fake vertical slices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .case_bundle import CaseBundle, BlobInfo, blob_numpy, build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from .comparison import ComparisonResult, compare_output
from .fake_target import FakeTargetTransport
from .hctp import ByteReader, ByteWriter, Frame, FrameDecoder, MessageType, SessionFrameValidator, encode_frame
from .measurement import NormalizedSample, RawCounterValue, RawSample, SampleStatistics, compute_sample_statistics, normalize_samples
from .transport import Transport


_COMPARISON_MODE_TO_CODE = {"exact_int": 1, "tolerant_int": 2, "float": 3, "bool": 4}


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


class HostSession:
    def __init__(self, transport: Transport, *, requested_counter_groups: tuple[str, ...] = ("cpu",)) -> None:
        self._transport = transport
        self._decoder = FrameDecoder(max_payload=4096)
        self._session_id: int | None = None
        self._incoming_validator: SessionFrameValidator | None = None
        self._outgoing_sequence_id = 0
        self._trace: list[str] = []
        self._frames: list[Frame] = []
        self._requested_counter_groups = requested_counter_groups
        self._last_sent_message_type: str | None = None

    def run(self, case_bundle: CaseBundle) -> SessionResult:
        return self.run_many([case_bundle])

    def run_many(self, case_bundles: list[CaseBundle]) -> SessionResult:
        hello = self._recv_one(MessageType.HELLO)
        self._decode_hello(hello.payload)
        self._session_id = hello.header.session_id
        self._incoming_validator = SessionFrameValidator(session_id=self._session_id, next_sequence_id=1)
        self._send(MessageType.HELLO_ACK, b"")
        self._send(MessageType.LOAD_PLAN, self._encode_plan(case_bundles))

        case_map = {bundle.case_id: bundle for bundle in case_bundles}
        results: dict[str, CaseRunResult] = {}
        current_case_id: str | None = None
        actual_output_bytes = bytearray()
        samples: list[SampleResult] = []
        comparison_result: ComparisonResult | None = None
        session_complete_cases = 0

        while True:
            frame = self._recv_any()
            if frame.header.message_type == MessageType.CAPABILITIES:
                continue
            if frame.header.message_type == MessageType.REQUEST_CASE:
                case_index = ByteReader(frame.payload).u16()
                bundle = case_bundles[case_index]
                current_case_id = bundle.case_id
                samples = []
                comparison_result = None
                actual_output_bytes = bytearray()
                self._send(MessageType.CASE_META, self._encode_case_meta(bundle))
            elif frame.header.message_type == MessageType.REQUEST_BLOB:
                if current_case_id is None:
                    raise RuntimeError("Target requested a blob before selecting a case.")
                self._handle_blob_request(frame.payload, case_map[current_case_id])
            elif frame.header.message_type == MessageType.CASE_READY:
                self._send(MessageType.RUN_CORRECTNESS, b"")
            elif frame.header.message_type == MessageType.CORRECTNESS_RESULT:
                pass
            elif frame.header.message_type == MessageType.OUTPUT_BEGIN:
                actual_output_bytes = bytearray()
            elif frame.header.message_type == MessageType.OUTPUT_CHUNK:
                reader = ByteReader(frame.payload)
                _offset = reader.u32()
                actual_output_bytes.extend(reader.raw())
            elif frame.header.message_type == MessageType.OUTPUT_END:
                if current_case_id is None:
                    raise RuntimeError("Received OUTPUT_END without an active case.")
                bundle = case_map[current_case_id]
                expected_output = blob_numpy(bundle.expected_output)
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
                results[current_case_id] = CaseRunResult(
                    case_bundle=case_map[current_case_id],
                    comparison=comparison_result,
                    output_bytes=bytes(actual_output_bytes),
                    samples=raw_samples,
                    normalized_samples=normalized_samples,
                    statistics=compute_sample_statistics(normalized_samples),
                )
            elif frame.header.message_type == MessageType.SESSION_COMPLETE:
                session_complete_cases = ByteReader(frame.payload).u16()
                break
            elif frame.header.message_type == MessageType.ERROR:
                raise RuntimeError(ByteReader(frame.payload).text())
            else:
                raise ValueError(f"Unhandled frame type: {frame.header.message_type}")

        ordered = tuple(results[bundle.case_id] for bundle in case_bundles)
        return SessionResult(cases=ordered, protocol_trace=tuple(self._trace), session_complete_cases=session_complete_cases)

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

    def _decode_hello(self, payload: bytes) -> dict[str, Any]:
        reader = ByteReader(payload)
        return {
            "build_id": reader.text(),
            "catalog_hash": reader.fixed(32).hex(),
            "max_frame_payload": reader.u32(),
            "runtime_arena_capacity": reader.u32(),
            "transfer_mode": reader.u8(),
            "output_mode": reader.u8(),
        }

    def _encode_plan(self, case_bundles: list[CaseBundle]) -> bytes:
        first = case_bundles[0]
        writer = ByteWriter()
        writer.u16(len(case_bundles))
        writer.u8(1)
        writer.u16(int(first.manifest["timing"]["warmups"]))
        writer.u16(int(first.manifest["timing"]["samples"]))
        writer.u32(int(first.manifest["timing"]["iterations_per_sample"]))
        writer.u32(int(first.manifest["timing"].get("min_cycles", 1024)))
        writer.u32(int(first.manifest["timing"].get("max_iterations", 256)))
        writer.u8(len(self._requested_counter_groups))
        for group in self._requested_counter_groups:
            writer.text(group)
        for case_bundle in case_bundles:
            writer.text(case_bundle.case_id)
            writer.u32(case_bundle.kernel_id)
        return writer.finish()

    def _encode_case_meta(self, case_bundle: CaseBundle) -> bytes:
        comparison = case_bundle.comparison
        scalar_parameters = case_bundle.manifest.get("serialized_scalar_parameters", {})
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
        reader = ByteReader(payload)
        sample_index = reader.u16()
        iterations = reader.u32()
        cycles = reader.u64()
        pass_name = reader.text()
        counter_count = reader.u8()
        counters = []
        for _ in range(counter_count):
            counters.append(
                {
                    "name": reader.text(),
                    "event_id": reader.u16(),
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



def run_fake_abs_vertical_slice(project_root: Path, *, output_root: Path | None = None) -> SessionResult:
    case_bundle = build_abs_s8_case_bundle(project_root, output_root=output_root)
    reloaded = load_case_bundle(case_bundle.manifest_path)
    transport = FakeTargetTransport(max_frame_payload=17, read_chunk_size=13)
    return HostSession(transport).run(reloaded)



def run_fake_convolve_vertical_slice(project_root: Path, *, output_root: Path | None = None) -> SessionResult:
    case_bundle = build_convolve_s8_case_bundle(project_root, output_root=output_root)
    reloaded = load_case_bundle(case_bundle.manifest_path)
    transport = FakeTargetTransport(max_frame_payload=19, read_chunk_size=11)
    return HostSession(transport, requested_counter_groups=("cpu", "memory", "mve")).run(reloaded)
