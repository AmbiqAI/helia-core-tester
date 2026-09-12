"""Fake HCTP target used for host-side streaming tests.

Speaks the protocol through the same wire.py codec as the host session: every
frame it receives is decoded with the host's encoder's counterpart and every frame
it sends is built with the host's decoder's counterpart, so the two sides can never
disagree about a payload layout without a round-trip test noticing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from helia_core_tester.generation.utils.tflite_utils import requantize_np

from .hctp import HEADER_SIZE, Frame, FrameDecoder, MessageType, SessionFrameValidator, encode_frame
from .measurement import (
    CounterDescriptor,
    CounterPass,
    RawCounterValue,
    RawSample,
    auto_calibrate_iterations,
)
from .pmu_catalog import CPU_CYCLES_EVENT_ID, CPU_CYCLES_NAME
from .transfer import ArenaTracker, BlobAccumulator, BlobTransferSpec, CaseTooLargeError
from .wire import (
    CAP_ABS_S8,
    CAP_CASE_STREAMING,
    CAP_CORRECTNESS,
    CAP_KERNEL_CATALOG,
    CAP_PERFORMANCE,
    CAP_PMU_ARMV8M,
    CAP_RTT_TRANSPORT,
    BlobDescriptor,
    CaseComplete,
    CaseMeta,
    CaseReady,
    CatalogEntry,
    CorrectnessResult,
    ErrorPayload,
    OutputBegin,
    OutputChunk,
    OutputEnd,
    RequestBlob,
    RequestCase,
    SessionComplete,
    SessionPlan,
    TargetInfo,
    decode_blob_chunk,
    decode_case_meta,
    decode_session_plan,
    encode_case_complete,
    encode_case_ready,
    encode_correctness_result,
    encode_error,
    encode_kernel_catalog,
    encode_output_begin,
    encode_output_chunk,
    encode_output_end,
    encode_request_blob,
    encode_request_case,
    encode_sample_result,
    encode_session_complete,
    encode_target_info,
    kernel_catalog_hash,
    output_checksum,
)

# Mirrors the real firmware's fixed limits (benchmark_server_session.h / _main.c) so
# host tests exercise the same bounds the target enforces. All of them are
# advertised in TARGET_INFO, exactly like the firmware does.
FAKE_PMU_COUNTER_SLOTS = 8
FAKE_RX_BUFFER_BYTES = 2048
FAKE_MAX_RX_PAYLOAD = FAKE_RX_BUFFER_BYTES - HEADER_SIZE
FAKE_MAX_CASES_PER_SESSION = 32
FAKE_MAX_PASSES = 16
FAKE_MAX_COUNTERS_PER_PASS = 4
EVENT_COUNTER_MASK = 0xFFFF  # one 16-bit slot
CHAINED_COUNTER_MASK = 0xFFFFFFFF  # two slots chained
CCNTR_MASK = 0xFFFFFFFF


class _TargetState(str, Enum):
    WAIT_TARGET_INFO_ACK = "wait_target_info_ack"
    WAIT_PLAN = "wait_plan"
    WAIT_CASE_META = "wait_case_meta"
    WAIT_BLOB_CHUNK = "wait_blob_chunk"
    WAIT_RUN_CORRECTNESS = "wait_run_correctness"
    WAIT_CORRECTNESS_ACK = "wait_correctness_ack"
    WAIT_RUN_PERFORMANCE = "wait_run_performance"
    COMPLETE = "complete"


class FakeKernelAdapter:
    entry: CatalogEntry
    supported_groups: tuple[str, ...] = ("cpu",)
    base_cycles_per_iteration: int = 1

    def invoke(self, blobs: dict[str, np.ndarray], scalar_parameters: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def measure(
        self,
        blobs: dict[str, np.ndarray],
        scalar_parameters: dict[str, Any],
        *,
        warmups: int,
        samples: int,
        iterations: int,
        counter_passes: list[CounterPass],
    ) -> tuple[int, list[RawSample]]:
        if iterations == 0:
            calibration = auto_calibrate_iterations(
                base_cycles=self._base_cycles(blobs),
                min_cycles=int(scalar_parameters.get("min_cycles", 1024)),
                max_iterations=int(scalar_parameters.get("max_iterations", 256)),
                stateful=not self.entry.repeated_invocation_safe,
            )
            iterations = calibration.iterations
        for _ in range(warmups):
            self.invoke(blobs, scalar_parameters)
        samples_out: list[RawSample] = []
        for counter_pass in counter_passes:
            for sample_index in range(samples):
                base_cycles = self._base_cycles(blobs) * iterations
                cycles = base_cycles + (sample_index * 11) + (counter_pass.pass_index * 37) + 100
                # Like the firmware: the CCNTR entry leads every sample. It reads a few
                # cycles above the DWT window because the PMU is started just before the
                # DWT start read and stopped just after the DWT end read.
                ccntr = cycles + 3
                counters = [
                    RawCounterValue(
                        name=CPU_CYCLES_NAME,
                        event_id=CPU_CYCLES_EVENT_ID,
                        value=ccntr & CCNTR_MASK,
                        overflow=ccntr > CCNTR_MASK,
                        supported=True,
                    )
                ]
                for counter in counter_pass.counters:
                    supported = counter.group in self.supported_groups
                    value = self._counter_value(counter, blobs, iterations, sample_index)
                    # Honour the real counter widths so tests can provoke an overflow:
                    # a 16-bit slot wraps unless the pass chains slot pairs into 32 bits.
                    mask = CHAINED_COUNTER_MASK if counter_pass.chained else EVENT_COUNTER_MASK
                    counters.append(
                        RawCounterValue(
                            name=counter.name,
                            event_id=counter.event_id,
                            value=(value & mask) if supported else 0,
                            overflow=supported and value > mask,
                            supported=supported,
                        )
                    )
                samples_out.append(
                    RawSample(
                        sample_index=sample_index,
                        iterations=iterations,
                        cycles=cycles,
                        counters=tuple(counters),
                        pass_name=counter_pass.name,
                    )
                )
        return iterations, samples_out

    def _base_cycles(self, blobs: dict[str, np.ndarray]) -> int:
        return max(1, int(sum(blob.size for blob in blobs.values() if blob.dtype != np.int32)) * self.base_cycles_per_iteration)

    def _counter_value(self, counter: CounterDescriptor, blobs: dict[str, np.ndarray], iterations: int, sample_index: int) -> int:
        return (self._base_cycles(blobs) * iterations) + counter.event_id + sample_index


class FakeAbsS8Adapter(FakeKernelAdapter):
    entry = CatalogEntry(1, "arm_abs_s8", "BasicMathFunctions", 1, "S8", 1, True, True, False, 0)
    supported_groups = ("cpu",)
    base_cycles_per_iteration = 3

    def invoke(self, blobs: dict[str, np.ndarray], scalar_parameters: dict[str, Any]) -> np.ndarray:
        return np.abs(blobs["input_0"].astype(np.int16)).astype(np.int8)


class FakeConvolveS8Adapter(FakeKernelAdapter):
    entry = CatalogEntry(2, "arm_convolve_s8", "ConvolutionFunctions", 1, "S8", 1, True, True, False, 64)
    supported_groups = ("cpu", "memory")
    base_cycles_per_iteration = 19

    def invoke(self, blobs: dict[str, np.ndarray], scalar_parameters: dict[str, Any]) -> np.ndarray:
        input_data = blobs["input_0"].astype(np.int32)
        weights = blobs["weights"].astype(np.int32)
        bias = blobs["bias"].astype(np.int32)
        multiplier = blobs["multiplier"].astype(np.int32)
        shift = blobs["shift"].astype(np.int32)
        stride_h = int(scalar_parameters["stride_h"])
        stride_w = int(scalar_parameters["stride_w"])
        padding = str(scalar_parameters["padding"])
        batch, in_h, in_w, in_c = input_data.shape
        filt_h, filt_w, filt_c, out_c = weights.shape
        assert batch == 1 and filt_c == in_c
        pad_top = pad_bottom = pad_left = pad_right = 0
        if padding == "SAME":
            out_h = (in_h + stride_h - 1) // stride_h
            out_w = (in_w + stride_w - 1) // stride_w
            pad_h = max((out_h - 1) * stride_h + filt_h - in_h, 0)
            pad_w = max((out_w - 1) * stride_w + filt_w - in_w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        padded = np.pad(input_data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
        out_h = ((padded.shape[1] - filt_h) // stride_h) + 1
        out_w = ((padded.shape[2] - filt_w) // stride_w) + 1
        activation_min = int(scalar_parameters.get("activation_min", -128))
        activation_max = int(scalar_parameters.get("activation_max", 127))
        output_offset = int(scalar_parameters.get("output_offset", 0))
        output = np.zeros((batch, out_h, out_w, out_c), dtype=np.int8)
        for oy in range(out_h):
            for ox in range(out_w):
                window = padded[0, oy * stride_h : oy * stride_h + filt_h, ox * stride_w : ox * stride_w + filt_w, :]
                for oc in range(out_c):
                    acc = int(np.sum(window * weights[:, :, :, oc])) + int(bias[oc])
                    value = int(requantize_np(np.array([acc], dtype=np.int32), int(multiplier[oc]), int(shift[oc]))[0])
                    output[0, oy, ox, oc] = np.int8(np.clip(value + output_offset, activation_min, activation_max))
        return output


class FakeTargetTransport:
    """Synchronous fake transport that simulates a target-side HCTP server."""

    def __init__(
        self,
        *,
        max_frame_payload: int = 64,
        read_chunk_size: int = 19,
        runtime_arena_capacity: int = 4096,
        pmu_present: bool = True,
        pmu_counter_slots: int = FAKE_PMU_COUNTER_SLOTS,
        max_rx_payload: int = FAKE_MAX_RX_PAYLOAD,
        max_cases_per_session: int = FAKE_MAX_CASES_PER_SESSION,
        max_passes: int = FAKE_MAX_PASSES,
    ) -> None:
        self._session_id = 0xC0DE1234
        self._pmu_present = pmu_present
        self._pmu_counter_slots = pmu_counter_slots if pmu_present else 0
        self._max_rx_payload = max_rx_payload
        self._max_cases_per_session = max_cases_per_session
        self._max_passes = max_passes
        self._target_sequence_id = 0
        self._decoder = FrameDecoder(max_payload=4096)
        self._validator = SessionFrameValidator(session_id=self._session_id)
        self._state = _TargetState.WAIT_TARGET_INFO_ACK
        self._outbound = bytearray()
        self._max_frame_payload = max_frame_payload
        self._read_chunk_size = read_chunk_size
        self._runtime_arena_capacity = runtime_arena_capacity
        self._arena = ArenaTracker(runtime_arena_capacity)
        self._flash_count = 1
        self._rewind_count = 0
        self._completed_case_count = 0
        self._case_workspace_history: list[int] = []
        self._current_case_index = 0
        self._adapters = {adapter.entry.kernel_id: adapter for adapter in (FakeAbsS8Adapter(), FakeConvolveS8Adapter())}
        self._catalog = tuple(adapter.entry for adapter in self._adapters.values())
        self._plan: SessionPlan | None = None
        self._case_meta: CaseMeta | None = None
        self._blob_specs: dict[int, BlobDescriptor] = {}
        self._accumulators: dict[int, BlobAccumulator] = {}
        self._blob_order: list[int] = []
        self._blob_index = 0
        self._pending_offset = 0
        self._current_blob_id: int | None = None
        self._computed_output = b""
        self._last_iterations = 0
        self._emit_target_info()

    @property
    def rewind_count(self) -> int:
        return self._rewind_count

    @property
    def flash_count(self) -> int:
        return self._flash_count

    @property
    def completed_case_count(self) -> int:
        return self._completed_case_count

    @property
    def arena_used_bytes(self) -> int:
        return self._arena.used_bytes

    @property
    def case_workspace_history(self) -> tuple[int, ...]:
        return tuple(self._case_workspace_history)

    def close(self) -> None:
        self._outbound.clear()

    def read(self, max_bytes: int = 4096) -> bytes:
        if not self._outbound:
            return b""
        size = min(len(self._outbound), max_bytes, self._read_chunk_size)
        chunk = bytes(self._outbound[:size])
        del self._outbound[:size]
        return chunk

    def write(self, payload: bytes) -> None:
        frames = self._decoder.feed(payload)
        for frame in frames:
            self._validator.accept(frame)
            self._handle_frame(frame)

    def _queue(self, message_type: MessageType, payload: bytes = b"", *, flags: int = 0) -> None:
        self._outbound.extend(
            encode_frame(message_type, payload, session_id=self._session_id, sequence_id=self._target_sequence_id, flags=flags)
        )
        self._target_sequence_id += 1

    def _emit_target_info(self) -> None:
        """TARGET_INFO, field for field what hct_build_target_info_frame() emits."""
        flags = CAP_CASE_STREAMING | CAP_CORRECTNESS | CAP_PERFORMANCE | CAP_RTT_TRANSPORT | CAP_KERNEL_CATALOG | CAP_ABS_S8
        if self._pmu_present:
            flags |= CAP_PMU_ARMV8M
        info = TargetInfo(
            build_id="fake-benchmark-server",
            catalog_hash=kernel_catalog_hash(self._catalog),
            max_frame_payload=self._max_frame_payload,
            runtime_arena_capacity=self._runtime_arena_capacity,
            transfer_mode=1,
            output_mode=1,
            board_id="fake_board",
            target_cpu="cortex-m55" if self._pmu_present else "cortex-m4",
            transport_kind=1,
            capability_flags=flags,
            pmu_counter_slots=self._pmu_counter_slots,
            max_rx_payload=self._max_rx_payload,
            max_cases_per_session=self._max_cases_per_session,
            max_passes=self._max_passes,
        )
        self._queue(MessageType.TARGET_INFO, encode_target_info(info))

    def _emit_kernel_catalog(self) -> None:
        """Emit the (small, unpaginated) fake catalog as a single KERNEL_CATALOG frame
        with HCTP_FLAG_MORE clear, matching the real firmware's paginated protocol
        contract (a single final chunk is a valid one-chunk "page")."""
        self._queue(MessageType.KERNEL_CATALOG, encode_kernel_catalog(self._catalog))

    def _queue_error(self, message: str) -> None:
        self._queue(MessageType.ERROR, encode_error(ErrorPayload(message)))
        self._state = _TargetState.COMPLETE

    def _admit_session_plan(self, payload: bytes) -> SessionPlan:
        """Decode a SESSION_PLAN and apply the same admission rules as the firmware's
        handle_session_plan(): it must fit the receive buffer, name at most
        max_cases_per_session cases and max_passes passes, and no pass may need more
        PMU slots than the target has."""
        if len(payload) > self._max_rx_payload:
            raise ValueError(f"SESSION_PLAN payload {len(payload)} exceeds the fake target's rx buffer ({self._max_rx_payload}).")
        plan = decode_session_plan(payload)
        if not plan.cases or len(plan.cases) > self._max_cases_per_session:
            raise ValueError(f"SESSION_PLAN names {len(plan.cases)} cases; fake target takes at most {self._max_cases_per_session}.")
        if len(plan.passes) > self._max_passes:
            raise ValueError(f"SESSION_PLAN carries {len(plan.passes)} PMU passes; fake target takes at most {self._max_passes}.")
        for counter_pass in plan.passes:
            slots = counter_pass.slots_required
            if len(counter_pass.counters) > FAKE_MAX_COUNTERS_PER_PASS or (self._pmu_present and slots > self._pmu_counter_slots):
                raise ValueError(f"PMU pass {counter_pass.name!r} needs {slots} slots; fake target has {self._pmu_counter_slots}.")
        return plan

    def _handle_frame(self, frame: Frame) -> None:
        if frame.header.message_type == MessageType.TARGET_INFO_ACK:
            self._state = _TargetState.WAIT_PLAN
            self._emit_kernel_catalog()
            return
        if frame.header.message_type == MessageType.SESSION_PLAN:
            self._plan = self._admit_session_plan(frame.payload)
            self._state = _TargetState.WAIT_CASE_META
            self._request_case()
            return
        if frame.header.message_type == MessageType.CASE_META:
            self._case_meta = decode_case_meta(frame.payload)
            try:
                self._prepare_case()
            except CaseTooLargeError as exc:
                self._queue_error(str(exc))
                return
            self._state = _TargetState.WAIT_BLOB_CHUNK
            self._request_blob()
            return
        if frame.header.message_type == MessageType.BLOB_CHUNK:
            self._handle_blob_chunk(frame.payload)
            return
        if frame.header.message_type == MessageType.RUN_CORRECTNESS:
            self._run_correctness()
            self._state = _TargetState.WAIT_CORRECTNESS_ACK
            return
        if frame.header.message_type == MessageType.CORRECTNESS_ACK:
            self._state = _TargetState.WAIT_RUN_PERFORMANCE
            return
        if frame.header.message_type == MessageType.RUN_PERFORMANCE:
            self._run_performance()
            return
        raise ValueError(f"Unsupported fake-target message: {frame.header.message_type}")

    def _request_case(self) -> None:
        self._queue(MessageType.REQUEST_CASE, encode_request_case(RequestCase(self._current_case_index)))

    def _scalar_parameters(self) -> dict[str, Any]:
        """The case's scalars as the fake adapters read them: ints, except the
        VALID/SAME padding enum which crosses the wire as 0/1."""
        assert self._case_meta is not None
        scalars: dict[str, Any] = {}
        for key, value in self._case_meta.scalar_parameters:
            scalars[key] = {0: "VALID", 1: "SAME"}[value] if key == "padding" else value
        return scalars

    def _prepare_case(self) -> None:
        assert self._case_meta is not None
        self._blob_specs = {blob.blob_id: blob for blob in self._case_meta.blobs}
        self._blob_order = [blob.blob_id for blob in self._case_meta.blobs]
        self._blob_index = 0
        self._pending_offset = 0
        self._current_blob_id = self._blob_order[0]
        self._accumulators = {
            blob_id: BlobAccumulator(
                BlobTransferSpec(
                    blob_id=blob_id,
                    byte_length=spec.byte_length,
                    expected_crc32=spec.crc32,
                    required_alignment=max(1, spec.alignment),
                )
            )
            for blob_id, spec in self._blob_specs.items()
        }
        for spec in self._case_meta.blobs:
            self._arena.reserve_aligned(spec.byte_length, max(1, spec.alignment))
        scratch_bytes = self._case_meta.scratch_bytes
        output_bytes = int(self._scalar_parameters()["output_capacity_bytes"])
        if scratch_bytes:
            self._arena.reserve_aligned(scratch_bytes, 16)
        if output_bytes:
            self._arena.reserve_aligned(output_bytes, 16)

    def _request_blob(self) -> None:
        assert self._current_blob_id is not None
        spec = self._blob_specs[self._current_blob_id]
        remaining = spec.byte_length - self._pending_offset
        request = RequestBlob(blob_id=spec.blob_id, offset=self._pending_offset, max_length=min(self._max_frame_payload, remaining))
        self._queue(MessageType.REQUEST_BLOB, encode_request_blob(request))

    def _handle_blob_chunk(self, payload: bytes) -> None:
        chunk = decode_blob_chunk(payload)
        if chunk.blob_id != self._current_blob_id:
            raise ValueError("Unexpected blob id.")
        self._accumulators[chunk.blob_id].add_chunk(chunk.offset, chunk.data)
        self._pending_offset = chunk.offset + len(chunk.data)
        if self._pending_offset < self._blob_specs[chunk.blob_id].byte_length:
            self._request_blob()
            return
        self._accumulators[chunk.blob_id].finish()
        self._blob_index += 1
        if self._blob_index < len(self._blob_order):
            self._current_blob_id = self._blob_order[self._blob_index]
            self._pending_offset = 0
            self._request_blob()
            return
        self._queue(MessageType.CASE_READY, encode_case_ready(CaseReady(blob_id=chunk.blob_id, bytes_received=self._pending_offset)))
        self._state = _TargetState.WAIT_RUN_CORRECTNESS

    def _case_blobs(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for blob_id, spec in self._blob_specs.items():
            payload = self._accumulators[blob_id].finish()
            dtype = {"S8": np.int8, "S16": np.int16, "S32": np.int32}[spec.dtype]
            arrays[spec.role] = np.frombuffer(payload, dtype=dtype).reshape(spec.dimensions)
        return arrays

    def _adapter(self) -> FakeKernelAdapter:
        assert self._case_meta is not None
        return self._adapters[self._case_meta.kernel_id]

    def _run_correctness(self) -> None:
        assert self._case_meta is not None
        blobs = self._case_blobs()
        output = self._adapter().invoke(blobs, self._scalar_parameters())
        self._computed_output = output.tobytes(order="C")
        self._queue(MessageType.CORRECTNESS_RESULT, encode_correctness_result(CorrectnessResult(status=0)))
        self._queue(MessageType.OUTPUT_BEGIN, encode_output_begin(OutputBegin(length=len(self._computed_output))))
        chunk_size = 11
        for offset in range(0, len(self._computed_output), chunk_size):
            part = self._computed_output[offset : offset + chunk_size]
            self._queue(MessageType.OUTPUT_CHUNK, encode_output_chunk(OutputChunk(offset=offset, data=part)))
        end = OutputEnd(length=len(self._computed_output), checksum=output_checksum(self._computed_output))
        self._queue(MessageType.OUTPUT_END, encode_output_end(end))

    def _as_reported(self, counter: RawCounterValue) -> RawCounterValue:
        """A DWT-only target still reports the cycle entry but marks every event
        counter unsupported, exactly like the firmware's __PMU_PRESENT == 0 path."""
        supported = counter.supported and (self._pmu_present or counter.event_id == CPU_CYCLES_EVENT_ID)
        return RawCounterValue(
            name=counter.name,
            event_id=counter.event_id,
            value=counter.value if supported else 0,
            overflow=bool(supported and counter.overflow),
            supported=supported,
        )

    def _run_performance(self) -> None:
        assert self._plan is not None
        assert self._case_meta is not None
        adapter = self._adapter()
        passes = list(self._plan.passes) or [CounterPass(group="cpu", pass_index=0, counters=())]
        scalar_parameters = self._scalar_parameters()
        scalar_parameters["min_cycles"] = self._plan.min_cycles
        scalar_parameters["max_iterations"] = self._plan.max_iterations
        iterations, samples = adapter.measure(
            self._case_blobs(),
            scalar_parameters,
            warmups=self._plan.warmups,
            samples=self._plan.samples,
            iterations=self._plan.iterations_per_sample,
            counter_passes=passes,
        )
        self._last_iterations = iterations
        for sample in samples:
            reported = RawSample(
                sample_index=sample.sample_index,
                iterations=sample.iterations,
                cycles=sample.cycles,
                counters=tuple(self._as_reported(counter) for counter in sample.counters),
                pass_name=sample.pass_name,
            )
            self._queue(MessageType.SAMPLE_RESULT, encode_sample_result(reported))
        complete = CaseComplete(case_id=self._case_meta.case_id, workspace_used_bytes=self._arena.used_bytes)
        self._queue(MessageType.CASE_COMPLETE, encode_case_complete(complete))
        self._case_workspace_history.append(self._arena.used_bytes)
        self._arena.rewind()
        self._rewind_count += 1
        self._completed_case_count += 1
        self._current_case_index += 1
        if self._current_case_index < len(self._plan.cases):
            self._state = _TargetState.WAIT_CASE_META
            self._request_case()
            return
        self._queue(MessageType.SESSION_COMPLETE, encode_session_complete(SessionComplete(self._completed_case_count)))
        self._state = _TargetState.COMPLETE
