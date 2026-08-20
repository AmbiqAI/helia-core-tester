"""Fake HCTP target used for host-side streaming tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any

import numpy as np

from helia_core_tester.generation.utils.tflite_utils import requantize_np

from .hctp import ByteReader, ByteWriter, Frame, FrameDecoder, MessageType, SessionFrameValidator, encode_frame
from .measurement import (
    CounterDescriptor,
    DEFAULT_COUNTERS,
    CounterPass,
    RawCounterValue,
    RawSample,
    auto_calibrate_iterations,
    plan_counter_passes,
    resolve_counter_selection,
)
from .transfer import ArenaTracker, BlobAccumulator, BlobTransferSpec, CaseTooLargeError


def _catalog_hash(entries: list[dict[str, Any]]) -> bytes:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).digest()


class _TargetState(str, Enum):
    WAIT_HELLO_ACK = "wait_hello_ack"
    WAIT_PLAN = "wait_plan"
    WAIT_CASE_META = "wait_case_meta"
    WAIT_BLOB_CHUNK = "wait_blob_chunk"
    WAIT_RUN_CORRECTNESS = "wait_run_correctness"
    WAIT_CORRECTNESS_ACK = "wait_correctness_ack"
    WAIT_RUN_PERFORMANCE = "wait_run_performance"
    COMPLETE = "complete"


@dataclass(frozen=True)
class KernelCatalogEntry:
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


class FakeKernelAdapter:
    entry: KernelCatalogEntry
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
                counters = []
                for counter in counter_pass.counters:
                    supported = counter.group in self.supported_groups
                    overflow = counter.name.endswith("INST_RETIRED") and sample_index == samples - 1 and counter_pass.pass_index > 0
                    value = self._counter_value(counter, blobs, iterations, sample_index)
                    counters.append(
                        RawCounterValue(
                            name=counter.name,
                            event_id=counter.event_id,
                            value=value,
                            overflow=overflow,
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
    entry = KernelCatalogEntry(1, "arm_abs_s8", "BasicMathFunctions", 1, "S8", 1, True, True, False, 0)
    supported_groups = ("cpu",)
    base_cycles_per_iteration = 3

    def invoke(self, blobs: dict[str, np.ndarray], scalar_parameters: dict[str, Any]) -> np.ndarray:
        return np.abs(blobs["input_0"].astype(np.int16)).astype(np.int8)


class FakeConvolveS8Adapter(FakeKernelAdapter):
    entry = KernelCatalogEntry(2, "arm_convolve_s8", "ConvolutionFunctions", 1, "S8", 1, True, True, False, 64)
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

    def __init__(self, *, max_frame_payload: int = 64, read_chunk_size: int = 19, runtime_arena_capacity: int = 4096) -> None:
        self._session_id = 0xC0DE1234
        self._target_sequence_id = 0
        self._decoder = FrameDecoder(max_payload=4096)
        self._validator = SessionFrameValidator(session_id=self._session_id)
        self._state = _TargetState.WAIT_HELLO_ACK
        self._outbound = bytearray()
        self._max_frame_payload = max_frame_payload
        self._read_chunk_size = read_chunk_size
        self._runtime_arena_capacity = runtime_arena_capacity
        self._arena = ArenaTracker(runtime_arena_capacity)
        self._flash_count = 1
        self._rewind_count = 0
        self._completed_case_count = 0
        self._current_case_index = 0
        self._catalog_entries = {adapter.entry.kernel_id: adapter for adapter in (FakeAbsS8Adapter(), FakeConvolveS8Adapter())}
        self._catalog = [self._catalog_dict(adapter.entry) for adapter in self._catalog_entries.values()]
        self._catalog_hash = _catalog_hash(self._catalog)
        self._plan: dict[str, Any] | None = None
        self._case_meta: dict[str, Any] | None = None
        self._blob_specs: dict[int, dict[str, Any]] = {}
        self._accumulators: dict[int, BlobAccumulator] = {}
        self._blob_order: list[int] = []
        self._blob_index = 0
        self._pending_offset = 0
        self._current_blob_id: int | None = None
        self._computed_output = b""
        self._last_iterations = 0
        self._emit_hello()

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

    def _catalog_dict(self, entry: KernelCatalogEntry) -> dict[str, Any]:
        return {
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

    def _queue(self, message_type: MessageType, payload: bytes = b"", *, flags: int = 0) -> None:
        self._outbound.extend(
            encode_frame(message_type, payload, session_id=self._session_id, sequence_id=self._target_sequence_id, flags=flags)
        )
        self._target_sequence_id += 1

    def _emit_hello(self) -> None:
        writer = ByteWriter()
        writer.text("fake-benchmark-server")
        writer.fixed(self._catalog_hash)
        writer.u32(self._max_frame_payload)
        writer.u32(self._runtime_arena_capacity)
        writer.u8(1)
        writer.u8(1)
        self._queue(MessageType.HELLO, writer.finish())

    def _emit_capabilities(self) -> None:
        """F008: emit the (small, unpaginated) fake catalog as a single CAPABILITIES
        frame with HCTP_FLAG_MORE clear, matching the real firmware's paginated
        protocol contract (a single final chunk is a valid one-chunk "page")."""
        writer = ByteWriter()
        writer.u16(len(self._catalog))
        for entry in self._catalog:
            writer.u32(int(entry["kernel_id"]))
            writer.text(str(entry["canonical_name"]))
            writer.text(str(entry["operator_family"]))
            writer.u16(int(entry["api_version"]))
            writer.text(str(entry["supported_dtype"]))
            writer.u16(int(entry["adapter_schema_version"]))
            writer.u8(1 if entry["stateless"] else 0)
            writer.u8(1 if entry["repeated_invocation_safe"] else 0)
            writer.u8(1 if entry["mutates_input"] else 0)
            writer.u32(int(entry["scratch_bytes"]))
        self._queue(MessageType.CAPABILITIES, writer.finish())

    def _queue_error(self, message: str) -> None:
        writer = ByteWriter()
        writer.text(message)
        self._queue(MessageType.ERROR, writer.finish())
        self._state = _TargetState.COMPLETE

    def _decode_plan(self, payload: bytes) -> dict[str, Any]:
        reader = ByteReader(payload)
        case_count = reader.u16()
        transfer_mode = reader.u8()
        warmups = reader.u16()
        samples = reader.u16()
        iterations = reader.u32()
        min_cycles = reader.u32()
        max_iterations = reader.u32()
        requested_groups = []
        group_count = reader.u8()
        for _ in range(group_count):
            requested_groups.append(reader.text())
        cases = []
        for _ in range(case_count):
            cases.append({"case_id": reader.text(), "kernel_id": reader.u32()})
        return {
            "transfer_mode": transfer_mode,
            "warmups": warmups,
            "samples": samples,
            "iterations": iterations,
            "min_cycles": min_cycles,
            "max_iterations": max_iterations,
            "requested_groups": requested_groups,
            "cases": cases,
        }

    def _decode_case_meta(self, payload: bytes) -> dict[str, Any]:
        reader = ByteReader(payload)
        case_id = reader.text()
        kernel_id = reader.u32()
        schema_version = reader.u16()
        comparison_mode = reader.u8()
        tolerance = reader.i32()
        atol_q16 = reader.u32()
        rtol_q16 = reader.u32()
        scalar_parameter_count = reader.u8()
        scalar_parameters = {}
        for _ in range(scalar_parameter_count):
            key = reader.text()
            value = reader.i32()
            if key == "padding":
                scalar_parameters[key] = {0: "VALID", 1: "SAME"}[value]
            else:
                scalar_parameters[key] = value
        blob_count = reader.u16()
        blobs = []
        for _ in range(blob_count):
            blob_id = reader.u32()
            role = reader.text()
            dtype = reader.text()
            rank = reader.u8()
            dims = tuple(reader.u32() for _ in range(6))[:rank]
            byte_length = reader.u32()
            alignment = reader.u32()
            crc32_value = reader.u32()
            flags = reader.u8()
            blobs.append({
                "blob_id": blob_id,
                "role": role,
                "dtype": dtype,
                "dimensions": dims,
                "byte_length": byte_length,
                "alignment": alignment,
                "crc32": crc32_value,
                "flags": flags,
            })
        scratch_bytes = reader.u32()
        return {
            "case_id": case_id,
            "kernel_id": kernel_id,
            "schema_version": schema_version,
            "comparison_mode": comparison_mode,
            "tolerance": tolerance,
            "atol_q16": atol_q16,
            "rtol_q16": rtol_q16,
            "scalar_parameters": scalar_parameters,
            "blobs": blobs,
            "scratch_bytes": scratch_bytes,
        }

    def _handle_frame(self, frame: Frame) -> None:
        if frame.header.message_type == MessageType.HELLO_ACK:
            self._state = _TargetState.WAIT_PLAN
            self._emit_capabilities()
            return
        if frame.header.message_type == MessageType.LOAD_PLAN:
            self._plan = self._decode_plan(frame.payload)
            self._state = _TargetState.WAIT_CASE_META
            self._request_case()
            return
        if frame.header.message_type == MessageType.CASE_META:
            self._case_meta = self._decode_case_meta(frame.payload)
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
        writer = ByteWriter()
        writer.u16(self._current_case_index)
        self._queue(MessageType.REQUEST_CASE, writer.finish())

    def _prepare_case(self) -> None:
        assert self._case_meta is not None
        self._blob_specs = {blob["blob_id"]: blob for blob in self._case_meta["blobs"]}
        self._blob_order = [blob["blob_id"] for blob in self._case_meta["blobs"]]
        self._blob_index = 0
        self._pending_offset = 0
        self._current_blob_id = self._blob_order[0]
        self._accumulators = {
            blob_id: BlobAccumulator(
                BlobTransferSpec(
                    blob_id=blob_id,
                    byte_length=int(spec["byte_length"]),
                    expected_crc32=int(spec["crc32"]),
                    required_alignment=max(1, int(spec["alignment"])),
                )
            )
            for blob_id, spec in self._blob_specs.items()
        }
        self._arena.reserve(int(self._case_meta["scratch_bytes"]))
        self._arena.reserve(sum(int(spec["byte_length"]) for spec in self._blob_specs.values()))

    def _request_blob(self) -> None:
        assert self._current_blob_id is not None
        spec = self._blob_specs[self._current_blob_id]
        remaining = int(spec["byte_length"]) - self._pending_offset
        writer = ByteWriter()
        writer.u32(int(spec["blob_id"]))
        writer.u32(self._pending_offset)
        writer.u16(min(self._max_frame_payload, remaining))
        self._queue(MessageType.REQUEST_BLOB, writer.finish())

    def _handle_blob_chunk(self, payload: bytes) -> None:
        reader = ByteReader(payload)
        blob_id = reader.u32()
        offset = reader.u32()
        chunk = reader.raw()
        if blob_id != self._current_blob_id:
            raise ValueError("Unexpected blob id.")
        self._accumulators[blob_id].add_chunk(offset, chunk)
        self._pending_offset = offset + len(chunk)
        if self._pending_offset < self._blob_specs[blob_id]["byte_length"]:
            self._request_blob()
            return
        self._accumulators[blob_id].finish()
        self._blob_index += 1
        if self._blob_index < len(self._blob_order):
            self._current_blob_id = self._blob_order[self._blob_index]
            self._pending_offset = 0
            self._request_blob()
            return
        writer = ByteWriter()
        writer.u32(blob_id)
        writer.u32(self._pending_offset)
        self._queue(MessageType.CASE_READY, writer.finish())
        self._state = _TargetState.WAIT_RUN_CORRECTNESS

    def _case_blobs(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for blob_id, spec in self._blob_specs.items():
            payload = self._accumulators[blob_id].finish()
            dtype = {"S8": np.int8, "S16": np.int16, "S32": np.int32}[spec["dtype"]]
            arrays[str(spec["role"])] = np.frombuffer(payload, dtype=dtype).reshape(spec["dimensions"])
        return arrays

    def _adapter(self) -> FakeKernelAdapter:
        assert self._case_meta is not None
        return self._catalog_entries[int(self._case_meta["kernel_id"])]

    def _run_correctness(self) -> None:
        assert self._case_meta is not None
        blobs = self._case_blobs()
        output = self._adapter().invoke(blobs, self._case_meta["scalar_parameters"])
        self._computed_output = output.tobytes(order="C")
        result_writer = ByteWriter()
        result_writer.i32(0)
        self._queue(MessageType.CORRECTNESS_RESULT, result_writer.finish())
        begin_writer = ByteWriter()
        begin_writer.u32(0)
        begin_writer.u32(len(self._computed_output))
        self._queue(MessageType.OUTPUT_BEGIN, begin_writer.finish())
        chunk_size = 11
        for offset in range(0, len(self._computed_output), chunk_size):
            part = self._computed_output[offset : offset + chunk_size]
            writer = ByteWriter()
            writer.u32(offset)
            writer.raw(part)
            self._queue(MessageType.OUTPUT_CHUNK, writer.finish())
        end_writer = ByteWriter()
        end_writer.u32(len(self._computed_output))
        end_writer.u32(np.uint32(np.frombuffer(self._computed_output, dtype=np.uint8).sum()).item())
        self._queue(MessageType.OUTPUT_END, end_writer.finish())

    def _run_performance(self) -> None:
        assert self._plan is not None
        assert self._case_meta is not None
        adapter = self._adapter()
        counters = resolve_counter_selection({group: "default" for group in self._plan["requested_groups"] or ["cpu"]})
        passes = plan_counter_passes(counters)
        scalar_parameters = dict(self._case_meta["scalar_parameters"])
        scalar_parameters["min_cycles"] = int(self._plan["min_cycles"])
        scalar_parameters["max_iterations"] = int(self._plan["max_iterations"])
        iterations, samples = adapter.measure(
            self._case_blobs(),
            scalar_parameters,
            warmups=int(self._plan["warmups"]),
            samples=int(self._plan["samples"]),
            iterations=int(self._plan["iterations"]),
            counter_passes=passes,
        )
        self._last_iterations = iterations
        for sample in samples:
            writer = ByteWriter()
            writer.u16(sample.sample_index)
            writer.u32(sample.iterations)
            writer.u64(sample.cycles)
            writer.text(sample.pass_name)
            writer.u8(len(sample.counters))
            for counter in sample.counters:
                writer.text(counter.name)
                writer.u16(counter.event_id)
                writer.u64(counter.value)
                writer.u8(1 if counter.overflow else 0)
                writer.u8(1 if counter.supported else 0)
            self._queue(MessageType.SAMPLE_RESULT, writer.finish())
        complete_writer = ByteWriter()
        complete_writer.text(str(self._case_meta["case_id"]))
        complete_writer.u8(1)
        complete_writer.u8(1)
        complete_writer.u32(self._arena.used_bytes)
        self._queue(MessageType.CASE_COMPLETE, complete_writer.finish())
        self._arena.rewind()
        self._rewind_count += 1
        self._completed_case_count += 1
        self._current_case_index += 1
        if self._current_case_index < len(self._plan["cases"]):
            self._state = _TargetState.WAIT_CASE_META
            self._request_case()
            return
        session_writer = ByteWriter()
        session_writer.u16(self._completed_case_count)
        self._queue(MessageType.SESSION_COMPLETE, session_writer.finish())
        self._state = _TargetState.COMPLETE
