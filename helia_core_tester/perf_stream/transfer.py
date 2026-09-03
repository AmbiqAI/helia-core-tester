"""Bounded blob transfer validation utilities for HCTP."""

from __future__ import annotations

from dataclasses import dataclass
import zlib


class TransferError(ValueError):
    pass


class DuplicateChunkError(TransferError):
    pass


class MissingChunkError(TransferError):
    pass


class OverlappingChunkError(TransferError):
    pass


class OutOfRangeChunkError(TransferError):
    pass


class BlobCrcMismatchError(TransferError):
    pass


class AlignmentError(TransferError):
    pass


class CaseTooLargeError(TransferError):
    pass


@dataclass(frozen=True)
class BlobTransferSpec:
    blob_id: int
    byte_length: int
    expected_crc32: int
    required_alignment: int = 1


class BlobAccumulator:
    def __init__(self, spec: BlobTransferSpec) -> None:
        self._spec = spec
        self._data = bytearray(spec.byte_length)
        self._received = [False] * spec.byte_length
        self._received_ranges: list[tuple[int, int]] = []

    def add_chunk(self, offset: int, payload: bytes) -> None:
        if offset < 0 or offset + len(payload) > self._spec.byte_length:
            raise OutOfRangeChunkError(
                f"Chunk [{offset}, {offset + len(payload)}) exceeds blob size {self._spec.byte_length}."
            )
        if self._spec.required_alignment > 1 and offset % self._spec.required_alignment != 0:
            raise AlignmentError(
                f"Chunk offset {offset} is not aligned to {self._spec.required_alignment} bytes."
            )
        start = offset
        end = offset + len(payload)
        if any(existing_start == start and existing_end == end for existing_start, existing_end in self._received_ranges):
            raise DuplicateChunkError(f"Duplicate chunk [{start}, {end}) for blob {self._spec.blob_id}.")
        for existing_start, existing_end in self._received_ranges:
            if start < existing_end and end > existing_start:
                raise OverlappingChunkError(
                    f"Chunk [{start}, {end}) overlaps existing [{existing_start}, {existing_end})."
                )
        self._data[start:end] = payload
        for index in range(start, end):
            self._received[index] = True
        self._received_ranges.append((start, end))
        self._received_ranges.sort()

    def is_complete(self) -> bool:
        return all(self._received)

    def finish(self) -> bytes:
        if not self.is_complete():
            missing = next(index for index, received in enumerate(self._received) if not received)
            raise MissingChunkError(f"Blob {self._spec.blob_id} is missing data at offset {missing}.")
        payload = bytes(self._data)
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != self._spec.expected_crc32:
            raise BlobCrcMismatchError(
                f"Blob {self._spec.blob_id} CRC mismatch: expected 0x{self._spec.expected_crc32:08x}, got 0x{actual_crc:08x}."
            )
        return payload


class ArenaTracker:
    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.used_bytes = 0

    def reserve(self, bytes_required: int) -> None:
        if self.used_bytes + bytes_required > self.capacity_bytes:
            raise CaseTooLargeError(
                f"Case requires {self.used_bytes + bytes_required} bytes, exceeds arena capacity {self.capacity_bytes}."
            )
        self.used_bytes += bytes_required

    def reserve_aligned(self, bytes_required: int, alignment: int) -> int:
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        aligned = (self.used_bytes + alignment - 1) & ~(alignment - 1)
        end = aligned + bytes_required
        if end > self.capacity_bytes:
            raise CaseTooLargeError(
                f"Case requires {end} bytes, exceeds arena capacity {self.capacity_bytes}."
            )
        self.used_bytes = end
        return aligned

    def rewind(self) -> None:
        self.used_bytes = 0
