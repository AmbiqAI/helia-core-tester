"""Bounded blob transfer validation utilities for HCTP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
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


# Mirror allocate_blob() MRAM row padding.
MRAM_ROW_BYTES = 16
MRAM_ROLES = frozenset({"weights", "bias"})
# Placements that program weights and bias into MRAM.
MRAM_PLACEMENTS = frozenset({"mram", "sram"})


# cmake/hardware/benchmark_server_mram.h: placement pool and cache line.
MRAM_LINE_BYTES = 32
MRAM_POOL_BYTES = 512 * 1024


def mram_extent(role: str, byte_length: int, placement: str) -> int:
    """MRAM pool bytes a blob takes (rows padded, from a cache line); 0 when it stays in the workspace."""
    if placement not in MRAM_PLACEMENTS or role not in MRAM_ROLES:
        return 0
    padded = -(-byte_length // MRAM_ROW_BYTES) * MRAM_ROW_BYTES
    return -(-padded // MRAM_LINE_BYTES) * MRAM_LINE_BYTES


# HCT_KERNEL_ID_FULLY_CONNECTED_PER_CHANNEL_PACKED_S8: the one kernel with a derived operand.
PACKED_FC_S8_KERNEL_ID = 182
_INT32_MAX = 2**31 - 1


def packed_fc_s8_size(k: int, n: int) -> int:
    """arm_fully_connected_per_channel_packed_s8_get_packed_size() for depth k, n outputs; 0 when unsupported."""
    if k <= 0 or n <= 0:
        return 0
    size = (n + 3) // 4 * (4 * (-(-k // 16) * 16) + 48)
    return 0 if size > _INT32_MAX else size


def derived_mram_extent(kernel_id: int, blobs: Iterable[Any], placement: str) -> int:
    """MRAM pool bytes of the operand the adapter derives (hct_derived_operand_bytes()); 0 when none."""
    if placement not in MRAM_PLACEMENTS or kernel_id != PACKED_FC_S8_KERNEL_ID:
        return 0
    weights = next((blob for blob in blobs if blob.role == "weights"), None)
    if weights is None:
        return 0
    dims = tuple(weights.dimensions) + (0, 0)
    return mram_extent("weights", packed_fc_s8_size(int(dims[1]), int(dims[0])), placement)


def staged_extent(role: str, byte_length: int, alignment: int, placement: str) -> tuple[int, int]:
    """Blob's firmware workspace length and alignment."""
    alignment = max(alignment, 1)
    if placement not in MRAM_PLACEMENTS or role not in MRAM_ROLES:
        return byte_length, alignment
    padded = -(-byte_length // MRAM_ROW_BYTES) * MRAM_ROW_BYTES
    return padded, max(alignment, MRAM_ROW_BYTES)


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
