"""RttRings: the host's direct SEGGER RTT ring access.

The J-Link DLL's RTT engine dropped the tail of a SAMPLE_RESULT burst when the host
polled slowly (it advanced the target's read offset past bytes it never returned),
which stalled full-catalog runs. The host now drains the rings itself; these tests
pin the ring arithmetic against a fake target memory.
"""

from __future__ import annotations

import struct

import pytest

from helia_core_tester.hardware.transport import RTT_ID, RttRings

BLOCK = 0x2000_0000
UP_BUFFER = 0x2000_1000
DOWN_BUFFER = 0x2000_2000


class _Memory:
    """Byte-addressed fake of pylink's memory calls."""

    def __init__(self) -> None:
        self.bytes: dict[int, int] = {}

    def memory_read(self, address: int, count: int) -> list[int]:
        return [self.bytes.get(address + i, 0) for i in range(count)]

    def memory_write(self, address: int, data: list[int]) -> None:
        for i, value in enumerate(data):
            self.bytes[address + i] = value

    def memory_read32(self, address: int, count: int) -> list[int]:
        raw = bytes(self.memory_read(address, 4 * count))
        return list(struct.unpack(f"<{count}I", raw))

    def memory_write32(self, address: int, words: list[int]) -> None:
        self.memory_write(address, list(struct.pack(f"<{len(words)}I", *words)))


def _target(*, up_size: int = 16, down_size: int = 8, max_up: int = 3, max_down: int = 3) -> _Memory:
    memory = _Memory()
    memory.memory_write(BLOCK, list(RTT_ID))
    memory.memory_write32(BLOCK + 16, [max_up, max_down])
    _set_ring(memory, 0, UP_BUFFER, up_size, 0, 0)
    _set_ring(memory, max_up, DOWN_BUFFER, down_size, 0, 0)
    return memory


def _set_ring(memory: _Memory, slot: int, buffer: int, size: int, write: int, read: int) -> None:
    memory.memory_write32(BLOCK + 24 + 24 * slot, [0, buffer, size, write, read, 0])


def _offsets(memory: _Memory, slot: int) -> tuple[int, int]:
    _, _, _, write, read, _ = memory.memory_read32(BLOCK + 24 + 24 * slot, 6)
    return write, read


def test_take_drains_a_wrapped_ring_and_frees_it() -> None:
    memory = _target()
    memory.memory_write(UP_BUFFER, list(b"WXYZ" + bytes(8) + b"abcd"))
    # Target wrote "abcd" at 12..15, then "WXYZ" at 0..3.
    _set_ring(memory, 0, UP_BUFFER, 16, write=4, read=12)
    rings = RttRings(memory, BLOCK)
    assert rings.take(4096) == b"abcdWXYZ"
    assert _offsets(memory, 0) == (4, 4)
    assert rings.take(4096) == b""


def test_take_stops_at_max_bytes_and_keeps_the_rest() -> None:
    memory = _target()
    memory.memory_write(UP_BUFFER, list(b"0123456789"))
    _set_ring(memory, 0, UP_BUFFER, 16, write=10, read=0)
    rings = RttRings(memory, BLOCK)
    assert rings.take(6) == b"012345"
    assert rings.take(6) == b"6789"
    assert _offsets(memory, 0) == (10, 10)


def test_put_keeps_one_slot_free_and_wraps() -> None:
    memory = _target(down_size=8)
    _set_ring(memory, 3, DOWN_BUFFER, 8, write=6, read=3)
    rings = RttRings(memory, BLOCK)
    # Two bytes reach the wrap point, then 0..1 (slot 2 stays free).
    assert rings.put(b"abcdef") == 2
    assert rings.put(b"cdef") == 2
    assert rings.put(b"ef") == 0
    assert bytes(memory.memory_read(DOWN_BUFFER, 8))[:2] == b"cd"
    assert bytes(memory.memory_read(DOWN_BUFFER + 6, 2)) == b"ab"
    assert _offsets(memory, 3) == (2, 3)


def test_rings_wait_for_the_firmware_block() -> None:
    memory = _Memory()
    rings = RttRings(memory, BLOCK)
    assert rings.take(4096) == b"" and rings.put(b"x") == 0
    for address, value in enumerate(_target().memory_read(BLOCK, 0x200)):
        memory.bytes[BLOCK + address] = value
    memory.memory_write(UP_BUFFER, list(b"hi"))
    _set_ring(memory, 0, UP_BUFFER, 16, write=2, read=0)
    assert rings.take(4096) == b"hi"


def test_rings_refuse_a_missing_channel_and_corrupt_offsets() -> None:
    with pytest.raises(RuntimeError, match="lacks the requested channels"):
        RttRings(_target(max_up=1), BLOCK, up_index=1).take(1)
    memory = _target()
    _set_ring(memory, 0, UP_BUFFER, 16, write=99, read=0)
    assert RttRings(memory, BLOCK).take(4096) == b""

