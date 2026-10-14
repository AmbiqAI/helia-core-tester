"""Direct SEGGER RTT ring access against fake target memory."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from helia_core_tester.hardware.nsx_app import RTT_BUFFER_SIZE_DOWN, RTT_BUFFER_SIZE_UP
from helia_core_tester.hardware.transport import (
    RTT_ID,
    SCB_CFSR,
    JLinkRttTransport,
    RttBufferSizeError,
    RttRings,
    TransportError,
)

BLOCK = 0x2000_0000
UP_BUFFER = 0x2000_1000
DOWN_BUFFER = 0x2000_2000


class _Memory:
    """Byte-addressed fake of pylink's memory calls."""

    def __init__(self) -> None:
        self.bytes: dict[int, int] = {}

    def memory_read(self, address: int, count: int) -> list[int]:
        return [self.bytes.get(address + i, 0) for i in range(count)]

    def memory_write(self, address: int, data: list[int]) -> int:
        for i, value in enumerate(data):
            self.bytes[address + i] = value
        return len(data)

    def memory_read32(self, address: int, count: int) -> list[int]:
        raw = bytes(self.memory_read(address, 4 * count))
        return list(struct.unpack(f"<{count}I", raw))

    def memory_write32(self, address: int, words: list[int]) -> int:
        # Like pylink on a real probe: bytes written.
        return self.memory_write(address, list(struct.pack(f"<{len(words)}I", *words)))


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
    # Two bytes to the wrap, two from 0; slot 2 stays free.
    assert rings.put(b"abcdef") == 4
    assert rings.put(b"ef") == 0
    assert bytes(memory.memory_read(DOWN_BUFFER, 8))[:2] == b"cd"
    assert bytes(memory.memory_read(DOWN_BUFFER + 6, 2)) == b"ab"
    assert _offsets(memory, 3) == (2, 3)


def test_rings_wait_for_the_firmware_block() -> None:
    memory = _Memory()
    rings = RttRings(memory, BLOCK)
    assert rings.take(4096) == b"" and rings.put(b"x") == 0 and rings.up_offsets() is None
    for address, value in enumerate(_target().memory_read(BLOCK, 0x200)):
        memory.bytes[BLOCK + address] = value
    assert rings.up_offsets() == (16, 0, 0)


def test_rings_refuse_a_missing_channel_and_corrupt_offsets() -> None:
    with pytest.raises(RuntimeError, match="lacks the requested channels"):
        RttRings(_target(max_up=1), BLOCK, up_index=1).take(1)
    memory = _target()
    _set_ring(memory, 0, UP_BUFFER, 16, write=99, read=0)
    assert RttRings(memory, BLOCK).take(4096) == b""


class _Probe(_Memory):
    """Fake J-Link core: registers plus memory."""

    def __init__(self) -> None:
        super().__init__()
        self.names = ["R13 (SP)", "R14", "R15 (PC)"]
        self.values = [0x20003F00, -7, 0x004967E0]  # pylink returns signed ints.
        self.calls: list[str] = []

    def register_list(self) -> list[int]:
        return list(range(len(self.names)))

    def register_name(self, index: int) -> str:
        return self.names[index]

    def register_read(self, index: int) -> int:
        return self.values[index]

    def halt(self) -> bool:
        self.calls.append("halt")
        return True

    def restart(self) -> None:
        self.calls.append("restart")


def test_target_state_reads_unsigned_registers_and_resumes() -> None:
    probe = _Probe()
    for address, value in enumerate(_target().memory_read(BLOCK, 0x200)):
        probe.bytes[BLOCK + address] = value
    probe.memory_write32(SCB_CFSR, [0x100, 0x40000000])
    transport = JLinkRttTransport.__new__(JLinkRttTransport)
    transport._jlink = probe
    transport._rings = RttRings(probe, BLOCK)
    assert transport.target_state() == {
        "pc": 0x004967E0, "lr": 0xFFFFFFF9, "sp": 0x20003F00, "cfsr": 0x100, "hfsr": 0x40000000,
        "rtt_size": 16, "rtt_write": 0, "rtt_read": 0,
    }
    assert probe.calls == ["halt", "restart"]


def test_take_stops_at_a_short_read() -> None:
    memory = _target()
    memory.memory_write(UP_BUFFER, list(b"WX" + bytes(10) + b"abcd"))
    _set_ring(memory, 0, UP_BUFFER, 16, write=2, read=12)
    rings = RttRings(memory, BLOCK)
    assert rings.up_offsets() == (16, 2, 12)
    full_read = memory.memory_read
    # The probe returns 3 bytes of each ring read.
    memory.memory_read = lambda address, count: full_read(address, min(count, 3) if address >= UP_BUFFER else count)  # type: ignore[method-assign]
    # Only acknowledge bytes actually copied, in order.
    assert rings.take(4096) == b"abc"
    assert _offsets(memory, 0) == (2, 15)


def test_target_state_refuses_a_running_core() -> None:
    probe = _Probe()
    probe.halt = lambda: False  # type: ignore[method-assign]
    transport = JLinkRttTransport.__new__(JLinkRttTransport)
    transport._jlink = probe
    transport._rings = RttRings(probe, BLOCK)
    with pytest.raises(RuntimeError, match="Core did not halt"):
        transport.target_state()


@pytest.mark.parametrize(
    ("ring", "short"),
    [("up", "memory_write32"), ("down", "memory_write32"), ("down", "memory_write")],
    ids=["up-rdoff", "down-wroff", "down-data"],
)
def test_short_write_fails_the_transport(ring: str, short: str) -> None:
    memory = _target()
    memory.memory_write(UP_BUFFER, list(b"abcd"))
    _set_ring(memory, 0, UP_BUFFER, 16, write=4, read=0)
    rings = RttRings(memory, BLOCK)
    assert rings.up_offsets() == (16, 4, 0)
    # The probe reports nothing written.
    setattr(memory, short, lambda address, data: 0)
    with pytest.raises(TransportError, match=r"^Short RTT write at 0x"):
        rings.take(4096) if ring == "up" else rings.put(b"xy")


def test_short_descriptor_read_fails_the_transport() -> None:
    memory = _target()
    rings = RttRings(memory, BLOCK)
    assert rings.up_offsets() == (16, 0, 0)
    memory.memory_read32 = lambda address, count: []  # type: ignore[method-assign]
    with pytest.raises(TransportError, match=r"^Short RTT read at 0x20000018: 0/6 words\.$"):
        rings.take(4096)


def test_rings_accept_the_expected_buffer_sizes() -> None:
    memory = _target(up_size=16, down_size=8)
    rings = RttRings(memory, BLOCK, expected_up_size=16, expected_down_size=8)
    assert rings.up_offsets() == (16, 0, 0)


@pytest.mark.parametrize(
    ("up_size", "down_size", "message"),
    [
        (1024, 8, r"^RTT up buffer 0 at 0x20000000 is 1024 bytes, expected 16;"),
        (15, 8, r"^RTT up buffer 0 at 0x20000000 is 15 bytes, expected 16;"),
        (17, 8, r"^RTT up buffer 0 at 0x20000000 is 17 bytes, expected 16;"),
        (16, 16, r"^RTT down buffer 0 at 0x20000000 is 16 bytes, expected 8;"),
        (16, 0, r"^RTT down buffer 0 at 0x20000000 is 0 bytes, expected 8;"),
    ],
    ids=["up-default-1k", "up-minus-1", "up-plus-1", "down-default-16", "down-zero"],
)
def test_rings_refuse_unexpected_buffer_sizes(up_size: int, down_size: int, message: str) -> None:
    memory = _target(up_size=up_size, down_size=down_size)
    memory.memory_write(UP_BUFFER, list(b"abcd"))
    _set_ring(memory, 0, UP_BUFFER, up_size, write=4 % max(up_size, 1), read=0)
    before = dict(memory.bytes)
    rings = RttRings(memory, BLOCK, expected_up_size=16, expected_down_size=8)
    for access in (lambda: rings.take(4096), lambda: rings.put(b"x"), rings.up_offsets):
        with pytest.raises(RttBufferSizeError, match=message):
            access()
    # Fail closed: no offset or data was touched.
    assert memory.bytes == before


def test_size_mismatch_reads_no_ring_offsets() -> None:
    memory = _target(up_size=1024, down_size=8)
    reads: list[tuple[int, int]] = []
    read32 = memory.memory_read32
    memory.memory_read32 = lambda address, count: reads.append((address, count)) or read32(address, count)  # type: ignore[method-assign]
    rings = RttRings(memory, BLOCK, expected_up_size=16, expected_down_size=8)
    with pytest.raises(RttBufferSizeError):
        rings.up_offsets()
    up = BLOCK + 24
    words = {address + 4 * i for address, count in reads for i in range(count)}
    assert up + 8 in words
    # WrOff, RdOff (and pBuffer) stay unread.
    assert words.isdisjoint({up + 4, up + 12, up + 16})


def test_buffer_size_error_is_not_a_transport_stall() -> None:
    # A mis-built image is not a stall: no target-state dump, no retry as one.
    assert issubclass(RttBufferSizeError, RuntimeError)
    assert not issubclass(RttBufferSizeError, TransportError)


def test_rings_skip_the_size_check_when_not_requested() -> None:
    assert RttRings(_target(up_size=1024, down_size=16), BLOCK).up_offsets() == (1024, 0, 0)


def test_rings_check_only_the_requested_direction() -> None:
    memory = _target(up_size=1024, down_size=8)
    assert RttRings(memory, BLOCK, expected_down_size=8).up_offsets() == (1024, 0, 0)
    with pytest.raises(RttBufferSizeError, match="up buffer"):
        RttRings(memory, BLOCK, expected_up_size=16).up_offsets()


def test_rings_check_the_selected_channel() -> None:
    memory = _target(up_size=16, down_size=8)
    _set_ring(memory, 1, UP_BUFFER, 64, 0, 0)
    _set_ring(memory, 3 + 2, DOWN_BUFFER, 32, 0, 0)
    rings = RttRings(memory, BLOCK, up_index=1, down_index=2, expected_up_size=64, expected_down_size=32)
    assert rings.up_offsets() == (64, 0, 0)
    with pytest.raises(RttBufferSizeError, match=r"^RTT up buffer 1 .* is 64 bytes, expected 16;"):
        RttRings(memory, BLOCK, up_index=1, expected_up_size=16).up_offsets()


@pytest.mark.parametrize("bad", [0, 1, -8192, True, False, 8192.0, "8192"], ids=repr)
@pytest.mark.parametrize("field", ["expected_up_size", "expected_down_size"])
def test_rings_reject_invalid_expected_sizes(field: str, bad: object) -> None:
    with pytest.raises(ValueError, match=rf"^{field} must be an int >= 2 or None, got "):
        RttRings(_Memory(), BLOCK, **{field: bad})


def test_rings_accept_the_smallest_expected_size() -> None:
    assert RttRings(_target(up_size=2, down_size=2), BLOCK, expected_up_size=2, expected_down_size=2).up_offsets() == (2, 0, 0)


def test_rings_defer_the_size_check_until_the_block_is_live() -> None:
    memory = _Memory()
    rings = RttRings(memory, BLOCK, expected_up_size=16, expected_down_size=8)
    assert rings.up_offsets() is None
    for address, value in enumerate(_target(up_size=1024).memory_read(BLOCK, 0x200)):
        memory.bytes[BLOCK + address] = value
    with pytest.raises(RttBufferSizeError, match="is 1024 bytes, expected 16"):
        rings.up_offsets()


class _FakePylink:
    class enums:
        class JLinkInterfaces:
            SWD = "swd"


class _FakeJLink(_Memory):
    def open(self, serial_no: int) -> None:
        pass

    def set_tif(self, tif: object) -> None:
        pass

    def connect(self, chip_name: str, speed: int, verbose: bool) -> None:
        pass

    def close(self) -> None:
        pass


def _open_transport(monkeypatch: pytest.MonkeyPatch, probe: _FakeJLink, **kwargs: object) -> JLinkRttTransport:
    import sys

    from helia_core_tester.hardware import jlink_library

    monkeypatch.setitem(sys.modules, "pylink", _FakePylink)
    monkeypatch.setattr(jlink_library, "open_jlink", lambda pylink: probe)
    return JLinkRttTransport(serial_no=1, chip_name="AP510", rtt_address=BLOCK, **kwargs)


def _probe_with(up_size: int, down_size: int) -> "_FakeJLink":
    probe = _FakeJLink()
    for address, value in enumerate(_target(up_size=up_size, down_size=down_size).memory_read(BLOCK, 0x200)):
        probe.bytes[BLOCK + address] = value
    return probe


def test_transport_without_expected_sizes_skips_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # None means "do not check", as on RttRings: no built-in sizes to trip over.
    transport = _open_transport(monkeypatch, _probe_with(1024, 16))
    assert transport._rings.up_offsets() == (1024, 0, 0)


def test_transport_refuses_segger_defaults_against_hctp_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (RTT_BUFFER_SIZE_UP, RTT_BUFFER_SIZE_DOWN) == (8192, 512)
    sizes = {"expected_up_size": RTT_BUFFER_SIZE_UP, "expected_down_size": RTT_BUFFER_SIZE_DOWN}
    good = _open_transport(monkeypatch, _probe_with(RTT_BUFFER_SIZE_UP, RTT_BUFFER_SIZE_DOWN), **sizes)
    assert good._rings.up_offsets() == (RTT_BUFFER_SIZE_UP, 0, 0)
    # SEGGER's own defaults: the regression #188 describes.
    stale = _open_transport(monkeypatch, _probe_with(1024, 16), **sizes)
    with pytest.raises(RttBufferSizeError, match=r"is 1024 bytes, expected 8192;"):
        stale._rings.up_offsets()


@pytest.mark.parametrize("size", [np.int64(8192), np.int32(8192)])
def test_numpy_integer_expected_sizes_are_accepted(size) -> None:
    rings = RttRings(_target(up_size=8192, down_size=512), BLOCK, expected_up_size=size, expected_down_size=512)
    assert rings.up_offsets() == (8192, 0, 0)


def test_transport_size_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _FakeJLink()
    for address, value in enumerate(_target(up_size=16, down_size=8).memory_read(BLOCK, 0x200)):
        probe.bytes[BLOCK + address] = value
    transport = _open_transport(monkeypatch, probe, expected_up_size=16, expected_down_size=8)
    assert transport._rings.up_offsets() == (16, 0, 0)
