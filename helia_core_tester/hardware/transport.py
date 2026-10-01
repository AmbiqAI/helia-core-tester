"""Transport primitives for HCTP host-side tests and loopback runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import subprocess
import time
from typing import Any, Iterable, Protocol


class Transport(Protocol):
    def write(self, payload: bytes) -> None:
        ...

    def read(self, max_bytes: int = 4096) -> bytes:
        ...

    def close(self) -> None:
        ...


class RttRings:
    """Channel rings of a SEGGER_RTT_CB, by memory access.

    `memory` is a pylink JLink. Bulk copies leave the access width to
    the DLL: memory_read8 costs one SWD access per byte.
    """

    def __init__(self, memory: Any, address: int, *, up_index: int = 0, down_index: int = 0) -> None:
        self._memory = memory
        self._address = address
        self._up_index = up_index
        self._down_index = down_index
        # Descriptor addresses, once the block is live.
        self._up: int | None = None
        self._down: int | None = None

    def _find(self) -> bool:
        """Locate both ring descriptors once the block is live."""
        if self._up is not None:
            return True
        block = bytes(self._memory.memory_read(self._address, RTT_ID_BYTES))
        if not block.startswith(RTT_ID):
            return False  # Firmware has not initialized RTT yet.
        max_up, max_down = self._memory.memory_read32(self._address + RTT_ID_BYTES, 2)
        if self._up_index >= max_up or self._down_index >= max_down:
            raise RuntimeError(f"RTT block at 0x{self._address:08x} lacks the requested channels.")
        rings = self._address + RTT_ID_BYTES + 8
        self._up = rings + RTT_RING_BYTES * self._up_index
        self._down = rings + RTT_RING_BYTES * (max_up + self._down_index)
        return True

    def _ring(self, descriptor: int) -> tuple[int, int, int, int] | None:
        """(buffer, size, write, read), or None when corrupt."""
        _, buffer, size, write, read, _ = (int(v) for v in self._memory.memory_read32(descriptor, 6))
        return (buffer, size, write, read) if write < size and read < size else None

    def take(self, max_bytes: int) -> bytes:
        """Drain up to `max_bytes` from the up ring."""
        ring = self._ring(self._up) if self._find() else None
        if ring is None:
            return b""
        buffer, size, write, read = ring
        # Copy to the wrap, then from 0.
        spans = [(read, write - read)] if read <= write else [(read, size - read), (0, write)]
        data = bytearray()
        for offset, length in spans:
            length = min(length, max_bytes - len(data))
            if length <= 0:
                break
            chunk = bytes(self._memory.memory_read(buffer + offset, length))
            data += chunk
            if len(chunk) < length:
                break  # Short read: keep the stream in order.
        if data:
            self._memory.memory_write32(self._up + 16, [(read + len(data)) % size])
        return bytes(data)

    def put(self, payload: bytes) -> int:
        """Copy what fits of `payload` into the down ring."""
        ring = self._ring(self._down) if self._find() else None
        if ring is None:
            return 0
        buffer, size, write, read = ring
        # Keep one slot free; stop at wrap.
        length = min(len(payload), (read - write - 1) % size, size - write)
        if length > 0:
            self._memory.memory_write(buffer + write, list(payload[:length]))
            self._memory.memory_write32(self._down + 12, [(write + length) % size])
        return length


class JLinkRttTransport:
    """HCTP over the SEGGER RTT rings, by plain J-Link memory access.

    Not the DLL's RTT engine: when the host polls slowly it can advance the
    target's read offset past bytes it never returns, dropping a burst's tail.
    """

    def __init__(
        self,
        *,
        serial_no: int,
        chip_name: str,
        rtt_address: int,
        speed_khz: int = 4000,
        up_buffer_index: int = 0,
        down_buffer_index: int = 0,
        reset_on_open: bool = False,
        reset_delay_s: float = 0.25,
        read_timeout_s: float = 2.0,
        poll_interval_s: float = 0.01,
    ) -> None:
        import pylink

        from .jlink_library import open_jlink

        self._pylink = pylink
        self._serial_no = serial_no
        self._chip_name = chip_name
        self._speed_khz = speed_khz
        self._reset_on_open = reset_on_open
        self._reset_delay_s = reset_delay_s
        self._read_timeout_s = read_timeout_s
        self._poll_interval_s = poll_interval_s
        self._jlink = open_jlink(pylink)
        self._rings = RttRings(self._jlink, rtt_address, up_index=up_buffer_index, down_index=down_buffer_index)
        self._jlink.open(serial_no=serial_no)
        try:
            self._jlink.set_tif(pylink.enums.JLinkInterfaces.SWD)
            self._jlink.connect(chip_name, speed=speed_khz, verbose=True)
            if reset_on_open:
                self._jlink.reset(halt=False)
                time.sleep(reset_delay_s)
        except Exception:
            # open() already claimed the JLink USB/DLL handle -- a failure in any step
            # after it (bad chip name, comm failure) must not leak that handle, or the
            # probe can require a process restart to reuse.
            self._jlink.close()
            raise

    def write(self, payload: bytes) -> None:
        remaining = payload
        deadline = time.monotonic() + self._read_timeout_s
        while remaining:
            written = self._rings.put(remaining)
            if written > 0:
                remaining = remaining[written:]
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out writing {len(payload)} RTT bytes.")
            time.sleep(self._poll_interval_s)

    def read(self, max_bytes: int = 4096) -> bytes:
        deadline = time.monotonic() + self._read_timeout_s
        while time.monotonic() < deadline:
            data = self._rings.take(max_bytes)
            if data:
                return data
            time.sleep(self._poll_interval_s)
        return b""

    def close(self) -> None:
        self._jlink.close()


# SEGGER_RTT_CB: id, ring counts, rings.
RTT_ID = b"SEGGER RTT\0"
RTT_ID_BYTES = 16
# Ring: name, buffer, size, WrOff, RdOff, flags.
RTT_RING_BYTES = 24


def symbol_address_from_elf(elf_path: str, symbol_name: str) -> int:
    from .toolchain import arm_tool

    result = subprocess.run(
        [arm_tool("arm-none-eabi-nm"), "-n", elf_path],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == symbol_name:
            return int(parts[0], 16)
    raise ValueError(f"Symbol not found in {elf_path}: {symbol_name}")


@dataclass
class LoopbackEndpoint:
    _incoming: bytearray
    _chunk_plan: deque[int]
    _peer: "LoopbackEndpoint | None" = None

    def connect(self, peer: "LoopbackEndpoint") -> None:
        self._peer = peer

    def write(self, payload: bytes) -> None:
        if self._peer is None:
            raise RuntimeError("Loopback endpoint is not connected.")
        self._peer._incoming.extend(payload)

    def read(self, max_bytes: int = 4096) -> bytes:
        if not self._incoming:
            return b""
        budget = max_bytes
        if self._chunk_plan:
            budget = min(budget, self._chunk_plan.popleft())
        size = min(len(self._incoming), budget)
        chunk = bytes(self._incoming[:size])
        del self._incoming[:size]
        return chunk

    def close(self) -> None:
        self._incoming.clear()


@dataclass(frozen=True)
class LoopbackPair:
    host: LoopbackEndpoint
    target: LoopbackEndpoint


def create_loopback_pair(
    *,
    host_read_chunks: Iterable[int] = (),
    target_read_chunks: Iterable[int] = (),
) -> LoopbackPair:
    host = LoopbackEndpoint(bytearray(), deque(host_read_chunks))
    target = LoopbackEndpoint(bytearray(), deque(target_read_chunks))
    host.connect(target)
    target.connect(host)
    return LoopbackPair(host=host, target=target)
