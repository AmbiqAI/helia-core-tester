"""Transport primitives for HCTP host-side tests and loopback runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import subprocess
import time
from typing import Iterable, Protocol


class Transport(Protocol):
    def write(self, payload: bytes) -> None:
        ...

    def read(self, max_bytes: int = 4096) -> bytes:
        ...

    def close(self) -> None:
        ...


class JLinkRttTransport:
    def __init__(
        self,
        *,
        serial_no: int,
        chip_name: str,
        speed_khz: int = 4000,
        rtt_address: int | None = None,
        up_buffer_index: int = 0,
        down_buffer_index: int = 0,
        reset_on_open: bool = False,
        reset_delay_s: float = 0.25,
        read_timeout_s: float = 2.0,
        poll_interval_s: float = 0.01,
    ) -> None:
        import pylink

        self._pylink = pylink
        self._serial_no = serial_no
        self._chip_name = chip_name
        self._speed_khz = speed_khz
        self._rtt_address = rtt_address
        self._up_buffer_index = up_buffer_index
        self._down_buffer_index = down_buffer_index
        self._reset_on_open = reset_on_open
        self._reset_delay_s = reset_delay_s
        self._read_timeout_s = read_timeout_s
        self._poll_interval_s = poll_interval_s
        self._jlink = pylink.JLink()
        self._jlink.open(serial_no=serial_no)
        try:
            self._jlink.set_tif(pylink.enums.JLinkInterfaces.SWD)
            self._jlink.connect(chip_name, speed=speed_khz, verbose=True)
            if reset_on_open:
                self._jlink.reset(halt=False)
                time.sleep(reset_delay_s)
            self._jlink.rtt_start(block_address=rtt_address)
        except Exception:
            # open() already claimed the JLink USB/DLL handle -- a failure in any step
            # after it (bad chip name, comm failure, RTT discovery timeout) must not leak
            # that handle, or the probe can require a process restart to reuse.
            self._jlink.close()
            raise

    def write(self, payload: bytes) -> None:
        remaining = payload
        deadline = time.monotonic() + self._read_timeout_s
        while remaining:
            written = int(self._jlink.rtt_write(self._down_buffer_index, list(remaining)))
            if written > 0:
                remaining = remaining[written:]
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out writing {len(payload)} RTT bytes.")
            time.sleep(self._poll_interval_s)

    def read(self, max_bytes: int = 4096) -> bytes:
        deadline = time.monotonic() + self._read_timeout_s
        while time.monotonic() < deadline:
            data = bytes(self._jlink.rtt_read(self._up_buffer_index, max_bytes))
            if data:
                return data
            time.sleep(self._poll_interval_s)
        return b""

    def close(self) -> None:
        try:
            self._jlink.rtt_stop()
        except Exception:
            pass
        self._jlink.close()


def symbol_address_from_elf(elf_path: str, symbol_name: str) -> int:
    result = subprocess.run(
        ["arm-none-eabi-nm", "-n", elf_path],
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
