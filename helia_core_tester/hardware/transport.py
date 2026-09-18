"""Transport primitives for HCTP host-side tests and loopback runs.

`JLinkRttTransport` opens one RTT session the way heliaPROFILER does
(`transport/rtt.py`):

0. (only when the control block has to be discovered) pre-clean stale RTT
   control blocks over SWD, because Apollo5 retains SRAM across reset and a
   block left by a previously flashed firmware can otherwise win discovery;
1. reset through a `JLinkExe` `r` / `g` script rather than `pylink.reset()` --
   the Apollo510 secure bootloader only hands control to the application when
   the debugger releases the probe, which the commander's exit does and pylink
   does not;
2. wait `SBL_SETTLE_S` for the unobservable SBL phase;
3. attach pylink, retrying until the target answers;
4. start RTT at the control-block address linked into the firmware (taken from
   the ELF), falling back to a host-side scan for the `SEGGER RTT` magic and
   then to J-Link's own auto-scan.

There is no heartbeat-based hang detection: HCTP has no periodic target signal
to time a gap against (the firmware speaks when a case, sample or blob request
is ready), so the existing per-read timeouts remain the only liveness bound.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
import subprocess
import time
from typing import Callable, Iterable, Protocol

from . import jlink_cli, rtt_control

#: Fixed floor between the reset and the first attach: the Apollo510 SBL phase
#: is unobservable, so a short wait costs less than a failed attach.
SBL_SETTLE_S = 0.25
#: How long to keep retrying the pylink attach after a reset.
ATTACH_TIMEOUT_S = 30.0
_ATTACH_RETRY_S = 0.1
#: `HCT_RTT_DISCOVERY=address|scan|auto` overrides how the control block is found.
DISCOVERY_ENV_VAR = "HCT_RTT_DISCOVERY"
DISCOVERY_MODES = ("auto", "address", "scan")


class Transport(Protocol):
    def write(self, payload: bytes) -> None:
        ...

    def read(self, max_bytes: int = 4096) -> bytes:
        ...

    def close(self) -> None:
        ...


def resolve_discovery_mode(
    requested: str | None = None,
    *,
    rtt_address: int | None = None,
    env: dict | None = None,
) -> str:
    """"address" when the linked control-block address is known, else "scan".

    `HCT_RTT_DISCOVERY` forces one or the other; it exists so the scan path
    (and with it the phase-0 pre-clean) can be exercised on real hardware
    without deleting the ELF the address comes from.
    """
    env = os.environ if env is None else env
    value = (requested or env.get(DISCOVERY_ENV_VAR, "") or "auto").strip().lower()
    if value not in DISCOVERY_MODES:
        raise ValueError(
            f"${DISCOVERY_ENV_VAR} must be one of {', '.join(DISCOVERY_MODES)} (got {value!r})."
        )
    if value != "auto":
        return value
    return "address" if rtt_address is not None else "scan"


def open_jlink_with_retry(
    jlink,
    pylink,
    *,
    serial_no: int,
    chip_name: str,
    speed_khz: int,
    timeout_s: float = ATTACH_TIMEOUT_S,
    interval_s: float = _ATTACH_RETRY_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Open, select the interface and connect, retrying until the target is ready.

    Immediately after a reset the Apollo510 can still be inside the SBL and
    refuse the connect; one attempt is therefore not enough to tell "board is
    booting" from "board is gone".
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            jlink.open(serial_no=serial_no)
            jlink.set_tif(pylink.enums.JLinkInterfaces.SWD)
            jlink.connect(chip_name, speed=speed_khz, verbose=True)
            return
        except Exception as exc:
            if not isinstance(exc, pylink.errors.JLinkException):
                raise
            try:
                jlink.close()
            except Exception:  # a failed close must not mask the connect error
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out attaching a J-Link session to {chip_name} on serial {serial_no} "
                    f"after {timeout_s:.0f}s ({exc}). Check the board is powered and the probe "
                    "is not in use by another process."
                ) from exc
            sleep(interval_s)


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
        reset_delay_s: float = SBL_SETTLE_S,
        read_timeout_s: float = 2.0,
        poll_interval_s: float = 0.01,
        scan_ranges: tuple[tuple[int, int], ...] = (),
        discovery: str | None = None,
        attach_timeout_s: float = ATTACH_TIMEOUT_S,
        block_wait_s: float = 5.0,
        reset_target: Callable[..., object] = jlink_cli.reset_target,
        echo: Callable[[str], None] | None = None,
    ) -> None:
        import pylink

        from .jlink_library import open_jlink

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
        self._scan_ranges = tuple(scan_ranges)
        self._block_wait_s = block_wait_s
        self._echo = echo or (lambda _message: None)
        self._mode = resolve_discovery_mode(discovery, rtt_address=rtt_address)
        if self._mode == "scan" and not self._scan_ranges:
            raise ValueError(
                "RTT control-block discovery needs SRAM scan ranges for this board "
                "(rtt_scan_ranges in assets/hardware_boards.yaml), but none were given."
            )

        if reset_on_open:
            if self._mode == "scan":
                self._preclean_stale_blocks(open_jlink, pylink, attach_timeout_s)
            # JLinkExe, never pylink: the commander's exit releases the probe so
            # the Apollo510 SBL starts the application (see the module docstring).
            reset_target(device=chip_name, serial_no=serial_no, speed_khz=speed_khz)
            time.sleep(reset_delay_s)

        self._jlink = open_jlink(pylink)
        open_jlink_with_retry(
            self._jlink, pylink, serial_no=serial_no, chip_name=chip_name,
            speed_khz=speed_khz, timeout_s=attach_timeout_s,
        )
        try:
            self._resume_if_halted()
            self._jlink.rtt_start(block_address=self._locate_control_block())
        except Exception:
            # open() already claimed the JLink USB/DLL handle -- a failure in any step
            # after it (bad chip name, comm failure, RTT discovery timeout) must not leak
            # that handle, or the probe can require a process restart to reuse.
            self._jlink.close()
            raise

    def _preclean_stale_blocks(self, open_jlink, pylink, attach_timeout_s: float) -> None:
        """Phase 0: blank every RTT control block in SRAM before the reset.

        Best effort by design: if this attach fails the reset still happens and
        discovery scoring picks the live block. Only reached in "scan" mode --
        with the linked address known, the firmware re-initialises exactly that
        address on boot and stale blocks elsewhere are unreachable, which is the
        same rule hpx applies to its `known_block_address` path.
        """
        jlink = open_jlink(pylink)
        try:
            open_jlink_with_retry(
                jlink, pylink, serial_no=self._serial_no, chip_name=self._chip_name,
                speed_khz=self._speed_khz, timeout_s=min(attach_timeout_s, 10.0),
            )
            try:
                jlink.halt()
            except Exception:  # halting only makes the wipe tidier
                pass
            wiped = rtt_control.wipe_control_blocks(jlink, self._scan_ranges)
            if wiped:
                self._echo(f"[hardware] RTT pre-clean blanked {wiped} stale control block(s).")
        except Exception as exc:
            self._echo(f"[hardware] RTT pre-clean skipped ({type(exc).__name__}: {exc}).")
        finally:
            try:
                jlink.close()
            except Exception:
                pass

    def _resume_if_halted(self) -> None:
        """Restart the core if the attach found it halted.

        A bare `JLinkExe r` (reset without `g`), an earlier debug session or a
        crashed firmware all leave the core halted, and a halted core publishes
        no RTT bytes -- the session would then fail as a protocol timeout
        instead of as the stopped target it is.
        """
        try:
            if not self._jlink.halted():
                return
            self._jlink.restart()
        except Exception as exc:  # not every DLL/target answers halted() the same way
            self._echo(f"[hardware] Could not check/resume the halted core ({type(exc).__name__}: {exc}).")
            return
        self._echo("[hardware] Target was halted on attach; resumed it.")
        time.sleep(self._reset_delay_s)

    def _wait_for_control_block(self, address: int, timeout_s: float) -> bool:
        """Poll until a valid control block appears at `address`, or time out.

        The firmware initialises its block early in boot, but "early" is after
        the SBL, so a single read right after the attach can legitimately see
        uninitialised `.bss`.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            if rtt_control.score_control_block(self._jlink, address) >= 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._poll_interval_s)

    def _locate_control_block(self, wait_s: float | None = None) -> int | None:
        """The control-block address to start RTT at, or None for J-Link auto-scan."""
        wait_s = self._block_wait_s if wait_s is None else wait_s
        if self._mode == "address":
            if not self._scan_ranges or self._wait_for_control_block(self._rtt_address, wait_s):
                return self._rtt_address
            # The ELF's address is still the best answer if the scan finds
            # nothing, so this only ever adds a candidate.
            # Scan only -- never wipe here. The phase-0 wipe works because a reset
            # follows it and the firmware republishes its block; wiping at this
            # point would erase the very block being looked for.
            self._echo(
                f"[hardware] No valid RTT control block at the linked address "
                f"0x{self._rtt_address:08x} after {wait_s:.0f}s; scanning SRAM for one."
            )
            found = rtt_control.find_control_block(self._jlink, self._scan_ranges)
            if found is None:
                self._echo("[hardware] RTT scan found no control block either; using the linked address.")
                return self._rtt_address
        else:
            found = rtt_control.find_control_block(self._jlink, self._scan_ranges)
            if found is None:
                self._echo("[hardware] RTT scan found no control block; falling back to J-Link auto-scan.")
                return None
        address, score = found
        self._echo(f"[hardware] RTT control block located at 0x{address:08x} (score {score}).")
        return address

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
