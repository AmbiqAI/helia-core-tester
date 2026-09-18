"""Opening an RTT session: reset through JLinkExe, attach, find the control block.

Everything runs against a fake J-Link whose "SRAM" is a byte array, so the
control-block scan, the stale-block pre-clean and the attach sequence are all
exercised without a probe.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from helia_core_tester.hardware import jlink_library, rtt_control, transport as transport_module
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.transport import JLinkRttTransport, resolve_discovery_mode

BOARD = resolve_board("apollo510_evb")
SERIAL = 1160003180
RAM_BASE = 0x20000000
RAM_SIZE = 0x8000
SCAN_RANGES = ((RAM_BASE, RAM_SIZE),)


class FakeMemory:
    """A flat little-endian memory the fake probe reads and writes."""

    def __init__(self, base: int = RAM_BASE, size: int = RAM_SIZE) -> None:
        self.base = base
        self.data = bytearray(size)

    def _offset(self, address: int, count: int) -> int:
        offset = address - self.base
        if offset < 0 or offset + count > len(self.data):
            raise ValueError(f"read outside the fake SRAM: 0x{address:08x}+{count}")
        return offset

    def write(self, address: int, payload: bytes) -> None:
        offset = self._offset(address, len(payload))
        self.data[offset:offset + len(payload)] = payload

    def read(self, address: int, count: int) -> bytes:
        offset = self._offset(address, count)
        return bytes(self.data[offset:offset + count])


def put_control_block(
    memory: FakeMemory,
    address: int,
    *,
    name: bytes = rtt_control.APP_UP_CHANNEL_NAME,
    size: int = 1024,
    write_offset: int = 8,
    read_offset: int = 0,
    magic: bytes = rtt_control.RTT_MAGIC,
) -> int:
    """Write a SEGGER RTT control block (id, counts, up-channel 0 descriptor)."""
    name_addr = address + 0x200
    buffer_addr = address + 0x400
    memory.write(name_addr, name + b"\0")
    memory.write(address, magic.ljust(16, b"\0"))
    memory.write(address + 16, struct.pack("<II", 1, 1))  # MaxNumUp/DownBuffers
    memory.write(address + 24, struct.pack("<IIIIII", name_addr, buffer_addr, size, write_offset, read_offset, 0))
    return address


class FakeJLink:
    def __init__(self, memory: FakeMemory, *, events: list[str], halted: bool = False, fail_opens: int = 0) -> None:
        self.memory = memory
        self.events = events
        self._halted = halted
        self._fail_opens = fail_opens
        self.rtt_block: int | None = -1
        self.closed = False

    # --- session lifecycle ---
    def open(self, serial_no=None):
        if self._fail_opens > 0:
            self._fail_opens -= 1
            import pylink

            self.events.append("open:refused")
            raise pylink.errors.JLinkException("Cannot connect to target.")
        self.events.append("open")

    def set_tif(self, interface):
        self.events.append("set_tif")

    def connect(self, chip_name, speed=None, verbose=False):
        self.events.append(f"connect:{chip_name}")

    def close(self):
        self.closed = True
        self.events.append("close")

    def halt(self):
        self._halted = True
        self.events.append("halt")

    def halted(self):
        return self._halted

    def restart(self):
        self._halted = False
        self.events.append("restart")

    def reset(self, halt=False):  # must never be used: pylink reset misses the Apollo510 SBL
        raise AssertionError("the RTT session must reset through JLinkExe, not pylink")

    # --- memory ---
    def memory_read8(self, address, count):
        return list(self.memory.read(address, count))

    def memory_read32(self, address, count):
        raw = self.memory.read(address, count * 4)
        return list(struct.unpack("<" + "I" * count, raw))

    def memory_write8(self, address, values):
        self.memory.write(address, bytes(values))

    def memory_write32(self, address, values):
        self.memory.write(address, struct.pack("<" + "I" * len(values), *values))

    # --- rtt ---
    def rtt_start(self, block_address=None):
        self.rtt_block = block_address
        self.events.append(f"rtt_start:{block_address if block_address is None else hex(block_address)}")

    def rtt_stop(self):
        self.events.append("rtt_stop")


@pytest.fixture
def fake_probe(monkeypatch):
    """Wire a fake pylink session and a recording JLinkExe reset into the transport."""
    state: dict = {"memory": FakeMemory(), "events": [], "resets": [], "jlinks": [], "halted": False, "fail_opens": 0}

    def _open_jlink(pylink_module=None, env=None):
        jlink = FakeJLink(
            state["memory"], events=state["events"],
            halted=state["halted"], fail_opens=state["fail_opens"],
        )
        state["fail_opens"] = 0
        state["jlinks"].append(jlink)
        return jlink

    def _reset(*, device, serial_no=None, speed_khz=4000, **kwargs):
        state["events"].append("jlinkexe_reset")
        state["resets"].append((device, serial_no, speed_khz))
        # The firmware republishes its control block on boot, which is what makes
        # the phase-0 wipe safe.
        for address in state.get("republish", ()):
            put_control_block(state["memory"], address)

    monkeypatch.setattr(jlink_library, "open_jlink", _open_jlink)
    monkeypatch.setattr(transport_module.time, "sleep", lambda _s: None)
    state["reset"] = _reset
    return state


def _open_transport(state, **kwargs) -> JLinkRttTransport:
    return JLinkRttTransport(
        serial_no=SERIAL, chip_name=BOARD.jlink_device, speed_khz=BOARD.swd_speed_khz,
        reset_on_open=True, reset_target=state["reset"], echo=state.setdefault("echoed", []).append,
        block_wait_s=0.05, **kwargs,
    )


# --- discovery mode ----------------------------------------------------------------


def test_discovery_defaults_to_the_linked_address_and_can_be_forced() -> None:
    assert resolve_discovery_mode(rtt_address=0x2000_0100, env={}) == "address"
    assert resolve_discovery_mode(rtt_address=None, env={}) == "scan"
    assert resolve_discovery_mode(rtt_address=0x2000_0100, env={"HCT_RTT_DISCOVERY": "scan"}) == "scan"
    assert resolve_discovery_mode("address", rtt_address=None, env={}) == "address"
    with pytest.raises(ValueError, match="HCT_RTT_DISCOVERY"):
        resolve_discovery_mode(rtt_address=None, env={"HCT_RTT_DISCOVERY": "magic"})


# --- control-block scanning --------------------------------------------------------


def test_the_live_block_outscores_a_stale_one_and_an_invalid_candidate() -> None:
    memory = FakeMemory()
    stale = put_control_block(memory, RAM_BASE + 0x1000, name=b"Terminal", write_offset=0, read_offset=0)
    live = put_control_block(memory, RAM_BASE + 0x2000, write_offset=64, read_offset=0)
    # A `SEGGER RTT` string that is not a control block at all (no up buffers).
    memory.write(RAM_BASE + 0x3000, rtt_control.RTT_MAGIC.ljust(16, b"\0") + struct.pack("<II", 0, 0))
    jlink = FakeJLink(memory, events=[])

    found = rtt_control.find_control_block(jlink, SCAN_RANGES)
    assert found is not None and found[0] == live
    assert found[1] >= rtt_control.LIVE_NAMED_SCORE
    assert rtt_control.score_control_block(jlink, stale) >= 0
    assert rtt_control.score_control_block(jlink, RAM_BASE + 0x3000) < 0
    assert rtt_control.up_channel0_name(jlink, live) == b"HCTP_UP"
    # Every structurally valid block is a candidate; the invalid one is not.
    assert [address for address, _ in rtt_control.scan_control_blocks(jlink, SCAN_RANGES)] == [live, stale]


def test_wiping_blanks_the_magic_and_can_spare_the_live_block() -> None:
    memory = FakeMemory()
    stale = put_control_block(memory, RAM_BASE + 0x1000, name=b"Terminal")
    live = put_control_block(memory, RAM_BASE + 0x2000)
    jlink = FakeJLink(memory, events=[])

    assert rtt_control.wipe_control_blocks(jlink, SCAN_RANGES, keep=(live,)) == 1
    assert memory.read(stale, 10) == b"\0" * 10
    assert memory.read(live, 10) == rtt_control.RTT_MAGIC
    assert rtt_control.find_control_block(jlink, SCAN_RANGES)[0] == live
    # And with nothing spared, discovery finds nothing at all.
    assert rtt_control.wipe_control_blocks(jlink, SCAN_RANGES) == 1
    assert rtt_control.find_control_block(jlink, SCAN_RANGES) is None


# --- opening a session -------------------------------------------------------------


def test_a_session_resets_through_jlinkexe_then_attaches_at_the_linked_address(fake_probe) -> None:
    address = put_control_block(fake_probe["memory"], RAM_BASE + 0x2000)
    transport = _open_transport(fake_probe, rtt_address=address, scan_ranges=SCAN_RANGES)

    assert fake_probe["resets"] == [(BOARD.jlink_device, SERIAL, BOARD.swd_speed_khz)]
    # The reset happens before any pylink session exists: JLinkExe needs the probe
    # to itself, and its exit is what lets the SBL start the application.
    assert fake_probe["events"][0] == "jlinkexe_reset"
    assert fake_probe["events"][1:5] == ["open", "set_tif", f"connect:{BOARD.jlink_device}", f"rtt_start:{hex(address)}"]
    transport.close()


def test_the_attach_is_retried_while_the_bootloader_is_still_running(fake_probe) -> None:
    address = put_control_block(fake_probe["memory"], RAM_BASE + 0x2000)
    fake_probe["fail_opens"] = 3
    transport = _open_transport(fake_probe, rtt_address=address, scan_ranges=SCAN_RANGES)

    assert fake_probe["events"].count("open:refused") == 3
    assert fake_probe["events"].count("open") == 1
    transport.close()


def test_a_halted_core_is_resumed_so_it_can_publish_rtt_bytes(fake_probe) -> None:
    """`JLinkExe r` without `g`, or an earlier debug session, leaves the core halted."""
    address = put_control_block(fake_probe["memory"], RAM_BASE + 0x2000)
    fake_probe["halted"] = True
    transport = _open_transport(fake_probe, rtt_address=address, scan_ranges=SCAN_RANGES)

    assert "restart" in fake_probe["events"]
    assert fake_probe["events"].index("restart") < fake_probe["events"].index(f"rtt_start:{hex(address)}")
    transport.close()


def test_scan_mode_precleans_stale_blocks_before_the_reset(fake_probe) -> None:
    """Phase 0: Apollo5 retains SRAM across reset, so a block from a previously
    flashed firmware can outlive it and race the live one during discovery."""
    memory = fake_probe["memory"]
    put_control_block(memory, RAM_BASE + 0x1000, name=b"Terminal")
    live = put_control_block(memory, RAM_BASE + 0x2000)
    fake_probe["republish"] = (live,)
    transport = _open_transport(fake_probe, rtt_address=None, scan_ranges=SCAN_RANGES, discovery="scan")

    events = fake_probe["events"]
    # Wipe (over its own short-lived session), then reset, then attach and scan.
    assert events.index("halt") < events.index("jlinkexe_reset") < events.index(f"rtt_start:{hex(live)}")
    assert any("pre-clean blanked 2" in line for line in fake_probe["echoed"])
    transport.close()


def test_a_dead_linked_address_falls_back_to_the_scan(fake_probe) -> None:
    """The ELF's address stays the primary answer; the scan only adds a candidate."""
    live = put_control_block(fake_probe["memory"], RAM_BASE + 0x2000)
    transport = JLinkRttTransport(
        serial_no=SERIAL, chip_name=BOARD.jlink_device, rtt_address=RAM_BASE + 0x6000,
        reset_on_open=True, reset_target=fake_probe["reset"], scan_ranges=SCAN_RANGES,
        echo=fake_probe.setdefault("echoed", []).append, block_wait_s=0.05,
    )
    assert transport._jlink.rtt_block == live
    assert any("scanning SRAM" in line for line in fake_probe["echoed"])
    transport.close()


def test_without_any_block_the_linked_address_is_still_used(fake_probe) -> None:
    transport = JLinkRttTransport(
        serial_no=SERIAL, chip_name=BOARD.jlink_device, rtt_address=RAM_BASE + 0x6000,
        reset_on_open=True, reset_target=fake_probe["reset"], scan_ranges=SCAN_RANGES,
        echo=fake_probe.setdefault("echoed", []).append, block_wait_s=0.05,
    )
    assert transport._jlink.rtt_block == RAM_BASE + 0x6000
    transport.close()


def test_scan_mode_without_ranges_is_refused_before_touching_the_probe(fake_probe) -> None:
    with pytest.raises(ValueError, match="rtt_scan_ranges"):
        _open_transport(fake_probe, rtt_address=None, scan_ranges=())
    assert fake_probe["events"] == []


def test_the_board_table_supplies_the_scan_window_to_the_session(monkeypatch, tmp_path: Path) -> None:
    """`open_rtt_session` passes the board's own SRAM window, not a hardcoded one."""
    from helia_core_tester.hardware import session_runner

    seen: dict = {}
    monkeypatch.setattr(session_runner, "symbol_address_from_elf", lambda elf, symbol: 0x2000_0100)
    monkeypatch.setattr(session_runner, "JLinkRttTransport", lambda **kwargs: seen.update(kwargs) or object())
    monkeypatch.setattr(session_runner, "HostSession", lambda transport, counter_passes=None: object())

    session_runner.open_rtt_session(BOARD, SERIAL, build_dir=tmp_path, counter_passes=())
    assert seen["scan_ranges"] == BOARD.rtt_scan_ranges == ((0x20000000, 0x80000),)
    assert seen["reset_on_open"] is True and seen["rtt_address"] == 0x2000_0100
