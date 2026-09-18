"""Find, score and blank SEGGER RTT control blocks over SWD.

The tester normally does not need any of this: it links the firmware itself, so
`_SEGGER_RTT`'s address comes straight out of the ELF and `rtt_start()` is given
that exact address. heliaPROFILER calls that its `known_block_address` path and
skips both the pre-clean and the discovery scan on it, for the same reason --
the firmware re-initialises that fixed address on every boot, so a stale block
somewhere else can never be selected.

What this module adds is the fallback for when that address is not usable: the
ELF is a different build than the board runs, `arm-none-eabi-nm` cannot be run,
or the block at the linked address does not validate. Then the host scans SRAM
for the `SEGGER RTT` magic exactly as hpx does, scores the candidates (up-channel
0 named `HCTP_UP` is the strong signal; recent write activity next; buffer size
as the tie-breaker) and attaches to the best one -- after blanking the magic of
every block it found, so a block left in retained SRAM by a previously flashed
firmware cannot win the race with the live one. Apollo5 retains SRAM across
reset, which is what makes that stale block possible at all.

The scan and wipe are ported from hpx's `transport/rtt_control.py`; the control
block layout is SEGGER's (`SEGGER_RTT_CB`: 16-byte id, MaxNumUpBuffers,
MaxNumDownBuffers, then the ring descriptors).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

log = logging.getLogger(__name__)

RTT_MAGIC = b"SEGGER RTT"
#: The firmware's up-channel 0 name (benchmark_server_transport_rtt.c).
APP_UP_CHANNEL_NAME = b"HCTP_UP"

_SCAN_CHUNK = 0x4000
_ID_SIZE = 16
_CB_HEADER_SIZE = 24  # acID[16] + MaxNumUpBuffers + MaxNumDownBuffers
_DESC_WORDS = 6  # sName, pBuffer, SizeOfBuffer, WrOff, RdOff, Flags
_NAME_MAX_LEN = 16
_NAME_MATCH_BONUS = 1 << 28
_ACTIVITY_BONUS = 1 << 26

#: Score of a block that is both named `HCTP_UP` and actively producing bytes.
LIVE_NAMED_SCORE = _NAME_MATCH_BONUS + _ACTIVITY_BONUS

ScanRanges = Tuple[Tuple[int, int], ...]


def up_channel0_name(jlink, block_address: int, max_len: int = _NAME_MAX_LEN) -> bytes:
    """The NUL-terminated name of up-channel 0, or `b""` when it cannot be read."""
    try:
        name_ptr = jlink.memory_read32(block_address + _CB_HEADER_SIZE, 1)[0]
        if name_ptr == 0:
            return b""
        raw = bytes(jlink.memory_read8(name_ptr, max_len))
    except Exception:  # name probing is best effort
        return b""
    nul = raw.find(0)
    return raw[:nul] if nul >= 0 else raw


def score_control_block(jlink, block_address: int) -> int:
    """Rank a candidate by how likely it is to be the live benchmark-server block.

    Negative means structurally invalid (no up buffers, impossible offsets) --
    i.e. a `SEGGER RTT` magic that is not a control block at all.
    """
    try:
        max_up_buffers = jlink.memory_read32(block_address + _ID_SIZE, 1)[0]
        if max_up_buffers <= 0:
            return -1
        name_ptr, buf_ptr, size, wr_off, rd_off, _flags = jlink.memory_read32(
            block_address + _CB_HEADER_SIZE, _DESC_WORDS
        )
    except Exception:  # an unreadable candidate is not a candidate
        return -1
    if buf_ptr == 0 or size <= 0 or wr_off > size or rd_off > size:
        return -1
    score = 1 if name_ptr != 0 else 0
    if up_channel0_name(jlink, block_address) == APP_UP_CHANNEL_NAME:
        score += _NAME_MATCH_BONUS
    if wr_off != rd_off:
        score += _ACTIVITY_BONUS
    return score + min(size, 1 << 20)


def _magic_addresses(jlink, ranges: ScanRanges):
    """Yield every address in `ranges` whose bytes start the `SEGGER RTT` magic."""
    seen: set[int] = set()
    for base, length in ranges:
        for offset in range(0, length, _SCAN_CHUNK):
            chunk_len = min(_SCAN_CHUNK, length - offset)
            try:
                chunk = bytes(jlink.memory_read8(base + offset, chunk_len))
            except Exception:  # a range the probe cannot read is simply skipped
                continue
            start = 0
            while True:
                index = chunk.find(RTT_MAGIC, start)
                if index < 0:
                    break
                address = base + offset + index
                start = index + 1
                if address not in seen:
                    seen.add(address)
                    yield address


def scan_control_blocks(jlink, ranges: ScanRanges) -> list[tuple[int, int]]:
    """Every structurally valid control block as `(address, score)`, best first."""
    candidates = [
        (address, score)
        for address, score in ((a, score_control_block(jlink, a)) for a in _magic_addresses(jlink, ranges))
        if score >= 0
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def find_control_block(jlink, ranges: ScanRanges) -> Optional[tuple[int, int]]:
    """The best-scoring control block and its score, or None when none was found."""
    candidates = scan_control_blocks(jlink, ranges)
    if not candidates:
        return None
    if len(candidates) > 1:
        log.debug(
            "RTT scan found %d control blocks; selected 0x%08X (score=%d) over 0x%08X (score=%d)",
            len(candidates), candidates[0][0], candidates[0][1], candidates[1][0], candidates[1][1],
        )
    return candidates[0]


def wipe_control_blocks(jlink, ranges: ScanRanges, *, keep: Sequence[int] = ()) -> int:
    """Blank the magic of every valid control block in `ranges`; returns how many.

    Blanking only the 16-byte id is enough to make a block invisible to both this
    scan and J-Link's own auto-discovery, and the live firmware rewrites its own
    id on the reset that follows. `keep` spares addresses the caller knows are
    the live ones (normally the linked `_SEGGER_RTT`).
    """
    zeros = [0] * _ID_SIZE
    spared = set(keep)
    wiped = 0
    for address in _magic_addresses(jlink, ranges):
        if address in spared:
            continue
        try:
            if score_control_block(jlink, address) < 0:
                continue
            jlink.memory_write8(address, zeros)
        except Exception:  # a failed wipe is not fatal: scoring still ranks the live block first
            continue
        wiped += 1
        log.debug("pre-clean blanked RTT control block at 0x%08X", address)
    return wiped
