"""Helia Core Tester Protocol framing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
import zlib

MAGIC = b"HCT1"
HEADER_SIZE = 32
SUPPORTED_VERSION = 1
DEFAULT_MAX_PAYLOAD = 64 * 1024

_HEADER_WITHOUT_CRC = struct.Struct("<4sHHIIIII")
_HEADER = struct.Struct("<4sHHIIIIII")


class HctpError(ValueError):
    """Base error for HCTP framing failures."""


class InvalidMagicError(HctpError):
    pass


class UnsupportedVersionError(HctpError):
    pass


class HeaderCrcError(HctpError):
    pass


class PayloadCrcError(HctpError):
    pass


class OversizedPayloadError(HctpError):
    pass


class SequenceMismatchError(HctpError):
    pass


class SessionMismatchError(HctpError):
    pass


class MessageType(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    LOAD_PLAN = 3
    CASE_META = 4
    BLOB_CHUNK = 5
    RUN_CORRECTNESS = 6
    CORRECTNESS_ACK = 7
    RUN_PERFORMANCE = 8
    ACK = 9
    NACK = 10
    ABORT_CASE = 11
    RESET_SESSION = 12
    PING = 13
    CAPABILITIES = 14
    REQUEST_CASE = 15
    REQUEST_BLOB = 16
    CASE_READY = 17
    CORRECTNESS_RESULT = 18
    OUTPUT_BEGIN = 19
    OUTPUT_CHUNK = 20
    OUTPUT_END = 21
    SAMPLE_RESULT = 22
    CASE_COMPLETE = 23
    SESSION_COMPLETE = 24
    ERROR = 25
    LOG = 26
    PONG = 27


@dataclass(frozen=True)
class FrameHeader:
    magic: bytes
    protocol_version: int
    message_type: MessageType
    flags: int
    session_id: int
    sequence_id: int
    payload_length: int
    payload_crc32: int
    header_crc32: int


@dataclass(frozen=True)
class Frame:
    header: FrameHeader
    payload: bytes


class ByteWriter:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def u8(self, value: int) -> None:
        self._buffer.extend(struct.pack("<B", value))

    def u16(self, value: int) -> None:
        self._buffer.extend(struct.pack("<H", value))

    def u32(self, value: int) -> None:
        self._buffer.extend(struct.pack("<I", value))

    def u64(self, value: int) -> None:
        self._buffer.extend(struct.pack("<Q", value))

    def i32(self, value: int) -> None:
        self._buffer.extend(struct.pack("<i", value))

    def raw(self, payload: bytes) -> None:
        self.u32(len(payload))
        self._buffer.extend(payload)

    def text(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u16(len(encoded))
        self._buffer.extend(encoded)

    def fixed(self, payload: bytes) -> None:
        self._buffer.extend(payload)

    def finish(self) -> bytes:
        return bytes(self._buffer)


class ByteReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def _take(self, size: int) -> bytes:
        end = self._offset + size
        if end > len(self._payload):
            raise HctpError("Unexpected end of payload.")
        chunk = self._payload[self._offset:end]
        self._offset = end
        return chunk

    def u8(self) -> int:
        return struct.unpack("<B", self._take(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def raw(self) -> bytes:
        return self._take(self.u32())

    def text(self) -> str:
        return self._take(self.u16()).decode("utf-8")

    def fixed(self, size: int) -> bytes:
        return self._take(size)

    def remaining(self) -> bytes:
        return self._payload[self._offset :]


class FrameDecoder:
    """Incremental HCTP frame decoder."""

    def __init__(self, *, max_payload: int = DEFAULT_MAX_PAYLOAD, supported_version: int = SUPPORTED_VERSION) -> None:
        self._buffer = bytearray()
        self._max_payload = max_payload
        self._supported_version = supported_version

    def feed(self, chunk: bytes) -> list[Frame]:
        self._buffer.extend(chunk)
        frames: list[Frame] = []
        while len(self._buffer) >= HEADER_SIZE:
            header_bytes = bytes(self._buffer[:HEADER_SIZE])
            header = decode_header(header_bytes, max_payload=self._max_payload, supported_version=self._supported_version)
            total_length = HEADER_SIZE + header.payload_length
            if len(self._buffer) < total_length:
                break
            payload = bytes(self._buffer[HEADER_SIZE:total_length])
            actual_payload_crc = crc32(payload)
            if actual_payload_crc != header.payload_crc32:
                raise PayloadCrcError(
                    f"Payload CRC mismatch for seq={header.sequence_id}: "
                    f"expected 0x{header.payload_crc32:08x}, got 0x{actual_payload_crc:08x}."
                )
            frames.append(Frame(header=header, payload=payload))
            del self._buffer[:total_length]
        return frames


class SessionFrameValidator:
    """Validate inbound session/sequence continuity."""

    def __init__(self, *, session_id: int, next_sequence_id: int = 0) -> None:
        self._session_id = session_id
        self._next_sequence_id = next_sequence_id

    def accept(self, frame: Frame) -> None:
        if frame.header.session_id != self._session_id:
            raise SessionMismatchError(
                f"Expected session {self._session_id}, got {frame.header.session_id}."
            )
        if frame.header.sequence_id != self._next_sequence_id:
            gap = frame.header.sequence_id - self._next_sequence_id
            raise SequenceMismatchError(
                f"Expected sequence {self._next_sequence_id}, got {frame.header.sequence_id} "
                f"(gap of {gap}) while receiving {frame.header.message_type.name}. "
                "This usually means one or more frames were dropped in transit "
                "(e.g. an RTT buffer overrun) rather than a protocol logic error."
            )
        self._next_sequence_id += 1


def crc32(payload: bytes) -> int:
    return zlib.crc32(payload) & 0xFFFFFFFF


def encode_header(
    *,
    message_type: MessageType,
    payload_length: int,
    payload_crc32: int,
    session_id: int,
    sequence_id: int,
    protocol_version: int = SUPPORTED_VERSION,
    flags: int = 0,
) -> bytes:
    partial = _HEADER_WITHOUT_CRC.pack(
        MAGIC,
        protocol_version,
        int(message_type),
        flags,
        session_id,
        sequence_id,
        payload_length,
        payload_crc32,
    )
    header_crc32 = crc32(partial)
    return _HEADER.pack(
        MAGIC,
        protocol_version,
        int(message_type),
        flags,
        session_id,
        sequence_id,
        payload_length,
        payload_crc32,
        header_crc32,
    )


def decode_header(
    payload: bytes,
    *,
    max_payload: int = DEFAULT_MAX_PAYLOAD,
    supported_version: int = SUPPORTED_VERSION,
) -> FrameHeader:
    if len(payload) != HEADER_SIZE:
        raise HctpError(f"Expected {HEADER_SIZE}-byte header, got {len(payload)} bytes.")
    unpacked = _HEADER.unpack(payload)
    header = FrameHeader(
        magic=unpacked[0],
        protocol_version=unpacked[1],
        message_type=MessageType(unpacked[2]),
        flags=unpacked[3],
        session_id=unpacked[4],
        sequence_id=unpacked[5],
        payload_length=unpacked[6],
        payload_crc32=unpacked[7],
        header_crc32=unpacked[8],
    )
    if header.magic != MAGIC:
        raise InvalidMagicError(f"Invalid HCTP magic: {header.magic!r}")
    if header.protocol_version != supported_version:
        raise UnsupportedVersionError(
            f"Unsupported HCTP version {header.protocol_version}; expected {supported_version}."
        )
    if header.payload_length > max_payload:
        raise OversizedPayloadError(
            f"Payload length {header.payload_length} exceeds maximum {max_payload}."
        )
    expected_crc = crc32(payload[:-4])
    if expected_crc != header.header_crc32:
        raise HeaderCrcError(
            f"Header CRC mismatch: expected 0x{header.header_crc32:08x}, got 0x{expected_crc:08x}."
        )
    return header


def encode_frame(
    message_type: MessageType,
    payload: bytes,
    *,
    session_id: int,
    sequence_id: int,
    protocol_version: int = SUPPORTED_VERSION,
    flags: int = 0,
) -> bytes:
    if len(payload) > DEFAULT_MAX_PAYLOAD:
        raise OversizedPayloadError(f"Payload length {len(payload)} exceeds maximum {DEFAULT_MAX_PAYLOAD}.")
    payload_crc = crc32(payload)
    header = encode_header(
        message_type=message_type,
        payload_length=len(payload),
        payload_crc32=payload_crc,
        session_id=session_id,
        sequence_id=sequence_id,
        protocol_version=protocol_version,
        flags=flags,
    )
    return header + payload
