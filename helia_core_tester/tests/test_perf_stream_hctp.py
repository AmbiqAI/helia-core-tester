from __future__ import annotations

from pathlib import Path
import struct

import pytest

from helia_core_tester.perf_stream.hctp import (
    HEADER_SIZE,
    ByteReader,
    ByteWriter,
    FrameDecoder,
    InvalidMagicError,
    HeaderCrcError,
    MessageType,
    OversizedPayloadError,
    PayloadCrcError,
    SequenceMismatchError,
    SessionFrameValidator,
    SessionMismatchError,
    UnsupportedVersionError,
    crc32,
    decode_header,
    encode_frame,
)
from helia_core_tester.perf_stream.transport import create_loopback_pair


SESSION_ID = 0x12345678



def _frame(message_type: MessageType, payload: bytes, *, seq: int = 0, session: int = SESSION_ID) -> bytes:
    return encode_frame(message_type, payload, session_id=session, sequence_id=seq)



def test_frame_roundtrip_little_endian() -> None:
    payload_writer = ByteWriter()
    payload_writer.u16(0x1234)
    payload_writer.u32(0xABCDEF12)
    payload_writer.text("ok")

    raw = _frame(MessageType.PING, payload_writer.finish(), seq=7)
    decoder = FrameDecoder()
    [frame] = decoder.feed(raw)

    assert frame.header.message_type is MessageType.PING
    assert frame.header.session_id == SESSION_ID
    assert frame.header.sequence_id == 7

    reader = ByteReader(frame.payload)
    assert reader.u16() == 0x1234
    assert reader.u32() == 0xABCDEF12
    assert reader.text() == "ok"



def test_invalid_magic_rejected() -> None:
    raw = bytearray(_frame(MessageType.PING, b"abc"))
    raw[0:4] = b"BAD!"
    with pytest.raises(InvalidMagicError):
        decode_header(bytes(raw[:HEADER_SIZE]))



def test_unsupported_version_rejected() -> None:
    raw = _frame(MessageType.PING, b"abc", seq=1)
    patched = bytearray(raw)
    struct.pack_into("<H", patched, 4, 99)
    patched[28:32] = struct.pack("<I", crc32(bytes(patched[:28])))
    with pytest.raises(UnsupportedVersionError):
        decode_header(bytes(patched[:HEADER_SIZE]))



def test_header_crc_rejected() -> None:
    raw = bytearray(_frame(MessageType.PING, b"abc"))
    raw[28:32] = struct.pack("<I", 0)
    with pytest.raises(HeaderCrcError):
        decode_header(bytes(raw[:HEADER_SIZE]))



def test_payload_crc_rejected() -> None:
    raw = bytearray(_frame(MessageType.PING, b"abcdef"))
    raw[-1] ^= 0x55
    decoder = FrameDecoder()
    with pytest.raises(PayloadCrcError):
        decoder.feed(bytes(raw))



def test_oversized_payload_rejected() -> None:
    raw = _frame(MessageType.PING, b"abc")
    patched = bytearray(raw[:HEADER_SIZE])
    struct.pack_into("<I", patched, 20, 4097)
    patched[28:32] = struct.pack("<I", crc32(bytes(patched[:28])))
    with pytest.raises(OversizedPayloadError):
        decode_header(bytes(patched), max_payload=4096)



def test_fragmented_frame_reception() -> None:
    raw = _frame(MessageType.PONG, b"fragmented")
    decoder = FrameDecoder()
    frames = []
    for chunk in (raw[:5], raw[5:17], raw[17:31], raw[31:]):
        frames.extend(decoder.feed(chunk))
    assert [frame.header.message_type for frame in frames] == [MessageType.PONG]
    assert frames[0].payload == b"fragmented"



def test_multiple_frames_in_one_read() -> None:
    decoder = FrameDecoder()
    frames = decoder.feed(_frame(MessageType.PING, b"one", seq=0) + _frame(MessageType.PONG, b"two", seq=1))
    assert [frame.header.message_type for frame in frames] == [MessageType.PING, MessageType.PONG]
    assert [frame.payload for frame in frames] == [b"one", b"two"]



def test_session_and_sequence_mismatch() -> None:
    decoder = FrameDecoder()
    frame_ok = decoder.feed(_frame(MessageType.PING, b"ok", seq=0))[0]
    frame_bad_seq = decoder.feed(_frame(MessageType.PONG, b"bad-seq", seq=2))[0]
    validator = SessionFrameValidator(session_id=SESSION_ID)
    validator.accept(frame_ok)
    with pytest.raises(SequenceMismatchError):
        validator.accept(frame_bad_seq)

    frame_bad_session = decoder.feed(_frame(MessageType.PONG, b"bad-session", seq=1, session=0x42))[0]
    validator = SessionFrameValidator(session_id=SESSION_ID)
    with pytest.raises(SessionMismatchError):
        validator.accept(frame_bad_session)



def test_loopback_transport_is_deterministic() -> None:
    pair = create_loopback_pair(host_read_chunks=[4, 4, 99])
    pair.target.write(b"abcdefghij")
    assert pair.host.read() == b"abcd"
    assert pair.host.read() == b"efgh"
    assert pair.host.read() == b"ij"
