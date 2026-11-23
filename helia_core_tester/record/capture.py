"""The record-mode capture file (written by src/test_runtime/hct_record.c).

Layout, all integers little-endian:

    header   b"HCTR", u16 version, u16 reserved (0)
    record   u8 kind, u8 dtype, u16 elem_size, u16 name_len, name (UTF-8), body
      tensor        u8 mode, u8 flags, u8 ndim, u8 reserved, u32 dims[ndim],
                    i64 tolerance, f64 atol, f64 rtol, u32 nbytes,
                    actual[nbytes], golden[nbytes], mask[count] when flags & 1
      status        i32 status, i32 expected
      unrecordable  (no body)
    end      u8 0xFF, u32 record_count, u32 crc32 (zlib) of every byte before it

The reader is fail-closed: a bad magic or version, an unknown kind or dtype, a size
that disagrees with the dtype, a truncated file, a missing end record, a wrong record
count or CRC, bytes after the end, or a tensor named twice is a CaptureError.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

MAGIC = b"HCTR"
VERSION = 1

KIND_TENSOR = 1
KIND_STATUS = 2
KIND_UNRECORDABLE = 3
KIND_END = 0xFF

MODE_EXACT = 1
MODE_TOLERANT = 2
MODE_FLOAT = 3
MODE_BOOL = 4
MODES = {MODE_EXACT: "exact", MODE_TOLERANT: "tolerant", MODE_FLOAT: "float", MODE_BOOL: "bool"}

FLAG_MASK = 1

DTYPES: Dict[int, Tuple[str, np.dtype]] = {
    1: ("int8", np.dtype("<i1")),
    2: ("uint8", np.dtype("<u1")),
    3: ("int16", np.dtype("<i2")),
    4: ("uint16", np.dtype("<u2")),
    5: ("int32", np.dtype("<i4")),
    6: ("uint32", np.dtype("<u4")),
    7: ("int64", np.dtype("<i8")),
    8: ("uint64", np.dtype("<u8")),
    9: ("float16", np.dtype("<f2")),
    10: ("float32", np.dtype("<f4")),
    11: ("float64", np.dtype("<f8")),
    12: ("bool", np.dtype("u1")),
}
DTYPE_CODES = {name: code for code, (name, _) in DTYPES.items()}
FLOAT_DTYPES = frozenset({"float16", "float32", "float64"})


class CaptureError(ValueError):
    """The capture file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    mode: str
    shape: Tuple[int, ...]
    tolerance: int
    atol: float
    rtol: float
    actual: np.ndarray
    golden: np.ndarray
    mask: Optional[np.ndarray] = None

    @property
    def symbol(self) -> str:
        """The golden's C identifier: the recorded expression without wrapping parentheses."""
        return _strip_parens(self.name)


@dataclass(frozen=True)
class StatusRecord:
    label: str
    status: int
    expected: int


@dataclass
class Capture:
    path: Path
    version: int
    tensors: List[TensorRecord] = field(default_factory=list)
    statuses: List[StatusRecord] = field(default_factory=list)
    unrecordable: List[str] = field(default_factory=list)

    def tensor(self, symbol: str) -> TensorRecord:
        for record in self.tensors:
            if record.symbol == symbol:
                return record
        raise KeyError(symbol)


def _strip_parens(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1].strip()
    return text


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth < 0:
            return False
    return depth == 0


class _Reader:
    def __init__(self, data: bytes, path: Path):
        self.data = data
        self.pos = 0
        self.path = path

    def fail(self, what: str) -> CaptureError:
        return CaptureError(f"capture {self.path}: {what} (at byte {self.pos})")

    def take(self, count: int, what: str) -> bytes:
        if count < 0 or self.pos + count > len(self.data):
            raise self.fail(f"truncated while reading {what}")
        chunk = self.data[self.pos:self.pos + count]
        self.pos += count
        return chunk

    def unpack(self, fmt: str, what: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt), what))


def read_capture(path: Union[str, Path]) -> Capture:
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise CaptureError(f"capture {path} does not exist (the harness wrote nothing)") from exc
    except OSError as exc:
        raise CaptureError(f"capture {path} is unreadable: {exc}") from exc
    return parse_capture(data, path)


def parse_capture(data: bytes, path: Union[str, Path] = "<bytes>") -> Capture:
    reader = _Reader(data, Path(path))
    magic = reader.take(4, "magic")
    if magic != MAGIC:
        raise reader.fail(f"bad magic {magic!r}, expected {MAGIC!r}")
    version, reserved = reader.unpack("HH", "header")
    if version != VERSION:
        raise reader.fail(f"unsupported version {version}, expected {VERSION}")
    if reserved != 0:
        raise reader.fail(f"header reserved field is {reserved}, expected 0")
    capture = Capture(path=Path(path), version=version)
    records = 0
    seen: Dict[str, int] = {}
    while True:
        (kind,) = reader.unpack("B", "record kind")
        if kind == KIND_END:
            (count,) = reader.unpack("I", "record count")
            crc_end = reader.pos
            (crc,) = reader.unpack("I", "crc")
            if count != records:
                raise reader.fail(f"end record counts {count} records, the file holds {records}")
            actual_crc = zlib.crc32(data[:crc_end]) & 0xFFFFFFFF
            if crc != actual_crc:
                raise reader.fail(f"crc 0x{crc:08x} does not match the contents (0x{actual_crc:08x})")
            if reader.pos != len(data):
                raise reader.fail(f"{len(data) - reader.pos} byte(s) after the end record")
            return capture
        dtype_code, elem_size, name_len = reader.unpack("BHH", "record head")
        if name_len == 0:
            raise reader.fail("record with an empty name")
        try:
            name = reader.take(name_len, "record name").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise reader.fail(f"record name is not UTF-8: {exc}") from exc
        records += 1
        if kind == KIND_TENSOR:
            record = _tensor(reader, name, dtype_code, elem_size)
            if record.symbol in seen:
                raise reader.fail(f"golden {record.symbol!r} recorded twice")
            seen[record.symbol] = records
            capture.tensors.append(record)
        elif kind == KIND_STATUS:
            if dtype_code != DTYPE_CODES["int32"] or elem_size != 4:
                raise reader.fail(f"status record {name!r} has dtype {dtype_code}/{elem_size}, expected int32/4")
            status, expected = reader.unpack("ii", "status record")
            capture.statuses.append(StatusRecord(name, status, expected))
        elif kind == KIND_UNRECORDABLE:
            if dtype_code != 0 or elem_size != 0:
                raise reader.fail(f"unrecordable record {name!r} carries dtype {dtype_code}/{elem_size}")
            if name not in capture.unrecordable:
                capture.unrecordable.append(name)
        else:
            raise reader.fail(f"unknown record kind {kind}")


def _tensor(reader: _Reader, name: str, dtype_code: int, elem_size: int) -> TensorRecord:
    if dtype_code not in DTYPES:
        raise reader.fail(f"tensor {name!r} has unknown dtype code {dtype_code}")
    dtype_name, np_dtype = DTYPES[dtype_code]
    if elem_size != np_dtype.itemsize:
        raise reader.fail(f"tensor {name!r}: element size {elem_size} does not match {dtype_name}")
    mode, flags, ndim, reserved = reader.unpack("BBBB", "tensor head")
    if mode not in MODES:
        raise reader.fail(f"tensor {name!r} has unknown validation mode {mode}")
    if flags & ~FLAG_MASK:
        raise reader.fail(f"tensor {name!r} has unknown flags 0x{flags:02x}")
    if reserved != 0:
        raise reader.fail(f"tensor {name!r} reserved field is {reserved}")
    if ndim == 0:
        raise reader.fail(f"tensor {name!r} has no dimensions")
    shape = reader.unpack("I" * ndim, "tensor dims")
    count = 1
    for dim in shape:
        count *= dim
    tolerance, atol, rtol, nbytes = reader.unpack("qddI", "tensor tolerance and size")
    if nbytes != count * elem_size:
        raise reader.fail(f"tensor {name!r}: {nbytes} bytes for shape {list(shape)} of {dtype_name}")
    actual = np.frombuffer(reader.take(nbytes, f"tensor {name!r} actual"), dtype=np_dtype).copy()
    golden = np.frombuffer(reader.take(nbytes, f"tensor {name!r} golden"), dtype=np_dtype).copy()
    mask = None
    if flags & FLAG_MASK:
        mask = np.frombuffer(reader.take(count, f"tensor {name!r} mask"), dtype=np.uint8).copy()
    if dtype_name == "bool":
        actual, golden = actual != 0, golden != 0
    return TensorRecord(
        name=name,
        dtype=dtype_name,
        mode=MODES[mode],
        shape=tuple(shape),
        tolerance=tolerance,
        atol=atol,
        rtol=rtol,
        actual=actual.reshape(shape),
        golden=golden.reshape(shape),
        mask=None if mask is None else mask.reshape(shape),
    )


def _head(kind: int, dtype: int, elem_size: int, name: str) -> bytes:
    raw = name.encode("utf-8")
    return struct.pack("<BBHH", kind, dtype, elem_size, len(raw)) + raw


def encode_tensor(
    name: str,
    actual: np.ndarray,
    golden: np.ndarray,
    mode: int = MODE_EXACT,
    mask: Optional[np.ndarray] = None,
    tolerance: int = 0,
    atol: float = 0.0,
    rtol: float = 0.0,
    dtype: Optional[str] = None,
) -> bytes:
    """One tensor record, as the C writer emits it (for tests and fixtures)."""
    dtype = dtype or ("bool" if np.asarray(actual).dtype == np.bool_ else np.asarray(actual).dtype.name)
    np_dtype = DTYPES[DTYPE_CODES[dtype]][1]
    flat_actual = np.asarray(actual).astype(np_dtype).reshape(-1)
    flat_golden = np.asarray(golden).astype(np_dtype).reshape(-1)
    body = struct.pack("<BBBBI", mode, FLAG_MASK if mask is not None else 0, 1, 0, flat_actual.size)
    body += struct.pack("<qddI", tolerance, atol, rtol, flat_actual.nbytes)
    body += flat_actual.tobytes() + flat_golden.tobytes()
    if mask is not None:
        body += np.asarray(mask, dtype=np.uint8).reshape(-1).tobytes()
    return _head(KIND_TENSOR, DTYPE_CODES[dtype], np_dtype.itemsize, name) + body


def encode_status(label: str, status: int, expected: int = 0) -> bytes:
    return _head(KIND_STATUS, DTYPE_CODES["int32"], 4, label) + struct.pack("<ii", status, expected)


def encode_unrecordable(golden: str) -> bytes:
    return _head(KIND_UNRECORDABLE, 0, 0, golden)


def encode_capture(records: Sequence[bytes], version: int = VERSION, count: Optional[int] = None) -> bytes:
    """A whole capture file around already-encoded records (for tests and fixtures)."""
    data = MAGIC + struct.pack("<HH", version, 0) + b"".join(records)
    data += struct.pack("<BI", KIND_END, len(records) if count is None else count)
    return data + struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)
