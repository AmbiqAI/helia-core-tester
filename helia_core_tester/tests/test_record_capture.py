"""The record-mode capture file: the C writer's output is what the Python reader
accepts, and every malformed or incomplete file is refused."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.record import capture as cap
from helia_core_tester.record.capture import CaptureError, encode_capture, encode_status, encode_tensor, parse_capture
from helia_core_tester.tests.library_stub import library_header_flags

PROJECT_ROOT = Path(__file__).resolve().parents[2]
C_HOST_DIR = Path(__file__).resolve().parent / "c_host"
HOST_FINISH = PROJECT_ROOT / "helia_core_tester" / "mutation" / "host" / "host_finish.c"

_PROGRAM = r"""
#include <stdlib.h>

#include "test_runtime/helia_test_runtime.h"

static const int8_t s8_expected_output[] = {-128, -1, 0, 127};
static int8_t s8_output[] = {-128, -1, 1, 127};
static const int16_t s16_expected_output[] = {-32768, 32767};
static int16_t s16_output[] = {-32767, 32767};
static const int32_t s32_expected_output[] = {1, 2, 3};
static int32_t s32_output[] = {1, 2, 3};
static const float f32_expected_output[] = {1.0f, -0.0f, 3.5f};
static float f32_output[] = {1.0f, 0.0f, 3.5f};
static const float masked_expected_output[] = {1.0f, 2.0f};
static float masked_output[] = {9.0f, 2.0f};
static const uint8_t masked_mask[] = {1, 0};
static const bool b_expected_output[] = {true, false};
static bool b_output[] = {true, true};
static const int8_t empty_expected_output[1] = {0};
static int8_t empty_output[1] = {0};
static const uint32_t bits_expected[] = {1u, 2u, 3u, 4u};
#if defined(HELIA_HAVE_FP16_TYPE)
static const __fp16 f16_expected_output[] = {(__fp16)1.5f, (__fp16)-2.0f};
static __fp16 f16_output[] = {(__fp16)1.5f, (__fp16)-2.0f};
#elif defined(HELIA_HAVE_FLOAT16_TYPE)
static const _Float16 f16_expected_output[] = {(_Float16)1.5f, (_Float16)-2.0f};
static _Float16 f16_output[] = {(_Float16)1.5f, (_Float16)-2.0f};
#endif

static int32_t run_case(void)
{
    int failures = 0;
    HELIA_VALIDATE_STATUS("Case", HCT_TEST_STATUS);
    HELIA_VALIDATE_OUTPUTS(EXACT_INT, s8_output, s8_expected_output, 4, 1, 0.0f, 0.0f, 20, failures);
    HELIA_VALIDATE_OUTPUTS(TOLERANT_INT, s16_output, s16_expected_output, 2, 2, 0.0f, 0.0f, 20, failures);
    HELIA_VALIDATE_EXACT_INTS(s32_output, (s32_expected_output), 3, 20, failures);
    HELIA_VALIDATE_OUTPUTS(FLOAT, f32_output, f32_expected_output, 3, 0, 1e-5f, 1.3e-6f, 20, failures);
    HELIA_VALIDATE_FLOATS_MASKED(masked_output, masked_expected_output, masked_mask, 2, 1e-5f, 0.0f, 20, failures);
    HELIA_VALIDATE_OUTPUTS(BOOL, b_output, b_expected_output, 2, 0, 0.0f, 0.0f, 20, failures);
    HELIA_VALIDATE_OUTPUTS(EXACT_INT, empty_output, empty_expected_output, 0, 0, 0.0f, 0.0f, 20, failures);
    HELIA_VALIDATE_OUTPUTS(NONE, s8_output, s8_expected_output, 4, 0, 0.0f, 0.0f, 20, failures);
#if defined(HELIA_HAVE_FP16_TYPE) || defined(HELIA_HAVE_FLOAT16_TYPE)
    HELIA_VALIDATE_OUTPUTS(FLOAT, f16_output, f16_expected_output, 2, 0, 1e-5f, 1e-3f, 20, failures);
#endif
    for (int i = 0; i < 4; ++i) {
        HELIA_VALIDATE_FLOAT_BITS(bits_expected[i], bits_expected[i], 0x7f800000u, 0, i, 20, failures);
    }
#ifdef HCT_TEST_CRASH
    abort();
#endif
    HELIA_VALIDATE_RETURN_FAILURES(failures);
}

int main(void)
{
    helia_test_finish(run_case());
}
"""


def _cc() -> str:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("host C compiler not available")
    return cc


@pytest.fixture(scope="module")
def record_binary(tmp_path_factory):
    def build(*defines: str) -> Path:
        work = tmp_path_factory.mktemp("record_capture")
        (work / "case.c").write_text(_PROGRAM, encoding="utf-8")
        binary = work / "case"
        subprocess.run(
            [_cc(), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
             "-DHCT_RECORD", *[f"-D{d}" for d in defines],
             "-I", str(C_HOST_DIR), "-I", str(PROJECT_ROOT / "src"), *library_header_flags(),
             str(work / "case.c"),
             "-Dhelia_test_finish=helia_test_finish_fvp_unused", "-c", "-o", str(work / "unused.o")],
            check=True, capture_output=True, text=True)
        runtime = work / "runtime.o"
        subprocess.run(
            [_cc(), "-std=c11", "-O1", "-c", "-Dhelia_test_finish=helia_test_finish_fvp_unused",
             "-I", str(C_HOST_DIR), "-I", str(PROJECT_ROOT / "src"), *library_header_flags(),
             str(PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.c"), "-o", str(runtime)],
            check=True, capture_output=True, text=True)
        subprocess.run(
            [_cc(), "-std=c11", "-O1", "-DHCT_RECORD", *[f"-D{d}" for d in defines],
             "-I", str(C_HOST_DIR), "-I", str(PROJECT_ROOT / "src"), *library_header_flags(),
             str(work / "case.c"), str(runtime), str(HOST_FINISH),
             str(PROJECT_ROOT / "src" / "test_runtime" / "hct_record.c"), "-lm", "-o", str(binary)],
            check=True, capture_output=True, text=True)
        return binary

    return build


def _run(binary: Path, capture: Path | None, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k != "HCT_RECORD_PATH"}
    if capture is not None:
        env["HCT_RECORD_PATH"] = str(capture)
    env.update(extra_env or {})
    return subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30)


def test_c_writer_round_trips_through_the_reader(record_binary, tmp_path) -> None:
    binary = record_binary("HCT_TEST_STATUS=ARM_CMSIS_NN_SUCCESS")
    path = tmp_path / "case.hctr"
    proc = _run(binary, path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = cap.read_capture(path)
    by_symbol = {t.symbol: t for t in got.tensors}

    s8 = by_symbol["s8_expected_output"]
    assert (s8.dtype, s8.mode, s8.shape) == ("int8", "exact", (4,))
    assert s8.actual.tolist() == [-128, -1, 1, 127] and s8.golden.tolist() == [-128, -1, 0, 127]
    s16 = by_symbol["s16_expected_output"]
    assert (s16.dtype, s16.mode, s16.tolerance) == ("int16", "tolerant", 2)
    assert s16.actual.tolist() == [-32767, 32767]
    assert by_symbol["s32_expected_output"].name == "(s32_expected_output)"
    f32 = by_symbol["f32_expected_output"]
    assert f32.dtype == "float32" and f32.atol == pytest.approx(1e-5) and f32.rtol == pytest.approx(1.3e-6)
    assert np.signbit(f32.golden[1]) and not np.signbit(f32.actual[1])
    masked = by_symbol["masked_expected_output"]
    assert masked.mask.tolist() == [1, 0]
    b = by_symbol["b_expected_output"]
    assert b.dtype == "bool" and b.actual.tolist() == [True, True] and b.golden.tolist() == [True, False]
    empty = by_symbol["empty_expected_output"]
    assert empty.shape == (0,) and empty.actual.size == 0
    if "f16_expected_output" in by_symbol:
        f16 = by_symbol["f16_expected_output"]
        assert f16.dtype == "float16" and f16.actual.tolist() == [1.5, -2.0]
    # NONE validates nothing, so it records nothing; a per-element validator records its call site once.
    assert len([t for t in got.tensors if t.symbol == "s8_expected_output"]) == 1
    assert got.unrecordable == ["bits_expected[i]"]
    assert [(s.label, s.status, s.expected) for s in got.statuses] == [("Case", 0, 0)]


def test_a_declined_status_is_recorded_and_still_ends_the_capture(record_binary, tmp_path) -> None:
    binary = record_binary("HCT_TEST_STATUS=ARM_CMSIS_NN_ARG_ERROR")
    path = tmp_path / "case.hctr"
    proc = _run(binary, path)
    assert proc.returncode == 1
    got = cap.read_capture(path)
    assert [(s.status, s.expected) for s in got.statuses] == [(-1, 0)]
    assert got.tensors == []


def test_missing_capture_path_fails_the_run(record_binary, tmp_path) -> None:
    proc = _run(record_binary("HCT_TEST_STATUS=ARM_CMSIS_NN_SUCCESS"), None)
    assert proc.returncode == 3
    assert "HCT_RECORD_ERROR: HCT_RECORD_PATH is not set" in proc.stderr


def test_unwritable_capture_path_fails_the_run(record_binary, tmp_path) -> None:
    proc = _run(record_binary("HCT_TEST_STATUS=ARM_CMSIS_NN_SUCCESS"), tmp_path / "missing" / "case.hctr")
    assert proc.returncode == 3
    assert "cannot open HCT_RECORD_PATH" in proc.stderr


def test_a_run_that_dies_leaves_a_capture_the_reader_refuses(record_binary, tmp_path) -> None:
    path = tmp_path / "case.hctr"
    proc = _run(record_binary("HCT_TEST_STATUS=ARM_CMSIS_NN_SUCCESS", "HCT_TEST_CRASH=1"), path)
    assert proc.returncode != 0
    with pytest.raises(CaptureError, match="truncated"):
        cap.read_capture(path)


def _valid() -> bytes:
    return encode_capture([
        encode_tensor("a_expected_output", np.array([1, 2], np.int8), np.array([1, 2], np.int8)),
        encode_status("Case", 0),
    ])


def test_python_encoder_matches_the_reader() -> None:
    got = parse_capture(_valid())
    assert [t.symbol for t in got.tensors] == ["a_expected_output"]
    assert got.statuses[0].label == "Case"


def test_empty_capture_is_valid() -> None:
    got = parse_capture(encode_capture([]))
    assert got.tensors == [] and got.statuses == [] and got.unrecordable == []


def test_every_truncation_is_refused() -> None:
    data = _valid()
    for length in range(len(data)):
        with pytest.raises(CaptureError):
            parse_capture(data[:length])


def _refit_crc(data: bytes) -> bytes:
    body = data[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: b"XXXX" + d[4:], "bad magic"),
        (lambda d: d[:4] + struct.pack("<H", 2) + d[6:], "unsupported version 2"),
        (lambda d: d[:6] + struct.pack("<H", 7) + d[8:], "reserved field is 7"),
        (lambda d: d[:-4] + struct.pack("<I", (struct.unpack("<I", d[-4:])[0] ^ 1)), "crc"),
        (lambda d: d + b"\0", "after the end record"),
        (lambda d: _refit_crc(d[:-8] + struct.pack("<I", 5) + d[-4:]), "counts 5 records"),
    ],
)
def test_corrupt_framing_is_refused(mutate, message) -> None:
    with pytest.raises(CaptureError, match=message):
        parse_capture(mutate(_valid()))


def test_flipped_payload_byte_is_caught_by_the_crc() -> None:
    data = bytearray(_valid())
    data[30] ^= 0x40
    with pytest.raises(CaptureError):
        parse_capture(bytes(data))


def _tensor_bytes(**overrides) -> bytes:
    fields = {"kind": cap.KIND_TENSOR, "dtype": 1, "elem": 1, "name": b"x_expected_output", "mode": cap.MODE_EXACT,
              "flags": 0, "ndim": 1, "reserved": 0, "dims": (2,), "nbytes": 2, "payload": b"\x01\x02\x01\x02"}
    fields.update(overrides)
    out = struct.pack("<BBHH", fields["kind"], fields["dtype"], fields["elem"], len(fields["name"])) + fields["name"]
    out += struct.pack("<BBBB", fields["mode"], fields["flags"], fields["ndim"], fields["reserved"])
    out += struct.pack("<" + "I" * len(fields["dims"]), *fields["dims"])
    out += struct.pack("<qddI", 0, 0.0, 0.0, fields["nbytes"]) + fields["payload"]
    return out


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"kind": 9}, "unknown record kind 9"),
        ({"dtype": 99}, "unknown dtype code 99"),
        ({"elem": 2}, "element size 2 does not match int8"),
        ({"mode": 0}, "unknown validation mode 0"),
        ({"flags": 2}, "unknown flags"),
        ({"reserved": 1}, "reserved field is 1"),
        ({"ndim": 0, "dims": ()}, "no dimensions"),
        ({"nbytes": 3}, "3 bytes for shape"),
        ({"name": b""}, "empty name"),
        ({"name": b"\xff\xfe"}, "not UTF-8"),
    ],
)
def test_malformed_tensor_record_is_refused(overrides, message) -> None:
    with pytest.raises(CaptureError, match=message):
        parse_capture(encode_capture([_tensor_bytes(**overrides)]))


def test_tensor_recorded_twice_is_refused() -> None:
    record = encode_tensor("(a_expected_output)", np.array([1], np.int8), np.array([1], np.int8))
    again = encode_tensor("a_expected_output", np.array([1], np.int8), np.array([1], np.int8))
    with pytest.raises(CaptureError, match="recorded twice"):
        parse_capture(encode_capture([record, again]))


def test_status_with_a_wrong_dtype_is_refused() -> None:
    bad = struct.pack("<BBHH", cap.KIND_STATUS, 1, 1, 1) + b"S" + struct.pack("<ii", 0, 0)
    with pytest.raises(CaptureError, match="expected int32/4"):
        parse_capture(encode_capture([bad]))


def test_unrecordable_with_a_dtype_is_refused() -> None:
    bad = struct.pack("<BBHH", cap.KIND_UNRECORDABLE, 1, 1, 1) + b"x"
    with pytest.raises(CaptureError, match="carries dtype"):
        parse_capture(encode_capture([bad]))


def test_missing_file_is_refused(tmp_path) -> None:
    with pytest.raises(CaptureError, match="does not exist"):
        cap.read_capture(tmp_path / "none.hctr")


@pytest.mark.parametrize(
    "name, symbol",
    [("x", "x"), ("(x)", "x"), (" ((x)) ", "x"), ("(a)+(b)", "(a)+(b)"), ("x[i]", "x[i]")],
)
def test_symbol_strips_only_wrapping_parentheses(name, symbol) -> None:
    record = parse_capture(encode_capture([encode_tensor(name, np.array([1], np.int8), np.array([1], np.int8))]))
    assert record.tensors[0].symbol == symbol
