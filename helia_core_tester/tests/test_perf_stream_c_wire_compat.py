from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from helia_core_tester.perf_stream.hctp import MessageType, encode_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]



def test_c_decoder_matches_python_wire_format(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")

    binary = tmp_path / "hctp_host_sanity"
    frame_path = tmp_path / "frame.bin"
    frame = encode_frame(
        MessageType.CASE_META,
        b"payload-for-c-wire-check",
        session_id=0x12345678,
        sequence_id=7,
        flags=0xA5A5A5A5,
    )
    frame_path.write_bytes(frame)

    subprocess.run(
        [
            cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(PROJECT_ROOT / "cmake" / "perf_stream"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "hctp_protocol.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "hctp_host_sanity_main.c"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    result = subprocess.run([str(binary), str(frame_path)], check=True, capture_output=True, text=True)
    stdout = result.stdout.strip()

    assert "magic=0x31544348" in stdout
    assert "version=1" in stdout
    assert f"type={int(MessageType.CASE_META)}" in stdout
    assert "flags=2779096485" in stdout
    assert "session=0x12345678" in stdout
    assert "sequence=7" in stdout
    assert "payload_length=24" in stdout
