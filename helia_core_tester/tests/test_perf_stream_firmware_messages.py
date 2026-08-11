from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from helia_core_tester.perf_stream.firmware_messages import decode_catalog_payload, decode_hello_payload
from helia_core_tester.perf_stream.hctp import FrameDecoder, MessageType

PROJECT_ROOT = Path(__file__).resolve().parents[2]



def test_firmware_hello_and_catalog_roundtrip_with_python_decoder(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")

    binary = tmp_path / "emit_boot_frames"
    hello_path = tmp_path / "hello.bin"
    catalog_path = tmp_path / "catalog.bin"

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
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_catalog.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_messages.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_host_emit_main.c"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    subprocess.run([str(binary), str(hello_path), str(catalog_path)], check=True, cwd=PROJECT_ROOT)

    decoder = FrameDecoder()
    [hello_frame] = decoder.feed(hello_path.read_bytes())
    [catalog_frame] = decoder.feed(catalog_path.read_bytes())

    assert hello_frame.header.message_type is MessageType.HELLO
    assert catalog_frame.header.message_type is MessageType.CAPABILITIES

    hello = decode_hello_payload(hello_frame.payload)
    catalog = decode_catalog_payload(catalog_frame.payload)

    assert hello.build_id == "hct-benchmark-server-v0"
    assert hello.board_id == "apollo510_evb"
    assert hello.target_cpu == "cortex-m55"
    assert hello.transport_kind == 1
    assert hello.max_frame_payload == 256
    assert hello.runtime_arena_capacity == 32768
    assert len(catalog) == 7
    assert catalog[0].canonical_name == "arm_abs_s8"
    assert catalog[1].canonical_name == "arm_convolve_s8"
    assert catalog[1].scratch_bytes == 64
    assert catalog[-1].canonical_name == "arm_maximum_s8"

    canonical = json.dumps(
        json.loads((PROJECT_ROOT / "cmake" / "perf_stream" / "kernel_catalog.json").read_text()),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hello.catalog_hash == hashlib.sha256(canonical).digest()
