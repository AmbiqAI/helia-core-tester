from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from helia_core_tester.perf_stream.firmware_messages import decode_catalog_payload, decode_hello_payload
from helia_core_tester.perf_stream.hctp import HCTP_FLAG_MORE, FrameDecoder, MessageType

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
    # F008: the emit tool concatenates every paginated CAPABILITIES chunk (each non-final
    # chunk carries HCTP_FLAG_MORE) into catalog.bin; decode them all and accumulate.
    catalog_frames = decoder.feed(catalog_path.read_bytes())
    assert len(catalog_frames) >= 1

    assert hello_frame.header.message_type is MessageType.HELLO
    entries_by_id: dict[int, object] = {}
    for index, frame in enumerate(catalog_frames):
        assert frame.header.message_type is MessageType.CAPABILITIES
        is_final = index == len(catalog_frames) - 1
        assert bool(frame.header.flags & HCTP_FLAG_MORE) != is_final
        for entry in decode_catalog_payload(frame.payload):
            assert entry.kernel_id not in entries_by_id, f"duplicate kernel_id {entry.kernel_id}"
            entries_by_id[entry.kernel_id] = entry
    catalog = [entries_by_id[kernel_id] for kernel_id in sorted(entries_by_id)]

    hello = decode_hello_payload(hello_frame.payload)

    assert hello.build_id == "hct-benchmark-server-v0"
    assert hello.board_id == "apollo510_evb"
    assert hello.target_cpu == "cortex-m55"
    assert hello.transport_kind == 1
    assert hello.max_frame_payload == 256
    assert hello.runtime_arena_capacity == 32768
    assert len(catalog) == 163
    assert catalog[0].kernel_id == 1
    assert catalog[0].canonical_name == "arm_abs_s8"
    assert catalog[5].kernel_id == 6
    assert catalog[5].operator_family == "BasicMathFunctions"
    assert catalog[5].canonical_name == "arm_maximum_s8"
    assert catalog[6].kernel_id == 7
    assert catalog[6].canonical_name == "arm_minimum_s8"

    canonical = json.dumps(
        json.loads((PROJECT_ROOT / "cmake" / "perf_stream" / "kernel_catalog.json").read_text()),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hello.catalog_hash == hashlib.sha256(canonical).digest()
