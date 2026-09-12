from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from helia_core_tester.perf_stream.firmware_messages import CAP_PMU_ARMV8M, decode_kernel_catalog, decode_target_info
from helia_core_tester.perf_stream.hctp import HCTP_FLAG_MORE, FrameDecoder, MessageType

PROJECT_ROOT = Path(__file__).resolve().parents[2]



def test_firmware_target_info_and_catalog_roundtrip_with_python_decoder(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")

    binary = tmp_path / "emit_boot_frames"
    target_info_path = tmp_path / "target_info.bin"
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

    subprocess.run([str(binary), str(target_info_path), str(catalog_path)], check=True, cwd=PROJECT_ROOT)

    decoder = FrameDecoder()
    [target_info_frame] = decoder.feed(target_info_path.read_bytes())
    # The emit tool concatenates every paginated KERNEL_CATALOG chunk (each non-final
    # chunk carries HCTP_FLAG_MORE) into catalog.bin; decode them all and accumulate.
    catalog_frames = decoder.feed(catalog_path.read_bytes())
    assert len(catalog_frames) >= 1

    assert target_info_frame.header.message_type is MessageType.TARGET_INFO
    entries_by_id: dict[int, object] = {}
    for index, frame in enumerate(catalog_frames):
        assert frame.header.message_type is MessageType.KERNEL_CATALOG
        is_final = index == len(catalog_frames) - 1
        assert bool(frame.header.flags & HCTP_FLAG_MORE) != is_final
        for entry in decode_kernel_catalog(frame.payload):
            assert entry.kernel_id not in entries_by_id, f"duplicate kernel_id {entry.kernel_id}"
            entries_by_id[entry.kernel_id] = entry
    catalog = [entries_by_id[kernel_id] for kernel_id in sorted(entries_by_id)]

    target_info = decode_target_info(target_info_frame.payload)

    assert target_info.build_id == "hct-benchmark-server-v0"
    assert target_info.board_id == "apollo510_evb"
    assert target_info.target_cpu == "cortex-m55"
    assert target_info.transport_kind == 1
    assert target_info.max_frame_payload == 256
    assert target_info.runtime_arena_capacity == 32768
    # A host compile has no __PMU_PRESENT, so the PMU capability is absent and no
    # event-counter slots are advertised; max_rx_payload is the 2 KiB rx buffer minus
    # the 32-byte frame header, and the session limits are the firmware's
    # HCT_SERVER_MAX_CASES / HCT_SERVER_MAX_PASSES.
    assert not target_info.capability_flags & CAP_PMU_ARMV8M
    assert target_info.has_pmu is False
    assert target_info.pmu_counter_slots == 0
    assert target_info.max_rx_payload == 2048 - 32
    assert target_info.max_cases_per_session == 32
    assert target_info.max_passes == 16
    assert len(catalog) == 173
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
    assert target_info.catalog_hash == hashlib.sha256(canonical).digest()
