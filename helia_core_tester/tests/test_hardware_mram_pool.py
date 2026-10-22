"""#377: a case whose MRAM weights and bias exceed the placement pool is refused up front."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.hardware import fake_target, session, transfer
from helia_core_tester.hardware.case_bundle import CaseBundle, build_convolve_s8_case_bundle, load_case_bundle
from helia_core_tester.hardware.fake_target import FakeTargetTransport
from helia_core_tester.hardware.session import HostSession
from helia_core_tester.hardware.transfer import MRAM_LINE_BYTES, MRAM_POOL_BYTES, MRAM_ROW_BYTES, mram_extent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARDWARE = PROJECT_ROOT / "cmake" / "hardware"


@pytest.mark.parametrize(
    ("role", "length", "placement", "expected"),
    [
        ("weights", 13, "mram", 32),
        ("weights", 32, "mram", 32),
        ("weights", 33, "mram", 64),
        ("bias", 48, "mram", 64),
        ("bias", 0, "mram", 0),
        ("input_0", 100, "mram", 0),
        ("weights", 100, "tcm", 0),
        ("weights", 13, "sram", 32),
        ("input_0", 100, "sram", 0),
    ],
)
def test_mram_extent_mirrors_the_firmware(role: str, length: int, placement: str, expected: int) -> None:
    assert mram_extent(role, length, placement) == expected


def _c_define(path: Path, name: str) -> str:
    match = re.search(rf"^#define {name} (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"{name} not in {path}"
    return match.group(1).strip()


def _c_int(expr: str) -> int:
    return eval(re.sub(r"(\d+)u\b", r"\1", expr), {"__builtins__": {}})


def test_pool_constants_match_the_firmware() -> None:
    mram_h = HARDWARE / "benchmark_server_mram.h"
    assert _c_int(_c_define(mram_h, "HCT_MRAM_POOL_BYTES")) == MRAM_POOL_BYTES
    assert _c_int(_c_define(mram_h, "HCT_MRAM_LINE_BYTES")) == MRAM_LINE_BYTES
    assert _c_int(_c_define(HARDWARE / "benchmark_server_session.c", "HCT_MRAM_ROW_BYTES")) == MRAM_ROW_BYTES
    protocol = (HARDWARE / "hctp_protocol.h").read_text(encoding="utf-8")
    assert re.search(rf"HCTP_STATUS_NO_SPACE = {fake_target.HCTP_STATUS_NO_SPACE}\b", protocol)


@pytest.fixture
def conv_bundle(tmp_path: Path) -> CaseBundle:
    return load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path).manifest_path)


def test_bundle_mram_bytes_sum_weights_and_bias(conv_bundle: CaseBundle) -> None:
    expected = sum(
        mram_extent(blob.role, blob.byte_length, "mram") for blob in conv_bundle.streamable_blobs
    )
    assert conv_bundle.mram_bytes_for("mram") == expected > 0
    assert conv_bundle.mram_bytes_for("tcm") == 0


def test_case_over_the_pool_fails_before_the_plan(conv_bundle: CaseBundle, monkeypatch) -> None:
    monkeypatch.setattr(CaseBundle, "mram_bytes_for", lambda self, placement: MRAM_POOL_BYTES + MRAM_LINE_BYTES)
    host = HostSession(FakeTargetTransport(placement="mram"))
    with pytest.raises(RuntimeError, match=rf"needs {MRAM_POOL_BYTES + MRAM_LINE_BYTES} bytes of MRAM.*holds {MRAM_POOL_BYTES}"):
        host.run(conv_bundle)
    assert "TX:SESSION_PLAN" not in host._trace


def test_case_filling_the_pool_exactly_runs(conv_bundle: CaseBundle, monkeypatch) -> None:
    monkeypatch.setattr(CaseBundle, "mram_bytes_for", lambda self, placement: MRAM_POOL_BYTES)
    result = HostSession(FakeTargetTransport(placement="mram")).run(conv_bundle)
    assert len(result.cases) == 1 and result.session_complete_cases == 1


def test_tcm_targets_skip_the_pool_check(conv_bundle: CaseBundle, monkeypatch) -> None:
    # No MRAM placement, no pool: a real tcm footprint is 0.
    monkeypatch.setattr(session, "MRAM_POOL_BYTES", 0)
    result = HostSession(FakeTargetTransport()).run(conv_bundle)
    assert len(result.cases) == 1 and result.session_complete_cases == 1


def test_firmware_refusal_reaches_the_host_as_pool_full(conv_bundle: CaseBundle, monkeypatch) -> None:
    # A host preflight that disagreed with the firmware: the fake (like plan_mram()) refuses CASE_META.
    monkeypatch.setattr(transfer, "MRAM_POOL_BYTES", MRAM_LINE_BYTES)
    host = HostSession(FakeTargetTransport(placement="mram"))
    with pytest.raises(RuntimeError, match=r"^message_type=\d+ status=-8 MRAM pool full"):
        host.run(conv_bundle)
    assert "TX:SESSION_PLAN" in host._trace
    assert "RX:ERROR" in host._trace


def test_workspace_overflow_is_reported_before_the_pool(conv_bundle: CaseBundle, monkeypatch) -> None:
    # Over both limits: like handle_case_meta(), the fake fails on the workspace before plan_mram().
    monkeypatch.setattr(CaseBundle, "workspace_bytes_for", lambda self, placement: 0)
    monkeypatch.setattr(transfer, "MRAM_POOL_BYTES", MRAM_LINE_BYTES)
    host = HostSession(FakeTargetTransport(placement="mram", runtime_arena_capacity=16))
    with pytest.raises(RuntimeError, match=r"exceeds arena capacity 16"):
        host.run(conv_bundle)
    assert "RX:ERROR" in host._trace


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [
        (1, 1, 4 * 16 + 48),
        (16, 4, 4 * 16 + 48),
        (17, 4, 4 * 32 + 48),
        (16, 5, 2 * (4 * 16 + 48)),
        (0, 4, 0),
        (16, 0, 0),
        (-16, 4, 0),
        (2**31, 4, 0),
    ],
)
def test_packed_fc_size_mirrors_ns_cmsis_nn(k: int, n: int, expected: int) -> None:
    assert transfer.packed_fc_s8_size(k, n) == expected


def test_packed_fc_kernel_id_matches_registry_and_firmware() -> None:
    from helia_core_tester.hardware.kernel_registry import lookup_entry_id

    assert lookup_entry_id(PROJECT_ROOT, "arm_fully_connected_per_channel_packed_s8") == transfer.PACKED_FC_S8_KERNEL_ID
    adapters_h = HARDWARE / "benchmark_server_adapters.h"
    assert _c_int(_c_define(adapters_h, "HCT_KERNEL_ID_FULLY_CONNECTED_PER_CHANNEL_PACKED_S8")) == transfer.PACKED_FC_S8_KERNEL_ID


class _Blob:
    def __init__(self, role: str, dimensions: tuple[int, ...]) -> None:
        self.role = role
        self.dimensions = dimensions


def test_derived_extent_counts_only_the_packed_fc_on_mram() -> None:
    # Weights [C=out, N=depth], as the generated hook reads them.
    blobs = [_Blob("input_0", (1, 20)), _Blob("weights", (5, 20)), _Blob("bias", (5,))]
    packed = transfer.packed_fc_s8_size(20, 5)
    expected = mram_extent("weights", packed, "mram")
    assert transfer.derived_mram_extent(transfer.PACKED_FC_S8_KERNEL_ID, blobs, "mram") == expected > packed - MRAM_LINE_BYTES
    assert transfer.derived_mram_extent(transfer.PACKED_FC_S8_KERNEL_ID, blobs, "tcm") == 0
    assert transfer.derived_mram_extent(transfer.PACKED_FC_S8_KERNEL_ID - 1, blobs, "mram") == 0
    assert transfer.derived_mram_extent(transfer.PACKED_FC_S8_KERNEL_ID, blobs[:1], "mram") == 0
    assert transfer.derived_mram_extent(transfer.PACKED_FC_S8_KERNEL_ID, [_Blob("weights", (5,))], "mram") == 0


def test_bundle_mram_bytes_count_the_packed_stream(conv_bundle: CaseBundle, monkeypatch) -> None:
    blobs_only = conv_bundle.mram_bytes_for("mram")
    packed = transfer.derived_mram_extent(transfer.PACKED_FC_S8_KERNEL_ID, conv_bundle.streamable_blobs, "mram")
    assert packed > 0
    monkeypatch.setattr(CaseBundle, "kernel_id", property(lambda self: transfer.PACKED_FC_S8_KERNEL_ID))
    assert conv_bundle.mram_bytes_for("mram") == blobs_only + packed
    assert conv_bundle.mram_bytes_for("tcm") == 0


def test_fake_target_refuses_a_packed_stream_over_the_pool(conv_bundle: CaseBundle, monkeypatch) -> None:
    # Blobs fit, blobs plus the packed stream do not: the fake (like plan_mram()) refuses CASE_META.
    blobs_only = sum(mram_extent(blob.role, blob.byte_length, "mram") for blob in conv_bundle.streamable_blobs)
    monkeypatch.setattr(transfer, "MRAM_POOL_BYTES", blobs_only)
    monkeypatch.setattr(fake_target, "derived_mram_extent", lambda kernel_id, blobs, placement: MRAM_LINE_BYTES)
    host = HostSession(FakeTargetTransport(placement="mram"))
    with pytest.raises(RuntimeError, match=r"^message_type=\d+ status=-8 MRAM pool full"):
        host.run(conv_bundle)
