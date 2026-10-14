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
