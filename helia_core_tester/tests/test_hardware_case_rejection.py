"""A case the kernel refuses is a per-case failure, not a lost session.

The firmware answers a kernel error with CASE_COMPLETE (performance_ran=0 plus the
kernel status) and moves on to the next case; the host records that case as
rejected and still writes the result bundle. A broken session (ERROR frame,
stalled transport) keeps failing loudly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from helia_core_tester.hardware import session_runner, wire
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.case_bundle import build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from helia_core_tester.hardware.fake_target import FakeTargetTransport
from helia_core_tester.hardware.measurement import CounterPass, counter_passes_for_selection
from helia_core_tester.hardware.pmu_catalog import CounterDescriptor
from helia_core_tester.hardware.session import HostSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSES = counter_passes_for_selection({"cpu": "default"})
ARG_ERROR = -1  # ARM_CMSIS_NN_ARG_ERROR


def _bundles(tmp_path: Path):
    return [
        load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_a").manifest_path),
        load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_b").manifest_path),
        load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_c").manifest_path),
    ]


def test_case_complete_carries_the_status_only_for_a_rejection() -> None:
    normal = wire.CaseComplete("c", 64)
    assert len(wire.encode_case_complete(normal)) == 2 + 1 + 1 + 1 + 4
    rejected = wire.CaseComplete("c", 64, correctness_ran=False, performance_ran=False, kernel_status=ARG_ERROR)
    payload = wire.encode_case_complete(rejected)
    assert len(payload) == 2 + 1 + 1 + 1 + 4 + 4
    assert wire.decode_case_complete(payload) == rejected


@pytest.mark.parametrize("stage", ["correctness", "performance"])
def test_rejected_case_fails_alone_and_the_session_continues(tmp_path: Path, stage: str) -> None:
    transport = FakeTargetTransport(rejections={"abs_a": (stage, ARG_ERROR)})
    seen: list[str] = []

    result = HostSession(transport, counter_passes=PASSES).run_many(
        _bundles(tmp_path), on_case_complete=lambda case: seen.append(case.case_bundle.case_id)
    )

    assert seen == ["abs_a", "conv_b", "abs_c"]
    assert result.session_complete_cases == 3
    rejected, *others = result.cases
    assert rejected.rejection is not None
    assert (rejected.rejection.kernel_status, rejected.rejection.stage) == (ARG_ERROR, stage)
    assert rejected.rejection.reason == f"kernel returned -1 in {stage} run"
    assert rejected.comparison.passed is False
    assert rejected.samples == () and rejected.statistics.sample_count == 0
    for case in others:
        assert case.rejection is None and case.comparison.passed
        assert case.statistics.sample_count > 0 and case.statistics.median_cycles > 0
    if stage == "correctness":
        # No output, ACK or sampling for the refused case.
        first_case = result.protocol_trace[: result.protocol_trace.index("RX:CASE_COMPLETE")]
        assert "RX:OUTPUT_END" not in first_case and "TX:RUN_PERFORMANCE" not in first_case


def test_runner_writes_the_bundle_with_the_rejected_case(tmp_path: Path, monkeypatch) -> None:
    transport = FakeTargetTransport(rejections={"conv_b": ("correctness", ARG_ERROR)})
    monkeypatch.setattr(
        session_runner, "open_rtt_session",
        lambda board, serial_no, *, build_dir, counter_passes: (HostSession(transport, counter_passes=counter_passes), transport, 0),
    )
    monkeypatch.setattr(session_runner, "generate_memory_report", lambda board, project_root=None, build_dir=None: tmp_path / "memory_report.json")
    (tmp_path / "memory_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cmake" / "hardware").mkdir(parents=True)
    (tmp_path / "cmake" / "hardware" / "kernel_catalog.json").write_text("[]", encoding="utf-8")

    result, bundle_root = session_runner.run_case_bundles(
        tmp_path, _bundles(tmp_path), board=resolve_board("apollo510_evb"), serial_no=1,
        counter_passes=PASSES, session_id="reject-session", build_dir=tmp_path,
    )

    assert [case.rejection is not None for case in result.cases] == [False, True, False]
    summary = json.loads((bundle_root / "session_summary.json").read_text())
    assert (summary["case_count"], summary["passed_cases"], summary["failed_cases"]) == (3, 2, 1)
    assert summary["rejected_cases"] == ["conv_b"]
    rows = {row["case_id"]: row for row in json.loads((bundle_root / "cases.json").read_text())}
    expected = {"kernel_status": ARG_ERROR, "stage": "correctness", "reason": "kernel returned -1 in correctness run"}
    assert rows["conv_b"]["rejection"] == expected
    assert rows["abs_a"]["rejection"] is None and rows["abs_c"]["rejection"] is None
    correctness = json.loads((bundle_root / "correctness" / "conv_b.json").read_text())
    assert correctness["passed"] is False and correctness["rejection"] == expected
    with (bundle_root / "case_summary.csv").open(encoding="utf-8") as handle:
        csv_rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    assert csv_rows["conv_b"]["comparison_passed"] == "false" and csv_rows["conv_b"]["sample_count"] == "0"
    assert all(csv_rows[case_id]["comparison_passed"] == "true" for case_id in ("abs_a", "abs_c"))
    assert float(csv_rows["abs_c"]["median_cycles"]) > 0
    failures = ElementTree.parse(bundle_root / "junit.xml").getroot().findall("testcase/failure")
    assert [(f.get("message"), f.text) for f in failures] == [("kernel rejected case", expected["reason"])]


class _StallingTarget(FakeTargetTransport):
    """Goes silent after a few reads."""

    def __init__(self, reads: int) -> None:
        super().__init__()
        self._reads_left = reads

    def read(self, max_bytes: int = 4096) -> bytes:
        self._reads_left -= 1
        return super().read(max_bytes) if self._reads_left >= 0 else b""


@pytest.mark.parametrize(
    ("transport", "passes", "message"),
    [
        # pmu_event_known() refuses an unmapped id: ERROR frame.
        (FakeTargetTransport(), (CounterPass("cpu", 0, (CounterDescriptor("vendor", 0x0C00, "cpu"),)),), r"^message_type=4 status=-1"),
        (_StallingTarget(reads=200), PASSES, r"^Transport stalled"),
    ],
    ids=["error-frame", "transport-stall"],
)
def test_broken_session_still_fails_without_a_bundle(tmp_path: Path, monkeypatch, transport, passes, message) -> None:
    monkeypatch.setattr(
        session_runner, "open_rtt_session",
        lambda board, serial_no, *, build_dir, counter_passes: (HostSession(transport, counter_passes=counter_passes), transport, 0),
    )
    monkeypatch.setattr(session_runner, "write_result_bundle", lambda *a, **k: pytest.fail("bundle written"))

    with pytest.raises(RuntimeError, match=message):
        session_runner.run_case_bundles(
            tmp_path, _bundles(tmp_path), board=resolve_board("apollo510_evb"), serial_no=1,
            counter_passes=passes, session_id="broken", build_dir=tmp_path,
        )
