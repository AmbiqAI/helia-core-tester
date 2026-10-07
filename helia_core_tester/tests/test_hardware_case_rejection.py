"""A case the kernel refuses is a per-case failure, not a lost session.

The firmware answers a kernel error with CASE_COMPLETE (performance_ran=0 plus the
kernel status) and moves on to the next case; the host records that case as
rejected and still writes the result bundle. A broken session (ERROR frame,
stalled transport) keeps failing loudly. Sampling sends nothing until every
PMU pass has run, so the host waits one read timeout per pass there.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from helia_core_tester.hardware import session_runner
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.case_bundle import build_abs_s8_case_bundle, build_convolve_s8_case_bundle, load_case_bundle
from helia_core_tester.hardware.fake_target import FakeTargetTransport
from helia_core_tester.hardware.hctp import MessageType
from helia_core_tester.hardware.measurement import CounterPass, counter_passes_for_selection
from helia_core_tester.hardware.pmu_catalog import CounterDescriptor
from helia_core_tester.hardware.session import OPERAND_CHANGED_STATUS, OUTPUT_CHANGED_STATUS, HostSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSES = counter_passes_for_selection({"cpu": "default"})
ARG_ERROR = -1  # ARM_CMSIS_NN_ARG_ERROR


def _bundles(tmp_path: Path):
    return [
        load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_a").manifest_path),
        load_case_bundle(build_convolve_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="conv_b").manifest_path),
        load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_c").manifest_path),
    ]


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
    first_case = result.protocol_trace[: result.protocol_trace.index("RX:CASE_COMPLETE")]
    if stage == "correctness":
        # No output, ACK or sampling for the refused case.
        assert "RX:OUTPUT_END" not in first_case and "TX:RUN_PERFORMANCE" not in first_case
    else:
        # Samples queued before the refusal are dropped.
        assert first_case.count("RX:SAMPLE_RESULT") > 0


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (OUTPUT_CHANGED_STATUS, "timed output differs from first call"),
        (OPERAND_CHANGED_STATUS, "kernel changed a read-only operand"),
    ],
)
def test_integrity_failure_names_the_cause(tmp_path: Path, status: int, reason: str) -> None:
    transport = FakeTargetTransport(rejections={"abs_a": ("performance", status)})

    rejected = HostSession(transport, counter_passes=PASSES).run_many(_bundles(tmp_path)).cases[0]

    assert rejected.rejection is not None and rejected.comparison.passed is False
    assert rejected.rejection.reason == reason


def test_runner_writes_the_bundle_with_the_rejected_case(tmp_path: Path, monkeypatch) -> None:
    transport = FakeTargetTransport(rejections={"conv_b": ("correctness", ARG_ERROR)})
    monkeypatch.setattr(
        session_runner, "open_rtt_session",
        lambda board, serial_no, *, build_dir, counter_passes: (HostSession(transport, counter_passes=counter_passes), transport, 0),
    )
    monkeypatch.setattr(session_runner, "generate_memory_report", lambda board, **_: tmp_path / "memory_report.json")
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


class _SilentSampling(FakeTargetTransport):
    """Answers RUN_PERFORMANCE after `silent` empty reads."""

    def __init__(self, silent: int) -> None:
        super().__init__()
        self._silent = silent
        self._left = 0

    def _handle_frame(self, frame) -> None:
        super()._handle_frame(frame)
        if frame.header.message_type == MessageType.RUN_PERFORMANCE:
            self._left = self._silent

    def read(self, max_bytes: int = 4096) -> bytes:
        if self._left > 0:
            self._left -= 1
            return b""
        return super().read(max_bytes)


@pytest.mark.parametrize(
    ("transport", "passes", "message"),
    [
        # pmu_event_known() refuses an unmapped id: ERROR frame.
        (FakeTargetTransport(), (CounterPass("cpu", 0, (CounterDescriptor("vendor", 0x0C00, "cpu"),)),), r"^message_type=4 status=-1"),
        # The stall names the case it interrupted.
        (_SilentSampling(silent=len(PASSES)), PASSES, r"^Transport stalled .* \(while running case_id='abs_a'\) \(batch 0, "),
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



def test_sampling_waits_one_read_timeout_per_pass(tmp_path: Path) -> None:
    # Firmware flushes samples only after every pass.
    passes = counter_passes_for_selection({"cpu": "all"})
    assert len(passes) > 2
    bundle = _bundles(tmp_path)[:1]
    result = HostSession(_SilentSampling(silent=len(passes) - 1), counter_passes=passes).run_many(bundle)
    assert result.cases[0].statistics.sample_count > 0
    with pytest.raises(RuntimeError, match="Last message sent to target: RUN_PERFORMANCE"):
        HostSession(_SilentSampling(silent=len(passes)), counter_passes=passes).run_many(bundle)
