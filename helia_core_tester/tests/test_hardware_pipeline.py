"""Host-only pieces of `hardware run`: --precision rules, the ELF-hash flash
decision, and the --json summary shape (driven through the fake target)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helia_core_tester.perf_stream import firmware_build
from helia_core_tester.perf_stream.boards import resolve_board
from helia_core_tester.perf_stream.case_bundle import build_abs_s8_case_bundle, load_case_bundle
from helia_core_tester.perf_stream.fake_target import FakeTargetTransport
from helia_core_tester.perf_stream.firmware_build import decide_flash, elf_path, flash_stamp_path, record_flash
from helia_core_tester.perf_stream.hardware_pipeline import (
    StreamOptions,
    apply_precision,
    parse_pmu_groups,
    run_hardware_pipeline,
    validate_fvp_gate,
)
from helia_core_tester.perf_stream.run_summary import build_json_summary
from helia_core_tester.perf_stream.session import HostSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- --precision -------------------------------------------------------------------


def test_precision_forces_float_suite_and_suffix_filter() -> None:
    assert apply_precision("fp16", "int", None) == ("float", "_f16")
    assert apply_precision("FP32", "float", None) == ("float", "_f32")
    assert apply_precision(None, "both", "conv") == ("both", "conv")


def test_precision_rejects_suite_both() -> None:
    with pytest.raises(ValueError, match=r"--precision cannot be combined with --suite both \(it selects float cases only\)\."):
        apply_precision("fp16", "both", None)


def test_precision_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match=r"--precision must be 'fp16' or 'fp32' \(got 'fp64'\)\."):
        apply_precision("fp64", "int", None)


def test_precision_rejects_test_name() -> None:
    with pytest.raises(ValueError, match=r"--precision and --test-name cannot be combined \(both filter via a single substring match\)\."):
        apply_precision("fp32", "float", "reshape")


def test_fvp_gate_and_pmu_groups_parsing() -> None:
    validate_fvp_gate(None)
    validate_fvp_gate("advisory")
    with pytest.raises(ValueError, match="--fvp-gate must be one of"):
        validate_fvp_gate("maybe")
    assert parse_pmu_groups("cpu, memory,,mve ") == ("cpu", "memory", "mve")


# --- flash-only-if-changed ---------------------------------------------------------


def _write_elf(build_dir: Path, payload: bytes) -> Path:
    elf = elf_path(build_dir)
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_bytes(payload)
    return elf


def test_flash_decision_follows_elf_hash(tmp_path: Path) -> None:
    build_dir = tmp_path / "build" / "perf_stream" / "apollo510_evb"
    _write_elf(build_dir, b"firmware-v1")
    serial = 1160002276

    first = decide_flash(build_dir, serial)
    assert first.needed and "no flash stamp" in first.reason

    stamp = record_flash(build_dir, serial, first.digest)
    assert stamp == flash_stamp_path(build_dir, serial) == build_dir / f".flashed-{serial}.sha256"
    assert stamp.read_text().strip() == first.digest

    unchanged = decide_flash(build_dir, serial)
    assert not unchanged.needed and "unchanged" in unchanged.reason

    # The stamp is keyed by serial: another probe still needs a flash.
    assert decide_flash(build_dir, 1160001958).needed

    assert decide_flash(build_dir, serial, force=True).needed

    _write_elf(build_dir, b"firmware-v2")
    changed = decide_flash(build_dir, serial)
    assert changed.needed and "changed" in changed.reason and changed.digest != first.digest


def test_flash_decision_requires_built_elf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        decide_flash(tmp_path, 1)


def test_flash_firmware_skips_flash_target_when_unchanged(tmp_path: Path, monkeypatch) -> None:
    board = resolve_board("apollo510_evb")
    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"firmware")
    built: list[str] = []
    monkeypatch.setattr(firmware_build, "configure", lambda *a, **k: None)
    monkeypatch.setattr(firmware_build, "build", lambda build_dir, target, jobs: built.append(target))

    first = firmware_build.flash_firmware(board, 7, build_dir=build_dir)
    assert first.needed
    assert built == [firmware_build.SERVER_TARGET, firmware_build.FLASH_TARGET]

    built.clear()
    second = firmware_build.flash_firmware(board, 7, build_dir=build_dir)
    assert not second.needed
    assert built == [firmware_build.SERVER_TARGET]

    built.clear()
    forced = firmware_build.flash_firmware(board, 7, build_dir=build_dir, force=True)
    assert forced.needed
    assert built == [firmware_build.SERVER_TARGET, firmware_build.FLASH_TARGET]


# --- --json summary ----------------------------------------------------------------


class _SkippedTest:
    def __init__(self, name: str) -> None:
        self.name = name


def test_json_summary_shape_from_fake_target_session(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_json").manifest_path)
    result = HostSession(FakeTargetTransport(), requested_counter_groups=("cpu",)).run_many([bundle])
    skipped = [(_SkippedTest("conv_x"), "conv_x: operator='Foo' is not bridgeable (bridged today: ['Abs']).")]

    summary = build_json_summary(
        result, skipped, session_id="apollo510_evb-20260912T000000Z", board_id="apollo510_evb",
        bundle=tmp_path / "artifacts" / "reports" / "performance_stream" / "apollo510_evb-20260912T000000Z",
    )
    encoded = json.loads(json.dumps(summary))  # must be JSON-serialisable as-is

    assert set(encoded) == {"session_id", "board", "bundle", "totals", "cases"}
    assert encoded["session_id"] == "apollo510_evb-20260912T000000Z"
    assert encoded["board"] == "apollo510_evb"
    assert encoded["bundle"].endswith("apollo510_evb-20260912T000000Z")
    assert encoded["totals"] == {"ran": 1, "passed": 1, "failed": 0, "skipped": 1}
    ran, skip = encoded["cases"]
    assert set(ran) == {"case_id", "passed", "median_cycles", "skipped_reason"}
    assert ran == {"case_id": "abs_json", "passed": True, "median_cycles": ran["median_cycles"], "skipped_reason": None}
    assert isinstance(ran["median_cycles"], float)
    assert skip["case_id"] == "conv_x" and skip["passed"] is None and skip["median_cycles"] is None
    assert skip["skipped_reason"].startswith("operator='Foo' is not bridgeable")
    assert "bridged today" not in skip["skipped_reason"]


# --- orchestration order ----------------------------------------------------------


def test_run_hardware_pipeline_generates_flashes_then_streams(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.perf_stream import hardware_pipeline

    board = resolve_board("apollo510_evb")
    order: list[str] = []

    def _generate(repo_root, spec, suite):
        order.append(f"generate:{spec.cpu}:{suite}")

    def _flash(spec, serial, *, build_dir, jobs, force_reconfigure):
        order.append(f"flash:{serial}:{build_dir.relative_to(tmp_path)}")
        return firmware_build.FlashDecision(True, "abc", "test")

    def _stream(repo_root, spec, serial, *, build_dir, options, echo, progress_to_stderr):
        order.append(f"stream:{options.suite}:{options.test_name}")
        return hardware_pipeline.HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[])

    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", _generate)
    monkeypatch.setattr(hardware_pipeline, "flash_firmware", _flash)
    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _stream)

    outcome = run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(suite="float", test_name="_f16"), echo=lambda _msg: None,
    )
    assert order == ["generate:cortex-m55:float", "flash:42:build/perf_stream/apollo510_evb", "stream:float:_f16"]
    assert outcome.flash is not None and outcome.flash.needed

    order.clear()
    run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(), skip_generate=True, skip_flash=True, echo=lambda _msg: None,
    )
    assert order == ["stream:int:None"]
