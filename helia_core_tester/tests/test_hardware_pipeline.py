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
    parse_pmu_counters,
    parse_pmu_groups,
    resolve_pmu_options,
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


def test_pmu_counters_parsing_and_deprecated_groups_alias() -> None:
    assert parse_pmu_counters(["mve:all", "cpu:default"]) == {"mve": "all", "cpu": "default"}
    assert parse_pmu_counters(["mve:ARM_PMU_MVE_STALL, ARM_PMU_MVE_PRED"]) == {"mve": ["ARM_PMU_MVE_STALL", "ARM_PMU_MVE_PRED"]}
    for bad, message in (
        (["mve"], "expects GROUP:SELECTION"),
        (["dsp:all"], "unknown group 'dsp'"),
        (["mve:ARM_PMU_NOPE"], "Unsupported counter 'ARM_PMU_NOPE' for group 'mve'. Valid names: ARM_PMU_MVE_INST_RETIRED"),
        (["cpu:all", "cpu:default"], "given more than once"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_pmu_counters(bad)

    # Neither flag: every group at its default. The deprecated --pmu-groups maps each
    # listed group to GROUP:default and warns.
    assert resolve_pmu_options([], None) == {"cpu": "default", "memory": "default", "mve": "default"}
    warnings: list[str] = []
    assert resolve_pmu_options([], "mve,cpu", warn=warnings.append) == {"mve": "default", "cpu": "default"}
    assert warnings and "deprecated" in warnings[0] and "--pmu-counters mve:default" in warnings[0]
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_pmu_options(["mve:all"], "cpu")
    assert StreamOptions().pmu_counters == {"cpu": "default", "memory": "default", "mve": "default"}


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
    result = HostSession(FakeTargetTransport()).run_many([bundle])
    skipped = [(_SkippedTest("conv_x"), "conv_x: operator='Foo' is not bridgeable (bridged today: ['Abs']).")]

    timing = {"generate_s": 0.0, "build_s": 2.5, "flash_s": 0.0, "stream_s": 1.25, "total_s": 3.75, "batch_count": 1, "cases": {"abs_json": 1.25}}
    summary = build_json_summary(
        result, skipped, session_id="apollo510_evb-20260912T000000Z", board_id="apollo510_evb",
        bundle=tmp_path / "artifacts" / "reports" / "performance_stream" / "apollo510_evb-20260912T000000Z",
        timing=timing,
    )
    encoded = json.loads(json.dumps(summary))  # must be JSON-serialisable as-is

    assert set(encoded) == {"session_id", "board", "bundle", "totals", "timing", "cases"}
    assert encoded["session_id"] == "apollo510_evb-20260912T000000Z"
    assert encoded["board"] == "apollo510_evb"
    assert encoded["bundle"].endswith("apollo510_evb-20260912T000000Z")
    assert encoded["totals"] == {"ran": 1, "passed": 1, "failed": 0, "skipped": 1}
    assert encoded["timing"] == timing
    ran, skip = encoded["cases"]
    assert set(ran) == {"case_id", "passed", "median_cycles", "valid_for_regression", "skipped_reason"}
    assert ran == {"case_id": "abs_json", "passed": True, "median_cycles": ran["median_cycles"], "valid_for_regression": True, "skipped_reason": None}
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


# --- --json keeps stdout clean ----------------------------------------------------


def test_stdout_to_stderr_covers_python_and_subprocess_output(capfd) -> None:
    import subprocess
    import sys

    from helia_core_tester.perf_stream.run_summary import stdout_to_stderr

    with stdout_to_stderr():
        print("python-line")
        subprocess.run([sys.executable, "-c", "print('child-line')"], check=True)
    print("after-line")
    out, err = capfd.readouterr()
    assert "python-line" in err and "child-line" in err
    assert "python-line" not in out and "child-line" not in out
    assert "after-line" in out
