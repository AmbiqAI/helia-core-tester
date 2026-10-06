from __future__ import annotations

import dataclasses
import subprocess

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware.boards import board_ids


runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_saved_options(monkeypatch, tmp_path) -> None:
    """Keep the real build dir out."""
    from helia_core_tester.hardware import cli as hardware_cli

    monkeypatch.setattr(hardware_cli, "repo_root", lambda: tmp_path / "repo")


def _result_text(result) -> str:
    text = ""
    for attr in ("output", "stdout", "stderr"):
        value = getattr(result, attr, "")
        if value:
            text += value
    return text


def test_gap_check_command_removed() -> None:
    result = runner.invoke(app, ["gap-check"])
    assert result.exit_code != 0
    assert "No such command" in _result_text(result)


def test_full_rejects_removed_skip_conversion_flag() -> None:
    result = runner.invoke(app, ["full", "--skip-conversion"])
    assert result.exit_code != 0
    assert "No such option" in _result_text(result)


def test_full_rejects_removed_skip_runners_flag() -> None:
    result = runner.invoke(app, ["full", "--skip-runners"])
    assert result.exit_code != 0
    assert "No such option" in _result_text(result)


def test_full_rejects_removed_regen_after_cleanup_flag() -> None:
    result = runner.invoke(app, ["full", "--regen-generated-tests-after-cleanup"])
    assert result.exit_code != 0
    assert "No such option" in _result_text(result)


def test_run_rejects_removed_report_dir_override() -> None:
    result = runner.invoke(app, ["run", "--report-dir", "artifacts/reports"])
    assert result.exit_code != 0
    assert "No such option" in _result_text(result)


def test_full_rejects_removed_include_float_flag() -> None:
    result = runner.invoke(app, ["full", "--include-float"])
    assert result.exit_code != 0
    assert "No such option" in _result_text(result)


# --- hardware CLI surface ------------------------------------------------------------


def test_perf_stream_group_removed() -> None:
    """The pre-cli-surface `perf-stream` group stays gone; `hardware` replaced it."""
    result = runner.invoke(app, ["perf-stream", "flash"])
    assert result.exit_code != 0
    assert "No such command" in _result_text(result)


def test_hardware_group_lists_expected_commands() -> None:
    result = runner.invoke(app, ["hardware", "--help"])
    assert result.exit_code == 0
    text = _result_text(result)
    for command in ("run", "build", "flash", "stream", "memory-report"):
        assert command in text
    for removed in ("build-firmware", "run-generated"):
        assert removed not in text


def test_hardware_commands_reject_removed_identity_flags() -> None:
    for args in (["hardware", "stream", "--chip-name", "X"], ["hardware", "stream", "--speed-khz", "1"], ["hardware", "build", "--cpu", "cortex-m55"]):
        result = runner.invoke(app, args)
        assert result.exit_code != 0, args
        assert "No such option" in _result_text(result), args


def test_boards_lists_table() -> None:
    result = runner.invoke(app, ["boards"])
    assert result.exit_code == 0
    text = _result_text(result)
    assert "apollo510_evb" in text and "cortex-m55" in text and "AP510NFA-CBR" in text and "4000" in text
    assert "core_clock" in text and "48 MHz" in text


def test_unknown_board_lists_known_ids() -> None:
    result = runner.invoke(app, ["hardware", "build", "--board", "nope_evb"])
    assert result.exit_code == 2
    text = _result_text(result)
    assert "Unknown board 'nope_evb'" in text and "apollo510_evb" in text


def test_probes_match_uses_env_serial(monkeypatch) -> None:
    monkeypatch.setenv("HPX_JLINK_SERIAL", "1160002276")
    result = runner.invoke(app, ["probes", "match", "--board", "apollo510_evb"])
    assert result.exit_code == 0
    assert "1160002276" in _result_text(result)


def test_probes_match_fails_without_probes(monkeypatch) -> None:
    from helia_core_tester.hardware import cli as hardware_cli

    monkeypatch.delenv("HPX_JLINK_SERIAL", raising=False)
    monkeypatch.setattr(hardware_cli, "resolve_serial", lambda explicit=None, **_: (_ for _ in ()).throw(
        hardware_cli.ProbeResolutionError("No connected J-Link probes detected.")))
    result = runner.invoke(app, ["probes", "match"])
    assert result.exit_code == 5
    assert "No connected J-Link probes" in _result_text(result)


def test_stream_precision_rules_are_enforced_before_hardware(monkeypatch) -> None:
    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    result = runner.invoke(app, ["hardware", "stream", "--precision", "fp16", "--suite", "both"])
    assert result.exit_code == 2
    assert "--precision cannot be combined with --suite both" in _result_text(result)
    result = runner.invoke(app, ["hardware", "run", "--precision", "fp32", "--test-name", "x", "--skip-generate", "--skip-flash"])
    assert result.exit_code == 2
    assert "--precision and --test-name cannot be combined" in _result_text(result)


def test_option_validation_runs_before_probe_resolution(monkeypatch) -> None:
    """With no --serial-no and no $HPX_JLINK_SERIAL, a bad option combination must
    still produce the option error -- never an enumeration/hardware error first."""
    from helia_core_tester.hardware import cli as hardware_cli

    monkeypatch.delenv("HPX_JLINK_SERIAL", raising=False)
    enumerated: list[str] = []

    def _resolve(explicit=None, **_):
        enumerated.append("probe")
        raise hardware_cli.ProbeResolutionError("No connected J-Link probes detected.")

    monkeypatch.setattr(hardware_cli, "resolve_serial", _resolve)
    for args, expected in (
        (["hardware", "stream", "--precision", "fp16", "--suite", "both"], "--precision cannot be combined with --suite both"),
        (["hardware", "run", "--precision", "fp16", "--suite", "both", "--skip-generate"], "--precision cannot be combined with --suite both"),
        (["hardware", "stream", "--suite", "nope"], "Invalid suite"),
        (["hardware", "run", "--fvp-gate", "maybe"], "--fvp-gate must be one of"),
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, args
        text = _result_text(result)
        assert expected in text and "No connected J-Link probes" not in text, (args, text)
    assert enumerated == []

    # Valid options: now the probe is resolved, and its error is what the user sees.
    result = runner.invoke(app, ["hardware", "stream", "--precision", "fp16"])
    assert result.exit_code == 5 and "No connected J-Link probes" in _result_text(result)
    assert enumerated == ["probe"]


@pytest.mark.parametrize(
    ("raise_factory", "expected_line"),
    [
        (lambda: subprocess.CalledProcessError(2, ["cmake", "--build", "build/x", "--target", "hct_benchmark_server_flash"]),
         "✗ Command failed with exit status 2: cmake --build build/x --target hct_benchmark_server_flash"),
        (lambda: __import__("pylink").JLinkException("Could not connect to the target device."),
         "✗ J-Link error: Could not connect to the target device."),
        (lambda: TimeoutError("Timed out writing 64 RTT bytes."), "✗ Timed out writing 64 RTT bytes."),
        (lambda: FileNotFoundError("Built firmware ELF not found: build/x/hardware/hct_benchmark_server.elf"),
         "✗ Built firmware ELF not found"),
    ],
)
def test_pipeline_failures_print_one_line_and_hide_the_traceback_unless_verbose(monkeypatch, raise_factory, expected_line) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    def _boom(*args, **kwargs):
        raise raise_factory()

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    monkeypatch.delenv("HELIA_CORE_TESTER_VERBOSITY", raising=False)
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _boom)
    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _boom)

    for args in (["hardware", "run", "--skip-generate"], ["hardware", "stream"]):
        result = runner.invoke(app, args)
        text = _result_text(result)
        assert result.exit_code == 5, (args, text)
        assert expected_line in text and "Traceback" not in text, (args, text)

        verbose = _result_text(runner.invoke(app, args + ["-v", "1"]))
        assert expected_line in verbose and "Traceback (most recent call last)" in verbose, (args, verbose)

    monkeypatch.setenv("HELIA_CORE_TESTER_VERBOSITY", "2")
    env_verbose = _result_text(runner.invoke(app, ["hardware", "run", "--skip-generate"]))
    assert "Traceback (most recent call last)" in env_verbose


def test_unexpected_exceptions_keep_their_traceback(monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    def _bug(*args, **kwargs):
        raise KeyError("case_id")

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _bug)
    result = runner.invoke(app, ["hardware", "stream"])
    assert result.exit_code != 0 and isinstance(result.exception, KeyError)


def test_run_precision_reaches_the_generate_step(monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    seen: dict = {}

    def _pipeline(repo_root, spec, serial, *, options, **kwargs):
        seen["options"] = options
        raise RuntimeError("stop here")

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _pipeline)
    runner.invoke(app, ["hardware", "run", "--precision", "fp16"])
    assert (seen["options"].suite, seen["options"].test_name, seen["options"].float_precision) == ("float", "_f16", "f16")
    runner.invoke(app, ["hardware", "run", "--suite", "float"])
    assert seen["options"].float_precision is None


def test_run_rejects_skip_flash_with_force_flash(monkeypatch) -> None:
    from helia_core_tester.hardware import cli as hardware_cli

    monkeypatch.setattr(hardware_cli, "resolve_serial", lambda explicit=None, **_: (_ for _ in ()).throw(
        AssertionError("probes must not be resolved before option validation")))
    result = runner.invoke(app, ["hardware", "run", "--skip-flash", "--force-flash"])
    assert result.exit_code == 2
    assert "--skip-flash and --force-flash cannot be combined" in _result_text(result)


def test_doctor_reports_hardware_section_without_failing_on_missing_tools(monkeypatch) -> None:
    from helia_core_tester.hardware import doctor as hw_doctor

    monkeypatch.setattr(hw_doctor, "_jlink_dll_check", lambda: hw_doctor.HardwareCheck("J-Link library (pylink)", False, "missing"))
    monkeypatch.setattr(hw_doctor, "find_jlink_exe", lambda: None)
    result = runner.invoke(app, ["doctor"])
    text = _result_text(result)
    assert "Hardware (helia_core_tester hardware ...)" in text
    assert "J-Link library (pylink): missing" in text
    assert "⚠ JLinkExe (flash target): not found: set $JLINK_PATH" in text
    assert "Board table" in text and f"{len(board_ids())} board(s)" in text


def test_doctor_reports_jlinkexe_path_and_source_and_missing_hpx_jlink_dll(monkeypatch, tmp_path) -> None:
    from helia_core_tester.hardware import doctor as hw_doctor
    from helia_core_tester.hardware.jlink_library import JLinkExecutable

    monkeypatch.setattr(hw_doctor, "find_jlink_exe", lambda: JLinkExecutable("/opt/SEGGER/JLink/JLinkExe", "$JLINK_PATH"))
    monkeypatch.setenv("HPX_JLINK_DLL", str(tmp_path / "gone.so"))
    result = runner.invoke(app, ["doctor"])
    text = _result_text(result)
    assert "✓ JLinkExe (flash target): /opt/SEGGER/JLink/JLinkExe (via $JLINK_PATH)" in text
    assert "⚠ J-Link library (pylink): $HPX_JLINK_DLL=" in text and "gone.so does not exist" in text


def test_precision_refuses_suite_both_in_any_spelling(monkeypatch) -> None:
    """`--suite BOTH` is normalised before the precision rules, so it is refused like `both`
    instead of slipping through as a float-only run."""
    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    for spelling in ("BOTH", "Both", " both "):
        for command in (["hardware", "stream"], ["hardware", "run", "--skip-generate"]):
            result = runner.invoke(app, [*command, "--precision", "fp16", "--suite", spelling])
            assert result.exit_code == 2, (command, spelling)
            assert "--precision cannot be combined with --suite both" in _result_text(result), (command, spelling)


def test_memory_report_missing_elf_is_a_one_line_error(tmp_path) -> None:
    result = runner.invoke(app, ["hardware", "memory-report", "--build-dir", str(tmp_path / "never-built")])
    text = _result_text(result)
    assert result.exit_code == 5 and isinstance(result.exception, SystemExit), text
    assert "✗ Built firmware ELF not found" in text and "hardware build" in text and "Traceback" not in text


def test_stream_requires_the_build_id_stamp_unless_allowed(monkeypatch, tmp_path) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    unstamped = tmp_path / "legacy"
    (unstamped / "hardware").mkdir(parents=True)
    (unstamped / "hardware" / "hct_benchmark_server.elf").write_bytes(b"legacy")
    result = runner.invoke(app, ["hardware", "stream", "--build-dir", str(unstamped)])
    text = _result_text(result)
    assert result.exit_code == 3 and "hct_build_id.txt not found" in text and "--allow-unverified-firmware" in text, text
    assert "Traceback" not in text

    seen: dict = {}

    def _stream(repo_root, spec, serial, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _stream)
    runner.invoke(app, ["hardware", "stream", "--build-dir", str(unstamped), "--allow-unverified-firmware"])
    assert seen["allow_unverified_firmware"] is True
    runner.invoke(app, ["hardware", "stream", "--build-dir", str(unstamped)])
    assert seen["allow_unverified_firmware"] is False

    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _stream)
    runner.invoke(app, ["hardware", "run", "--skip-generate", "--skip-flash", "--allow-unverified-firmware"])
    assert seen["allow_unverified_firmware"] is True


@pytest.mark.parametrize("command", ["build", "flash", "run"])
def test_inline_asm_flag_is_tri_state(monkeypatch, tmp_path, command) -> None:
    """Unset reuses the build dir's saved setting."""
    from helia_core_tester.hardware import firmware_build, hardware_pipeline, nsx_app

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen["options"] = kwargs.get("app_options") or kwargs.get("options")
        raise RuntimeError("stop here")

    monkeypatch.setattr(firmware_build, "build_firmware", _capture)
    monkeypatch.setattr(firmware_build, "flash_firmware", _capture)
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _capture)
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    app_dir.mkdir(parents=True)
    built = nsx_app.AppOptions(cmsis_nn_ref="v9", cmsis_nn_ref_explicit=True, requantize_inline_asm=False)
    nsx_app.save_options(app_dir, built)
    base = ["hardware", command, "--build-dir", str(tmp_path)]
    for flags, inline_asm in (([], False), (["--inline-asm"], True), (["--no-inline-asm"], False)):
        runner.invoke(app, base + flags)
        assert seen["options"] == dataclasses.replace(built, requantize_inline_asm=inline_asm), flags


def test_stream_only_run_skips_option_resolution(monkeypatch, tmp_path) -> None:
    """A gone kernel root cannot block streaming."""
    from helia_core_tester.hardware import firmware_build, hardware_pipeline, nsx_app

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    seen: dict = {}

    def _pipeline(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _pipeline)
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    app_dir.mkdir(parents=True)
    nsx_app.save_options(app_dir, nsx_app.AppOptions(cmsis_nn_root=tmp_path / "moved", requantize_inline_asm=False))
    result = runner.invoke(app, ["hardware", "run", "--build-dir", str(tmp_path), "--skip-generate", "--skip-flash"])
    assert seen["app_options"] is None and "inline asm off" in _result_text(result)


def _capture_run(monkeypatch, seen: dict) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    def _pipeline(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _pipeline)


def test_skip_flash_generates_from_the_built_kernels(monkeypatch, tmp_path) -> None:
    """Generation matches the flashed firmware."""
    from helia_core_tester.hardware import firmware_build, nsx_app

    seen: dict = {}
    _capture_run(monkeypatch, seen)
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    app_dir.mkdir(parents=True)
    # Defaulted, off the pin: still kept.
    built = nsx_app.AppOptions(cmsis_nn_ref="v9", requantize_inline_asm=False)
    nsx_app.save_options(app_dir, built)
    base = ["hardware", "run", "--build-dir", str(tmp_path), "--skip-flash"]
    result = runner.invoke(app, base)
    assert seen["app_options"] == built and "Options changed" not in _result_text(result)
    seen.clear()
    result = runner.invoke(app, base + ["--no-inline-asm", "--cmsis-nn-ref", "v9"])
    assert seen["app_options"] == built


@pytest.mark.parametrize("flags", [["--cmsis-nn-ref", "v10"], ["--inline-asm"]])
def test_skip_flash_refuses_new_kernel_flags(monkeypatch, tmp_path, flags) -> None:
    """New flags would not reach the firmware."""
    from helia_core_tester.hardware import firmware_build, nsx_app

    seen: dict = {}
    _capture_run(monkeypatch, seen)
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    app_dir.mkdir(parents=True)
    nsx_app.save_options(app_dir, nsx_app.AppOptions(cmsis_nn_ref="v9", requantize_inline_asm=False))
    result = runner.invoke(app, ["hardware", "run", "--build-dir", str(tmp_path), "--skip-flash", *flags])
    assert result.exit_code != 0 and not seen
    assert "--skip-flash keeps the built kernels" in _result_text(result)


def test_skip_flash_needs_a_saved_build(monkeypatch, tmp_path) -> None:
    seen: dict = {}
    _capture_run(monkeypatch, seen)
    result = runner.invoke(app, ["hardware", "run", "--build-dir", str(tmp_path), "--skip-flash"])
    assert result.exit_code != 0 and not seen
    assert "needs a saved build" in _result_text(result)


def test_skip_flash_refusal_is_one_line(monkeypatch, tmp_path) -> None:
    """Refusals are errors, not tracebacks."""
    from helia_core_tester.hardware import firmware_build, nsx_app

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    app_dir = firmware_build.nsx_app_dir(tmp_path)
    app_dir.mkdir(parents=True)
    nsx_app.save_options(app_dir, nsx_app.AppOptions(cmsis_nn_ref="v9"))
    result = runner.invoke(app, ["hardware", "run", "--build-dir", str(tmp_path), "--skip-flash"])
    assert result.exit_code == 3
    assert "Kernels changed since the build; rebuild first." in _result_text(result)
    assert "Traceback" not in _result_text(result)


def test_golden_refusal_exits_refused(monkeypatch) -> None:
    """Refusals exit 3, not the correctness 1."""
    from helia_core_tester.hardware import hardware_pipeline
    from helia_core_tester.hardware.errors import RunRefused

    def _refuse(*args, **kwargs):
        raise RunRefused("Golden run failed these cases: c")

    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", _refuse)
    result = runner.invoke(app, ["hardware", "run", "--skip-generate"])
    assert result.exit_code == 3 and "Golden run failed these cases" in _result_text(result)


def test_correctness_failure_exits_one(monkeypatch, tmp_path) -> None:
    """Exit 1 means a case failed."""
    from helia_core_tester.hardware import hardware_pipeline, run_summary

    outcome = hardware_pipeline.HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[])
    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    monkeypatch.setattr(hardware_pipeline, "run_hardware_pipeline", lambda *a, **k: outcome)
    monkeypatch.setattr(run_summary, "print_run_report", lambda *a, **k: True)
    result = runner.invoke(app, ["hardware", "run", "--skip-generate"])
    assert result.exit_code == 1 and "failed correctness" in _result_text(result)
