from __future__ import annotations

from typer.testing import CliRunner

from helia_core_tester.cli import app


runner = CliRunner()


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


def test_unknown_board_lists_known_ids() -> None:
    result = runner.invoke(app, ["hardware", "build", "--board", "nope_evb"])
    assert result.exit_code == 1
    text = _result_text(result)
    assert "Unknown board 'nope_evb'" in text and "apollo510_evb" in text


def test_probes_match_uses_env_serial(monkeypatch) -> None:
    monkeypatch.setenv("HPX_JLINK_SERIAL", "1160002276")
    result = runner.invoke(app, ["probes", "match", "--board", "apollo510_evb"])
    assert result.exit_code == 0
    assert "1160002276" in _result_text(result)


def test_probes_match_fails_without_probes(monkeypatch) -> None:
    from helia_core_tester.perf_stream import cli as hardware_cli

    monkeypatch.delenv("HPX_JLINK_SERIAL", raising=False)
    monkeypatch.setattr(hardware_cli, "resolve_serial", lambda explicit=None, **_: (_ for _ in ()).throw(
        hardware_cli.ProbeResolutionError("No connected J-Link probes detected.")))
    result = runner.invoke(app, ["probes", "match"])
    assert result.exit_code == 1
    assert "No connected J-Link probes" in _result_text(result)


def test_stream_precision_rules_are_enforced_before_hardware(monkeypatch) -> None:
    monkeypatch.setenv("HPX_JLINK_SERIAL", "1")
    result = runner.invoke(app, ["hardware", "stream", "--precision", "fp16", "--suite", "both"])
    assert result.exit_code == 1
    assert "--precision cannot be combined with --suite both" in _result_text(result)
    result = runner.invoke(app, ["hardware", "run", "--precision", "fp32", "--test-name", "x", "--skip-generate", "--skip-flash"])
    assert result.exit_code == 1
    assert "--precision and --test-name cannot be combined" in _result_text(result)


def test_doctor_reports_hardware_section_without_failing_on_missing_tools(monkeypatch) -> None:
    from helia_core_tester.perf_stream import doctor as hw_doctor

    monkeypatch.setattr(hw_doctor, "_jlink_dll_check", lambda: hw_doctor.HardwareCheck("J-Link library (pylink)", False, "missing"))
    result = runner.invoke(app, ["doctor"])
    text = _result_text(result)
    assert "Hardware (helia_core_tester hardware ...)" in text
    assert "J-Link library (pylink): missing" in text
    assert "Board table" in text and "1 board(s)" in text
