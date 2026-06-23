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
