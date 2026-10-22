"""--report-formats: comma and repeated forms, rejection before any step runs, and forwarding to the FVP child."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester import cli as cli_module
from helia_core_tester.core.config import VALID_REPORT_FORMATS, Config, parse_report_formats
from helia_core_tester.core.errors import ConfigurationError
from helia_core_tester.core.steps.run import RunStep
from helia_core_tester.fvp.cli import build_arg_parser

EXPECTED = "expected a comma-separated list of: json, html, md, junit"


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "tester"
    (root / "helia_core_tester" / "generation").mkdir(parents=True)
    return root


def _flat(result) -> str:
    text = "".join(getattr(result, attr, "") or "" for attr in ("output", "stdout", "stderr"))
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", text).split())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HELIA_CORE_TESTER_REPORT_FORMATS", raising=False)
    monkeypatch.delenv("HELIA_CORE_TESTER_CONFIG", raising=False)


@pytest.fixture
def captured(monkeypatch) -> list[Config]:
    seen: list[Config] = []

    class _FakeStep:
        def __init__(self, config, *args, **kwargs):
            seen.append(config)

        def plan(self):
            return None

        def execute(self):
            raise AssertionError("step must not run")

    class _FakePipeline:
        def __init__(self, config):
            seen.append(config)

        def print_plan(self):
            pass

        def run(self):
            raise AssertionError("pipeline must not run")

    monkeypatch.setattr(cli_module, "RunStep", _FakeStep)
    monkeypatch.setattr(cli_module, "FullTestPipeline", _FakePipeline)
    monkeypatch.setattr(cli_module, "_print_plan_item", lambda item: None)
    return seen


def _invoke(tmp_path: Path, command: str, *args: str, plan: bool = True):
    argv = [command, "--repo-root", str(_repo(tmp_path)), *args]
    if plan:
        argv.append("--plan")
    return CliRunner().invoke(cli_module.app, argv)


@pytest.mark.parametrize("command", ["run", "full"])
@pytest.mark.parametrize(
    "args",
    [
        ["--report-formats", "json,html,md"],
        ["--report-formats", "json", "--report-formats", "html", "--report-formats", "md"],
        ["--report-formats", " json , html ,md "],
        ["--report-formats", "json,html", "--report-formats", "md,json"],
        ["--report-formats", "json,json,html,md,html"],
    ],
    ids=["comma", "repeated", "whitespace", "mixed", "duplicates"],
)
def test_comma_and_repeated_forms_give_the_same_list(tmp_path, captured, command, args) -> None:
    result = _invoke(tmp_path, command, *args)
    assert result.exit_code == 0, _flat(result)
    assert captured[0].report_formats == ["json", "html", "md"]


@pytest.mark.parametrize("command", ["run", "full"])
def test_default_is_json(tmp_path, captured, command) -> None:
    result = _invoke(tmp_path, command)
    assert result.exit_code == 0, _flat(result)
    assert captured[0].report_formats == ["json"]


@pytest.mark.parametrize("command", ["run", "full"])
def test_every_valid_format_is_accepted(tmp_path, captured, command) -> None:
    result = _invoke(tmp_path, command, "--report-formats", ",".join(VALID_REPORT_FORMATS))
    assert result.exit_code == 0, _flat(result)
    assert captured[0].report_formats == list(VALID_REPORT_FORMATS)


@pytest.mark.parametrize("command", ["run", "full"])
@pytest.mark.parametrize(
    "value, message",
    [
        ("json,xml", f"unknown report format 'xml' in 'json,xml'; {EXPECTED}"),
        ("JSON", f"unknown report format 'JSON' in 'JSON'; {EXPECTED}"),
        ("json,,md", f"empty report format in 'json,,md'; {EXPECTED}"),
        ("json,", f"empty report format in 'json,'; {EXPECTED}"),
        (",json", f"empty report format in ',json'; {EXPECTED}"),
        ("", f"empty report format in ''; {EXPECTED}"),
        ("   ", f"empty report format in '   '; {EXPECTED}"),
    ],
    ids=["unknown", "case", "empty-middle", "trailing-comma", "leading-comma", "empty-string", "blank"],
)
def test_bad_formats_are_rejected_before_the_pipeline(tmp_path, captured, command, value, message) -> None:
    result = _invoke(tmp_path, command, "--report-formats", value, plan=False)
    assert result.exit_code == 2
    # Rich wraps the error box, so compare with whitespace collapsed on both sides.
    assert " ".join(f"Invalid value for '--report-formats': {message}".split()) in _flat(result)
    assert captured == []


def test_bad_token_in_a_later_repeat_is_rejected(tmp_path, captured) -> None:
    result = _invoke(tmp_path, "full", "--report-formats", "json", "--report-formats", "pdf", plan=False)
    assert result.exit_code == 2
    assert f"unknown report format 'pdf' in 'pdf'; {EXPECTED}" in _flat(result)
    assert captured == []


def test_help_shows_both_forms() -> None:
    result = CliRunner().invoke(cli_module.app, ["full", "--help"], env={"COLUMNS": "400"})
    assert result.exit_code == 0
    text = _flat(result)
    assert "--report-formats json,html" in text
    assert "--report-formats json --report-formats html" in text


def test_parser_rejects_non_string_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="at least one report format is required"):
        parse_report_formats([])
    with pytest.raises(ValueError, match="report format must be a string, got 1"):
        parse_report_formats(["json", 1])
    with pytest.raises(ValueError, match="report formats must be a string or a list of strings, got 5"):
        parse_report_formats(5)
    assert parse_report_formats("md, json") == ["md", "json"]


@pytest.mark.parametrize(
    "toml_value, expected",
    [
        ('"json,html"', ["json", "html"]),
        ('["json,junit", "md", "json"]', ["json", "junit", "md"]),
    ],
)
def test_config_file_values_are_normalised(tmp_path, toml_value, expected) -> None:
    root = _repo(tmp_path)
    (root / "helia_core_tester.toml").write_text(f"[helia_core_tester]\nreport_formats = {toml_value}\n")
    assert Config(project_root=root).report_formats == expected


@pytest.mark.parametrize(
    "toml_value, message",
    [
        ('"json,xml"', "unknown report format 'xml'"),
        ('["json", ""]', "empty report format in ''"),
        ("[]", "at least one report format is required"),
        ('["json", 3]', "report format must be a string, got 3"),
    ],
)
def test_config_file_rejects_bad_values(tmp_path, toml_value, message) -> None:
    root = _repo(tmp_path)
    (root / "helia_core_tester.toml").write_text(f"[helia_core_tester]\nreport_formats = {toml_value}\n")
    with pytest.raises(ConfigurationError, match=re.escape(f"report_formats: {message}")):
        Config(project_root=root)


def test_env_value_is_normalised(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HELIA_CORE_TESTER_REPORT_FORMATS", " junit , json,junit ")
    assert Config(project_root=_repo(tmp_path)).report_formats == ["junit", "json"]


@pytest.mark.parametrize("value", ["json,,md", "", "json,pdf"])
def test_env_rejects_bad_values(tmp_path, monkeypatch, value) -> None:
    monkeypatch.setenv("HELIA_CORE_TESTER_REPORT_FORMATS", value)
    with pytest.raises(ConfigurationError, match="Invalid env override HELIA_CORE_TESTER_REPORT_FORMATS"):
        Config(project_root=_repo(tmp_path))


def test_run_step_forwards_formats_as_separate_argv_entries(tmp_path) -> None:
    root = _repo(tmp_path)
    cfg = Config(
        project_root=root,
        report_formats=["json,html", "junit"],
        _explicit_overrides={"project_root", "report_formats"},
    )
    commands = RunStep(cfg)._run_commands()
    assert commands
    for cmd in commands:
        idx = cmd.index("--report-formats")
        assert cmd[idx + 1 : idx + 4] == ["json", "html", "junit"]
        assert not any("," in arg for arg in cmd[idx + 1 : idx + 4])
        args = build_arg_parser(tmp_path, tmp_path).parse_args(cmd[cmd.index("--no-setup") :])
        assert args.report_formats == ["json", "html", "junit"]


def test_fvp_child_accepts_exactly_the_validated_formats(tmp_path) -> None:
    parser = build_arg_parser(tmp_path, tmp_path)
    action = next(a for a in parser._actions if "--report-formats" in a.option_strings)
    assert tuple(action.choices) == VALID_REPORT_FORMATS
