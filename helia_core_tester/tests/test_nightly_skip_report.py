"""scripts/nightly_skip_report.py: per-family skip counts and growth in the nightly summary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nightly_skip_report.py"
_SELECTION = {"suite": "both", "limit": None}


def _load():
    spec = importlib.util.spec_from_file_location("nightly_skip_report", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load()


def _leg(root: Path, board: str, run: int, skipped: dict[str, int], untimed: list[str] | None = None, *,
         selection=_SELECTION, date: str = "20261006") -> Path:
    """One hct-hardware artifact the way the nightly stages it."""
    artifact = root / f"hct-hardware-{board}-{date}-{run}"
    session = f"nightly-{run}-{board}"
    bundle = artifact / "hardware" / session
    bundle.mkdir(parents=True)
    (artifact / "hardware-nightly-run.json").write_text(json.dumps({"session_id": session, "selection": selection}))
    cases = [
        {"case_id": f"{family}_{i}", "family": family, "suite": "int", "reason": "no adapter"}
        for family, count in skipped.items() for i in range(count)
    ]
    (bundle / "session_summary.json").write_text(json.dumps({"skipped_cases": cases}))
    if untimed is not None:
        (bundle / "coverage.json").write_text(json.dumps({"deployed": {"missing": untimed}}))
    return artifact


def _run(tmp_path: Path, boards: list[str], *, previous: bool = True, note: str | None = None) -> tuple[str, list[str]]:
    return report.render(tmp_path / "cur", tmp_path / "prev" if previous else None, boards, note)


def test_counts_per_family_and_flags_growth(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "apollo510_evb", 2, {"Add": 5, "Concat": 3, "Gather": 1}, ["arm_b", "arm_a", "arm_c"])
    _leg(tmp_path / "prev" / "1", "apollo510_evb", 1, {"Add": 2, "Concat": 4, "Fill": 2}, ["arm_a", "arm_b"])

    markdown, warnings = _run(tmp_path, ["apollo510_evb"])

    assert "Baseline: `hct-hardware-apollo510_evb-20261006-1`." in markdown
    assert "| Add | 5 | +3 ⚠ |" in markdown
    assert "| Concat | 3 | -1 |" in markdown
    assert "| Fill | 0 | -2 |" in markdown
    assert "| Gather | 1 | new ⚠ |" in markdown
    assert "| **Total** | **9** | +1 |" in markdown
    assert "<summary>Untimed deployed entry points (3, 1 new)</summary>" in markdown
    assert "- `arm_a`\n- `arm_b`\n- `arm_c` (new)\n\n</details>" in markdown
    assert warnings == [
        "apollo510_evb: Add skipped cases grew from 2 to 5",
        "apollo510_evb: Gather skipped cases grew from 0 to 1",
        "apollo510_evb: arm_c is newly untimed",
    ]


def test_unchanged_counts_raise_no_warning(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 2}, [])
    _leg(tmp_path / "prev" / "1", "b", 1, {"Add": 2}, [])

    markdown, warnings = _run(tmp_path, ["b"])

    assert "| Add | 2 | 0 |" in markdown
    assert "<summary>Untimed deployed entry points (0)</summary>" in markdown
    assert warnings == []


def test_zero_skips_render_an_empty_table(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 2, {}, [])
    _leg(tmp_path / "prev" / "1", "b", 1, {}, [])

    markdown, warnings = _run(tmp_path, ["b"])

    assert "| **Total** | **0** | 0 |" in markdown
    assert warnings == []


def test_different_selection_is_not_compared(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 9}, ["arm_x"])
    _leg(tmp_path / "prev" / "1", "b", 1, {"Add": 1}, [], selection={"suite": "int", "limit": 2})

    markdown, warnings = _run(tmp_path, ["b"])

    assert ("No baseline: none of the 1 earlier run(s) searched has a comparable result for this board "
            "(1 used a different selection).") in markdown
    assert "| Add | 9 |  |" in markdown
    assert "(new)" not in markdown
    assert warnings == []


def test_missing_baseline_reports_counts_with_the_note(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 1}, [])

    markdown, warnings = _run(tmp_path, ["b"], previous=False, note="no earlier scheduled run")

    assert "No baseline: no earlier scheduled run." in markdown
    assert "| Add | 1 |  |" in markdown
    assert warnings == []


def test_board_without_current_result(tmp_path: Path) -> None:
    (tmp_path / "cur").mkdir()
    _leg(tmp_path / "prev" / "1", "b", 1, {"Add": 1})

    markdown, warnings = _run(tmp_path, ["b"])

    assert "#### b\n\nNo TCM-leg result." in markdown
    assert warnings == ["b: no current TCM-leg result; skip growth not checked"]


def test_missing_current_result_keeps_other_boards(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "a", 2, {"Add": 2})
    _leg(tmp_path / "prev" / "1", "a", 1, {"Add": 1})

    markdown, warnings = _run(tmp_path, ["a", "c"])

    assert "| Add | 2 | +1 ⚠ |" in markdown and "#### c\n\nNo TCM-leg result." in markdown
    assert warnings == ["a: Add skipped cases grew from 1 to 2",
                        "c: no current TCM-leg result; skip growth not checked"]


def test_baseline_runs_without_results_warn_once(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 9, {"Add": 2}, [])
    (tmp_path / "prev" / "8").mkdir(parents=True)
    _leg(tmp_path / "prev" / "7", "other", 7, {"Add": 1}, [])

    markdown, warnings = _run(tmp_path, ["b"])

    why = "none of the 2 earlier run(s) searched has a comparable result for this board (2 had no result)"
    assert f"No baseline: {why}." in markdown
    assert warnings == [f"b: no baseline because earlier runs left no result; skip growth not checked ({why})"]


def test_missing_and_other_selection_baselines_still_warn(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 9, {"Add": 2}, [])
    (tmp_path / "prev" / "8").mkdir(parents=True)
    _leg(tmp_path / "prev" / "7", "b", 7, {"Add": 1}, [], selection={"suite": "int", "limit": 1})

    markdown, warnings = _run(tmp_path, ["b"])

    assert "(1 had no result, 1 used a different selection)" in markdown
    assert len(warnings) == 1 and warnings[0].startswith("b: no baseline because earlier runs left no result")


def test_empty_baseline_dir_with_note_does_not_warn(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 1}, [])
    (tmp_path / "prev").mkdir()

    markdown, warnings = _run(tmp_path, ["b"], note="no earlier scheduled run")

    assert "No baseline: no earlier scheduled run." in markdown
    assert warnings == []


def test_artifact_match_is_exact_per_board_and_takes_the_newest_run(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "apollo3p_evb", 5, {"Add": 1})
    _leg(tmp_path / "cur", "apollo3p_evb", 7, {"Add": 4})
    _leg(tmp_path / "cur", "apollo3p_evb2", 9, {"Add": 50})
    (tmp_path / "cur" / "hct-mram-apollo3p_evb-20261006-9").mkdir()

    leg = report.load_leg(tmp_path / "cur", "apollo3p_evb")

    assert leg.artifact == "hct-hardware-apollo3p_evb-20261006-7"
    assert leg.skipped_by_family == {"Add": 4}


def test_missing_family_counts_as_unknown(tmp_path: Path) -> None:
    artifact = _leg(tmp_path / "cur", "b", 1, {})
    summary = artifact / "hardware" / "nightly-1-b" / "session_summary.json"
    summary.write_text(json.dumps({"skipped_cases": [{"case_id": "x"}, {"case_id": "y", "family": ""}]}))

    assert report.load_leg(tmp_path / "cur", "b").skipped_by_family == {report.UNKNOWN_FAMILY: 2}


def test_no_coverage_file_is_reported(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 1}, None)

    markdown, _ = _run(tmp_path, ["b"], previous=False)

    assert "Untimed deployed entry points: no coverage.json." in markdown


def test_artifact_without_run_json_has_no_result(tmp_path: Path) -> None:
    artifact = _leg(tmp_path / "cur", "b", 1, {"Add": 1})
    (artifact / "hardware-nightly-run.json").unlink()

    assert report.load_leg(tmp_path / "cur", "b") is None


@pytest.mark.parametrize(
    ("relative", "content", "message"),
    [
        ("hardware-nightly-run.json", "{not json", "hardware-nightly-run.json"),
        ("hardware-nightly-run.json", json.dumps({"session_id": "../escape"}), "no usable session_id"),
        ("hardware-nightly-run.json", json.dumps({"session_id": ".."}), "no usable session_id"),
        ("hardware-nightly-run.json", json.dumps([]), "no usable session_id"),
        ("hardware/nightly-1-b/session_summary.json", json.dumps({}), "no skipped_cases list"),
        ("hardware/nightly-1-b/session_summary.json", json.dumps({"skipped_cases": ["x"]}), "non-object"),
        ("hardware/nightly-1-b/session_summary.json", "", "session_summary.json"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": {"missing": "arm_a"}}), "deployed.missing"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": {"missing": [1]}}), "deployed.missing"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({}), "deployed is not an object"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": ["invalid"]}), "deployed is not an object"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": []}), "deployed is not an object"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": None}), "deployed is not an object"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": "x"}), "deployed is not an object"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": {}}), "deployed.missing"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": {"missing": None}}), "deployed.missing"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": {"missing": {"a": 1}}}), "deployed.missing"),
        ("hardware/nightly-1-b/coverage.json", json.dumps({"deployed": {"missing": ["a", None]}}), "deployed.missing"),
        ("hardware/nightly-1-b/coverage.json", json.dumps([]), "coverage.json: not an object"),
        ("hardware/nightly-1-b/coverage.json", json.dumps(None), "coverage.json: not an object"),
        ("hardware/nightly-1-b/coverage.json", "", "coverage.json"),
        ("hardware/nightly-1-b/session_summary.json", json.dumps([]), "no skipped_cases list"),
        ("hardware/nightly-1-b/session_summary.json", json.dumps({"skipped_cases": None}), "no skipped_cases list"),
        ("hardware/nightly-1-b/session_summary.json", json.dumps({"skipped_cases": {}}), "no skipped_cases list"),
        ("hardware/nightly-1-b/session_summary.json", json.dumps({"skipped_cases": [None]}), "non-object"),
        ("hardware-nightly-run.json", json.dumps({}), "no usable session_id"),
        ("hardware-nightly-run.json", json.dumps({"session_id": ""}), "no usable session_id"),
        ("hardware-nightly-run.json", json.dumps({"session_id": 5}), "no usable session_id"),
        ("hardware-nightly-run.json", "", "hardware-nightly-run.json"),
    ],
)
def test_corrupt_artifact_is_refused(tmp_path: Path, relative: str, content: str, message: str) -> None:
    artifact = _leg(tmp_path / "cur", "b", 1, {"Add": 1}, [])
    (artifact / relative).write_text(content)

    with pytest.raises(report.ReportError, match=message):
        report.load_leg(tmp_path / "cur", "b")


def test_missing_summary_file_is_refused(tmp_path: Path) -> None:
    artifact = _leg(tmp_path / "cur", "b", 1, {"Add": 1})
    (artifact / "hardware" / "nightly-1-b" / "session_summary.json").unlink()

    with pytest.raises(report.ReportError, match="session_summary.json"):
        report.load_leg(tmp_path / "cur", "b")


def test_unreadable_current_result_warns_and_keeps_other_boards(tmp_path: Path) -> None:
    bad = _leg(tmp_path / "cur", "a", 2, {"Add": 1})
    (bad / "hardware-nightly-run.json").write_text("{")
    _leg(tmp_path / "cur", "b", 2, {"Add": 1}, [])

    markdown, warnings = _run(tmp_path, ["a", "b"], previous=False)

    assert "#### a\n\nTCM-leg result unreadable; see warnings." in markdown
    assert "#### b" in markdown and "| Add | 1 |  |" in markdown
    assert len(warnings) == 1 and warnings[0].startswith("a: current result unreadable: hardware-nightly-run.json")


def test_malformed_coverage_in_one_board_keeps_the_others(tmp_path: Path) -> None:
    bad = _leg(tmp_path / "cur", "a", 2, {"Add": 1}, [])
    (bad / "hardware" / "nightly-2-a" / "coverage.json").write_text(json.dumps({"deployed": ["invalid"]}))
    _leg(tmp_path / "cur", "c", 2, {"Add": 1}, [])
    _leg(tmp_path / "prev" / "1", "c", 1, {"Add": 1}, [])

    markdown, warnings = _run(tmp_path, ["a", "c"])

    assert "#### a\n\nTCM-leg result unreadable; see warnings." in markdown
    assert "Baseline: `hct-hardware-c-20261006-1`." in markdown
    assert warnings == ["a: current result unreadable: coverage.json: deployed is not an object"]


def test_malformed_baseline_coverage_falls_back_to_an_older_run(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 3, {"Add": 2}, [])
    bad = _leg(tmp_path / "prev" / "2", "b", 2, {"Add": 2}, [])
    (bad / "hardware" / "nightly-2-b" / "coverage.json").write_text(json.dumps({"deployed": ["invalid"]}))
    _leg(tmp_path / "prev" / "1", "b", 1, {"Add": 1}, [])

    markdown, warnings = _run(tmp_path, ["b"])

    assert "Baseline: `hct-hardware-b-20261006-1`." in markdown
    assert warnings == ["b: baseline run 2 result unreadable: coverage.json: deployed is not an object",
                        "b: Add skipped cases grew from 1 to 2"]


def test_unreadable_baseline_still_reports_current_counts(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 3}, [])
    bad = _leg(tmp_path / "prev" / "1", "b", 1, {"Add": 1}, [])
    (bad / "hardware" / "nightly-1-b" / "session_summary.json").write_text("[")

    markdown, warnings = _run(tmp_path, ["b"], note="should be replaced")

    assert ("No baseline: none of the 1 earlier run(s) searched has a comparable result for this board "
            "(1 unreadable).") in markdown
    assert "| Add | 3 |  |" in markdown
    assert len(warnings) == 1 and warnings[0].startswith("b: baseline run 1 result unreadable: session_summary.json")


def test_main_appends_markdown_and_prints_warnings(tmp_path: Path, capsys) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 2})
    _leg(tmp_path / "prev" / "1", "b", 1, {"Add": 1})
    out = tmp_path / "summary.md"
    out.write_text("### Hardware nightly\n")

    code = report.main(["--current", str(tmp_path / "cur"), "--previous", str(tmp_path / "prev"),
                        "--boards", json.dumps(["b"]), "--out", str(out)])

    assert code == 0
    text = out.read_text()
    assert text.startswith("### Hardware nightly\n### Skipped cases")
    assert "| Add | 2 | +1 ⚠ |" in text
    assert capsys.readouterr().out == "::warning::b: Add skipped cases grew from 1 to 2\n"


def test_main_without_downloaded_baseline_says_so(tmp_path: Path, capsys) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 2})
    out = tmp_path / "summary.md"

    report.main(["--current", str(tmp_path / "cur"), "--previous", str(tmp_path / "absent"),
                 "--boards", '["b"]', "--out", str(out)])

    assert "No baseline: no earlier run's artifacts were downloaded." in out.read_text()
    assert capsys.readouterr().out == (
        "::warning::b: no baseline because earlier runs left no result; skip growth not checked "
        "(no earlier run's artifacts were downloaded)\n"
    )


def test_main_with_baseline_note_does_not_warn(tmp_path: Path, capsys) -> None:
    _leg(tmp_path / "cur", "b", 2, {"Add": 2})
    out = tmp_path / "summary.md"

    report.main(["--current", str(tmp_path / "cur"), "--previous", str(tmp_path / "absent"),
                 "--baseline-note", "no earlier scheduled run", "--boards", '["b"]', "--out", str(out)])

    assert "No baseline: no earlier scheduled run." in out.read_text()
    assert capsys.readouterr().out == ""


def test_warning_text_is_escaped_for_workflow_commands() -> None:
    assert report._escape_command("a%b\nc\rd") == "a%25b%0Ac%0Dd"


@pytest.mark.parametrize("boards", ["not json", "{}", "[]", "[]x", '[""]', "[1]", '"b"'])
def test_main_rejects_bad_boards(tmp_path: Path, boards: str) -> None:
    with pytest.raises(SystemExit) as exc:
        report.main(["--current", str(tmp_path), "--boards", boards, "--out", str(tmp_path / "o.md")])
    assert exc.value.code == 2


def test_baseline_skips_a_newer_run_without_a_result(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 9, {"Add": 6}, ["arm_a", "arm_b"])
    _leg(tmp_path / "prev" / "8", "b", 8, {"Add": 6}, ["arm_a", "arm_b"])
    (tmp_path / "prev" / "8" / "hct-hardware-b-20261006-8" / "hardware-nightly-run.json").unlink()
    _leg(tmp_path / "prev" / "7", "b", 7, {"Add": 1}, ["arm_a"])

    markdown, warnings = _run(tmp_path, ["b"])

    assert "Baseline: `hct-hardware-b-20261006-7`." in markdown
    assert warnings == ["b: Add skipped cases grew from 1 to 6", "b: arm_b is newly untimed"]


def test_baseline_skips_unreadable_and_other_selection_runs_newest_first(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "b", 20, {"Add": 4}, [])
    bad = _leg(tmp_path / "prev" / "19", "b", 19, {"Add": 4}, [])
    (bad / "hardware-nightly-run.json").write_text("{")
    _leg(tmp_path / "prev" / "18", "b", 18, {"Add": 4}, [], selection={"suite": "int", "limit": 1})
    _leg(tmp_path / "prev" / "17", "b", 17, {"Add": 3}, [])
    _leg(tmp_path / "prev" / "9", "b", 9, {"Add": 0}, [])
    (tmp_path / "prev" / "notes").mkdir()

    markdown, warnings = _run(tmp_path, ["b"])

    assert "Baseline: `hct-hardware-b-20261006-17`." in markdown
    assert "| Add | 4 | +1 ⚠ |" in markdown
    assert warnings[0].startswith("b: baseline run 19 result unreadable")
    assert warnings[1:] == ["b: Add skipped cases grew from 3 to 4"]


def test_baselines_are_chosen_per_board(tmp_path: Path) -> None:
    _leg(tmp_path / "cur", "a", 3, {"Add": 2})
    _leg(tmp_path / "cur", "c", 3, {"Add": 2})
    _leg(tmp_path / "prev" / "2", "a", 2, {"Add": 2})
    _leg(tmp_path / "prev" / "1", "c", 1, {"Add": 1})

    markdown, warnings = _run(tmp_path, ["a", "c"])

    assert "Baseline: `hct-hardware-a-20261006-2`." in markdown
    assert "Baseline: `hct-hardware-c-20261006-1`." in markdown
    assert warnings == ["c: Add skipped cases grew from 1 to 2"]


def test_baseline_runs_order_numerically(tmp_path: Path) -> None:
    for name in ("9", "10", "x1"):
        (tmp_path / name).mkdir()
    (tmp_path / "11").write_text("")

    assert [run.name for run in report.baseline_runs(tmp_path)] == ["10", "9"]
    assert report.baseline_runs(tmp_path / "absent") == []
    assert report.baseline_runs(None) == []
