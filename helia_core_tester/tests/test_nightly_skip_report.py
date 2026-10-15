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


def _pages(*pages):
    """fetch_page over fixed pages of (id, conclusion); a page past the end is empty."""
    calls: list[int] = []

    def fetch(page: int):
        calls.append(page)
        runs = pages[page - 1] if page <= len(pages) else []
        return [{"id": run_id, "conclusion": conclusion} for run_id, conclusion in runs]

    return fetch, calls


def test_select_skips_newer_and_cancelled_runs_across_pages() -> None:
    # A rerun of 50: the first page is all newer or cancelled runs.
    fetch, calls = _pages([(60, "success"), (55, "failure"), (49, "cancelled")],
                          [(48, "success"), (47, "cancelled"), (46, "failure")],
                          [(45, "success"), (44, "success")])

    assert report.select_baseline_runs(fetch, 50, 3, 10, per_page=3) == ([48, 46, 45], False)
    assert calls == [1, 2, 3]


def test_select_stops_at_the_end_of_history() -> None:
    fetch, calls = _pages([(9, "success"), (8, "cancelled"), (7, "failure")], [(6, "success")])

    assert report.select_baseline_runs(fetch, 10, 3, 10, per_page=3) == ([9, 7, 6], False)
    fetch, calls = _pages([(9, "success"), (8, "cancelled"), (7, "cancelled")], [(6, "cancelled")])
    assert report.select_baseline_runs(fetch, 10, 3, 10, per_page=3) == ([9], True)
    assert calls == [1, 2]


def test_select_full_last_page_needs_one_more_empty_page() -> None:
    fetch, calls = _pages([(9, "cancelled"), (8, "cancelled")])

    assert report.select_baseline_runs(fetch, 10, 3, 10, per_page=2) == ([], True)
    assert calls == [1, 2]


def test_select_page_cap_bounds_the_search() -> None:
    fetch, calls = _pages(*[[(1000 - page, "cancelled")] for page in range(50)])

    assert report.select_baseline_runs(fetch, 2000, 3, 4, per_page=1) == ([], False)
    assert calls == [1, 2, 3, 4]


def test_select_ignores_duplicates_from_shifting_pages() -> None:
    fetch, _ = _pages([(9, "success"), (8, "success")], [(8, "success"), (7, "success")], [])

    assert report.select_baseline_runs(fetch, 10, 3, 10, per_page=2) == ([9, 8, 7], False)


@pytest.mark.parametrize("page", [None, {}, "x", [None], [{}], [{"id": "9"}], [{"id": True}], [{"id": 1.5}]])
def test_select_refuses_malformed_pages(page) -> None:
    with pytest.raises(report.ReportError, match="page 1"):
        report.select_baseline_runs(lambda _: page, 10, 3, 10)


@pytest.mark.parametrize(("count", "max_pages", "per_page"), [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_select_rejects_non_positive_bounds(count: int, max_pages: int, per_page: int) -> None:
    with pytest.raises(report.ReportError, match="positive"):
        report.select_baseline_runs(lambda _: [], 10, count, max_pages, per_page)


def _select(tmp_path: Path, fetch, *extra: str) -> tuple[int, str]:
    out = tmp_path / "github_output"
    code = report.select_main(["--repo", "o/r", "--before", "50", "--output", str(out), *extra], fetch)
    return code, out.read_text() if out.exists() else ""


def test_select_main_writes_runs(tmp_path: Path, capsys) -> None:
    pages = {1: [{"id": 51, "conclusion": "success"}, {"id": 49, "conclusion": "success"}]}
    code, output = _select(tmp_path, lambda repo, page: pages.get(page, []))

    assert code == 0 and output == "runs=49\n"
    assert capsys.readouterr().out == ""


def test_select_main_notes_an_empty_history(tmp_path: Path, capsys) -> None:
    code, output = _select(tmp_path, lambda repo, page: [{"id": 51, "conclusion": "success"}])

    assert code == 0 and output == "runs=\nnote=no earlier scheduled run\n"
    assert capsys.readouterr().out == ""


def test_select_main_warns_when_the_page_cap_hits(tmp_path: Path, capsys) -> None:
    full = [{"id": 60, "conclusion": "success"}] * report.PER_PAGE
    code, output = _select(tmp_path, lambda repo, page: full, "--max-pages", "2")

    assert code == 0 and output == "runs=\n"
    assert capsys.readouterr().out == "::warning::baseline run selection stopped after 2 page(s) with 0 of 3 run(s)\n"


def test_select_main_fails_loudly_on_api_errors(tmp_path: Path, capsys) -> None:
    def fetch(repo: str, page: int):
        raise report.ReportError(f"gh api page {page} failed: HTTP 502")

    code, output = _select(tmp_path, fetch)

    assert code == 1 and output == ""
    assert capsys.readouterr().out == "::error::baseline run selection failed: gh api page 1 failed: HTTP 502\n"


def test_main_dispatches_select_baselines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report, "_gh_runs_page", lambda repo, page: [{"id": 7, "conclusion": "success"}])

    code = report.main(["select-baselines", "--repo", "o/r", "--before", "8", "--output", str(tmp_path / "o")])

    assert code == 0 and (tmp_path / "o").read_text() == "runs=7\n"


@pytest.mark.parametrize("args", [
    ["--repo", "o/r", "--before", "0"], ["--repo", "o/r", "--before", "x"], ["--repo", "o r", "--before", "5"],
    ["--repo", "o/r/x", "--before", "5"], ["--repo", "o/r", "--before", "5", "--max-pages", "0"], ["--before", "5"],
])
def test_select_main_rejects_bad_arguments(tmp_path: Path, args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        report.select_main([*args, "--output", str(tmp_path / "o")], lambda repo, page: [])
    assert exc.value.code == 2


class _Done:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_gh_runs_page_queries_the_nightly(monkeypatch) -> None:
    seen = {}

    def run(command, **kwargs):
        seen["command"], seen["timeout"] = command, kwargs["timeout"]
        return _Done(0, json.dumps({"workflow_runs": [{"id": 3}]}))

    monkeypatch.setattr(report.subprocess, "run", run)

    assert report._gh_runs_page("o/r", 2) == [{"id": 3}]
    assert seen["command"][4] == "repos/o/r/actions/workflows/hardware-nightly.yml/runs"
    assert {"event=schedule", "branch=main", "status=completed", "per_page=100", "page=2"} <= set(seen["command"])
    assert seen["timeout"] == report.GH_TIMEOUT_S


@pytest.mark.parametrize(("outcome", "message"), [
    (_Done(1, stderr="HTTP 502"), "failed: HTTP 502"),
    (_Done(0, "{not json"), "gh api page 1"),
    (OSError("no gh"), "no gh"),
    (report.subprocess.TimeoutExpired("gh", 1), "timed out"),
])
def test_gh_runs_page_raises_on_failure(monkeypatch, outcome, message: str) -> None:
    def run(command, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(report.subprocess, "run", run)

    with pytest.raises(report.ReportError, match=message):
        report._gh_runs_page("o/r", 1)


def test_gh_runs_page_without_runs_is_refused_by_the_selection(monkeypatch) -> None:
    monkeypatch.setattr(report.subprocess, "run", lambda command, **kwargs: _Done(0, json.dumps({"message": "x"})))

    with pytest.raises(report.ReportError, match="not a list of runs"):
        report.select_baseline_runs(lambda page: report._gh_runs_page("o/r", page), 10, 3, 1)
