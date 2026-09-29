"""discover_or_skip() skips only for an absent generated input and fails for a broken one (#150).

Each case records which outcome the helper produced rather than expecting one inside
pytest.raises(): a skip raised where a failure is expected would otherwise make this
test itself report SKIPPED, which is the silent pass these guards exist to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.tests.generated_inputs import discover_or_skip, generated_family_dir

FAMILY = "BasicMathFunctions"

VALID_DESCRIPTOR = """\
name: abs_default_s8
operator: Abs
resolved_tensor_dtypes:
  input: S8
  output: S8
"""


def _case_dir(project_root: Path, name: str) -> Path:
    directory = generated_family_dir(project_root, suite="int", cpu="cortex-m55", family=FAMILY) / name
    directory.mkdir(parents=True)
    return directory


def _outcome(project_root: Path, name_filter: str) -> tuple[str, object]:
    try:
        return "returned", discover_or_skip(project_root, family=FAMILY, name_filter=name_filter)
    except pytest.skip.Exception as exc:
        return "skipped", str(exc)
    except pytest.fail.Exception as exc:
        return "failed", str(exc)


def test_absent_corpus_skips_naming_the_input(tmp_path: Path) -> None:
    outcome, detail = _outcome(tmp_path, "abs_default_s8")
    assert outcome == "skipped", detail
    assert "int/cortex-m55/BasicMathFunctions case matching 'abs_default_s8'" in detail


def test_other_generated_cases_do_not_satisfy_the_filter(tmp_path: Path) -> None:
    (_case_dir(tmp_path, "add_default_s8") / "descriptor.yaml").write_text(VALID_DESCRIPTOR.replace("abs", "add"))
    outcome, detail = _outcome(tmp_path, "abs_default_s8")
    assert outcome == "skipped", detail


def test_present_case_without_descriptor_fails(tmp_path: Path) -> None:
    _case_dir(tmp_path, "abs_default_s8")
    outcome, detail = _outcome(tmp_path, "abs_default_s8")
    assert outcome == "failed", detail
    assert "['abs_default_s8']" in detail and "present but not discoverable" in detail


def test_present_case_with_unreadable_descriptor_fails(tmp_path: Path) -> None:
    case_dir = _case_dir(tmp_path, "abs_default_s8")
    (case_dir / "descriptor.yaml").write_text("name: [unterminated\n")
    outcome, detail = _outcome(tmp_path, "abs_default_s8")
    assert outcome == "failed", detail
    assert f"{case_dir / 'descriptor.yaml'} is unreadable" in detail


def test_present_case_is_returned(tmp_path: Path) -> None:
    (_case_dir(tmp_path, "abs_default_s8") / "descriptor.yaml").write_text(VALID_DESCRIPTOR)
    outcome, detail = _outcome(tmp_path, "abs_default_s8")
    assert outcome == "returned", detail
    assert [case.name for case in detail] == ["abs_default_s8"]
