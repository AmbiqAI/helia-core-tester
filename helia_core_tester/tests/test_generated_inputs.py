"""discover_or_skip() skips only for an absent generated input and fails for a broken one (#150)."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.tests.generated_inputs import discover_or_skip, generated_family_dir

FAMILY = "BasicMathFunctions"


def _case_dir(project_root: Path, name: str) -> Path:
    directory = generated_family_dir(project_root, suite="int", cpu="cortex-m55", family=FAMILY) / name
    directory.mkdir(parents=True)
    return directory


def _write_descriptor(directory: Path, text: str) -> None:
    (directory / "descriptor.yaml").write_text(text)


VALID_DESCRIPTOR = """\
name: abs_default_s8
operator: Abs
resolved_tensor_dtypes:
  input: S8
  output: S8
"""


def test_absent_corpus_skips_naming_the_input(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match=r"int/cortex-m55/BasicMathFunctions case matching 'abs_default_s8'"):
        discover_or_skip(tmp_path, family=FAMILY, name_filter="abs_default_s8")


def test_other_generated_cases_do_not_satisfy_the_filter(tmp_path: Path) -> None:
    _write_descriptor(_case_dir(tmp_path, "add_default_s8"), VALID_DESCRIPTOR.replace("abs", "add"))
    with pytest.raises(pytest.skip.Exception):
        discover_or_skip(tmp_path, family=FAMILY, name_filter="abs_default_s8")


def test_present_case_without_descriptor_fails(tmp_path: Path) -> None:
    _case_dir(tmp_path, "abs_default_s8")
    with pytest.raises(pytest.fail.Exception, match="present but not discoverable"):
        discover_or_skip(tmp_path, family=FAMILY, name_filter="abs_default_s8")


def test_present_case_with_unreadable_descriptor_fails(tmp_path: Path) -> None:
    _write_descriptor(_case_dir(tmp_path, "abs_default_s8"), "name: [unterminated\n")
    with pytest.raises(pytest.fail.Exception, match="unreadable"):
        discover_or_skip(tmp_path, family=FAMILY, name_filter="abs_default_s8")


def test_present_case_is_returned(tmp_path: Path) -> None:
    _write_descriptor(_case_dir(tmp_path, "abs_default_s8"), VALID_DESCRIPTOR)
    cases = discover_or_skip(tmp_path, family=FAMILY, name_filter="abs_default_s8")
    assert [case.name for case in cases] == ["abs_default_s8"]
