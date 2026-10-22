"""scripts/conformance_leg.py: the contract-conformance CI legs cannot pass by omission."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "conformance_leg.py"
spec = importlib.util.spec_from_file_location("conformance_leg", SCRIPT)
leg = importlib.util.module_from_spec(spec)
sys.modules["conformance_leg"] = leg
spec.loader.exec_module(leg)

NAMES = ["a_conformance", "b_conformance", "c_conformance"]


def _manifest(tmp_path: Path, tests=(), skipped=()) -> Path:
    path = tmp_path / "int" / "cortex-m55" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"tests": [{"name": n} for n in tests], "skipped": list(skipped)}))
    return path


def _skip(name: str, status: str, reason: str = "") -> dict:
    return {"name": name, "status": status, **({"reason": reason} if reason else {})}


def test_the_grid_is_listed() -> None:
    names = leg.conformance_names()
    assert names and all("conformance" in n for n in names) and len(set(names)) == len(names)


def test_a_present_leg_needs_generated_cases_and_only_capability_skips(tmp_path: Path) -> None:
    path = _manifest(tmp_path, tests=NAMES[:2], skipped=[_skip(NAMES[2], "skipped_capability")])
    assert leg.check("present", path, NAMES) == ["2 generated, 1 capability-skipped"]


@pytest.mark.parametrize("tests, skipped, message", [
    (NAMES[:2], [_skip(NAMES[2], "skipped_conformance", "no planner")], "missing planner on a present leg"),
    (NAMES[:2], [_skip(NAMES[2], "skipped_kernel_symbol")], "other than a capability"),
    ((), [_skip(n, "skipped_capability") for n in NAMES], "no conformance case was generated"),
    (NAMES[:2], [], "neither generated nor skipped"),
])
def test_a_present_leg_fails_closed(tmp_path: Path, tests, skipped, message: str) -> None:
    with pytest.raises(leg.LegError, match=message):
        leg.check("present", _manifest(tmp_path, tests, skipped), NAMES)


def test_an_absent_leg_needs_every_case_skipped_with_a_reason(tmp_path: Path) -> None:
    path = _manifest(tmp_path, skipped=[_skip(n, "skipped_conformance", "no planner") for n in NAMES])
    assert leg.check("absent", path, NAMES) == ["3 skipped for a missing planner"]


@pytest.mark.parametrize("tests, skipped, message", [
    (NAMES[:1], [_skip(n, "skipped_conformance", "x") for n in NAMES[1:]], "generated without a planner"),
    ((), [_skip(n, "skipped_conformance") for n in NAMES], "with a reason"),
    ((), [_skip(n, "skipped_capability", "x") for n in NAMES], "with a reason"),
])
def test_an_absent_leg_fails_closed(tmp_path: Path, tests, skipped, message: str) -> None:
    with pytest.raises(leg.LegError, match=message):
        leg.check("absent", _manifest(tmp_path, tests, skipped), NAMES)


def test_an_unreadable_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(leg.LegError, match="cannot be read"):
        leg.check("present", tmp_path / "missing.json", NAMES)
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(leg.LegError, match="cannot be read"):
        leg.check("absent", bad, NAMES)


def test_the_cli_exits_nonzero_on_a_failed_leg(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _manifest(tmp_path)
    argv = ["check", "--expect", "present", "--cpu", "cortex-m55", "--generated-tests-dir", str(tmp_path)]
    assert leg.main(argv) == 1
    assert "neither generated nor skipped" in capsys.readouterr().err
