"""Version tags come only from merged release-please release PRs (#135).

scripts/release_tag.py makes the decision that .github/workflows/tag-release.yml acts
on, so an ordinary merged pull request is exercised here without GitHub: it must not
produce a tag.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release_tag.py"
WORKFLOW = ROOT / ".github" / "workflows" / "tag-release.yml"
RELEASE_BRANCH = "release-please--branches--main--components--helia-core-tester"
MANIFEST = {".": "0.3.0"}

_spec = importlib.util.spec_from_file_location("release_tag", SCRIPT)
release_tag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_tag)


def _event(*, merged: bool = True, labels=("autorelease: pending",), head: str = RELEASE_BRANCH, base: str = "main") -> dict:
    return {
        "action": "closed",
        "pull_request": {
            "merged": merged,
            "labels": [{"name": name} for name in labels],
            "head": {"ref": head},
            "base": {"ref": base},
        },
    }


def test_merged_release_pull_request_is_tagged_from_the_manifest() -> None:
    assert release_tag.release_tag(_event(), MANIFEST) == "v0.3.0"


@pytest.mark.parametrize(
    "event",
    [
        _event(labels=(), head="feat/some-feature"),
        _event(labels=("enhancement",), head="fix/some-bug"),
        _event(merged=False),
        _event(labels=()),
        _event(head="feat/labelled-by-hand"),
        _event(base="dev/next"),
        {"action": "opened", "pull_request": _event()["pull_request"]},
    ],
    ids=[
        "ordinary-merge",
        "labelled-ordinary-merge",
        "release-pr-closed-unmerged",
        "release-branch-without-label",
        "label-without-release-branch",
        "release-pr-into-other-base",
        "not-a-close",
    ],
)
def test_anything_but_a_merged_release_pull_request_gets_no_tag(event: dict) -> None:
    assert release_tag.release_tag(event, MANIFEST) is None


@pytest.mark.parametrize("manifest", [{}, {".": ""}, {".": "0.3"}, {".": "v0.3.0"}, {".": 3}])
def test_release_without_a_valid_manifest_version_fails(manifest: dict) -> None:
    with pytest.raises(ValueError, match="manifest"):
        release_tag.release_tag(_event(), manifest)


@pytest.mark.parametrize(("event", "expected"), [(_event(), "tag=v0.3.0"), (_event(labels=(), head="feat/x"), "tag=")])
def test_cli_prints_github_output(tmp_path: Path, event: dict, expected: str) -> None:
    event_path = tmp_path / "event.json"
    manifest_path = tmp_path / "manifest.json"
    event_path.write_text(json.dumps(event))
    manifest_path.write_text(json.dumps(MANIFEST))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(event_path), str(manifest_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected


def test_repository_manifest_holds_a_valid_version() -> None:
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text())
    assert release_tag.release_tag(_event(), manifest) is not None


def test_workflow_tags_only_through_the_release_decision() -> None:
    """The workflow must gate the job on the release label, take the tag from the
    script, and create nothing unless the script returned a tag."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["tag"]
    assert "merged == true" in job["if"]
    assert "autorelease: pending" in job["if"]
    steps = {step.get("name"): step for step in job["steps"]}
    assert "scripts/release_tag.py" in steps["Decide the release tag"]["run"]
    for name in ("Create and push tag", "Mark the release PR tagged"):
        assert steps[name]["if"] == "steps.release.outputs.tag != ''"
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_no_other_workflow_creates_tags() -> None:
    for path in sorted(WORKFLOW.parent.glob("*.yml")):
        if path == WORKFLOW:
            continue
        assert "git tag" not in path.read_text(), f"{path.name} creates a git tag outside tag-release.yml"
