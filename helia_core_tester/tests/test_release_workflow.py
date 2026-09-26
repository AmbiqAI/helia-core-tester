"""Tags and GitHub releases come only from release-please, for merged release PRs (#135).

release-please creates a release only for a merged pull request labelled
`autorelease: pending` on its release branch, and it does so before it builds the next
release PR in the same run, so that run already sees the new release. These tests pin
the workflow configuration that behaviour depends on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"


def _load(path: Path) -> dict:
    workflow = yaml.safe_load(path.read_text())
    # PyYAML reads the bare `on:` key as the boolean True.
    if True in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _release_please_step() -> dict:
    steps = _load(RELEASE_WORKFLOW)["jobs"]["release-please"]["steps"]
    return next(step for step in steps if str(step.get("uses", "")).startswith("googleapis/release-please-action@"))


def test_release_please_creates_the_tag_and_release() -> None:
    options = _release_please_step().get("with") or {}
    assert options.get("skip-github-release") in (None, False, "false")
    assert options.get("skip-github-pull-request") in (None, False, "false")
    assert options["config-file"] == "release-please-config.json"
    assert options["manifest-file"] == ".release-please-manifest.json"


def test_release_runs_only_for_main_and_by_hand() -> None:
    triggers = _load(RELEASE_WORKFLOW)["on"]
    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"] == {"branches": ["main"]}


def test_release_runs_share_one_queue_and_never_cancel() -> None:
    """Parallel runs could build a release PR before the previous one is marked
    released, and a cancelled run could stop between creating a release and
    relabelling its PR. The group must not depend on the ref a run was started from."""
    concurrency = _load(RELEASE_WORKFLOW)["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert "${{" not in concurrency["group"]


def test_release_workflow_can_write_tags_releases_and_labels() -> None:
    permissions = _load(RELEASE_WORKFLOW)["permissions"]
    assert permissions == {"contents": "write", "issues": "write", "pull-requests": "write"}


def test_no_other_workflow_creates_tags_or_releases() -> None:
    assert not (WORKFLOWS / "tag-release.yml").exists()
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        if path == RELEASE_WORKFLOW:
            continue
        text = path.read_text()
        for marker in (
            "git tag",
            "refs/tags/",
            "--tags",
            "git/refs",
            "gh release",
            "/releases",
            "release-please-action",
            "action-gh-release",
            "release-action",
            "create-release",
        ):
            assert marker not in text, f"{path.name} touches tags or releases ({marker!r})"


def test_release_tags_are_v_and_the_manifest_version() -> None:
    config = json.loads((ROOT / "release-please-config.json").read_text())
    package = config["packages"]["."]
    assert package["include-v-in-tag"] is True
    assert package["include-component-in-tag"] is False
    assert config.get("skip-github-release") in (None, False)
    assert package.get("skip-github-release") in (None, False)
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text())
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["."])
