"""Version tags come only from merged release-please release PRs (#135).

scripts/release_tag.py makes the decision that .github/workflows/tag-release.yml acts
on, so an ordinary merged pull request is exercised here without GitHub: it must not
produce a tag.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
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


@pytest.mark.parametrize("head", [RELEASE_BRANCH, "release-please--branches--main"])
def test_merged_release_pull_request_is_tagged_from_the_manifest(head: str) -> None:
    assert release_tag.release_tag(_event(head=head), MANIFEST) == "v0.3.0"


@pytest.mark.parametrize(
    "event",
    [
        _event(labels=(), head="feat/some-feature"),
        _event(labels=("enhancement",), head="fix/some-bug"),
        _event(merged=False),
        _event(labels=()),
        _event(head="feat/labelled-by-hand"),
        _event(head="release-please--branches--mainline"),
        _event(base="dev/next"),
        {"action": "opened", "pull_request": _event()["pull_request"]},
    ],
    ids=[
        "ordinary-merge",
        "labelled-ordinary-merge",
        "release-pr-closed-unmerged",
        "release-branch-without-label",
        "label-without-release-branch",
        "branch-prefix-without-boundary",
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


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _step(name: str) -> dict:
    return next(step for step in _workflow()["jobs"]["tag"]["steps"] if step.get("name") == name)


def test_workflow_tags_only_through_the_release_decision() -> None:
    """The job is gated on a merged, release-labelled PR; the tag comes from the
    script; nothing is created unless the script returned a tag."""
    job = _workflow()["jobs"]["tag"]
    assert job["if"] == (
        "github.event.pull_request.merged == true && "
        "contains(github.event.pull_request.labels.*.name, 'autorelease: pending')"
    )
    assert "scripts/release_tag.py" in _step("Decide the release tag")["run"]
    for name in ("Create and push tag", "Mark the release PR tagged"):
        assert _step(name)["if"] == "steps.release.outputs.tag != ''"


def test_release_runs_share_no_concurrency_group() -> None:
    """Every closed PR runs this workflow, so a shared group would let a later
    ordinary merge cancel a queued release run before it tags."""
    workflow = _workflow()
    assert "concurrency" not in workflow
    assert all("concurrency" not in job for job in workflow["jobs"].values())


def test_no_other_workflow_creates_tags() -> None:
    for path in sorted(WORKFLOW.parent.glob("*.yml")):
        if path == WORKFLOW:
            continue
        text = path.read_text()
        for marker in ("git tag", "refs/tags/", "--tags"):
            assert marker not in text, f"{path.name} touches tags ({marker!r}) outside tag-release.yml"


@pytest.fixture
def tag_repo(tmp_path: Path) -> dict:
    """A clone of a bare origin with two commits, for running the workflow's tag step."""
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("git and bash required to run the workflow tag step")
    origin = tmp_path / "origin.git"
    clone = tmp_path / "clone"
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": os.environ["PATH"], "HOME": str(tmp_path)}

    def git(*args: str, cwd: Path = clone) -> str:
        return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "--bare", "-q", str(origin), cwd=tmp_path)
    git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
    git("commit", "-q", "--allow-empty", "-m", "first")
    first = git("rev-parse", "HEAD")
    git("commit", "-q", "--allow-empty", "-m", "second")
    second = git("rev-parse", "HEAD")
    git("push", "-q", "origin", "HEAD")
    return {"git": git, "env": env, "clone": clone, "first": first, "second": second}


def _run_tag_step(repo: dict, tag: str, sha: str) -> subprocess.CompletedProcess:
    script = _step("Create and push tag")["run"]
    env = dict(repo["env"], TAG=tag, SHA=sha)
    return subprocess.run(["bash", "-e", "-c", script], cwd=repo["clone"], env=env, capture_output=True, text=True)


def _origin_tag(repo: dict, tag: str) -> str:
    return repo["git"]("ls-remote", "origin", f"refs/tags/{tag}^{{}}", f"refs/tags/{tag}")


def test_tag_step_creates_a_new_tag_on_the_merge_commit(tag_repo: dict) -> None:
    result = _run_tag_step(tag_repo, "v0.3.0", tag_repo["first"])
    assert result.returncode == 0, result.stderr
    assert tag_repo["first"] in _origin_tag(tag_repo, "v0.3.0")


def test_tag_step_rerun_on_the_same_commit_is_a_no_op(tag_repo: dict) -> None:
    assert _run_tag_step(tag_repo, "v0.3.0", tag_repo["first"]).returncode == 0
    result = _run_tag_step(tag_repo, "v0.3.0", tag_repo["first"])
    assert result.returncode == 0, result.stderr
    assert "already points at" in result.stdout


def test_tag_step_refuses_a_tag_on_another_commit(tag_repo: dict) -> None:
    assert _run_tag_step(tag_repo, "v0.3.0", tag_repo["first"]).returncode == 0
    result = _run_tag_step(tag_repo, "v0.3.0", tag_repo["second"])
    assert result.returncode != 0
    assert "already exists at" in result.stdout
    assert tag_repo["first"] in _origin_tag(tag_repo, "v0.3.0")
