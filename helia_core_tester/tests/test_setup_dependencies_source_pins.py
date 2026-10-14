"""CMSIS-5 / ethos-u-core-platform are fetched at, and verified against, pinned commits."""

import shutil
import subprocess
from pathlib import Path

import pytest

from helia_core_tester.scripts import setup_dependencies as sd

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")

NAME = "CMSIS_5"


def _git(cwd: Path, *args: str) -> str:
    env_args = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    return subprocess.run(
        ["git", *env_args, *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    """A local repo with two commits; returns (path, first_sha, second_sha)."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f.txt").write_text("one\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "one")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "f.txt").write_text("two\n")
    _git(repo, "commit", "-q", "-am", "two")
    second = _git(repo, "rev-parse", "HEAD")
    return repo, first, second


@pytest.fixture
def pin_to(monkeypatch, upstream):
    repo, _, _ = upstream

    def _pin(commit: str, url: str | None = None) -> sd.SourcePin:
        pin = sd.SourcePin("CMSIS-5", url or repo.as_uri(), commit)
        monkeypatch.setitem(sd.SOURCE_PINS, NAME, pin)
        return pin

    return _pin


def _head(path: Path) -> str:
    return _git(path, "rev-parse", "HEAD")


def _leftover_staging(downloads: Path) -> list:
    return [p for p in downloads.iterdir() if p.name.startswith(f".{NAME}-")]


def test_shipped_pins_are_full_commits_on_expected_hosts():
    assert set(sd.SOURCE_PINS) == {"CMSIS_5", "ethos-u-core-platform"}
    for pin in sd.SOURCE_PINS.values():
        assert sd._COMMIT_RE.fullmatch(pin.commit), pin
        assert pin.url.startswith("https://"), pin


def test_fresh_install_lands_on_pinned_commit_not_tip(tmp_path, upstream, pin_to):
    _, first, second = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == first != second
    assert (downloads / NAME / "f.txt").read_text() == "one\n"
    assert _leftover_staging(downloads) == []


def test_existing_checkout_at_pin_is_reused(tmp_path, upstream, pin_to, capsys):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    sd.setup_cmsis5(downloads)
    marker = downloads / NAME / "untracked.txt"
    marker.write_text("kept")
    sd.setup_cmsis5(downloads)
    assert marker.exists()
    assert "already at pinned" in capsys.readouterr().out


def test_existing_checkout_at_other_commit_is_moved_to_pin(tmp_path, upstream, pin_to):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == second
    pin_to(first)
    sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == first


def test_dirty_checkout_is_rejected_and_left_alone(tmp_path, upstream, pin_to):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    (downloads / NAME / "f.txt").write_text("local edit\n")
    pin_to(first)
    with pytest.raises(sd.SourcePinError, match="local changes"):
        sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == second
    assert (downloads / NAME / "f.txt").read_text() == "local edit\n"


def test_dirty_checkout_is_replaced_with_force(tmp_path, upstream, pin_to):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    (downloads / NAME / "f.txt").write_text("local edit\n")
    pin_to(first)
    sd.setup_cmsis5(downloads, force=True)
    assert _head(downloads / NAME) == first
    assert (downloads / NAME / "f.txt").read_text() == "one\n"


@pytest.mark.parametrize("make", ["plain_dir", "empty_git", "file"])
def test_unverifiable_directory_is_rejected(tmp_path, upstream, pin_to, make):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    dest = downloads / NAME
    downloads.mkdir()
    if make == "plain_dir":
        dest.mkdir()
        (dest / "CMSIS").mkdir()
    elif make == "empty_git":
        dest.mkdir()
        _git(dest, "init", "-q")
    else:
        dest.write_text("not a dir")
    with pytest.raises(sd.SourcePinError, match="not a git checkout"):
        sd.setup_cmsis5(downloads)
    assert dest.exists()


def test_unverifiable_directory_is_replaced_with_force(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    (downloads / NAME / "CMSIS").mkdir(parents=True)
    sd.setup_cmsis5(downloads, force=True)
    assert _head(downloads / NAME) == first
    assert not (downloads / NAME / "CMSIS").exists()


def test_unknown_commit_fails_and_leaves_nothing(tmp_path, pin_to):
    pin_to("0" * 40)
    downloads = tmp_path / "downloads"
    with pytest.raises(sd.SourcePinError, match="fetch"):
        sd.setup_cmsis5(downloads)
    assert not (downloads / NAME).exists()
    assert _leftover_staging(downloads) == []


def test_failed_force_fetch_keeps_existing_checkout(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    downloads = tmp_path / "downloads"
    pin_to(first)
    sd.setup_cmsis5(downloads)
    pin_to("0" * 40)
    with pytest.raises(sd.SourcePinError):
        sd.setup_cmsis5(downloads, force=True)
    assert _head(downloads / NAME) == first
    assert _leftover_staging(downloads) == []


def test_unreachable_remote_fails_closed(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    pin_to(first, url=(tmp_path / "missing").as_uri())
    downloads = tmp_path / "downloads"
    with pytest.raises(sd.SourcePinError, match="fetch"):
        sd.setup_cmsis5(downloads)
    assert not (downloads / NAME).exists()


def test_failed_move_of_existing_checkout_keeps_old_head(tmp_path, upstream, pin_to):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    pin_to("1" * 40)
    with pytest.raises(sd.SourcePinError, match="fetch"):
        sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == second


def test_checkout_landing_elsewhere_is_detected(tmp_path, upstream, pin_to, monkeypatch):
    _, first, second = upstream
    pin_to(first)
    monkeypatch.setattr(sd, "_checkout_head", lambda path: second)
    with pytest.raises(sd.SourcePinError, match="expected pinned"):
        sd.setup_cmsis5(tmp_path / "downloads")
    assert not (tmp_path / "downloads" / NAME).exists()


@pytest.mark.parametrize(
    "commit",
    ["", "abc123", "main", "A" * 40, "g" * 40, "0" * 39, "0" * 41, " " + "0" * 40],
)
def test_malformed_pin_is_rejected_before_any_git_call(tmp_path, pin_to, monkeypatch, commit):
    pin_to(commit)

    def no_git(*a, **k):
        raise AssertionError("git must not run for a malformed pin")

    monkeypatch.setattr(sd, "_git", no_git)
    with pytest.raises(sd.SourcePinError, match="40-hex"):
        sd.setup_cmsis5(tmp_path / "downloads")
    assert not (tmp_path / "downloads").exists()


def test_ethos_platform_uses_its_own_pin(tmp_path, upstream, monkeypatch):
    repo, first, _ = upstream
    monkeypatch.setitem(
        sd.SOURCE_PINS, "ethos-u-core-platform", sd.SourcePin("Ethos", repo.as_uri(), first)
    )
    sd.setup_ethos_u_platform(tmp_path / "downloads")
    assert _head(tmp_path / "downloads" / "ethos-u-core-platform") == first
