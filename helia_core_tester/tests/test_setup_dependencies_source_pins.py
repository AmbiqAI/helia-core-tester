"""CMSIS-5 / ethos-u-core-platform are fetched at, and verified against, pinned commits."""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from helia_core_tester.scripts import setup_dependencies as sd

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")

NAME = "CMSIS_5"


def _git(cwd: Path, *args: str) -> str:
    env_args = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "safe.directory=*"]
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
        pin = sd.SourcePin("CMSIS-5", url or repo.as_uri(), commit, "CMSIS")
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
    assert sd.SOURCE_PINS["CMSIS_5"].marker == "CMSIS"
    assert sd.SOURCE_PINS["ethos-u-core-platform"].marker == "core_platform"


def test_readme_does_not_restate_the_pinned_commits():
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    assert "SOURCE_PINS" in readme
    for pin in sd.SOURCE_PINS.values():
        assert pin.commit[:12] not in readme, pin


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
    sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == first
    assert "already at pinned" in capsys.readouterr().out


def test_untracked_file_counts_as_a_local_change(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    sd.setup_cmsis5(downloads)
    stray = downloads / NAME / "stray.h"
    stray.write_text("#define EXTRA 1\n")
    with pytest.raises(sd.SourcePinError, match="local changes"):
        sd.setup_cmsis5(downloads)
    assert stray.exists()
    sd.setup_cmsis5(downloads, force=True)
    assert not stray.exists()
    assert _head(downloads / NAME) == first


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
    with pytest.raises(sd.SourcePinError, match="not a git checkout git can read"):
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


def test_ethos_platform_uses_its_own_pin(tmp_path, upstream, monkeypatch):
    repo, first, _ = upstream
    monkeypatch.setitem(
        sd.SOURCE_PINS, "ethos-u-core-platform", sd.SourcePin("Ethos", repo.as_uri(), first, "core_platform")
    )
    sd.setup_ethos_u_platform(tmp_path / "downloads")
    assert _head(tmp_path / "downloads" / "ethos-u-core-platform") == first


def test_full_clone_moved_to_pin_stays_full(tmp_path, upstream, pin_to):
    repo, first, second = upstream
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    _git(downloads, "clone", "-q", repo.as_uri(), NAME)
    assert _head(downloads / NAME) == second
    pin_to(first)
    sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == first
    assert _git(downloads / NAME, "rev-parse", "--is-shallow-repository") == "false"
    assert _git(downloads / NAME, "rev-list", "--count", second) == "2"


def test_shallow_clone_fetches_missing_pin(tmp_path, upstream, pin_to):
    repo, first, second = upstream
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    _git(downloads, "clone", "-q", "--depth=1", repo.as_uri(), NAME)
    assert _git(downloads / NAME, "rev-parse", "--is-shallow-repository") == "true"
    pin_to(first)
    sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == first


# --- warn mode (the setup every build/run starts) -----------------------------


def test_warn_mode_fetches_a_missing_source_at_the_pin(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    pin_to(first)
    sd.setup_cmsis5(tmp_path / "downloads", pin_mode="warn")
    assert _head(tmp_path / "downloads" / NAME) == first


def test_warn_mode_keeps_a_checkout_at_another_commit(tmp_path, upstream, pin_to, capsys):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    pin_to(first)
    sd.setup_cmsis5(downloads, pin_mode="warn")
    assert _head(downloads / NAME) == second
    assert "WARNING" in capsys.readouterr().out


def test_warn_mode_keeps_a_dirty_checkout(tmp_path, upstream, pin_to, capsys):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    (downloads / NAME / "f.txt").write_text("local edit\n")
    pin_to(first)
    sd.setup_cmsis5(downloads, pin_mode="warn")
    assert (downloads / NAME / "f.txt").read_text() == "local edit\n"
    assert "local changes" in capsys.readouterr().out


def test_warn_mode_keeps_a_non_git_install_with_its_marker(tmp_path, upstream, pin_to, capsys):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    (downloads / NAME / "CMSIS").mkdir(parents=True)
    sd.setup_cmsis5(downloads, pin_mode="warn")
    assert (downloads / NAME / "CMSIS").is_dir()
    assert not (downloads / NAME / ".git").exists()
    assert "cannot be verified" in capsys.readouterr().out


@pytest.mark.parametrize("make", ["empty_dir", "file"])
def test_warn_mode_rejects_an_unusable_directory(tmp_path, upstream, pin_to, make):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    dest = downloads / NAME
    dest.mkdir() if make == "empty_dir" else dest.write_text("x")
    with pytest.raises(sd.SourcePinError, match="not a git checkout git can read"):
        sd.setup_cmsis5(downloads, pin_mode="warn")
    assert dest.exists()


def test_unknown_pin_mode_is_rejected(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    pin_to(first)
    with pytest.raises(ValueError, match="pin_mode"):
        sd.setup_cmsis5(tmp_path / "downloads", pin_mode="lenient")
    assert not (tmp_path / "downloads").exists()


def test_run_setup_enforces_the_pins(monkeypatch, tmp_path):
    from helia_core_tester.fvp import env as fvp_env

    calls = []
    monkeypatch.setattr(fvp_env.subprocess, "call", lambda cmd, **kwargs: calls.append(cmd) or 0)
    fvp_env.call_setup_dependencies(tmp_path)
    (cmd,) = calls
    assert "--source-pins" not in cmd  # default: enforce


# --- git environment --------------------------------------------------------------


def test_dubious_ownership_is_trusted_for_our_checkout(tmp_path, upstream, pin_to, monkeypatch):
    # GIT_TEST_ASSUME_DIFFERENT_OWNER makes git refuse every repo as another user's,
    # as a root container does with a bind-mounted workspace.
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    # Ignore the host's system/global git config: a safe.directory entry there
    # (the GitHub runner trusts the repo) hides the refusal this test needs.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("GIT_CONFIG_SYSTEM", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=downloads / NAME, check=True, capture_output=True)
    sd.setup_cmsis5(downloads)
    pin_to(first)
    sd.setup_cmsis5(downloads)
    assert _head(downloads / NAME) == first


def test_git_runs_without_prompts_or_stdin(monkeypatch, tmp_path):
    seen = []
    real_run = subprocess.run

    def spy(cmd, **kwargs):
        seen.append(kwargs)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(sd.subprocess, "run", spy)
    sd._git(["init", "--quiet"], tmp_path)
    (kwargs,) = seen
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_run_command_returns_stdout_and_quiet_stays_silent(capsys):
    assert sd.run_command(["git", "--version"], quiet=True).startswith("git version")
    with pytest.raises(subprocess.CalledProcessError):
        sd.run_command(["git", "no-such-subcommand"], quiet=True)
    assert capsys.readouterr().out == ""


# --- replacement, staging and locking ------------------------------------------------


def test_failed_swap_restores_the_old_checkout(tmp_path, upstream, pin_to, monkeypatch):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    pin_to(first)
    real_replace = sd.os.replace

    def fail_stage_move(src, dst):
        if Path(src).name == NAME and Path(src).parent != downloads:
            raise OSError("simulated move failure")
        return real_replace(src, dst)

    monkeypatch.setattr(sd.os, "replace", fail_stage_move)
    with pytest.raises(OSError, match="simulated"):
        sd.setup_cmsis5(downloads, force=True)
    assert _head(downloads / NAME) == second
    assert _leftover_staging(downloads) == []


def test_unremovable_old_checkout_does_not_fail_the_swap(tmp_path, upstream, pin_to, monkeypatch, capsys):
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    pin_to(first)
    real_rmtree = shutil.rmtree
    attempts = []

    def rmtree(path, *args, **kwargs):
        if Path(path).name == f"{NAME}.old":
            attempts.append(path)
            raise PermissionError("simulated read-only file")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(sd.shutil, "rmtree", rmtree)
    sd.setup_cmsis5(downloads, force=True)
    assert attempts, "the old checkout was never removed explicitly"
    assert _head(downloads / NAME) == first
    assert "could not remove the old checkout" in capsys.readouterr().out


def test_stale_staging_dirs_are_removed(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    stale = downloads / f".{NAME}-killed" / NAME
    stale.mkdir(parents=True)
    (stale / "partial.pack").write_text("x")
    sd.setup_cmsis5(downloads)
    assert _leftover_staging(downloads) == []
    assert _head(downloads / NAME) == first


@pytest.mark.skipif(sd.fcntl is None, reason="no fcntl")
def test_concurrent_setup_waits_for_the_lock(tmp_path, upstream, pin_to):
    _, first, _ = upstream
    pin_to(first)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    errors = []
    with open(downloads / f".{NAME}.lock", "a") as handle:
        sd.fcntl.flock(handle, sd.fcntl.LOCK_EX)
        worker = threading.Thread(target=lambda: _capture(errors, sd.setup_cmsis5, downloads))
        worker.start()
        time.sleep(0.3)
        assert worker.is_alive()
        assert not (downloads / NAME).exists()
        sd.fcntl.flock(handle, sd.fcntl.LOCK_UN)
    worker.join(timeout=30)
    assert not worker.is_alive() and errors == []
    assert _head(downloads / NAME) == first


def _capture(errors: list, fn, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:  # surfaced by the asserting test
        errors.append(exc)


@pytest.mark.parametrize(
    "commit",
    ["", "abc123", "main", "A" * 40, "g" * 40, "0" * 39, "0" * 41, " " + "0" * 40, "0" * 40 + "\n"],
)
def test_malformed_pin_is_rejected_before_any_git_call(tmp_path, pin_to, monkeypatch, commit):
    pin_to(commit)

    def no_git(*a, **k):
        raise AssertionError("git must not run for a malformed pin")

    monkeypatch.setattr(sd, "_git", no_git)
    with pytest.raises(sd.SourcePinError, match="40-hex"):
        sd.setup_cmsis5(tmp_path / "downloads")
    assert not (tmp_path / "downloads" / NAME).exists()


def test_without_a_lock_stale_staging_is_left_alone(tmp_path, upstream, pin_to, monkeypatch):
    # A concurrent setup's live staging dir looks the same as a killed one's.
    _, first, _ = upstream
    pin_to(first)
    monkeypatch.setattr(sd, "fcntl", None)
    monkeypatch.setattr(sd, "msvcrt", None)
    downloads = tmp_path / "downloads"
    live = downloads / f".{NAME}-other" / NAME
    live.mkdir(parents=True)
    sd.setup_cmsis5(downloads)
    assert live.is_dir()
    assert _head(downloads / NAME) == first


class _FakeMsvcrt:
    LK_LOCK, LK_UNLCK = 1, 0

    def __init__(self, busy_attempts: int):
        self.busy_attempts = busy_attempts
        self.calls = []

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes))
        if mode == self.LK_LOCK and self.busy_attempts:
            self.busy_attempts -= 1
            raise OSError("locked by another process")


def test_windows_lock_retries_until_held_and_releases(tmp_path, upstream, pin_to, monkeypatch):
    _, first, _ = upstream
    pin_to(first)
    fake = _FakeMsvcrt(busy_attempts=2)
    monkeypatch.setattr(sd, "fcntl", None)
    monkeypatch.setattr(sd, "msvcrt", fake)
    downloads = tmp_path / "downloads"
    stale = downloads / f".{NAME}-killed"
    stale.mkdir(parents=True)
    sd.setup_cmsis5(downloads)
    assert fake.calls == [(fake.LK_LOCK, 1)] * 3 + [(fake.LK_UNLCK, 1)]
    assert not stale.exists()
    assert _head(downloads / NAME) == first


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
@pytest.mark.parametrize("move", [False, True], ids=["reuse", "move"])
def test_hidden_index_entry_counts_as_a_local_change(tmp_path, upstream, pin_to, flag, move):
    # git status reports such a tree as clean, and checkout keeps its bytes.
    _, first, second = upstream
    downloads = tmp_path / "downloads"
    pin_to(second)
    sd.setup_cmsis5(downloads)
    dest = downloads / NAME
    _git(dest, "update-index", flag, "f.txt")
    (dest / "f.txt").write_text("hidden edit\n")
    assert _git(dest, "status", "--porcelain") == ""
    if move:
        pin_to(first)
    with pytest.raises(sd.SourcePinError, match="skip-worktree"):
        sd.setup_cmsis5(downloads)
    assert _head(dest) == second
    assert (dest / "f.txt").read_text() == "hidden edit\n"
    sd.setup_cmsis5(downloads, force=True)
    assert (dest / "f.txt").read_text() == ("one\n" if move else "two\n")
