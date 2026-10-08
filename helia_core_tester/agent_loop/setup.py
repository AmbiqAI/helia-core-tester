"""`agent-loop init`: build a campaign workspace."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from helia_core_tester.hardware.boards import repo_root, resolve_board
from helia_core_tester.hardware.candidate_check import CheckError
from helia_core_tester.hardware.candidate_check import _git as check_git
from helia_core_tester.hardware.harness_lock import tester_state

from .agent import WRAPPERS, agent_settings, write_wrappers
from .config import Campaign, ConfigError
from .judge import apply_patch, object_sizes
from .prompt import baseline_rows, render_prompt
from .workspace import Workspace, kernel_lib

SECRET_BYTES = 32
Echo = Callable[[str], None]


class InitError(RuntimeError):
    """A step of init failed."""


def _run(cmd: list[str], log: Path, cwd: Path | None = None) -> None:
    """Run; stdout and stderr to log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        rc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise InitError(f"{cmd[0]} {' '.join(cmd[1:4])} ... exited {rc}; see {log}")


def _git(*args: str, cwd: Path) -> str:
    try:
        return check_git(cwd, *args).decode().strip()
    except CheckError as exc:
        raise InitError(f"git {' '.join(args[:3])}: {str(exc)[-300:]}") from exc


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def check_paths(ws: Workspace, campaign: Campaign) -> None:
    """Secrets live apart from everything else."""
    secrets_dir, root = campaign.secrets_dir, ws.root
    for other in (root, campaign.kernels_repo, repo_root()):
        if _inside(secrets_dir, other) or _inside(other, secrets_dir):
            raise ConfigError(f"secrets_dir must sit outside {other}")
    if _inside(root, campaign.kernels_repo) or _inside(root, repo_root()):
        raise ConfigError("workspace must sit outside the repos")


def pin_tester(ws: Workspace, sha: str | None) -> str:
    """Detached tester worktree at a clean commit."""
    if ws.tester.exists():
        head = _git("rev-parse", "HEAD", cwd=ws.tester)
        if sha and head != sha:
            raise InitError(f"{ws.tester} is at {head[:12]}, not {sha[:12]}")
        return head
    source = repo_root()
    state = tester_state(source)
    if state["dirty"] is not False or not state["commit"]:
        raise InitError(f"Tester {source} is dirty or unknown; commit it.")
    sha = sha or state["commit"]
    # A deleted W/tester blocks re-adding.
    _git("worktree", "prune", cwd=source)
    _git("worktree", "add", "-q", "--detach", str(ws.tester), sha, cwd=source)
    downloads = source / "artifacts" / "downloads"
    if downloads.is_dir():
        # Reuse the toolchain cache.
        (ws.tester / "artifacts").mkdir(exist_ok=True)
        (ws.tester / "artifacts" / "downloads").symlink_to(downloads.resolve())
    _run(["uv", "--directory", str(ws.tester), "sync", "-q"], ws.logs / "uv-sync.log")
    return sha


def tree_at(dest: Path, sha: str, tag: bool) -> bool:
    """Clean checkout at sha, tagged if asked."""
    try:
        head = _git("rev-parse", "HEAD", cwd=dest)
        tagged = not tag or _git("rev-parse", "base^{commit}", cwd=dest) == sha
        clean = not _git("status", "--porcelain", cwd=dest)
    except InitError:
        return False
    return head == sha and tagged and clean


def shallow_tree(repo: Path, sha: str, dest: Path, tag: bool = False) -> None:
    """Standalone one-commit clone, no remote."""
    if dest.exists():
        if tree_at(dest, sha, tag):
            return
        # Interrupted earlier; start over.
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", str(dest), cwd=dest.parent)
    _git("fetch", "-q", "--depth", "1", f"file://{repo}", sha, cwd=dest)
    _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=dest)
    if tag:
        _git("tag", "base", cwd=dest)


def make_hidden(ws: Workspace, campaign: Campaign) -> None:
    """Fresh secret and hidden set, private."""
    hidden, seed = ws.hidden_dir(campaign), ws.seed_file(campaign)
    if (hidden / "done").is_file():
        return
    root = campaign.secrets_dir
    # Resume reuses this campaign's seed.
    if not seed.is_file():
        if root.exists() and any(root.iterdir()):
            raise InitError(f"{root} is not empty; use a fresh secrets_dir")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        fd = os.open(seed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_hex(SECRET_BYTES))
    shutil.rmtree(hidden, ignore_errors=True)
    cpu = resolve_board(campaign.board).cpu
    _run([*ws.tester_cmd(), "generate", "--cpu", cpu, "--random-shapes", str(campaign.hidden_shapes),
          "--hidden-dir", str(hidden), "--hidden-seed-file", str(seed)], ws.logs / "hidden.log")
    hidden.chmod(0o700)
    (hidden / "done").write_text("", encoding="utf-8")


def make_baseline(ws: Workspace, campaign: Campaign, leg: str) -> None:
    """candidate baseline under the board lock."""
    out = ws.baseline(leg)
    if (out / "baseline.json").is_file():
        return
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bench-agent", "run", campaign.bench_id, "--reason", f"agent-loop {campaign.name} baseline {leg}",
           "--timeout", str(campaign.lock_timeout_s), "--", *ws.tester_cmd(), "candidate", "baseline",
           "--kernels", str(ws.base), "--board", campaign.board, "--out", str(out), "--repeats", str(campaign.repeats),
           "--placement", leg, "--op", campaign.op, "--dtype", campaign.dtype]
    for case_id in campaign.case_ids:
        cmd += ["--case-id", case_id]
    if campaign.hidden_shapes:
        cmd += ["--hidden-set", str(ws.hidden_dir(campaign))]
    _run(cmd, ws.logs / f"baseline-{leg}.log", cwd=ws.tester)


def make_size_ref(ws: Workspace, campaign: Campaign) -> None:
    """Base kernel code bytes per object."""
    if ws.size_ref.is_file():
        return
    _run([*ws.tester_cmd(), "hardware", "build", "--board", campaign.board, "--cmsis-nn-root", str(ws.base),
          "--build-dir", str(ws.size_build)], ws.logs / "size-ref.log")
    sizes = object_sizes(kernel_lib(ws.size_build))
    ws.size_ref.write_text(json.dumps(sizes, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def write_agent_files(ws: Workspace, campaign: Campaign, start_diff: bytes | None) -> None:
    """Prompt, settings and wrappers."""
    rows = baseline_rows(ws.baseline(campaign.legs[0]))
    paths = {name: ws.bin / name for name in WRAPPERS} | {"results": ws.results}
    ws.prompt.write_text(render_prompt(campaign, rows, paths, start_diff), encoding="utf-8")
    settings = agent_settings(ws, campaign, extra_denies=[repo_root()])
    ws.settings.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    write_wrappers(ws)
    ws.results.mkdir(exist_ok=True)
    ws.ledger.mkdir(exist_ok=True)


def init_workspace(ws: Workspace, campaign: Campaign, echo: Echo = print) -> dict:
    """Every init step; done steps are skipped."""
    check_paths(ws, campaign)
    facts: dict = {}
    if ws.state.is_file():
        saved, facts = ws.load()
        if saved != campaign:
            raise InitError(f"{ws.root} holds a different campaign")
        echo("Resuming init.")
    elif ws.root.exists() and any(ws.root.iterdir()):
        raise InitError(f"{ws.root} is not empty")
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.logs.mkdir(exist_ok=True)
    facts.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    facts["source_tester"] = str(repo_root())
    # Saved first, so any failure resumes.
    ws.save(campaign, facts)
    facts["tester_commit"] = pin_tester(ws, facts.get("tester_commit"))
    echo(f"Tester pinned at {facts['tester_commit'][:12]} in {ws.tester}")
    base = facts.get("base_commit") or _git("rev-parse", "--verify", f"{campaign.base_ref}^{{commit}}",
                                            cwd=campaign.kernels_repo)
    facts["base_commit"] = base
    ws.save(campaign, facts)
    shallow_tree(campaign.kernels_repo, base, ws.base)
    try:
        start_diff = campaign.start_patch.read_bytes() if campaign.start_patch else None
    except OSError as exc:
        raise InitError(f"Cannot read start_patch {campaign.start_patch}: {exc.strerror}") from exc
    if not ws.agent.exists():
        # Patched aside, then moved in.
        staging = ws.root / "agent.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        shallow_tree(campaign.kernels_repo, base, staging, tag=True)
        if start_diff is not None:
            apply_patch(staging, start_diff)
        staging.rename(ws.agent)
    echo(f"Base and agent trees at {base[:12]}")
    if campaign.hidden_shapes:
        echo("Generating the hidden set...")
        make_hidden(ws, campaign)
    for leg in campaign.legs:
        echo(f"Recording the {leg} baseline on {campaign.bench_id}...")
        make_baseline(ws, campaign, leg)
    echo("Building the size reference...")
    make_size_ref(ws, campaign)
    write_agent_files(ws, campaign, start_diff)
    # Other commands require this.
    facts["ready"] = True
    ws.save(campaign, facts)
    return facts
