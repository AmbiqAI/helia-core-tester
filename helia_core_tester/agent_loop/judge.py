"""Trusted judge: check, submit and disasm for the agent."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from helia_core_tester.hardware.candidate_check import _git
from helia_core_tester.hardware.candidate_eval import SNAPSHOT_TREES, VERDICT_EXITS, CopyBudget, TooLarge, copy_tree
from helia_core_tester.hardware.candidate_scan import run_binutil

from helia_core_tester.hardware.toolchain import toolchain_spec

from .config import Campaign
from .ledger import (EXIT_BUDGET, SKIPPED, Ledger, LockBusy, agent_view, file_lock, is_infra, ledger_row,
                     merge_legs, next_note, passing_evals, skip_reason)
from .workspace import Workspace, kernel_lib

EDIT_TREES = ("Source", "Include")
FN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
DISASM_LINES = 600
BUILD_ERRORS = 40
RETRY_PAUSE_S = 30
# `timeout` exits 124 on expiry.
TIMED_OUT = 124
# Least lock wait and eval time.
MIN_LOCK_S = 10
MIN_EVAL_S = 60


def object_sizes(lib: Path) -> dict[str, int]:
    """Text bytes per object in an archive."""
    sizes = {}
    for line in run_binutil("arm-none-eabi-size", [str(lib)]).splitlines()[1:]:
        cols = line.split()
        if len(cols) >= 6:
            sizes[cols[5]] = int(cols[0])
    return sizes


def size_delta(ref: dict[str, int], cur: dict[str, int]) -> dict[str, Any]:
    """Kernel code bytes vs the base."""
    changed = {k: [ref.get(k, 0), cur.get(k, 0)] for k in sorted(set(ref) | set(cur)) if ref.get(k, 0) != cur.get(k, 0)}
    return {"kernel_text_bytes": [sum(ref.values()), sum(cur.values())],
            "delta_bytes": sum(cur.values()) - sum(ref.values()), "changed_objects": changed}


def tree_diff(ws: Workspace, tree: Path) -> bytes:
    """base -> tree diff over the edit trees."""
    out = b""
    for name in EDIT_TREES:
        new = (tree / name).relative_to(ws.root)
        proc = subprocess.run(["diff", "-ruN", f"base/{name}", str(new)], cwd=ws.root, capture_output=True)
        out += proc.stdout
    return out


def stage_tree(ws: Workspace, base_sha: str, area: Path) -> Path:
    """Fresh base clone with the agent's trees."""
    tree = area / "tree"
    shutil.rmtree(tree, ignore_errors=True)
    area.mkdir(parents=True, exist_ok=True)
    _git(area, "clone", "-q", "--no-checkout", str(ws.base), tree.name)
    _git(tree, "checkout", "-q", base_sha)
    budget = CopyBudget()
    for name in SNAPSHOT_TREES:
        src, dst = ws.agent / name, tree / name
        shutil.rmtree(dst, ignore_errors=True)
        if src.is_symlink():
            # candidate check flags it.
            dst.symlink_to(src.readlink())
        elif src.is_dir():
            copy_tree(src, dst, budget)
    return tree


def _trim(lines: list[str], ws: Workspace, area: Path, build_dir: Path) -> list[str]:
    prefixes = (f"{build_dir}/nsx_app/modules/nsx-cmsis-nn/", f"{area / 'tree'}/", f"{ws.root}/")
    for prefix in prefixes:
        lines = [line.replace(prefix, "") for line in lines]
    return lines


class OutOfTime(Exception):
    """A bounded command hit the deadline."""


def run_bounded(cmd: list[str], deadline: float, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) -> subprocess.CompletedProcess:
    """Run until deadline; kill the whole group."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise OutOfTime
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=left)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise OutOfTime from None
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run_check(ws: Workspace, campaign: Campaign, base_sha: str, area: Path,
              deadline: Optional[float] = None) -> tuple[bool, dict[str, Any]]:
    """Stage area/tree; rules, build, size."""
    if deadline is None:
        deadline = time.monotonic() + campaign.submit_deadline_s
    try:
        return _check_steps(ws, campaign, base_sha, area, deadline)
    except OutOfTime:
        return False, {"check": {"ok": False, "error": "check or build timed out"}, "build": "timed out"}


def _check_steps(ws: Workspace, campaign: Campaign, base_sha: str, area: Path, deadline: float
                 ) -> tuple[bool, dict[str, Any]]:
    out: dict[str, Any] = {}
    with file_lock(area.with_name(f".{area.name}-tree.lock")):
        try:
            tree = stage_tree(ws, base_sha, area)
        except TooLarge as exc:
            out["check"] = {"ok": False, "error": str(exc)}
            return False, out
        proc = run_bounded([*ws.tester_cmd(), "candidate", "check", str(tree), "--base", base_sha], deadline)
        (area / "check.err").write_text(proc.stderr, encoding="utf-8")
        try:
            report = json.loads(proc.stdout)
            out["check"] = {k: report.get(k) for k in ("ok", "findings", "error") if k in report}
        except ValueError:
            tail = _trim(proc.stderr.splitlines()[-20:], ws, area, area / "build")
            out["check"] = {"ok": False, "error": "\n".join(tail)[-1500:]}
        if proc.returncode != 0 or not out["check"].get("ok"):
            out["check"]["ok"] = False
            out["build"] = "skipped: check failed"
            return False, out
        builds, sizes = {}, {}
        for toolchain in campaign.toolchains:
            builds[toolchain], sizes[toolchain] = _build_one(ws, campaign, tree, area, toolchain, deadline)
        # One toolchain keeps the flat shape.
        keyed = len(campaign.toolchains) > 1
        out["build"] = builds if keyed else builds[campaign.toolchains[0]]
        if all(b == "ok" for b in builds.values()):
            out["code_size"] = sizes if keyed else sizes[campaign.toolchains[0]]
            return True, out
    return False, out


def _build_one(ws: Workspace, campaign: Campaign, tree: Path, area: Path, toolchain: str, deadline: float
               ) -> tuple[Any, dict]:
    """Build with one toolchain; size vs its ref."""
    spec = toolchain_spec(toolchain)
    build_dir = spec.build_dir(area / "build")
    log = area / f"build{spec.dir_suffix}.log"
    with log.open("w", encoding="utf-8") as handle:
        build = run_bounded(
            [*ws.tester_cmd(), "hardware", "build", "--board", campaign.board, "--cmsis-nn-root", str(tree),
             "--toolchain", toolchain, "--build-dir", str(build_dir)], deadline, stdout=handle, stderr=subprocess.STDOUT,
        )
    if build.returncode != 0:
        lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if re.search(r"error|warning", line)]
        return {"ok": False, "errors": _trim(lines[:BUILD_ERRORS], ws, area, build_dir)}, {}
    try:
        ref = json.loads(ws.size_ref_of(toolchain).read_text(encoding="utf-8"))
        return "ok", size_delta(ref, object_sizes(kernel_lib(build_dir)))
    except (OSError, ValueError) as exc:
        return "ok", {"error": str(exc)}


def check(ws: Workspace) -> int:
    """Print the no-board check view."""
    campaign, facts = ws.load()
    ok, out = run_check(ws, campaign, facts["base_commit"], ws.check_dir)
    out = {"result": "ok" if ok else "not_ok", "evals_used": 0, **out}
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


def disasm(ws: Workspace, name: str, toolchain: str = "") -> int:
    """One function from the check build."""
    campaign, _ = ws.load()
    toolchain = toolchain or campaign.toolchains[0]
    if not FN_RE.fullmatch(name or "") or toolchain not in campaign.toolchains:
        print(f"usage: disasm <function_name> [--toolchain {'|'.join(campaign.toolchains)}]   (run check first)")
        return 2
    try:
        lib = kernel_lib(toolchain_spec(toolchain).build_dir(ws.check_dir / "build"))
        text = run_binutil(toolchain_spec(toolchain).objdump(), ["-d", "--no-show-raw-insn", str(lib)])
    except FileNotFoundError:
        print("no build yet: run check first")
        return 2
    except (OSError, ValueError):
        print("objdump failed on the check build")
        return 2
    lines, on = [], False
    for line in text.splitlines():
        if line.endswith(f"<{name}>:"):
            on = True
        if on:
            if not line.strip():
                break
            lines.append(line)
    if not lines:
        print(f"function {name} not found in the kernel library")
        return 1
    print("\n".join(lines[:DISASM_LINES]))
    if len(lines) > DISASM_LINES:
        print(f"... truncated: {len(lines)} lines total")
    return 0


def toolchain_drift(campaign: Campaign, facts: dict) -> bool:
    """A compiler differs from the baselines'."""
    for toolchain in campaign.toolchains:
        old, now = facts["toolchains"].get(toolchain), toolchain_spec(toolchain).installed()
        if old and now and old != now:
            return True
    return False


def leg_command(ws: Workspace, campaign: Campaign, eid: str, leg: str, lock_s: int, eval_s: int) -> list[str]:
    """bench-agent wrapped `candidate eval`."""
    cmd = ["bench-agent", "run", campaign.bench_id, "--reason", f"agent-loop {campaign.name} {eid} {leg}",
           "--timeout", str(lock_s), "--", "timeout", str(eval_s), *ws.tester_cmd(), "candidate", "eval",
           "--kernels", str(ws.submit_dir / "tree"), "--baseline", str(ws.baseline(leg))]
    if campaign.min_score is not None:
        cmd += ["--min-score", str(campaign.min_score)]
    return cmd


def _budget(campaign: Campaign, deadline: float) -> Optional[tuple[int, int]]:
    """Lock wait and eval time left."""
    left = int(deadline - time.monotonic())
    eval_s = min(campaign.eval_timeout_s, left - MIN_LOCK_S)
    if eval_s < MIN_EVAL_S:
        return None
    return min(campaign.lock_timeout_s, left - eval_s), eval_s


def run_leg(ws: Workspace, campaign: Campaign, eid: str, leg: str, deadline: float,
            runner=subprocess.run) -> tuple[dict, int]:
    """One leg; retry errors while time lasts."""
    verdict: Optional[dict] = None
    # A charged error outlives later busy retries.
    charged: Optional[dict] = None
    attempt = 0
    for attempt in range(1, campaign.retries + 2):
        if attempt > 1:
            time.sleep(RETRY_PAUSE_S)
        times = _budget(campaign, deadline)
        if times is None:
            break
        err = ws.ledger / f"{eid}.{leg}.err"
        with err.open("a", encoding="utf-8") as handle:
            proc = runner(leg_command(ws, campaign, eid, leg, *times), cwd=ws.tester, stdout=subprocess.PIPE,
                          stderr=handle, text=True)
        (ws.ledger / f"{eid}.{leg}.{attempt}.json").write_text(proc.stdout or "", encoding="utf-8")
        try:
            verdict = json.loads(proc.stdout)
        except (TypeError, ValueError):
            verdict = None
        if verdict is None and proc.returncode == TIMED_OUT:
            # A hang is the candidate's.
            return {"verdict": "error", "stage": "run", "reason": f"eval timed out after {times[1]} s"}, attempt
        if is_infra(verdict):
            continue
        if verdict.get("verdict") != "error":
            return verdict, attempt
        charged = verdict
    verdict = charged or verdict
    if not isinstance(verdict, dict) or not verdict.get("verdict"):
        # Never pass raw stderr on.
        verdict = {"verdict": "error", "stage": "board", "reason": "board busy or eval failed"}
    return {k: verdict.get(k) for k in ("verdict", "stage", "reason") if verdict.get(k)}, attempt


def _emit(ws: Workspace, eid: str, view: dict) -> int:
    text = json.dumps(view, indent=1)
    ws.results.mkdir(parents=True, exist_ok=True)
    (ws.results / f"{eid}.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return view["exit_code"]


def submit(ws: Workspace, runner=subprocess.run, checker=run_check) -> int:
    """Freeze, check, run every leg, record."""
    campaign, facts = ws.load()
    deadline = time.monotonic() + campaign.submit_deadline_s
    ledger = Ledger(ws.ledger)
    try:
        return _submit_locked(ws, campaign, facts, ledger, deadline, runner, checker)
    except LockBusy:
        # Another submit holds the lock.
        ledger.append({"eval": None, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdict": "error",
                       "charged": False, "infra": True, "note": "submit lock busy"})
        print(json.dumps({"verdict": "error", "exit_code": VERDICT_EXITS["error"],
                          "evals_left": campaign.evals - ledger.charged(),
                          "note": "Another submit is running; not charged."}))
        return VERDICT_EXITS["error"]


def _submit_locked(ws: Workspace, campaign: Campaign, facts: dict, ledger: Ledger, deadline: float, runner,
                   checker) -> int:
    with file_lock(ws.root / ".submit.lock", deadline - MIN_EVAL_S):
        if ledger.charged() >= campaign.evals:
            print(json.dumps({"verdict": "budget_spent", "exit_code": EXIT_BUDGET, "evals_left": 0}))
            return EXIT_BUDGET
        if ledger.infra_streak() >= campaign.max_infra_errors:
            print(json.dumps({"verdict": "error", "exit_code": VERDICT_EXITS["error"],
                              "note": "The board keeps failing. Stop and write your summary."}))
            return VERDICT_EXITS["error"]
        if facts.get("toolchains") and toolchain_drift(campaign, facts):
            ledger.append({"eval": None, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdict": "error",
                           "charged": False, "infra": True, "note": "toolchain changed"})
            print(json.dumps({"verdict": "error", "exit_code": VERDICT_EXITS["error"],
                              "note": "Compiler changed since init; not charged. Stop."}))
            return VERDICT_EXITS["error"]
        eid = ledger.next_id()
        # Legs judge this frozen copy.
        ok, checked = checker(ws, campaign, facts["base_commit"], ws.submit_dir, deadline)
        diff = tree_diff(ws, ws.submit_dir / "tree")
        (ws.ledger / f"{eid}.diff").write_bytes(diff)
        size = checked.get("code_size") or {"error": "no build"}
        if not ok:
            ledger.append(ledger_row(eid, "rejected", {}, charged=False, infra=False, size=size, diff=diff, attempts={},
                                     runs=campaign.runs))
            view = {"verdict": "rejected", "exit_code": VERDICT_EXITS["rejected"],
                    "evals_left": campaign.evals - ledger.charged(),
                    "note": "Failed check before the board; not charged.", **checked}
            return _emit(ws, eid, view)
        legs: dict[str, Optional[dict]] = {}
        attempts: dict[str, int] = {}
        infra = False
        # tcm first, so it gates mram.
        order = sorted(campaign.runs, key=lambda r: (campaign.toolchains.index(r.toolchain), r.placement != "tcm"))
        for run in order:
            reason = skip_reason(run, legs, campaign.runs)
            if reason:
                legs[run.name] = {"verdict": SKIPPED, "reason": reason}
                continue
            legs[run.name], attempts[run.name] = run_leg(ws, campaign, eid, run.name, deadline, runner)
            if is_infra(legs[run.name]) or legs[run.name].get("stage") == "board":
                infra = True
                break
        overall = "error" if infra else merge_legs(legs, campaign.leg_names)
        ledger.append(ledger_row(eid, overall, legs, charged=not infra, infra=infra, size=size, diff=diff,
                                 attempts=attempts, runs=campaign.runs))
        note = None
        if infra:
            # No free scores from earlier legs.
            legs = {}
            note = "Board busy or failing; not charged. Submit again."
        elif overall == "error":
            note = "The board run failed; check for faults or hangs."
        left = campaign.evals - ledger.charged()
        next_step = None if infra else next_note(passing_evals(ledger.rows(), campaign.runs), left,
                                                 campaign.size_budget, overall == "pass")
        view = agent_view(overall, legs, evals_left=left, size=size,
                          runs=campaign.runs, note=note, next_step=next_step)
        return _emit(ws, eid, view)


HUNK_RE = re.compile(rb"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
# Git extended headers we accept.
GIT_META = (b"diff ", b"index ", b"new file mode ", b"deleted file mode ")


def _rewrite_path(mark: bytes, line: bytes) -> bytes:
    """Map any diff path to a/<tree>/..."""
    body = line.rstrip(b"\r\n")
    path, sep, tail = body[4:].partition(b"\t")
    tail += line[len(body):]
    if path == b"/dev/null":
        return line
    for tree in EDIT_TREES:
        key = tree.encode() + b"/"
        at = path.find(b"/" + key)
        rel = path[at + 1:] if at >= 0 else path if path.startswith(key) else None
        if rel is not None and b"/../" not in b"/" + rel + b"/":
            side = b"a/" if mark == b"--- " else b"b/"
            return mark + side + rel + sep + tail
    raise ValueError(f"patch touches {path.decode(errors='replace')}, outside Source/Include")


def clean_patch(diff: bytes) -> bytes:
    """Strict parse; every header pair checked."""
    lines = diff.splitlines(keepends=True)
    out, i, headers = [], 0, False
    while i < len(lines):
        line = lines[i]
        if line.startswith(b"--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith(b"+++ "):
                raise ValueError("patch: --- without +++")
            out += [_rewrite_path(b"--- ", line), _rewrite_path(b"+++ ", lines[i + 1])]
            i, headers = i + 2, True
            continue
        hunk = HUNK_RE.match(line)
        if hunk:
            if not headers:
                raise ValueError("patch: hunk before file headers")
            old = int(hunk.group(1) or 1)
            new = int(hunk.group(2) or 1)
            out.append(line)
            i += 1
            # Consume exactly the hunk body.
            while old > 0 or new > 0:
                if i >= len(lines):
                    raise ValueError("patch: hunk ends early")
                body, kind = lines[i], lines[i][:1]
                if kind == b" ":
                    old, new = old - 1, new - 1
                elif kind == b"-":
                    old -= 1
                elif kind == b"+":
                    new -= 1
                elif kind != b"\\":
                    raise ValueError("patch: bad hunk line")
                if old < 0 or new < 0:
                    raise ValueError("patch: hunk longer than its header")
                out.append(body)
                i += 1
            while i < len(lines) and lines[i].startswith(b"\\"):
                out.append(lines[i])
                i += 1
            continue
        if line.startswith(GIT_META):
            out.append(line)
            i += 1
            continue
        raise ValueError(f"patch: unexpected line {line[:60]!r}")
    return b"".join(out)


def apply_patch(tree: Path, diff: bytes) -> None:
    """Apply a base -> agent diff to tree."""
    proc = subprocess.run(["patch", "-p1", "--batch", "--forward", "-d", str(tree)], input=clean_patch(diff),
                          capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"start patch does not apply: {proc.stdout.decode(errors='replace')[-500:]}")
