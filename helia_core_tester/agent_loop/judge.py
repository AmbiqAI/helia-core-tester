"""Trusted judge: check, submit and disasm for the agent."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from helia_core_tester.hardware.candidate_check import _git
from helia_core_tester.hardware.candidate_eval import SNAPSHOT_TREES, VERDICT_EXITS, CopyBudget, TooLarge, copy_tree
from helia_core_tester.hardware.candidate_scan import run_binutil

from .config import Campaign
from .ledger import EXIT_BUDGET, Ledger, agent_view, file_lock, is_infra, ledger_row, merge_legs, scored
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


def _trim(lines: list[str], ws: Workspace, area: Path) -> list[str]:
    prefixes = (f"{area / 'build'}/nsx_app/modules/nsx-cmsis-nn/", f"{area / 'tree'}/", f"{ws.root}/")
    for prefix in prefixes:
        lines = [line.replace(prefix, "") for line in lines]
    return lines


def run_check(ws: Workspace, campaign: Campaign, base_sha: str, area: Path) -> tuple[bool, dict[str, Any]]:
    """Stage area/tree; rules, build, size."""
    out: dict[str, Any] = {}
    with file_lock(area.with_name(f".{area.name}-tree.lock")):
        try:
            tree = stage_tree(ws, base_sha, area)
        except TooLarge as exc:
            out["check"] = {"ok": False, "error": str(exc)}
            return False, out
        proc = subprocess.run([*ws.tester_cmd(), "candidate", "check", str(tree), "--base", base_sha],
                              capture_output=True, text=True)
        (area / "check.err").write_text(proc.stderr, encoding="utf-8")
        try:
            report = json.loads(proc.stdout)
            out["check"] = {k: report.get(k) for k in ("ok", "findings", "error") if k in report}
        except ValueError:
            tail = _trim(proc.stderr.splitlines()[-20:], ws, area)
            out["check"] = {"ok": False, "error": "\n".join(tail)[-1500:]}
        if proc.returncode != 0 or not out["check"].get("ok"):
            out["check"]["ok"] = False
            out["build"] = "skipped: check failed"
            return False, out
        log = area / "build.log"
        with log.open("w", encoding="utf-8") as handle:
            build = subprocess.run(
                [*ws.tester_cmd(), "hardware", "build", "--board", campaign.board, "--cmsis-nn-root", str(tree),
                 "--build-dir", str(area / "build")], stdout=handle, stderr=subprocess.STDOUT,
            )
        if build.returncode != 0:
            lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if re.search(r"error|warning", line)]
            out["build"] = {"ok": False, "errors": _trim(lines[:BUILD_ERRORS], ws, area)}
            return False, out
        out["build"] = "ok"
        try:
            ref = json.loads(ws.size_ref.read_text(encoding="utf-8"))
            out["code_size"] = size_delta(ref, object_sizes(kernel_lib(area / "build")))
        except (OSError, ValueError) as exc:
            out["code_size"] = {"error": str(exc)}
    return True, out


def check(ws: Workspace) -> int:
    """Print the no-board check view."""
    campaign, facts = ws.load()
    ok, out = run_check(ws, campaign, facts["base_commit"], ws.check_dir)
    out = {"result": "ok" if ok else "not_ok", "evals_used": 0, **out}
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


def disasm(ws: Workspace, name: str) -> int:
    """One function from the check build."""
    if not FN_RE.fullmatch(name or ""):
        print("usage: disasm <function_name>   (run check first)")
        return 2
    try:
        lib = kernel_lib(ws.check_dir / "build")
        text = run_binutil("arm-none-eabi-objdump", ["-d", "--no-show-raw-insn", str(lib)])
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
        (ws.ledger / f"{eid}.{leg}.json").write_text(proc.stdout or "", encoding="utf-8")
        try:
            verdict = json.loads(proc.stdout)
        except (TypeError, ValueError):
            verdict = None
        if verdict is None and proc.returncode == TIMED_OUT:
            # A hang is the candidate's.
            return {"verdict": "error", "stage": "run", "reason": f"eval timed out after {times[1]} s"}, attempt
        if not is_infra(verdict) and verdict.get("verdict") != "error":
            return verdict, attempt
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
    with file_lock(ws.root / ".submit.lock"):
        if ledger.charged() >= campaign.evals:
            print(json.dumps({"verdict": "budget_spent", "exit_code": EXIT_BUDGET, "evals_left": 0}))
            return EXIT_BUDGET
        if ledger.infra_streak() >= campaign.max_infra_errors:
            print(json.dumps({"verdict": "error", "exit_code": VERDICT_EXITS["error"],
                              "note": "The board keeps failing. Stop and write your summary."}))
            return VERDICT_EXITS["error"]
        eid = ledger.next_id()
        # Legs judge this frozen copy.
        ok, checked = checker(ws, campaign, facts["base_commit"], ws.submit_dir)
        diff = tree_diff(ws, ws.submit_dir / "tree")
        (ws.ledger / f"{eid}.diff").write_bytes(diff)
        size = checked.get("code_size") or {"error": "no build"}
        if not ok:
            ledger.append(ledger_row(eid, "rejected", {}, charged=False, infra=False, size=size, diff=diff, attempts={}))
            view = {"verdict": "rejected", "exit_code": VERDICT_EXITS["rejected"],
                    "evals_left": campaign.evals - ledger.charged(),
                    "note": "Failed check before the board; not charged.", **checked}
            return _emit(ws, eid, view)
        legs: dict[str, Optional[dict]] = {}
        attempts: dict[str, int] = {}
        infra = False
        for leg in campaign.legs:
            # Later legs need a scored leg.
            if legs and not all(scored(v) for v in legs.values()):
                break
            legs[leg], attempts[leg] = run_leg(ws, campaign, eid, leg, deadline, runner)
            if is_infra(legs[leg]) or legs[leg].get("stage") == "board":
                infra = True
                break
        overall = "error" if infra else merge_legs(legs, campaign.legs)
        ledger.append(ledger_row(eid, overall, legs, charged=not infra, infra=infra, size=size, diff=diff,
                                 attempts=attempts))
        note = None
        if infra:
            # No free scores from earlier legs.
            legs = {}
            note = "Board busy or failing; not charged. Submit again."
        elif overall == "error":
            note = "The board run failed; check for faults or hangs."
        view = agent_view(overall, legs, evals_left=campaign.evals - ledger.charged(), size=size,
                          first_leg=campaign.legs[0], note=note)
        return _emit(ws, eid, view)


def _rewrite_header(line: bytes) -> bytes:
    """Map any diff path to a/<tree>/..."""
    body = line.rstrip(b"\r\n")
    mark, rest = body[:4], body[4:]
    path, sep, tail = rest.partition(b"\t")
    tail += line[len(body):]
    if path == b"/dev/null":
        return line
    for tree in EDIT_TREES:
        key = tree.encode() + b"/"
        at = path.find(b"/" + key)
        rel = path[at + 1:] if at >= 0 else path if path.startswith(key) else None
        if rel is not None:
            side = b"a/" if mark == b"--- " else b"b/"
            return mark + side + rel + sep + tail
    raise ValueError(f"patch touches {path.decode(errors='replace')}, outside Source/Include")


def apply_patch(tree: Path, diff: bytes) -> None:
    """Apply a base -> agent diff to tree."""
    lines, header = [], True
    for line in diff.splitlines(keepends=True):
        # Headers sit between diff and @@.
        if line.startswith(b"diff "):
            header = True
        elif line.startswith(b"@@"):
            header = False
        lines.append(_rewrite_header(line) if header and line.startswith((b"--- ", b"+++ ")) else line)
    proc = subprocess.run(["patch", "-p1", "--batch", "--forward", "-d", str(tree)], input=b"".join(lines),
                          capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"start patch does not apply: {proc.stdout.decode(errors='replace')[-500:]}")
