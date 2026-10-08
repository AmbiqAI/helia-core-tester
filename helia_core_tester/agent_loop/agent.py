"""Agent side: settings, wrappers, launch, status, selftest."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import Campaign
from .ledger import Ledger
from .workspace import Workspace

TOOLS = "Read,Edit,Write,Glob,Grep,Bash"
WRAPPERS = ("submit", "check", "disasm")
SELFTEST_MODEL = "haiku"


def _rule_path(path: Path) -> str:
    """Absolute paths need a // prefix."""
    return "/" + str(path)


def agent_settings(ws: Workspace, campaign: Campaign, extra_denies: Iterable[Path] = ()) -> dict[str, Any]:
    """Permission rules with absolute paths."""
    root = ws.root
    allow = [
        f"Read({_rule_path(ws.agent)}/**)",
        f"Read({_rule_path(ws.results)}/**)",
        f"Edit({_rule_path(ws.agent / 'Source')}/**)",
        f"Edit({_rule_path(ws.agent / 'Include')}/**)",
        f"Bash({ws.bin / 'submit'})",
        f"Bash({ws.bin / 'check'})",
        f"Bash({ws.bin / 'disasm'}:*)",
    ]
    dirs = (ws.tester, ws.base, root / "baselines", ws.ledger, ws.check_dir, ws.submit_dir, ws.logs, ws.size_build,
            root / "agent.tmp")
    files = (ws.size_ref, ws.state, ws.run_meta, ws.prompt, ws.settings)
    deny = [f"Read({_rule_path(d)}/**)" for d in dirs] + [f"Read({_rule_path(f)})" for f in files]
    deny += [f"Read({_rule_path(ws.agent / 'Tests')}/**)", f"Read({_rule_path(campaign.secrets_dir)}/**)",
             f"Read({_rule_path(campaign.kernels_repo)}/**)", "Read(~/.claude/**)"]
    deny += [f"Read({_rule_path(p)}/**)" for p in extra_denies]
    deny += ["WebFetch", "WebSearch"]
    return {"permissions": {"allow": allow, "deny": deny}}


def wrapper_text(ws: Workspace, name: str) -> str:
    """Tiny script the allow rules match."""
    cmd = shlex.join([*ws.tester_cmd(), "agent-loop", name, "--workspace", str(ws.root)])
    arg = ' "$1"' if name == "disasm" else ""
    return f"#!/bin/sh\n# Agent bridge to the trusted judge.\nexec {cmd}{arg}\n"


def write_wrappers(ws: Workspace) -> None:
    ws.bin.mkdir(parents=True, exist_ok=True)
    for name in WRAPPERS:
        path = ws.bin / name
        path.write_text(wrapper_text(ws, name), encoding="utf-8")
        path.chmod(0o755)


def claude_args(ws: Workspace, model: str, prompt: str) -> list[str]:
    """Headless claude, locked-down tools."""
    return [
        "claude", "-p", prompt, "--model", model, "--tools", TOOLS, "--settings", str(ws.settings),
        "--setting-sources", "project", "--permission-mode", "dontAsk", "--strict-mcp-config",
        "--output-format", "stream-json", "--verbose",
    ]


def pid_alive(pid: Optional[int]) -> bool:
    """Running and still a claude process."""
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"claude" in cmdline


def read_meta(ws: Workspace) -> dict[str, Any]:
    try:
        return json.loads(ws.run_meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


RESUME_PROMPT = "Continue the work plan from where you stopped."
NOTE_HEAD = "Operator note:"
NOTE_MAX_CHARS = 8000
STOP_WAITS = ((signal.SIGINT, 20.0), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0))

# USD per MTok: in, out, cache read, 5m write, 1h write.
# platform.claude.com/docs/en/about-claude/pricing, 2026-10.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0, 8.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5, 4.0),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25, 2.0),
}


def note_prompt(text: str) -> str:
    """Prefixed, size-capped operator note."""
    text = text.strip()
    if not text:
        raise RuntimeError("Note is empty")
    if len(text) > NOTE_MAX_CHARS:
        raise RuntimeError(f"Note exceeds {NOTE_MAX_CHARS} characters")
    return f"{NOTE_HEAD}\n{text}"


def model_prices(model: str) -> Optional[tuple]:
    """Longest price-table prefix match."""
    keys = [k for k in PRICES if model.startswith(k)]
    return PRICES[max(keys, key=len)] if keys else None


def _visible_chars(content: list) -> int:
    n = 0
    for block in content:
        if isinstance(block, dict):
            n += len(block.get("text") or block.get("thinking") or "")
            n += len(json.dumps(block["input"])) if "input" in block else 0
    return n


def usage_cost(events: list[dict]) -> Optional[float]:
    """Estimate from message usage; None if unpriced."""
    msgs: dict[str, dict] = {}
    for n, ev in enumerate(events):
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message") or {}
        if msg.get("model") == "<synthetic>":
            continue
        prices, usage = model_prices(msg.get("model") or ""), msg.get("usage")
        if prices is None or not usage:
            return None
        # Content blocks repeat the message id.
        entry = msgs.setdefault(msg.get("id") or f"#{n}", {"prices": prices, "usage": usage, "chars": 0})
        entry["chars"] += _visible_chars(msg.get("content") or [])
    return sum(_price(e["prices"], e["usage"], e["chars"]) for e in msgs.values())


def _price(prices: tuple, u: dict, chars: int = 0) -> float:
    p_in, p_out, p_read, p_5m, p_1h = prices
    w5 = (u.get("cache_creation") or {}).get("ephemeral_5m_input_tokens", 0)
    # Unsplit writes count at the 1h rate.
    w1 = u.get("cache_creation_input_tokens", 0) - w5
    # Streamed usage under-reports output tokens.
    out = max(u.get("output_tokens", 0), math.ceil(chars / 2))
    return (u.get("input_tokens", 0) * p_in + out * p_out + u.get("cache_read_input_tokens", 0) * p_read
            + w5 * p_5m + w1 * p_1h) / 1e6


def result_cost(result: dict) -> Optional[float]:
    """Price this run's own result usage."""
    prices = [model_prices(m) for m in result.get("modelUsage") or {}]
    if not prices or None in prices or not result.get("usage"):
        return None
    return _price(max(prices, key=lambda x: x[1]), result["usage"])


def spend_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Exact, estimated, assumed and unpriced spend."""
    runs, restored = [], 0.0
    for log in meta.get("logs") or []:
        events = stream_events(Path(log))
        result = next((e for e in reversed(events) if e.get("type") == "result"), {})
        total = result.get("total_cost_usd") or 0.0
        if total > 0:
            # Resume restores earlier cost into total_cost_usd.
            own = result_cost(result)
            runs.append({"log": log, "cost_usd": own if own is not None else total - min(restored, total),
                         "estimated": False})
            restored = total
        else:
            runs.append({"log": log, "cost_usd": usage_cost(events), "estimated": True})
    assumed = meta.get("assumed") or []
    covered = {log for a in assumed for log in a.get("logs") or []}
    unpriced = [r["log"] for r in runs if r["cost_usd"] is None and r["log"] not in covered]
    priced = [r for r in runs if r["cost_usd"] is not None]
    estimated = sum(r["cost_usd"] for r in priced if r["estimated"])
    assumed_usd = sum(a.get("usd") or 0.0 for a in assumed)
    total = sum(r["cost_usd"] for r in priced) + assumed_usd
    return {"usd": round(total, 4), "estimated_usd": round(estimated, 4), "assumed_usd": assumed_usd,
            "unpriced": unpriced, "runs": runs}


def launch(ws: Workspace, popen=subprocess.Popen, resume: bool = False, note: Optional[str] = None,
           note_path: Optional[Path] = None, assume_spent: Optional[float] = None) -> dict[str, Any]:
    """Start or resume the agent detached."""
    campaign, _ = ws.load()
    meta = read_meta(ws)
    if pid_alive(meta.get("pid")):
        raise RuntimeError(f"Agent already running as pid {meta['pid']}")
    if resume and not meta.get("session_id"):
        raise RuntimeError("No session to resume; launch first")
    if not resume and (note is not None or assume_spent is not None):
        raise RuntimeError("--note and --assume-spent need --resume")
    left, assumed = campaign.cost_usd, []
    if resume:
        spend, assumed = spend_summary(meta), list(meta.get("assumed") or [])
        if assume_spent is not None:
            if not spend["unpriced"]:
                raise RuntimeError("Every run is priced; drop --assume-spent")
            assumed.append({"usd": assume_spent, "logs": spend["unpriced"]})
            spend = {**spend, "usd": spend["usd"] + assume_spent, "unpriced": []}
        if spend["unpriced"]:
            raise RuntimeError("Run cost unknown; pass --assume-spent USD")
        left -= spend["usd"]
    if left < 0.01:
        raise RuntimeError("Cost cap spent; raise cost_usd to resume")
    prompt = ws.prompt.read_text(encoding="utf-8")
    if resume:
        prompt = RESUME_PROMPT if note is None else note_prompt(note)
    ws.logs.mkdir(parents=True, exist_ok=True)
    session = meta["session_id"] if resume else str(uuid.uuid4())
    # Same locked-down flags either way.
    args = [*claude_args(ws, campaign.model, prompt), *(["--resume", session] if resume else ["--session-id", session]),
            "--max-budget-usd", f"{left:.2f}", "--name", f"agent-loop {campaign.name}"]
    run = len(meta.get("logs") or []) + 1 if resume else 1
    log = ws.logs / f"agent-run-{session[:8]}-{run}.jsonl"
    with log.open("wb") as out:
        # Own session; outlives this process.
        proc = popen(args, cwd=ws.agent, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                     start_new_session=True)
    logs = [*(meta.get("logs") or []), str(log)] if resume else [str(log)]
    meta = {"pid": proc.pid, "session_id": session, "model": campaign.model, "max_budget_usd": round(left, 2),
            "log": str(log), "logs": logs, "assumed": assumed}
    if note is not None:
        meta["note"] = {"path": str(note_path) if note_path else None, "chars": len(note.strip()),
                        "sha256": hashlib.sha256(note.encode("utf-8")).hexdigest()}
    ws.run_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while pid_alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)
    return True


def stop(ws: Workspace) -> str:
    """SIGINT, then SIGTERM, then SIGKILL."""
    pid = read_meta(ws).get("pid")
    if not pid_alive(pid):
        return "Agent is not running."
    pgid = os.getpgid(pid)
    for sig, wait in STOP_WAITS:
        # SIGINT lets claude record the turn.
        if sig == signal.SIGINT:
            os.kill(pid, sig)
        else:
            os.killpg(pgid, sig)
        if _wait_gone(pid, wait):
            if sig == signal.SIGINT:
                # End leftover tools, e.g. submit.
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return f"Agent pid {pid} stopped after {sig.name}."
    return f"Agent pid {pid} still alive after SIGKILL."


def stream_events(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def run_cost(events: list[dict]) -> dict[str, Any]:
    """Cost and turns from the result event."""
    results = [e for e in events if e.get("type") == "result"]
    if not results:
        return {"finished": False}
    last = results[-1]
    return {"finished": True, "subtype": last.get("subtype"), "cost_usd": last.get("total_cost_usd"),
            "turns": last.get("num_turns"), "denials": len(last.get("permission_denials") or [])}


def readable(events: list[dict]) -> list[str]:
    """Tool calls and agent text, one line each."""
    out = []
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text" and block["text"].strip():
                out.append(f"[agent] {block['text'].strip()[:300]}")
            elif block.get("type") == "tool_use":
                i = block.get("input") or {}
                out.append(f"[tool] {block.get('name')}: {i.get('command') or i.get('file_path') or i.get('pattern') or ''}")
    return out


def status(ws: Workspace, tail: int = 10) -> dict[str, Any]:
    """Ledger, process and cost."""
    campaign, _ = ws.load()
    ledger = Ledger(ws.ledger)
    meta = read_meta(ws)
    events = stream_events(Path(meta["log"])) if meta.get("log") else []
    spend = spend_summary(meta)
    rows = [{"eval": r["eval"], "verdict": r["verdict"], "charged": r.get("charged"),
             "geomean": {leg: v.get("geomean") for leg, v in (r.get("legs") or {}).items()},
             "size_delta": r.get("size_delta")} for r in ledger.rows()]
    return {"campaign": campaign.name, "evals_used": ledger.charged(), "evals": campaign.evals, "rows": rows,
            "pid": meta.get("pid"), "running": pid_alive(meta.get("pid")), "session_id": meta.get("session_id"),
            "cost": run_cost(events), "spend": spend, "spent_usd": round(spend["usd"], 2), "note": meta.get("note"),
            "recent": readable(events)[-tail:] if tail else []}


# --- selftest ---------------------------------------------------------------------------


def probes(ws: Workspace, campaign: Campaign, token: str) -> list[dict[str, Any]]:
    """Tool calls with the expected outcome."""
    hidden_summary = next(iter(sorted(campaign.secrets_dir.glob("hidden/**/summary.json"))), campaign.secrets_dir / "x")
    # Unique names: cleanup touches only these.
    scratch = ws.agent / "Source" / f".selftest-{token}"
    return [
        {"tool": "Read", "arg": str(ws.agent / "README.md"), "allow": True},
        {"tool": "Write", "arg": str(scratch), "allow": True},
        {"tool": "Bash", "arg": f"{ws.bin / 'disasm'} selftest_probe", "allow": True},
        {"tool": "Read", "arg": str(ws.ledger / "ledger.jsonl"), "allow": False},
        {"tool": "Read", "arg": str(ws.state), "allow": False},
        {"tool": "Read", "arg": str(ws.baseline(campaign.legs[0]) / "baseline.json"), "allow": False},
        {"tool": "Read", "arg": str(ws.tester / "README.md"), "allow": False},
        {"tool": "Read", "arg": str(hidden_summary), "allow": False},
        {"tool": "Write", "arg": str(ws.agent / "nsx" / f"selftest-{token}.txt"), "allow": False},
        {"tool": "Bash", "arg": f"cat {ws.state}", "allow": False},
        {"tool": "Bash", "arg": "curl -sI https://example.com", "allow": False},
        {"tool": "Bash", "arg": f"touch {ws.root / f'selftest-{token}'}", "allow": False},
        {"tool": "Bash", "arg": "git -C .. status", "allow": False},
        {"tool": "Bash", "arg": f"git diff --no-index {ws.state} /dev/null", "allow": False},
        {"tool": "Bash", "arg": "cat ../campaign.json", "allow": False},
    ]


def selftest_prompt(items: list[dict]) -> str:
    lines = ["This is a permission test. Make each tool call below exactly once, in order,",
             "one call per message. Do not retry, explain or do anything else.",
             "For Write calls, write the text `selftest`.", ""]
    lines += [f"{n}. {p['tool']}: {p['arg']}" for n, p in enumerate(items, 1)]
    lines += ["", "When done, reply DONE."]
    return "\n".join(lines)


def _call_arg(block: dict) -> str:
    i = block.get("input") or {}
    return str(i.get("command") or i.get("file_path") or "")


def judge_selftest(items: list[dict], events: list[dict]) -> list[dict[str, Any]]:
    """Match tool calls to probes; denied or ran."""
    denied = set()
    for ev in events:
        if ev.get("type") == "result":
            denied |= {d.get("tool_use_id") for d in ev.get("permission_denials") or []}
    errors = {}
    for ev in events:
        if ev.get("type") == "user":
            for block in ev.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    errors[block.get("tool_use_id")] = bool(block.get("is_error"))
    calls = [b for ev in events if ev.get("type") == "assistant"
             for b in ev.get("message", {}).get("content", []) if b.get("type") == "tool_use"]
    out = []
    for item in items:
        call = next((c for c in calls if c.get("name") == item["tool"] and _call_arg(c).strip() == item["arg"]), None)
        if call is None:
            outcome = "not_attempted"
        elif call.get("id") in denied:
            outcome = "denied"
        else:
            outcome = "error" if errors.get(call.get("id")) else "allowed"
        # Errors still mean it ran.
        ran = outcome in ("allowed", "error")
        ok = outcome != "not_attempted" and ran == item["allow"]
        out.append({"tool": item["tool"], "arg": item["arg"], "expect": "allow" if item["allow"] else "deny",
                    "outcome": outcome, "ok": ok})
    return out


def selftest(ws: Workspace, model: str = SELFTEST_MODEL, runner=subprocess.run) -> list[dict[str, Any]]:
    """Haiku permission probe; no session saved."""
    campaign, _ = ws.load()
    token = uuid.uuid4().hex[:12]
    items = probes(ws, campaign, token)
    args = [*claude_args(ws, model, selftest_prompt(items)), "--no-session-persistence", "--max-budget-usd", "1"]
    try:
        proc = runner(args, cwd=ws.agent, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    finally:
        # Undo allowed writes.
        for item in items:
            if item["tool"] == "Write":
                Path(item["arg"]).unlink(missing_ok=True)
        (ws.root / f"selftest-{token}").unlink(missing_ok=True)
    return judge_selftest(items, events)
