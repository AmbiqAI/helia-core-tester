"""Agent side: settings, wrappers, launch, status, selftest."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
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


def launch(ws: Workspace, popen=subprocess.Popen) -> dict[str, Any]:
    """Start the agent detached; record its pid."""
    campaign, _ = ws.load()
    meta = read_meta(ws)
    if pid_alive(meta.get("pid")):
        raise RuntimeError(f"Agent already running as pid {meta['pid']}")
    ws.logs.mkdir(parents=True, exist_ok=True)
    session = str(uuid.uuid4())
    args = [*claude_args(ws, campaign.model, ws.prompt.read_text(encoding="utf-8")),
            "--session-id", session, "--max-budget-usd", f"{campaign.cost_usd:g}", "--name", f"agent-loop {campaign.name}"]
    with ws.run_jsonl.open("ab") as out:
        # Own session; outlives this process.
        proc = popen(args, cwd=ws.agent, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                     start_new_session=True)
    meta = {"pid": proc.pid, "session_id": session, "model": campaign.model, "max_budget_usd": campaign.cost_usd,
            "log": str(ws.run_jsonl), "resume": f"cd {ws.agent} && claude --resume {session}"}
    ws.run_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def stop(ws: Workspace) -> str:
    """Terminate the agent's process group."""
    pid = read_meta(ws).get("pid")
    if not pid_alive(pid):
        return "Agent is not running."
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    return f"Sent SIGTERM to agent pid {pid}."


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
    events = stream_events(ws.run_jsonl)
    rows = [{"eval": r["eval"], "verdict": r["verdict"], "charged": r.get("charged"),
             "geomean": {leg: v.get("geomean") for leg, v in (r.get("legs") or {}).items()},
             "size_delta": r.get("size_delta")} for r in ledger.rows()]
    return {"campaign": campaign.name, "evals_used": ledger.charged(), "evals": campaign.evals, "rows": rows,
            "pid": meta.get("pid"), "running": pid_alive(meta.get("pid")), "session_id": meta.get("session_id"),
            "cost": run_cost(events), "recent": readable(events)[-tail:] if tail else []}


# --- selftest ---------------------------------------------------------------------------


def probes(ws: Workspace, campaign: Campaign) -> list[dict[str, Any]]:
    """Tool calls with the expected outcome."""
    hidden_summary = next(iter(sorted(campaign.secrets_dir.glob("hidden/**/summary.json"))), campaign.secrets_dir / "x")
    scratch = ws.agent / "Source" / ".selftest"
    return [
        {"tool": "Read", "arg": str(ws.agent / "README.md"), "allow": True},
        {"tool": "Write", "arg": str(scratch), "allow": True},
        {"tool": "Bash", "arg": f"{ws.bin / 'disasm'} selftest_probe", "allow": True},
        {"tool": "Read", "arg": str(ws.ledger / "ledger.jsonl"), "allow": False},
        {"tool": "Read", "arg": str(ws.state), "allow": False},
        {"tool": "Read", "arg": str(ws.baseline(campaign.legs[0]) / "baseline.json"), "allow": False},
        {"tool": "Read", "arg": str(ws.tester / "README.md"), "allow": False},
        {"tool": "Read", "arg": str(hidden_summary), "allow": False},
        {"tool": "Write", "arg": str(ws.agent / "nsx" / "selftest.txt"), "allow": False},
        {"tool": "Bash", "arg": f"cat {ws.state}", "allow": False},
        {"tool": "Bash", "arg": "curl -sI https://example.com", "allow": False},
        {"tool": "Bash", "arg": f"touch {ws.root / 'selftest-touch'}", "allow": False},
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
    items = probes(ws, campaign)
    args = [*claude_args(ws, model, selftest_prompt(items)), "--no-session-persistence", "--max-budget-usd", "1"]
    try:
        proc = runner(args, cwd=ws.agent, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    finally:
        # Undo allowed writes.
        for item in items:
            if item["tool"] == "Write":
                Path(item["arg"]).unlink(missing_ok=True)
        (ws.root / "selftest-touch").unlink(missing_ok=True)
    return judge_selftest(items, events)
