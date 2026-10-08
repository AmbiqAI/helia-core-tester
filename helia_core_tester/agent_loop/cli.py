"""`helia_core_tester agent-loop` commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import ConfigError, load_campaign
from .workspace import Workspace

agent_loop_app = typer.Typer(help="Run a kernel optimization agent campaign.", no_args_is_help=True)

WS_OPT = typer.Option(..., "--workspace", "-w", file_okay=False, resolve_path=True, help="Campaign workspace dir.")


def _ws(path: Path) -> Workspace:
    """A workspace whose init finished."""
    ws = Workspace(path)
    try:
        _, facts = ws.load()
    except FileNotFoundError:
        facts = {}
    if not facts.get("ready"):
        typer.echo(f"✗ Init of {path} is not done; run agent-loop init.", err=True)
        raise typer.Exit(2)
    return ws


@agent_loop_app.command("init")
def init_command(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True, help="Campaign YAML."),
    workspace: Path = WS_OPT,
) -> None:
    """Pin the tester, clone trees, record baselines, write agent files."""
    from .setup import InitError, init_workspace

    try:
        campaign = load_campaign(config)
        facts = init_workspace(Workspace(workspace), campaign, echo=typer.echo)
    except (ConfigError, InitError, ValueError) as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(1)
    ws = Workspace(workspace)
    typer.echo(f"✓ Campaign {campaign.name} ready at {workspace}")
    typer.echo(f"  tester {facts['tester_commit'][:12]}, base {facts['base_commit'][:12]}")
    typer.echo(f"  next: helia_core_tester agent-loop selftest -w {ws.root}")
    typer.echo(f"        helia_core_tester agent-loop launch -w {ws.root}")


@agent_loop_app.command("validate")
def validate_command(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True, help="Campaign YAML."),
) -> None:
    """Check a campaign YAML without side effects."""
    try:
        campaign = load_campaign(config)
    except ConfigError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(campaign.to_json(), indent=2))


@agent_loop_app.command("submit")
def submit_command(workspace: Path = WS_OPT) -> None:
    """Judge the agent tree on the board (agent wrapper)."""
    from .judge import submit

    raise typer.Exit(submit(_ws(workspace)))


@agent_loop_app.command("check")
def check_command(workspace: Path = WS_OPT) -> None:
    """Rules, build and size, no board (agent wrapper)."""
    from .judge import check

    raise typer.Exit(check(_ws(workspace)))


@agent_loop_app.command("disasm")
def disasm_command(
    function: str = typer.Argument("", help="Kernel function name."),
    workspace: Path = WS_OPT,
    toolchain: str = typer.Option("", "--toolchain", help="gcc or atfe build (default: first)."),
) -> None:
    """Disassemble one function from the check build (agent wrapper)."""
    from .judge import disasm

    raise typer.Exit(disasm(_ws(workspace), function, toolchain))


@agent_loop_app.command("launch")
def launch_command(
    workspace: Path = WS_OPT,
    resume: bool = typer.Option(False, "--resume", help="Continue the saved session; cap the remaining cost."),
) -> None:
    """Start the agent detached, with the cost cap."""
    from .agent import launch

    try:
        meta = launch(_ws(workspace), resume=resume)
    except RuntimeError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(meta, indent=2))


@agent_loop_app.command("status")
def status_command(
    workspace: Path = WS_OPT,
    tail: int = typer.Option(10, "--tail", min=0, help="Recent agent lines to show."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Ledger, agent pid and cost so far."""
    from .agent import status

    info = status(_ws(workspace), tail)
    if as_json:
        typer.echo(json.dumps(info, indent=2))
        return
    state = "running" if info["running"] else "not running"
    typer.echo(f"{info['campaign']}: {info['evals_used']}/{info['evals']} evals, agent {state} (pid {info['pid']})")
    cost = info["cost"]
    if cost["finished"]:
        typer.echo(f"finished {cost['subtype']}: ${cost['cost_usd']}, {cost['turns']} turns, {cost['denials']} denials")
    typer.echo(f"cost of finished runs: ${info['spent_usd']}")
    for row in info["rows"]:
        charged = "" if row["charged"] else " (free)"
        means = ", ".join(f"{leg} {fam} {g:.3f}" for leg, fams in row["geomean"].items()
                          for fam, g in fams.items() if g is not None)
        typer.echo(f"  {row['eval']} {row['verdict']:<14}{charged} size {row['size_delta']} {means}")
        for name, gain in (row.get("toolchains") or {}).items():
            typer.echo(f"      {name}: geomean {gain['geomean']}, size {gain['size_delta']}")
    for line in info["recent"]:
        typer.echo(f"  {line}")


@agent_loop_app.command("stop")
def stop_command(workspace: Path = WS_OPT) -> None:
    """Stop the running agent."""
    from .agent import stop

    typer.echo(stop(_ws(workspace)))


@agent_loop_app.command("selftest")
def selftest_command(
    workspace: Path = WS_OPT,
    model: str = typer.Option("haiku", "--model", help="Cheap model for the probe."),
) -> None:
    """Probe the permission rules with a cheap model."""
    from .agent import selftest

    results = selftest(_ws(workspace), model)
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        typer.echo(f"{mark} {r['tool']:<5} expect {r['expect']:<5} got {r['outcome']:<13} {r['arg']}")
    raise typer.Exit(0 if all(r["ok"] for r in results) else 1)
