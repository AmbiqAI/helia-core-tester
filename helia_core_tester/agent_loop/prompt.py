"""Render the agent prompt from config and baseline facts."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import jinja2

from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.candidate_eval import read_baseline
from helia_core_tester.hardware.pmu_explain import classify_route, load_ceilings
from helia_core_tester.hardware.score import load_bundle

from .config import Campaign

TEMPLATE = Path(__file__).with_name("prompt.md.j2")
LEG_TEXT = {
    "tcm": "all operands in tightly coupled memory, no cache. Pure compute.",
    "mram": ("weights in MRAM behind the data cache, cache cold before each call. "
             "This is how models deploy; memory access order and tiling show here."),
}
TOOLCHAIN_TEXT = {"gcc": " Built with gcc.", "atfe": " Built with ATfE clang."}
# Spaced, lowercase op names.
OP_TEXT = {"DepthwiseConv": "depthwise convolution", "Convolve": "convolution", "FullyConnected": "fully connected"}


def _number(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _ceiling(symbol: str, cpu: str) -> Optional[dict]:
    op, dtype = classify_route(symbol)
    entry = ((load_ceilings()["cpus"].get(cpu) or {}).get("ops", {}).get(op) or {}).get(dtype)
    return dict(entry, key=f"{op} {dtype}") if entry else None


def route_facts(rows: list[dict], cpu: str) -> list[dict[str, Any]]:
    """Per route: cases, median and best cycles/MAC."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        if row.get("hidden") == "true":
            continue
        key = (row.get("timed_symbol") or "?", row.get("inner_symbol") or "")
        counts[key] += 1
        if (cpm := _number(row.get("cycles_per_mac"))) is not None:
            groups[key].append(cpm)
    out = []
    for (timed, inner), n in counts.items():
        cpms = groups[(timed, inner)]
        ceiling = _ceiling(inner or timed, cpu)
        out.append({
            "timed": timed, "inner": inner, "cases": n,
            "median": round(statistics.median(cpms), 2) if cpms else "n/a",
            "best": round(min(cpms), 2) if cpms else "n/a",
            "ceiling": ceiling["cycles_per_mac"] if ceiling else None,
            "basis": f"{ceiling['key']}: {ceiling.get('basis')}" if ceiling else None,
            "ceiling_text": f"{ceiling['cycles_per_mac']} ({ceiling['key']})" if ceiling else None,
            "weight": sum(cpms),
        })
    return sorted(out, key=lambda r: -r["weight"])


def baseline_rows(baseline: Path) -> list[dict]:
    """The first baseline run's case rows."""
    first = read_baseline(baseline)["sessions"][0]
    return list(load_bundle(baseline / "bundles" / first).rows.values())


def _patch_files(diff: bytes) -> list[str]:
    found = re.findall(rb"^\+\+\+ \S*?((?:Source|Include)/\S+)", diff, flags=re.M)
    return sorted({f.decode(errors="replace") for f in found})


def render_prompt(campaign: Campaign, rows: list[dict], paths: dict[str, Path],
                  start_diff: Optional[bytes] = None) -> str:
    """Fill the template; paths has submit, check, disasm, results."""
    board = resolve_board(campaign.board)
    routes = route_facts(rows, board.cpu)
    public = [r for r in rows if r.get("hidden") != "true"]
    cpms = [c for r in public if (c := _number(r.get("cycles_per_mac"))) is not None]
    ceilings = sorted({r["ceiling_text"] for r in routes if r["ceiling_text"]})
    cpu_text = board.cpu.replace("cortex-m", "Cortex-M")
    extra = ["Helium/MVE"] if board.has_mve else []
    if board.cpu == "cortex-m55":
        extra.append("dual-beat")
    many = len(campaign.toolchains) > 1
    start = None
    if start_diff is not None:
        start = {"files": _patch_files(start_diff), "lines": start_diff.count(b"\n"), "notes": campaign.start_notes}
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True, autoescape=False)
    template = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
    return template.render(
        base_ref=campaign.base_ref, op=campaign.op, dtype=campaign.dtype, dtype_lower=campaign.dtype.lower(),
        op_text=OP_TEXT.get(campaign.op, campaign.op), cpu=cpu_text,
        cpu_text=f"{cpu_text} (Ambiq {board.soc}{', ' + ', '.join(extra) if extra else ''})",
        public_cases=len(public), hidden_cases=len(rows) - len(public), routes=routes,
        median=round(statistics.median(cpms), 2) if cpms else None,
        ceilings=", ".join(ceilings), bases=sorted({r["basis"] for r in routes if r["basis"]}),
        is_depthwise=campaign.op == "DepthwiseConv", has_mve=board.has_mve,
        legs=[{"name": leg.name, "text": LEG_TEXT[leg.placement] + (TOOLCHAIN_TEXT[leg.toolchain] if many else "")}
              for leg in campaign.runs],
        first_leg=campaign.runs[0].name, toolchains=campaign.toolchains if many else (),
        evals=campaign.evals, start=start, **{k: str(v) for k, v in paths.items()},
    )
