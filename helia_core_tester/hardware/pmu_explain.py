"""Turn bundle PMU counters into short tuning feedback.

`explain_case` reads one case row (a cases.json entry or a case_summary.csv
row) and returns % of peak, where cycles go, a diagnosis and ranked hints.
Rules are the `_rule_*` functions below; each returns a Finding whose
`impact` estimates the share of cycles at stake, and findings rank by it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from .boards import repo_root

SCHEMA = "hct.pmu_explain"
SCHEMA_VERSION = 1

# Counters the agent loop should request.
AGENT_PMU_SELECTION: dict[str, list[str]] = {
    "cpu": ["ARM_PMU_INST_RETIRED", "ARM_PMU_STALL_FRONTEND", "ARM_PMU_STALL_BACKEND"],
    "memory": ["ARM_PMU_L1D_CACHE_REFILL"],
    "mve": [
        "ARM_PMU_MVE_INST_RETIRED", "ARM_PMU_MVE_INT_MAC_RETIRED", "ARM_PMU_MVE_FP_MAC_RETIRED",
        "ARM_PMU_MVE_PRED", "ARM_PMU_MVE_STALL_RESOURCE_MEM", "ARM_PMU_MVE_STALL_DEPENDENCY",
    ],
}

_WANTED = tuple(name for names in AGENT_PMU_SELECTION.values() for name in names)
_CEILINGS_PATH = repo_root() / "assets" / "scoring" / "ceilings.yaml"
_OPS = ("conv", "depthwise", "fc")
_DTYPE_RE = re.compile(r"_(s4|s8|s16|f16|f32)(?=_|$)")
# Cortex-M55 L1D line size.
_LINE_BYTES = 32

NEAR_PEAK = 0.6
BACKEND_BOUND = 0.3
LOW_IPC = 0.6
UNDERFILLED = 1.5
SCALAR = 0.5
LOW_MVE_SHARE = 0.3
DEP_STALL = 0.05
FRONTEND_STALL = 0.1
PREPARE_HEAVY = 0.5
OVERHEAD = 1.5


@lru_cache(maxsize=None)
def load_ceilings(path: Path = _CEILINGS_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("schema") != "hct.scoring.ceilings":
        raise ValueError(f"{path}: not a ceilings file")
    return data


def classify_route(symbol: str) -> tuple[Optional[str], Optional[str]]:
    """Map a kernel symbol to (op, dtype) ceiling keys."""
    if "depthwise" in symbol:
        op = "depthwise"
    elif "fully_connected" in symbol or "batch_matmul" in symbol:
        op = "fc"
    elif "conv" in symbol:
        op = "conv"
    else:
        op = None
    match = _DTYPE_RE.search(symbol)
    dtype = match.group(1) if match else None
    # s4 weights unpack to s8 MACs.
    return op, ("s8" if dtype == "s4" else dtype)


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    return num / den if num is not None and den else None


@dataclass
class Finding:
    rule: str
    impact: float
    diagnosis: str
    hint: str


@dataclass
class Explanation:
    case_id: str
    cpu: str
    placement: Optional[str]
    op: Optional[str]
    dtype: Optional[str]
    route: Optional[str]
    timing_status: Optional[str]
    cycles: Optional[float]
    macs: Optional[float]
    cycles_per_mac: Optional[float]
    ceiling: Optional[dict[str, Any]]
    pct_of_peak: Optional[float]
    metrics: dict[str, Optional[float]]
    missing_counters: list[str]
    findings: list[Finding] = field(default_factory=list)

    @property
    def diagnosis(self) -> str:
        return self.findings[0].diagnosis

    @property
    def hints(self) -> list[str]:
        return [finding.hint for finding in self.findings[:3]]

    def lines(self) -> list[str]:
        head = f"{self.case_id}: {self.op or '?'} {self.dtype or '?'} via {self.route or '?'}"
        if self.pct_of_peak is not None:
            head += (f", {self.cycles_per_mac:.3f} cyc/MAC vs {self.ceiling['cycles_per_mac']}"
                     f" = {self.pct_of_peak:.0%} of peak")
        elif self.cycles is not None:
            head += f", {self.cycles:.0f} cycles"
        if self.timing_status not in (None, "valid"):
            head += f" [timing {self.timing_status}]"
        hints = " ".join(f"{i}) {hint}" for i, hint in enumerate(self.hints, 1))
        lines = [head]
        if where := _where_line(self.metrics):
            lines.append(f"  cycles: {where}")
        return lines + [f"  diagnosis: {self.diagnosis}", f"  hints: {hints}"]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(diagnosis=self.diagnosis, hints=self.hints, lines=self.lines())
        return data


_WHERE_FORMAT = (
    ("ipc", "IPC {:.2f}"), ("mve_share", "MVE {:.0%} of inst"), ("mac_instr_ratio", "MVE MACs {:.2f}x ideal"),
    ("inst_per_mac_instr", "{:.1f} inst/MVE MAC"), ("stall_frontend", "FE stall {:.0%}"),
    ("stall_backend", "BE stall {:.0%}"), ("mve_stall_mem", "MVE mem stall {:.0%}"),
    ("mve_stall_dep", "MVE dep stall {:.0%}"), ("refill_kb", "L1D refill {:.1f} KB"),
    ("prepare_share", "prepare {:.0%}"),
)


def _where_line(metrics: Mapping[str, Optional[float]]) -> str:
    return ", ".join(fmt.format(metrics[key]) for key, fmt in _WHERE_FORMAT if metrics.get(key) is not None)


def _metrics(row: Mapping[str, Any], cycles: Optional[float], macs: Optional[float], lanes: int, dtype: Optional[str]):
    nested = row.get("counters")
    source = nested if isinstance(nested, Mapping) else row
    counter = lambda name: _number(source.get(f"ARM_PMU_{name}"))  # noqa: E731
    inst, mve_inst = counter("INST_RETIRED"), counter("MVE_INST_RETIRED")
    mve_mac = counter("MVE_FP_MAC_RETIRED" if dtype in ("f16", "f32") else "MVE_INT_MAC_RETIRED")
    refills = counter("L1D_CACHE_REFILL")
    metrics = {
        "ipc": _ratio(inst, cycles),
        "mve_share": _ratio(mve_inst, inst) if lanes else None,
        "mac_instr_ratio": _ratio(mve_mac, macs / lanes) if lanes and macs else None,
        "inst_per_mac_instr": _ratio(inst, mve_mac) if lanes else None,
        "stall_frontend": _ratio(counter("STALL_FRONTEND"), cycles),
        "stall_backend": _ratio(counter("STALL_BACKEND"), cycles),
        "mve_stall_mem": _ratio(counter("MVE_STALL_RESOURCE_MEM"), cycles),
        "mve_stall_dep": _ratio(counter("MVE_STALL_DEPENDENCY"), cycles),
        "pred_cycle_share": _ratio(counter("MVE_PRED"), cycles),
        "refill_kb": refills * _LINE_BYTES / 1024 if refills is not None else None,
        "prepare_share": _ratio(_number(row.get("prepare_cycles")), cycles),
    }
    return metrics, [name for name in _WANTED if _number(source.get(name)) is None]


Rule = Callable[[Explanation], Optional[Finding]]


def _rule_near_peak(e: Explanation) -> Optional[Finding]:
    pct = e.pct_of_peak
    if pct is None or pct < NEAR_PEAK:
        return None
    return Finding("near_peak", pct, f"Near ceiling at {pct:.0%} of peak",
                   "Only fewer MACs or fused work helps now")


def _rule_memory_bound(e: Explanation) -> Optional[Finding]:
    m = e.metrics
    be, ipc = m["stall_backend"], m["ipc"]
    if be is None or ipc is None or be < BACKEND_BOUND or ipc >= LOW_IPC:
        return None
    if e.placement == "mram" and m["refill_kb"]:
        return Finding("memory_bound", be, f"MRAM-bound: {be:.0%} backend stall, {m['refill_kb']:.0f} KB refilled",
                       "Prefetch weights or stage them in TCM")
    return Finding("memory_bound", be, f"Memory-bound: {be:.0%} of cycles in backend stall",
                   "Interleave loads with MACs; keep operands in DTCM")


def _rule_overhead(e: Explanation) -> Optional[Finding]:
    ratio = e.metrics["inst_per_mac_instr"]
    target = e.ceiling.get("target_inst_per_mac_instr") if e.ceiling else None
    fill = e.metrics["mac_instr_ratio"]
    # Scalar MACs explain this better.
    if ratio is None or not target or ratio < target * OVERHEAD or (fill is not None and fill < SCALAR):
        return None
    scalar = 1 - (e.metrics["mve_share"] or 0)
    return Finding("instruction_overhead", 1 - target / ratio,
                   f"{ratio:.1f} inst per MVE MAC (best {target}), {scalar:.0%} scalar",
                   "Cut non-MAC work: hoist address math, reuse loaded vectors")


def _rule_underfilled(e: Explanation) -> Optional[Finding]:
    ratio = e.metrics["mac_instr_ratio"]
    if ratio is None or ratio < UNDERFILLED:
        return None
    lanes = e.ceiling["lanes"]
    pred = e.metrics["pred_cycle_share"] or 0
    return Finding("underfilled_vectors", 1 - 1 / ratio,
                   f"MVE MACs fill {1 / ratio:.0%} of {lanes} lanes, {pred:.0%} cycles predicated",
                   "Block channels so each MAC fills all lanes")


def _rule_scalar_macs(e: Explanation) -> Optional[Finding]:
    ratio = e.metrics["mac_instr_ratio"]
    if ratio is None or ratio >= SCALAR:
        return None
    return Finding("scalar_macs", 1 - ratio, f"Only {ratio:.0%} of MACs use MVE MAC instructions",
                   "Vectorise the inner MAC loop or check route")


def _rule_low_mve_share(e: Explanation) -> Optional[Finding]:
    share = e.metrics["mve_share"]
    if share is None or share >= LOW_MVE_SHARE:
        return None
    return Finding("low_mve_share", 1 - share, f"Scalar code is {1 - share:.0%} of instructions",
                   "Trim tail, edge and setup code around MVE loops")


def _rule_dependency(e: Explanation) -> Optional[Finding]:
    dep = e.metrics["mve_stall_dep"]
    if dep is None or dep < DEP_STALL:
        return None
    return Finding("dependency_stall", dep, f"MVE dependency stalls take {dep:.0%} of cycles",
                   "Use more independent accumulators; interleave MAC chains")


def _rule_frontend(e: Explanation) -> Optional[Finding]:
    fe = e.metrics["stall_frontend"]
    if fe is None or fe < FRONTEND_STALL:
        return None
    return Finding("frontend_stall", fe, f"Front-end stalls take {fe:.0%} of cycles",
                   "Unroll hot loops; cut branches and mispredicts")


def _rule_prepare(e: Explanation) -> Optional[Finding]:
    share = e.metrics["prepare_share"]
    if share is None or share < PREPARE_HEAVY:
        return None
    return Finding("prepare_heavy", share / (1 + share), f"Prepare costs {share:.0%} of kernel cycles",
                   "Move setup out of the per-call path")


RULES: tuple[Rule, ...] = (
    _rule_near_peak, _rule_memory_bound, _rule_overhead, _rule_underfilled, _rule_scalar_macs,
    _rule_low_mve_share, _rule_dependency, _rule_frontend, _rule_prepare,
)


def explain_case(
    row: Mapping[str, Any], *, cpu: str, placement: Optional[str] = None, ceilings: Optional[Mapping[str, Any]] = None
) -> Explanation:
    """Explain one case row for `cpu` (e.g. cortex-m55)."""
    cpu_table = (ceilings or load_ceilings())["cpus"].get(cpu, {})
    route = row.get("inner_symbol") or row.get("timed_symbol") or None
    op, dtype = classify_route(route or "")
    cycles, macs = _number(row.get("median_cycles")), _number(row.get("macs"))
    entry = cpu_table.get("ops", {}).get(op, {}).get(dtype)
    ceiling = None
    if entry:
        ceiling = dict(entry, key=f"{cpu}/{op}/{dtype}",
                       target_inst_per_mac_instr=cpu_table.get("target_inst_per_mac_instr"))
    cpm = _number(row.get("cycles_per_mac")) or _ratio(cycles, macs)
    lanes = int(ceiling["lanes"]) if ceiling else 0
    metrics, missing = _metrics(row, cycles, macs, lanes, dtype)
    result = Explanation(
        case_id=str(row.get("case_id")), cpu=cpu, placement=placement, op=op, dtype=dtype, route=route,
        timing_status=row.get("timing_status"), cycles=cycles, macs=macs, cycles_per_mac=cpm, ceiling=ceiling,
        pct_of_peak=_ratio(ceiling["cycles_per_mac"], cpm) if ceiling and cpm else None,
        metrics=metrics, missing_counters=missing,
    )
    if result.timing_status not in (None, "valid"):
        result.findings = [Finding("timing_invalid", 0.0, f"Timing {result.timing_status}: no diagnosis",
                                   "Fix the case before tuning it")]
        return result
    if len(missing) == len(_WANTED):
        result.findings = [Finding("no_counters", 0.0, "Cycles only: no PMU counters in bundle",
                                   "Run on a Cortex-M55 board for counters")]
        return result
    findings = [finding for rule in RULES if (finding := rule(result))]
    findings.sort(key=lambda finding: finding.impact, reverse=True)
    result.findings = findings or [Finding("no_bottleneck", 0.0, "No dominant bottleneck in counters",
                                           "Compare the inner loop with the best route")]
    return result


def _selected(row: Mapping[str, Any], cases: tuple[str, ...], ops: tuple[str, ...], all_cases: bool) -> bool:
    if cases and not any(c.lower() in str(row.get("case_id", "")).lower() for c in cases):
        return False
    route = row.get("inner_symbol") or row.get("timed_symbol") or ""
    text = f"{row.get('case_id')} {row.get('timed_symbol')} {route}".lower()
    op = classify_route(route)[0]
    # Op names match exactly, else substring.
    if ops and not any(o == op if o in _OPS else o.lower() in text for o in ops):
        return False
    return bool(cases) or all_cases or bool(_number(row.get("macs")))


def explain_bundle(
    bundle: Path, cases: tuple[str, ...] = (), ops: tuple[str, ...] = (), all_cases: bool = False
) -> dict[str, Any]:
    """Explain the selected cases of one bundle."""
    target = json.loads((bundle / "session_manifest.json").read_text(encoding="utf-8"))["target"]
    placement = (target.get("placement") or {}).get("name")
    rows = json.loads((bundle / "cases.json").read_text(encoding="utf-8"))
    return {
        "bundle": str(bundle), "board": target.get("board"), "cpu": target["cpu"], "placement": placement,
        "cases": [explain_case(row, cpu=target["cpu"], placement=placement)
                  for row in rows if _selected(row, cases, ops, all_cases)],
    }
