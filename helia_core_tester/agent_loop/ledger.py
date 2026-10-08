"""Eval ledger, verdict merge and the agent's view."""

from __future__ import annotations

import fcntl
import hashlib
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from helia_core_tester.hardware.candidate_eval import EXIT_ERROR, VERDICT_EXITS

from .config import Leg

# Worst first; unknown counts as error.
ORDER = ("error", "rejected", "refused", "not_comparable", "fail", "no_gain", "pass")
EXIT_BUDGET = 6
# Stages the agent cannot cause.
INFRA_STAGES = ("tester", "baseline")
CASE_COLUMNS = ["case_id", "baseline_cycles", "candidate_cycles", "speedup", "band_pct", "cycles_per_mac"]
TOP_HINTS = 10


class LockBusy(Exception):
    """The lock stayed held past the deadline."""


LOCK_POLL_S = 0.5


@contextmanager
def file_lock(path: Path, deadline: Optional[float] = None) -> Iterator[None]:
    """Hold an exclusive flock on path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if deadline is not None else 0))
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LockBusy(str(path)) from None
                time.sleep(LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Ledger:
    """ledger/: counter, rows and per-eval files."""

    def __init__(self, root: Path):
        self.root = root
        self.rows_path = root / "ledger.jsonl"
        self.counter = root / "next_id"

    def next_id(self) -> str:
        """Take the next id; call under the lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            n = int(self.counter.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            n = 1
        self.counter.write_text(f"{n + 1}\n", encoding="utf-8")
        return f"{n:03d}"

    def rows(self) -> list[dict]:
        try:
            lines = self.rows_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    def append(self, row: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def charged(self) -> int:
        return sum(1 for row in self.rows() if row.get("charged"))

    def infra_streak(self) -> int:
        """Infra rows since the last other row."""
        streak = 0
        for row in reversed(self.rows()):
            if not row.get("infra"):
                break
            streak += 1
        return streak


def is_infra(verdict: Optional[dict]) -> bool:
    """No verdict, or a setup refusal."""
    if not isinstance(verdict, dict) or not verdict.get("verdict"):
        return True
    return verdict["verdict"] == "refused" and verdict.get("stage") in INFRA_STAGES


def merge_legs(legs: dict[str, Optional[dict]], wanted: tuple[str, ...]) -> str:
    """Worst leg wins; pass needs every leg."""
    verdicts = [v.get("verdict") if v.get("verdict") in ORDER else "error" for v in legs.values() if v is not None]
    if not verdicts:
        return "error"
    overall = min(verdicts, key=ORDER.index)
    if overall == "pass" and any(legs.get(leg) is None for leg in wanted):
        return "error"
    return overall


def scored(verdict: Optional[dict]) -> bool:
    """The leg got as far as scoring."""
    return isinstance(verdict, dict) and verdict.get("stage") == "score"


def leg_view(v: dict, hints: bool) -> dict[str, Any]:
    """One leg, compact: touched rows only."""
    keep = ("verdict", "stage", "reason", "findings", "failures", "score", "hidden")
    out = {k: v.get(k) for k in keep if v.get(k) is not None}
    fam_keys = ("cases", "geomean_speedup", "regression", "untouched_cases", "untouched_geomean")
    out["families"] = {f: {k: d.get(k) for k in fam_keys} for f, d in (v.get("families") or {}).items()}
    cases = v.get("cases") or []
    touched = [c for c in cases if c.get("touched") is not False]
    out["cases"] = [[c.get("case_id"), c.get("baseline_cycles"), c.get("candidate_cycles"),
                     round(c["speedup"], 4) if c.get("speedup") else None, c.get("band_pct"),
                     c.get("cycles_per_mac_candidate")] for c in touched]
    rest = [c["speedup"] for c in cases if c.get("touched") is False and c.get("speedup")]
    out["untouched_cases"] = {"count": len(rest), "speedup_min": round(min(rest), 4) if rest else None,
                              "speedup_max": round(max(rest), 4) if rest else None}
    if hints:
        # Ten slowest touched cases.
        cycles = {c.get("case_id"): c.get("candidate_cycles") or 0 for c in touched}
        top = sorted((h for h in v.get("hints") or [] if h.get("case_id") in cycles),
                     key=lambda h: -cycles[h["case_id"]])
        out["hints"] = top[:TOP_HINTS]
    return out


def family_geomeans(v: Optional[dict]) -> dict[str, Any]:
    return {f: d.get("geomean_speedup") for f, d in ((v or {}).get("families") or {}).items()}


def diff_digest(diff: bytes) -> dict[str, Any]:
    return {"diff_sha": hashlib.sha256(diff).hexdigest()[:12], "diff_lines": diff.count(b"\n")}


def toolchain_names(runs: tuple[Leg, ...]) -> list[str]:
    return list(dict.fromkeys(leg.toolchain for leg in runs))


def size_deltas(size: dict, runs: tuple[Leg, ...]) -> Any:
    """Delta bytes; keyed when several toolchains."""
    if len(toolchain_names(runs)) < 2:
        return size.get("delta_bytes")
    return {name: (size.get(name) or {}).get("delta_bytes") for name in toolchain_names(runs)}


def toolchain_gains(legs: dict, runs: tuple[Leg, ...], size: dict) -> Optional[dict[str, Any]]:
    """Geomean and code bytes per toolchain."""
    if len(toolchain_names(runs)) < 2:
        return None
    deltas, out = size_deltas(size, runs), {}
    for name in toolchain_names(runs):
        means = [g for leg in runs if leg.toolchain == name for g in family_geomeans(legs.get(leg.name)).values() if g]
        out[name] = {"geomean": round(statistics.geometric_mean(means), 4) if means else None,
                     "size_delta": deltas[name]}
    return out


def ledger_row(eid: str, overall: str, legs: dict, *, charged: bool, infra: bool, size: dict, diff: bytes,
               attempts: dict[str, int], runs: tuple[Leg, ...] = ()) -> dict[str, Any]:
    """One JSONL row per submit."""
    row = {
        "eval": eid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdict": overall, "charged": charged,
        "infra": infra, "attempts": attempts,
        "legs": {k: {"verdict": v.get("verdict"), "stage": v.get("stage"), "score": v.get("score"),
                     "geomean": family_geomeans(v), "hidden": v.get("hidden")} for k, v in legs.items() if v},
        "size_delta": size_deltas(size, runs), **diff_digest(diff),
    }
    gains = toolchain_gains(legs, runs, size)
    if gains:
        row["toolchains"] = gains
    return row


def row_gains(row: dict, runs: tuple[Leg, ...]) -> dict[str, dict]:
    """Geomean and bytes per toolchain."""
    if row.get("toolchains"):
        return row["toolchains"]
    means = [g for v in (row.get("legs") or {}).values() for g in (v.get("geomean") or {}).values() if g]
    return {toolchain_names(runs)[0]: {"geomean": round(statistics.geometric_mean(means), 4) if means else None,
                                       "size_delta": row.get("size_delta")}}


def _known(pick: dict) -> bool:
    return all(g.get("geomean") is not None and isinstance(g.get("size_delta"), int)
               for g in pick["toolchains"].values())


def _speed(pick: dict) -> float:
    return statistics.geometric_mean([g["geomean"] for g in pick["toolchains"].values()])


def _bytes(pick: dict) -> int:
    return sum(g["size_delta"] for g in pick["toolchains"].values())


def _dominates(a: dict, b: dict) -> bool:
    """a is no worse anywhere, better somewhere."""
    pairs = [(a["toolchains"][t], b["toolchains"][t]) for t in b["toolchains"]]
    if not all(x["geomean"] >= y["geomean"] and x["size_delta"] <= y["size_delta"] for x, y in pairs):
        return False
    return any(x["geomean"] > y["geomean"] or x["size_delta"] < y["size_delta"] for x, y in pairs)


def passing_evals(rows: list[dict], runs: tuple[Leg, ...]) -> list[dict[str, Any]]:
    """Passing evals marked fastest, smallest, pareto."""
    picks = [{"eval": r["eval"], "toolchains": row_gains(r, runs)} for r in rows if r.get("verdict") == "pass"]
    known = [p for p in picks if _known(p)]
    # Ties go to the earlier eval.
    fastest = max(known, key=_speed, default=None)
    smallest = min(known, key=_bytes, default=None)
    for pick in picks:
        pick["fastest"], pick["smallest"] = pick is fastest, pick is smallest
        pick["pareto"] = pick in known and not any(_dominates(q, pick) for q in known)
    return picks


def pick_text(pick: dict) -> str:
    """+3,088 B gcc / +2,154 B atfe at 2.78x / 3.42x"""
    gains = pick["toolchains"]
    sizes = " / ".join(f"{g['size_delta']:+,} B {name}" for name, g in gains.items())
    speeds = " / ".join(f"{g['geomean']:.2f}x" for g in gains.values())
    return f"{sizes} at {speeds}"


def size_note(picks: list[dict]) -> Optional[str]:
    """Point a pass at the size phase."""
    fastest = next((p for p in picks if p["fastest"]), None)
    smallest = next((p for p in picks if p["smallest"]), None)
    if fastest is None or smallest is None:
        return None
    if fastest is smallest:
        return f"next: shrink code; best pass {fastest['eval']} {pick_text(fastest)}"
    return (f"next: shrink code; fastest pass {fastest['eval']} {pick_text(fastest)}; "
            f"smallest pass {smallest['eval']} {pick_text(smallest)}")


def agent_view(overall: str, legs: dict, *, evals_left: int, size: dict, runs: tuple[Leg, ...],
               note: Optional[str] = None, next_step: Optional[str] = None) -> dict[str, Any]:
    """What submit prints for the agent."""
    out: dict[str, Any] = {"verdict": overall, "exit_code": VERDICT_EXITS.get(overall, EXIT_ERROR), "evals_left": evals_left}
    if note:
        out["note"] = note
    if next_step:
        out["next"] = next_step
    out["code_size"] = size
    gains = toolchain_gains(legs, runs, size)
    if gains:
        out["toolchains"] = gains
    out["cases_columns"] = CASE_COLUMNS
    out["legs"] = {k: leg_view(v, hints=(k == runs[0].name)) for k, v in legs.items() if v}
    return out
