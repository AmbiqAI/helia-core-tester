"""Score a kernel candidate against a baseline on hardware.

Each side takes one or more result bundles (repeat runs); per-case
cycles pool as the median of per-run medians. Bundles from another
board, placement or harness are refused. The candidate fails on a failed
comparison, a dropped case, an input digest change, or a timed case that
is slower than its noise band: max(board floor, mad_k * 1.4826 * MAD /
baseline median), where MAD is the larger of the in-run MAD and the
spread of repeat medians. Only timing_status "valid" cases are timed.
The score is sum(weight * ln(family geomean speedup)) with weights from
assets/scoring/family_weights.yaml; with weights that sum to 1 it
approximates ln(whole-model speedup). JSON schema: see `score_bundles`.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from .nsx_app import CMSIS_NN_MODULE

SCHEMA = "hct.score"
SCHEMA_VERSION = 1
SCORING_DIR = Path(__file__).resolve().parents[2] / "assets" / "scoring"
MAD_SIGMA = 1.4826
EXIT_PASS, EXIT_FAIL, EXIT_REFUSED = 0, 1, 3
# INST_RETIRED plus every MVE retired counter.
_RETIRED = re.compile(r"^ARM_PMU_(INST|MVE_\w+)_RETIRED$")


@dataclass(frozen=True)
class Bundle:
    path: Path
    manifest: dict
    rows: dict[str, dict[str, str]]
    digests: dict[str, str]
    symbols: dict[str, str]

    @property
    def session_id(self) -> str:
        return str(self.manifest.get("session_id") or self.path.name)

    def symbol(self, case_id: str) -> str:
        row = self.rows[case_id]
        return row.get("timed_symbol") or self.symbols.get(row.get("kernel_id", ""), "")


def _find_root(path: Path) -> Path:
    """The bundle dir at or under `path`."""
    if (path / "case_summary.csv").is_file():
        return path
    found = sorted(path.rglob("case_summary.csv"))
    if len(found) != 1:
        raise typer.BadParameter(f"{path}: expected one bundle, found {len(found)}")
    return found[0].parent


def load_bundle(path: Path) -> Bundle:
    root = _find_root(path)
    with (root / "case_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    manifest_path = root / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    cases_path, catalog_path = root / "cases.json", root / "kernel_catalog.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8")) if cases_path.exists() else []
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []
    digests = {case["case_id"]: case["input_digest"] for case in cases if case.get("input_digest")}
    symbols = {str(entry["kernel_id"]): str(entry.get("canonical_name", "")) for entry in catalog}
    return Bundle(root, manifest, rows, digests, symbols)


def harness_print(manifest: dict) -> dict:
    """Everything in the build except kernels."""
    if manifest.get("harness_digest"):
        return {"harness_digest": manifest["harness_digest"]}
    build = manifest.get("build") or {}
    return {
        "options": {k: v for k, v in (build.get("options") or {}).items() if not k.startswith("cmsis_nn_") and k != "placement"},
        "modules": [m for m in build.get("modules") or [] if m.get("name") != CMSIS_NN_MODULE],
        "neuralspotx_version": build.get("neuralspotx_version"),
        "toolchain": build.get("toolchain"),
        "core_clock_hz": (manifest.get("boot") or {}).get("core_clock_hz"),
        "strict_compare": (manifest.get("compare") or {}).get("strict"),
    }


def _identity(bundle: Bundle) -> dict:
    target = bundle.manifest.get("target") or {}
    return {
        "board": target.get("board"),
        "placement": (target.get("placement") or {}).get("name"),
        "harness": harness_print(bundle.manifest),
    }


def refusals(baselines: list[Bundle], candidates: list[Bundle]) -> list[str]:
    """Why these bundles cannot be compared."""
    reasons = []
    first = _identity(baselines[0])
    for bundle in baselines[1:] + candidates:
        other = _identity(bundle)
        for key in ("board", "placement"):
            if other[key] != first[key]:
                reasons.append(f"{bundle.session_id}: {key} {other[key]} != {first[key]}")
        changed = sorted(k for k in first["harness"].keys() | other["harness"].keys() if first["harness"].get(k) != other["harness"].get(k))
        if changed:
            reasons.append(f"{bundle.session_id}: harness differs in {', '.join(changed)}")
    for side, bundles in (("baseline", baselines), ("candidate", candidates)):
        trees = {((b.manifest.get("build") or {}).get("kernels") or {}).get("tree_hash") for b in bundles}
        if len(trees) > 1:
            reasons.append(f"{side} repeats built different kernels")
    return reasons


def _num(row: dict, key: str) -> float | None:
    cell = row.get(key)
    return None if cell in ("", None) else float(cell)


def _status(row: dict) -> str:
    """timing_status; legacy bundles infer it."""
    if row.get("timing_status"):
        return row["timing_status"]
    return "valid" if row.get("valid_for_regression") == "true" else "legacy_invalid"


def _pool(rows: list[dict], key: str) -> float | None:
    values = [v for v in (_num(row, key) for row in rows) if v is not None]
    return statistics.median(values) if len(values) == len(rows) else None


def _spread(rows: list[dict]) -> float:
    """Pooled MAD in cycles, scaled to sigma."""
    medians = [_num(row, "median_cycles") or 0.0 for row in rows]
    center = statistics.median(medians)
    repeat_mad = statistics.median(abs(m - center) for m in medians)
    run_mad = statistics.median(_num(row, "mad_cycles") or 0.0 for row in rows)
    return MAD_SIGMA * max(repeat_mad, run_mad)


def load_scoring(board: str | None, directory: Path = SCORING_DIR) -> dict:
    """Weights, family rules and noise floor for `board`."""
    weights = yaml.safe_load((directory / "family_weights.yaml").read_text(encoding="utf-8"))
    floors = yaml.safe_load((directory / "noise_floors.yaml").read_text(encoding="utf-8"))
    weights_board = board if board in weights["boards"] else weights["fallback_board"]
    floor_row = floors["boards"].get(board) or {}
    return {
        "weights_version": weights["version"],
        "weights_board": weights_board,
        "weights": dict(weights["boards"][weights_board]["weights"]),
        "families": weights["families"],
        "default_family": weights["default_family"],
        "floor_pct": float(floor_row.get("floor_pct", floors["default_floor_pct"])),
        "mad_k": float(floors["mad_k"]),
    }


def family_of(symbol: str, scoring: dict) -> str:
    return next((rule["name"] for rule in scoring["families"] if rule["match"] in symbol), scoring["default_family"])


def _retired(base: list[dict], cand: list[dict]) -> dict:
    out = {}
    for name in base[0]:
        if _RETIRED.match(name) and name in cand[0]:
            a, b = _pool(base, name), _pool(cand, name)
            if a is not None and b is not None:
                out[name] = {"baseline": a, "candidate": b, "delta": b - a}
    return out


def _per_mac(cycles: float | None, row: dict) -> float | None:
    macs = _num(row, "macs")
    return round(cycles / macs, 4) if cycles and macs else None


def _case(case_id: str, baselines: list[Bundle], candidates: list[Bundle], scoring: dict) -> dict:
    base = [b.rows[case_id] for b in baselines if case_id in b.rows]
    cand = [c.rows[case_id] for c in candidates]
    statuses = sorted({_status(row) for row in base + cand} - {"valid"})
    if any(row.get("comparison_passed") != "true" for row in base + cand):
        statuses.insert(0, "comparison_failed")
    symbol = baselines[0].symbol(case_id) if case_id in baselines[0].rows else candidates[0].symbol(case_id)
    a, b = _pool(base, "median_cycles"), _pool(cand, "median_cycles")
    case: dict[str, Any] = {
        "case_id": case_id,
        "family": family_of(symbol, scoring),
        "timed_symbol": symbol,
        "inner_symbol": cand[0].get("inner_symbol") or None,
        "eligible": not statuses and bool(a) and bool(b),
        "excluded_by": statuses[0] if statuses else None,
        "baseline_cycles": a,
        "candidate_cycles": b,
        "cycles_per_mac_baseline": _per_mac(a, base[0]),
        "cycles_per_mac_candidate": _per_mac(b, cand[0]),
        "retired": _retired(base, cand),
    }
    if case["eligible"]:
        band = max(scoring["floor_pct"], scoring["mad_k"] * max(_spread(base), _spread(cand)) / a * 100.0)
        delta = (b - a) / a * 100.0
        case.update(speedup=a / b, delta_pct=delta, band_pct=band, within_noise=abs(delta) <= band, regression=delta > band)
    elif case["excluded_by"] is None:
        case["excluded_by"] = "zero_cycles"
    return case


def score_bundles(baselines: list[Bundle], candidates: list[Bundle], scoring: dict) -> dict:
    """The score report; `schema_version` bumps on breaking change.

    Keys: schema, schema_version, verdict (pass | fail | not_comparable),
    score (null unless comparable), board, placement, baseline and
    candidate session ids, settings, families {name: {weight, cases,
    geomean_speedup, contribution}}, cases (eligible and excluded rows),
    failures [{kind, case_id, reason}]. Failure kinds: not_comparable,
    comparison_failed, missing_case, input_digest, regression,
    no_eligible_cases.
    """
    identity = _identity(baselines[0])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "verdict": "pass",
        "score": None,
        "board": identity["board"],
        "placement": identity["placement"],
        "baseline": [b.session_id for b in baselines],
        "candidate": [c.session_id for c in candidates],
        "settings": {k: scoring[k] for k in ("weights_version", "weights_board", "weights", "floor_pct", "mad_k")},
        "families": {},
        "cases": [],
        "failures": [],
    }
    failures = report["failures"]
    refused = refusals(baselines, candidates)
    if refused:
        report["verdict"] = "not_comparable"
        failures.extend({"kind": "not_comparable", "case_id": None, "reason": reason} for reason in refused)
        return report

    wanted = list(dict.fromkeys(case_id for b in baselines for case_id in b.rows))
    for c in candidates:
        for case_id, row in c.rows.items():
            if row.get("comparison_passed") != "true":
                failures.append({"kind": "comparison_failed", "case_id": case_id, "reason": f"{c.session_id}: output mismatch"})
    present = []
    for case_id in wanted:
        missing = [c.session_id for c in candidates if case_id not in c.rows]
        if missing:
            failures.append({"kind": "missing_case", "case_id": case_id, "reason": f"absent from {', '.join(missing)}"})
            continue
        present.append(case_id)
        digests = {b.digests[case_id] for b in baselines + candidates if case_id in b.digests}
        if len(digests) > 1:
            failures.append({"kind": "input_digest", "case_id": case_id, "reason": "inputs differ between runs"})

    report["cases"] = cases = [_case(case_id, baselines, candidates, scoring) for case_id in present]
    eligible = [case for case in cases if case["eligible"]]
    for case in eligible:
        if case["regression"]:
            reason = f"{case['delta_pct']:+.2f}% slower than band {case['band_pct']:.2f}%"
            failures.append({"kind": "regression", "case_id": case["case_id"], "reason": reason})
    if not eligible:
        failures.append({"kind": "no_eligible_cases", "case_id": None, "reason": "no case has valid timing"})

    total = 0.0
    for family in sorted({case["family"] for case in eligible}):
        speedups = [case["speedup"] for case in eligible if case["family"] == family]
        geomean = math.exp(statistics.fmean(math.log(s) for s in speedups))
        weight = scoring["weights"].get(family, 0.0)
        contribution = weight * math.log(geomean)
        total += contribution
        report["families"][family] = {"weight": weight, "cases": len(speedups), "geomean_speedup": geomean, "contribution": contribution}
    report["score"] = total
    if failures:
        report["verdict"] = "fail"
    return report


def _cell(value: Any, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def format_report(report: dict) -> str:
    """Human table for a score report."""
    lines = [
        f"baseline:  {', '.join(report['baseline'])}",
        f"candidate: {', '.join(report['candidate'])}",
        f"board {report['board']}  placement {report['placement']}  floor {report['settings']['floor_pct']}%  mad_k {report['settings']['mad_k']}",
    ]
    if report["families"]:
        lines += ["", f"{'family':<16} {'weight':>6} {'cases':>5} {'geomean':>8} {'contrib':>9}"]
        for name, fam in report["families"].items():
            lines.append(f"{name:<16} {fam['weight']:>6.2f} {fam['cases']:>5} {fam['geomean_speedup']:>8.4f} {fam['contribution']:>+9.5f}")
    timed = [case for case in report["cases"] if case["eligible"]]
    if timed:
        lines += ["", f"{'case':<60} {'route':<36} {'A cycles':>11} {'B cycles':>11} {'A/B':>7} {'band%':>6} {'noise':>5} {'cpm A':>7} {'cpm B':>7} {'dINST':>8} {'dMVE':>8}"]
        for case in timed:
            retired = case["retired"]
            inst = retired.get("ARM_PMU_INST_RETIRED", {}).get("delta")
            mve = retired.get("ARM_PMU_MVE_INST_RETIRED", {}).get("delta")
            lines.append(
                f"{case['case_id'][:60]:<60} {(case['inner_symbol'] or case['timed_symbol'])[:36]:<36} "
                f"{case['baseline_cycles']:>11.1f} {case['candidate_cycles']:>11.1f} {case['speedup']:>7.4f} {case['band_pct']:>6.2f} "
                f"{'yes' if case['within_noise'] else 'NO':>5} {_cell(case['cycles_per_mac_baseline'], '.4f'):>7} "
                f"{_cell(case['cycles_per_mac_candidate'], '.4f'):>7} {_cell(inst, '+.0f'):>8} {_cell(mve, '+.0f'):>8}"
            )
    excluded: dict[str, int] = {}
    for case in report["cases"]:
        if not case["eligible"]:
            excluded[case["excluded_by"]] = excluded.get(case["excluded_by"], 0) + 1
    if excluded:
        lines += ["", "excluded: " + ", ".join(f"{k} {v}" for k, v in sorted(excluded.items()))]
    if report["failures"]:
        lines += ["", f"failures ({len(report['failures'])}):"]
        lines += [f"  {f['kind']}: {f['case_id'] or '-'}: {f['reason']}" for f in report["failures"]]
    score = "-" if report["score"] is None else f"{report['score']:+.5f}"
    lines += ["", f"== {report['verdict'].upper()}  score {score}  timed cases {len(timed)}"]
    return "\n".join(lines)


def score(
    baseline: list[Path] = typer.Argument(..., help="Baseline bundle dirs (repeat runs pool)."),
    candidate: list[Path] = typer.Option(..., "--candidate", help="Candidate bundle dir (repeatable)."),
    as_json: bool = typer.Option(False, "--json", help="Print the JSON report."),
    floor_pct: Optional[float] = typer.Option(None, "--floor-pct", min=0.0, help="Noise floor in percent (default: per board)."),
    mad_k: Optional[float] = typer.Option(None, "--mad-k", min=0.0, help="MAD multiplier for the band."),
) -> None:
    """Score a kernel candidate against a baseline.

    Exit 0 on pass, 1 on fail, 3 when the bundles cannot be compared.
    """
    baselines = [load_bundle(path) for path in baseline]
    candidates = [load_bundle(path) for path in candidate]
    scoring = load_scoring(_identity(baselines[0])["board"])
    if floor_pct is not None:
        scoring["floor_pct"] = floor_pct
    if mad_k is not None:
        scoring["mad_k"] = mad_k
    report = score_bundles(baselines, candidates, scoring)
    typer.echo(json.dumps(report, indent=2) if as_json else format_report(report))
    raise typer.Exit({"pass": EXIT_PASS, "fail": EXIT_FAIL}.get(report["verdict"], EXIT_REFUSED))
