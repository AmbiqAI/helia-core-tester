"""Score a kernel candidate against a baseline on hardware.

Each side takes one or more result bundles (repeat runs); per-case
cycles pool as the median of per-run medians. Bundles from another
board, placement or harness are refused. The candidate fails on a failed
comparison, a dropped case, an input digest change, or a timed case that
is slower than its noise band: max(board floor, mad_k * 1.4826 * MAD /
baseline median), with the baseline's MAD: the larger of its in-run
MAD and the spread of its repeat medians. A case timed valid in the
baseline but not in the candidate fails. Only timing_status "valid" cases are timed.
Code layout moves untouched kernels, so given the touched case set
(cases whose kernel code changed) the case and family gates, family
geomeans and the score cover only those; untouched drift is reported
per family but never fails. Correctness gates cover every case.
With repeat baselines the floor is the small session floor instead of
the board floor. Each family also fails when its geomean is slower than
max(family floor, median case band / sqrt(cases)). Prepare cycles move
with code layout, so a case fails on them only when they go missing or
grow past their band (max(prepare floor %, k * repeat spread) of
prepare, prepare_timed_pct of timed) and also either pass
prepare_max_ratio times the baseline or could pay for a timed gain:
the case got faster than max(board floor, k * timed spread) and
prepare grew past prepare_share_pct of the cycles saved. MLPerf layer
cases weigh mlperf_case_weight in geomeans. The score is sum(weight * ln(family geomean speedup)) with
weights from assets/scoring/family_weights.yaml; with weights that sum
to 1 it approximates ln(whole-model speedup). --focus limits the score
(not the gates) to some routes or dtypes, weights renormalized. A
candidate with no failures but score <= min_score (default 0.005, about
ln 1.005) gets verdict no_gain. JSON schema: see `score_bundles`.

A case missing its input digest on either side fails. Trust:
`--check` takes the `candidate check` report; scoring refuses unless it
passed, its tree hash is the candidate build's, its base commit is the
baseline's kernel commit, and the candidate compared strictly.
`--no-check` skips this for humans scoring harness changes.
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

from .boards import repo_root
from .harness_lock import kernel_digest
from .nsx_app import CMSIS_NN_MODULE
from .pmu_explain import _DTYPE_RE, classify_route, load_ceilings
from .work_count import per_unit

SCHEMA = "hct.score"
SCHEMA_VERSION = 3
SCORING_DIR = repo_root() / "assets" / "scoring"
MAD_SIGMA = 1.4826
# Typer usage errors already exit 2.
EXIT_PASS, EXIT_FAIL, EXIT_REFUSED, EXIT_NO_GAIN = 0, 1, 3, 4
# About a 0.5 % weighted speedup.
DEFAULT_MIN_SCORE = 0.005
EXITS = {"pass": EXIT_PASS, "fail": EXIT_FAIL, "not_comparable": EXIT_REFUSED, "no_gain": EXIT_NO_GAIN}
SETTINGS = (
    "weights_version", "weights_board", "weights", "floor_pct", "family_floor_pct", "session_floor_pct",
    "prepare_floor_pct", "prepare_timed_pct", "prepare_share_pct", "prepare_max_ratio", "mad_k", "mlperf_weight", "min_score",
)
# INST_RETIRED plus every MVE retired counter.
_RETIRED = re.compile(r"^ARM_PMU_(INST|MVE_\w+)_RETIRED$")


@dataclass(frozen=True)
class Bundle:
    path: Path
    manifest: dict
    rows: dict[str, dict[str, str]]
    digests: dict[str, str]
    symbols: dict[str, str]
    hidden: frozenset[str] = frozenset()
    commitment: str | None = None

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
    hidden = frozenset(case_id for case_id, row in rows.items() if row.get("hidden") == "true")
    summary_path = root / "session_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    hidden_set = (summary.get("selection") or {}).get("hidden_set") or {}
    return Bundle(root, manifest, rows, digests, symbols, hidden, hidden_set.get("seed_commitment"))


def harness_print(manifest: dict) -> dict:
    """Everything in the build except kernels."""
    # Run options stay beside the digest.
    # Compare mode is checked in refusals.
    runtime = {"core_clock_hz": (manifest.get("boot") or {}).get("core_clock_hz")}
    if manifest.get("harness_digest"):
        return {"harness_digest": manifest["harness_digest"], **runtime}
    build = manifest.get("build") or {}
    return {
        "options": {k: v for k, v in (build.get("options") or {}).items() if not k.startswith("cmsis_nn_") and k != "placement"},
        "modules": [m for m in build.get("modules") or [] if m.get("name") != CMSIS_NN_MODULE],
        "neuralspotx_version": build.get("neuralspotx_version"),
        "toolchain": build.get("toolchain"),
        **runtime,
    }


def _identity(bundle: Bundle) -> dict:
    target = bundle.manifest.get("target") or {}
    return {
        "board": target.get("board"),
        "placement": (target.get("placement") or {}).get("name"),
        "harness": harness_print(bundle.manifest),
    }


def refusals(baselines: list[Bundle], candidates: list[Bundle], check: dict | None = None) -> list[str]:
    """Why these bundles cannot be compared.

    `check` is the candidate check report; None skips trust.
    """
    reasons = [] if check is None else trust_refusals(baselines, candidates, check)
    first = _identity(baselines[0])
    for bundle in baselines[1:] + candidates:
        other = _identity(bundle)
        for key in ("board", "placement"):
            if other[key] != first[key]:
                reasons.append(f"{bundle.session_id}: {key} {other[key]} != {first[key]}")
        changed = sorted(k for k in first["harness"].keys() | other["harness"].keys() if first["harness"].get(k) != other["harness"].get(k))
        if changed:
            reasons.append(f"{bundle.session_id}: harness differs in {', '.join(changed)}")
    reasons += _compare_refusals(baselines, candidates)
    # Hidden sets must match exactly.
    for bundle in baselines[1:] + candidates:
        if bundle.commitment != baselines[0].commitment:
            reasons.append(f"{bundle.session_id}: hidden seed commitment differs")
        elif bundle.hidden != baselines[0].hidden:
            reasons.append(f"{bundle.session_id}: hidden case set differs")
    for side, bundles in (("baseline", baselines), ("candidate", candidates)):
        kernels = [_kernel_id(b) for b in bundles]
        if len(bundles) > 1 and None in kernels:
            reasons.append(f"{side} repeats lack kernel identity")
        elif len(set(kernels)) > 1:
            reasons.append(f"{side} repeats built different kernels")
    return reasons


def _compare_refusals(baselines: list[Bundle], candidates: list[Bundle]) -> list[str]:
    """Candidates judged no looser than baseline.

    The kernel loop runs the baseline on default goldens and the candidate
    with --golden-from that baseline (strict), so stricter is allowed.
    """
    reasons = []
    if len({bool(_compare(b).get("strict")) for b in baselines}) > 1:
        reasons.append("baseline repeats differ in compare mode")
    strict_base = any(_compare(b).get("strict") for b in baselines)
    # Goldens must come from these baselines.
    sources = {b.session_id for b in baselines} | {_compare(b).get("golden_session_id") for b in baselines}
    for c in candidates:
        if strict_base and not _compare(c).get("strict"):
            reasons.append(f"{c.session_id}: looser compare than baseline")
        source = _compare(c).get("golden_session_id")
        if source and source not in sources:
            reasons.append(f"{c.session_id}: goldens from {source}, not a baseline")
    return reasons


def _compare(bundle: Bundle) -> dict:
    return bundle.manifest.get("compare") or {}


def _kernels(bundle: Bundle) -> dict:
    return (bundle.manifest.get("build") or {}).get("kernels") or {}


def _module_commit(bundle: Bundle) -> str | None:
    modules = (bundle.manifest.get("build") or {}).get("modules") or []
    return next((m.get("commit") for m in modules if m.get("name") == CMSIS_NN_MODULE), None)


def kernel_commit(bundle: Bundle) -> str | None:
    """The clean commit a bundle built."""
    kernels = _kernels(bundle)
    # Local roots need a known clean HEAD.
    if any(kernels.get(k) is not None for k in ("root", "root_head", "root_dirty")):
        return kernels.get("root_head") if kernels.get("root_dirty") is False else None
    return kernels.get("commit") or _module_commit(bundle)


def trust_refusals(baselines: list[Bundle], candidates: list[Bundle], check: dict) -> list[str]:
    """Tie the candidate to a passed check."""
    if check.get("schema") != "hct.candidate_check":
        return ["check report has the wrong schema"]
    reasons = [] if check.get("ok") is True else ["candidate check did not pass"]
    tree, base = check.get("tree_hash"), check.get("base_commit")
    if not tree:
        reasons.append("check report lacks tree_hash")
    if not base:
        reasons.append("check report lacks base_commit")
    for b in baselines:
        commit = kernel_commit(b)
        if base and commit != base:
            reasons.append(f"{b.session_id}: kernel commit {commit} != check base {base}")
    for c in candidates:
        built = kernel_digest(c.manifest)
        if tree and built != tree:
            reasons.append(f"{c.session_id}: tree hash {built} != checked {tree}")
        if not _compare(c).get("strict"):
            reasons.append(f"{c.session_id}: tolerant compare; run with --golden-from")
    return reasons


def _kernel_id(bundle: Bundle) -> tuple | None:
    """Kernel tree identity; None when unknown."""
    kernels = _kernels(bundle)
    if kernels.get("tree_hash"):
        return ("tree_hash", kernels["tree_hash"])
    known = (kernels.get("commit") or _module_commit(bundle), kernels.get("ref"), kernels.get("root_head"), kernels.get("root_dirty"))
    return ("provenance", known) if any(v is not None for v in known) else None


def _num(row: dict, key: str) -> float | None:
    """A finite number, else None."""
    cell = row.get(key)
    if cell in ("", None):
        return None
    try:
        value = float(cell)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _lost(lost: list[str], base: list[dict], a: float | None, b: float | None) -> str | None:
    """Why a valid baseline case lost timing."""
    if not all(_status(row) == "valid" for row in base):
        return None
    if lost:
        return lost[0]
    # Zero, missing or non-finite cycles.
    return "no_cycles" if a and not b else None


def _status(row: dict) -> str:
    """timing_status; legacy bundles infer it."""
    if row.get("timing_status"):
        return row["timing_status"]
    return "valid" if row.get("valid_for_regression") == "true" else "legacy_invalid"


def _pool(rows: list[dict], key: str) -> float | None:
    values = [v for v in (_num(row, key) for row in rows) if v is not None]
    return statistics.median(values) if len(values) == len(rows) else None


def _spread(rows: list[dict], key: str = "median_cycles", mad_key: str | None = "mad_cycles") -> float:
    """Pooled MAD in cycles, scaled to sigma."""
    medians = [_num(row, key) or 0.0 for row in rows]
    center = statistics.median(medians)
    repeat_mad = statistics.median(abs(m - center) for m in medians)
    run_mad = statistics.median(_num(row, mad_key) or 0.0 for row in rows) if mad_key else 0.0
    return MAD_SIGMA * max(repeat_mad, run_mad)


# v2 focus and gates; floors v3 prepare intent.
SCORING_SCHEMAS = {"family_weights.yaml": 2, "noise_floors.yaml": 3}


def _scoring_file(path: Path) -> dict:
    """A scoring YAML at the current schema."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    want = SCORING_SCHEMAS[path.name]
    if data.get("schema_version") != want:
        raise ValueError(f"{path}: need schema_version {want}, got {data.get('schema_version')}")
    return data


def load_scoring(board: str | None, directory: Path = SCORING_DIR) -> dict:
    """Weights, family rules and noise floor for `board`."""
    weights, floors = (_scoring_file(directory / name) for name in ("family_weights.yaml", "noise_floors.yaml"))
    weights_board = board if board in weights["boards"] else weights["fallback_board"]
    floor_row = floors["boards"].get(board) or {}
    return {
        "weights_version": weights["version"],
        "weights_board": weights_board,
        "weights": dict(weights["boards"][weights_board]["weights"]),
        "families": weights["families"],
        "default_family": weights["default_family"],
        "floor_pct": float(floor_row.get("floor_pct", floors["default_floor_pct"])),
        "family_floor_pct": float(floor_row.get("family_floor_pct", floors["default_family_floor_pct"])),
        "session_floor_pct": float(floors["session_floor_pct"]),
        "prepare_floor_pct": float(floors["prepare_floor_pct"]),
        "prepare_timed_pct": float(floors["prepare_timed_pct"]),
        "prepare_share_pct": float(floors["prepare_share_pct"]),
        "prepare_max_ratio": float(floors["prepare_max_ratio"]),
        "mad_k": float(floors["mad_k"]),
        "mlperf_weight": float(weights["mlperf_case_weight"]),
        "min_score": DEFAULT_MIN_SCORE,
    }


def _dtype(route: str, case_id: str) -> str | None:
    """The dtype token in the route or case id."""
    match = _DTYPE_RE.search(route) or _DTYPE_RE.search(case_id)
    return match.group(1) if match else None


def in_focus(route: str, symbol: str, dtype: str | None, focus: dict | None) -> bool:
    """Case matches every focus axis given."""
    if not focus:
        return True
    routes, dtypes = focus.get("routes") or [], focus.get("dtypes") or []
    return (not routes or route in routes or symbol in routes) and (not dtypes or dtype in dtypes)


def _peak_pct(cpu: str | None, route: str, cycles_per_mac: float | None) -> float | None:
    """Percent of the MAC ceiling reached."""
    op, dtype = classify_route(route)
    entry = (((load_ceilings()["cpus"].get(cpu) or {}).get("ops") or {}).get(op) or {}).get(dtype)
    return entry["cycles_per_mac"] / cycles_per_mac * 100.0 if entry and cycles_per_mac else None


def _prepare_cause(a: float, b: float | None, allowed: float, saved: float, scoring: dict) -> str | None:
    """Why prepare growth fails, or None."""
    if b is None:
        return "missing"
    growth = b - a
    if growth <= allowed:
        return None
    if b > a * scoring["prepare_max_ratio"]:
        return "blowup"
    # Growth that could buy the gain.
    return "pays_for_gain" if saved > 0 and growth > saved * scoring["prepare_share_pct"] / 100.0 else None


def _prepare(base: list[dict], cand: list[dict], timed: tuple[float | None, float | None], scoring: dict) -> dict | None:
    """Prepare cycles; None when the baseline lacks them."""
    a = _pool(base, "prepare_cycles")
    if a is None:
        return None
    b = _pool(cand, "prepare_cycles")
    band = max(scoring["prepare_floor_pct"], scoring["mad_k"] * _spread(base, "prepare_cycles", None) / a * 100.0) if a else 0.0
    before, after = timed
    allowed = max(a * band / 100.0, (before or 0.0) * scoring["prepare_timed_pct"] / 100.0)
    # Gains inside cross-build noise don't count.
    noise = max(scoring["floor_pct"], scoring["mad_k"] * _spread(base) / before * 100.0) if before else 0.0
    saved = before - after if before and after and before - after > before * noise / 100.0 else 0.0
    cause = _prepare_cause(a, b, allowed, saved, scoring)
    return {"baseline": a, "candidate": b, "band_pct": band, "cause": cause, "regression": cause is not None}


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


def _case(
    case_id: str, baselines: list[Bundle], candidates: list[Bundle], scoring: dict, focus: dict | None = None,
    touched: frozenset[str] | None = None,
) -> dict:
    base = [b.rows[case_id] for b in baselines if case_id in b.rows]
    cand = [c.rows[case_id] for c in candidates]
    statuses = sorted({_status(row) for row in base + cand} - {"valid"})
    # A baseline repeat lacks this case.
    if len(base) != len(baselines):
        statuses.insert(0, "missing_baseline_repeat")
    # Valid baseline, untimed candidate: a failure.
    lost = sorted({_status(row) for row in cand} - {_status(row) for row in base} - {"valid"})
    if any(row.get("comparison_passed") != "true" for row in base + cand):
        statuses.insert(0, "comparison_failed")
    symbol = baselines[0].symbol(case_id) if case_id in baselines[0].rows else candidates[0].symbol(case_id)
    a, b = _pool(base, "median_cycles"), _pool(cand, "median_cycles")
    inner = cand[0].get("inner_symbol") or None
    # Baseline route: the candidate cannot move it.
    base_route = (base[0].get("inner_symbol") if base else None) or symbol
    cpm_a, cpm_b = per_unit(a, _num(base[0], "macs")), per_unit(b, _num(cand[0], "macs"))
    cpu = (baselines[0].manifest.get("target") or {}).get("cpu")
    mlperf = "mlperf" in case_id
    case: dict[str, Any] = {
        "case_id": case_id,
        "family": family_of(symbol, scoring),
        "timed_symbol": symbol,
        "inner_symbol": inner,
        "dtype": _dtype(base_route, case_id),
        "mlperf": mlperf,
        "hidden": case_id in baselines[0].hidden,
        "weight": scoring["mlperf_weight"] if mlperf else 1.0,
        "eligible": not statuses and bool(a) and bool(b),
        "excluded_by": statuses[0] if statuses else None,
        "timing_lost": _lost(lost, base, a, b),
        "baseline_cycles": a,
        "candidate_cycles": b,
        "cycles_per_mac_baseline": cpm_a,
        "cycles_per_mac_candidate": cpm_b,
        "pct_of_peak_baseline": _peak_pct(cpu, base_route, cpm_a),
        "pct_of_peak_candidate": _peak_pct(cpu, inner or symbol, cpm_b),
        "prepare": _prepare(base, cand, (a, b), scoring),
        "touched": None if touched is None else case_id in touched,
        "retired": _retired(base, cand),
    }
    case["in_focus"] = in_focus(base_route, symbol, case["dtype"], focus)
    if case["eligible"]:
        # Repeat baselines measure this session's noise.
        floor = scoring["session_floor_pct"] if len(baselines) > 1 else scoring["floor_pct"]
        band = max(floor, scoring["mad_k"] * _spread(base) / a * 100.0)
        delta = (b - a) / a * 100.0
        case.update(speedup=a / b, delta_pct=delta, band_pct=band, within_noise=abs(delta) <= band,
                    regression=_gated(case) and delta > band)
    elif case["excluded_by"] is None:
        case["excluded_by"] = "zero_cycles"
    return case


def _gated(case: dict) -> bool:
    """Untouched cases move with layout only."""
    return case["touched"] is not False


def _geomean(cases: list[dict]) -> float:
    """Case-weighted geomean speedup."""
    total = sum(case["weight"] for case in cases)
    return math.exp(sum(case["weight"] * math.log(case["speedup"]) for case in cases) / total)


def _family_band(cases: list[dict], scoring: dict) -> float:
    """Median case band over sqrt(n), floored."""
    spread = statistics.median(case["band_pct"] for case in cases) / math.sqrt(len(cases))
    return max(scoring["family_floor_pct"], spread)


def _contributions(cases: list[dict], scoring: dict, renorm: bool) -> dict[str, float]:
    """Weighted ln geomean per family."""
    weights = {case["family"]: scoring["weights"].get(case["family"], 0.0) for case in cases}
    norm = sum(weights.values()) if renorm else 1.0
    return {
        family: weight / norm * math.log(_geomean([c for c in cases if c["family"] == family])) if norm else 0.0
        for family, weight in sorted(weights.items())
    }


# Documented in score_bundles; tests pin it.
FAMILY_KEYS = ("weight", "cases", "geomean_speedup", "band_pct", "gates", "regression", "focus_cases", "focus_geomean",
               "contribution", "untouched_cases", "untouched_geomean")
GATE_KEYS = ("subset", "cases", "slowdown_pct", "band_pct", "regression")


def _family_gate(subset: str, cases: list[dict], scoring: dict) -> dict:
    """Fail a subset slower than its band."""
    band = _family_band(cases, scoring)
    slowdown = (1.0 / _geomean(cases) - 1.0) * 100.0
    return {"subset": subset, "cases": len(cases), "slowdown_pct": slowdown, "band_pct": band, "regression": slowdown > band}


def score_bundles(
    baselines: list[Bundle], candidates: list[Bundle], scoring: dict, check: dict | None = None, focus: dict | None = None,
    touched: frozenset[str] | None = None,
) -> dict:
    """The score report; `schema_version` bumps on breaking change.

    Keys: schema, schema_version, verdict (pass | fail | not_comparable | no_gain),
    score (null unless comparable), board, placement, baseline and
    candidate session ids, settings, families {name: {weight, cases,
    geomean_speedup, band_pct, gates [{subset, cases, slowdown_pct,
    band_pct, regression}], regression, focus_cases, focus_geomean,
    contribution, untouched_cases, untouched_geomean}}, subscores ({public, hidden: {cases, score}} when the
    run has hidden cases, else null; each renormalized), cases (eligible
    and excluded rows), failures [{kind,
    case_id, reason}]. settings.check is the trusted {tree_hash,
    base_commit}, or null when unchecked; settings.focus is {routes,
    dtypes} or null; settings.case_gate is "touched" when `touched`
    (case ids whose kernel code changed) is given, else "all". Each
    case's `touched` is a bool, or null when unknown. Untouched cases
    never fail a speed gate and leave cases, geomean_speedup, band_pct,
    focus and contribution; untouched_cases and untouched_geomean
    report their drift (0 and null when every case is gated). A family
    with no gated case has null geomean_speedup and band_pct and no
    gates. Failure kinds: not_comparable, comparison_failed,
    missing_case, input_digest, timing_lost, regression,
    family_regression, prepare_regression, no_eligible_cases.

    Correctness and prepare gates cover every case, the case and
    family gates every touched case; under
    focus the family gate judges the focus and rest subsets apart. The score covers
    focus cases only (by baseline route), with family weights
    renormalized over the families they hit.
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
        "settings": {
            **{k: scoring[k] for k in SETTINGS},
            "band_source": "session" if len(baselines) > 1 else "board",
            "check": None if check is None else {k: check.get(k) for k in ("tree_hash", "base_commit")},
            "focus": focus or None,
            "case_gate": "all" if touched is None else "touched",
        },
        "families": {},
        "subscores": None,
        "cases": [],
        "failures": [],
    }
    failures = report["failures"]
    refused = refusals(baselines, candidates, check)
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
        unhashed = [b.session_id for b in baselines + candidates if case_id not in b.digests]
        if unhashed:
            failures.append({"kind": "input_digest", "case_id": case_id, "reason": f"no input digest in {', '.join(unhashed)}"})
            continue
        base_digests = {b.digests[case_id] for b in baselines}
        cand_digests = {c.digests[case_id] for c in candidates}
        if len(base_digests) > 1 or len(cand_digests) > 1:
            failures.append({"kind": "input_digest", "case_id": case_id, "reason": "inputs differ between repeats"})
        elif base_digests != cand_digests:
            failures.append({"kind": "input_digest", "case_id": case_id, "reason": "inputs differ between runs"})

    report["cases"] = cases = [_case(case_id, baselines, candidates, scoring, focus, touched) for case_id in present]
    eligible = [case for case in cases if case["eligible"]]
    for case in cases:
        if case["timing_lost"]:
            failures.append({"kind": "timing_lost", "case_id": case["case_id"], "reason": f"candidate timing is {case['timing_lost']}"})
    for case in eligible:
        if case["regression"]:
            reason = f"{case['delta_pct']:+.2f}% slower than band {case['band_pct']:.2f}%"
            failures.append({"kind": "regression", "case_id": case["case_id"], "reason": reason})
    for case in cases:
        prepare = case["prepare"]
        if prepare and prepare["regression"] and case["excluded_by"] != "comparison_failed":
            reason = f"prepare cycles {prepare['baseline']:.0f} -> {_cell(prepare['candidate'], '.0f')}, {prepare['cause']}"
            failures.append({"kind": "prepare_regression", "case_id": case["case_id"], "reason": reason})
    if not any(case["in_focus"] for case in eligible):
        reason = "no focus case has valid timing" if focus else "no case has valid timing"
        failures.append({"kind": "no_eligible_cases", "case_id": None, "reason": reason})

    # Layout drift never fails a candidate.
    gated = [case for case in eligible if _gated(case)]
    focused = [case for case in gated if case["in_focus"]]
    for family in sorted({case["family"] for case in eligible}):
        members = [case for case in gated if case["family"] == family]
        drift = [case for case in eligible if case["family"] == family and not _gated(case)]
        hits = [case for case in members if case["in_focus"]]
        # Gate focus and rest apart.
        subsets = {"focus": hits, "rest": [c for c in members if not c["in_focus"]]} if focus else {"all": members}
        gates = [_family_gate(name, part, scoring) for name, part in subsets.items() if part]
        for gate in gates:
            if gate["regression"]:
                reason = f"{family} {gate['subset']}: {gate['slowdown_pct']:+.2f}% slower than band {gate['band_pct']:.2f}%"
                failures.append({"kind": "family_regression", "case_id": None, "reason": reason})
        report["families"][family] = {
            "weight": scoring["weights"].get(family, 0.0), "cases": len(members),
            "geomean_speedup": _geomean(members) if members else None,
            "band_pct": _family_band(members, scoring) if members else None, "gates": gates,
            "regression": any(g["regression"] for g in gates),
            "focus_cases": len(hits), "focus_geomean": _geomean(hits) if hits else None, "contribution": 0.0,
            "untouched_cases": len(drift), "untouched_geomean": _geomean(drift) if drift else None,
        }
    for family, contribution in _contributions(focused, scoring, bool(focus)).items():
        report["families"][family]["contribution"] = contribution
    report["score"] = total = sum(fam["contribution"] for fam in report["families"].values())
    if any(case["hidden"] for case in cases):
        parts = {name: [c for c in focused if c["hidden"] == hidden] for name, hidden in (("public", False), ("hidden", True))}
        report["subscores"] = {
            name: {"cases": len(part), "score": sum(_contributions(part, scoring, True).values()) if part else None}
            for name, part in parts.items()
        }
    if failures:
        report["verdict"] = "fail"
    elif total <= scoring["min_score"]:
        report["verdict"] = "no_gain"
    return report


def _cell(value: Any, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def format_report(report: dict) -> str:
    """Human table for a score report."""
    lines = [
        f"baseline:  {', '.join(report['baseline'])}",
        f"candidate: {', '.join(report['candidate'])}",
        f"board {report['board']}  placement {report['placement']}  floor {report['settings']['floor_pct']}%  "
        f"mad_k {report['settings']['mad_k']}  band {report['settings']['band_source']}",
    ]
    if report["settings"]["focus"]:
        lines.append(f"focus: {json.dumps(report['settings']['focus'])}")
    if report["families"]:
        lines += ["", f"{'family':<16} {'weight':>6} {'cases':>5} {'geomean':>8} {'band%':>6} {'focus':>5} {'f geo':>8} {'contrib':>9} {'untch':>5} {'u geo':>8}"]
        for name, fam in report["families"].items():
            lines.append(
                f"{name:<16} {fam['weight']:>6.2f} {fam['cases']:>5} {_cell(fam['geomean_speedup'], '.4f'):>8} "
                f"{_cell(fam['band_pct'], '.2f'):>6} "
                f"{fam['focus_cases']:>5} {_cell(fam['focus_geomean'], '.4f'):>8} {fam['contribution']:>+9.5f} "
                f"{fam['untouched_cases']:>5} {_cell(fam['untouched_geomean'], '.4f'):>8}"
            )
    timed = [case for case in report["cases"] if case["eligible"]]
    if timed:
        lines += ["", f"{'case':<60} {'route':<36} {'A cycles':>11} {'B cycles':>11} {'A/B':>7} {'band%':>6} {'noise':>5} {'cpm A':>7} {'cpm B':>7} {'peak%':>6} {'dINST':>8} {'dMVE':>8}"]
        for case in timed:
            retired = case["retired"]
            inst = retired.get("ARM_PMU_INST_RETIRED", {}).get("delta")
            mve = retired.get("ARM_PMU_MVE_INST_RETIRED", {}).get("delta")
            mark = "*" if case["in_focus"] and report["settings"]["focus"] else " "
            lines.append(
                f"{mark}{case['case_id'][:59]:<59} {(case['inner_symbol'] or case['timed_symbol'])[:36]:<36} "
                f"{case['baseline_cycles']:>11.1f} {case['candidate_cycles']:>11.1f} {case['speedup']:>7.4f} {case['band_pct']:>6.2f} "
                f"{'yes' if case['within_noise'] else 'NO':>5} {_cell(case['cycles_per_mac_baseline'], '.4f'):>7} "
                f"{_cell(case['cycles_per_mac_candidate'], '.4f'):>7} {_cell(case['pct_of_peak_candidate'], '.1f'):>6} {_cell(inst, '+.0f'):>8} {_cell(mve, '+.0f'):>8}"
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
    for name, part in (report["subscores"] or {}).items():
        lines.append(f"{name} score {_cell(part['score'], '+.5f')}  cases {part['cases']}")
    score = "-" if report["score"] is None else f"{report['score']:+.5f}"
    lines += ["", f"== {report['verdict'].upper()}  score {score}  timed cases {len(timed)}"]
    return "\n".join(lines)


def parse_focus(terms: list[str]) -> dict | None:
    """Split --focus terms into routes and dtypes."""
    if not terms:
        return None
    dtypes = sorted({t for t in terms if _DTYPE_RE.fullmatch(f"_{t}")})
    return {"routes": sorted(set(terms) - set(dtypes)), "dtypes": dtypes}


def unknown_focus(focus: dict | None, baselines: list[Bundle]) -> list[str]:
    """Focus terms no baseline case has."""
    if not focus:
        return []
    known = set()
    for b in baselines:
        for case_id, row in b.rows.items():
            symbol = b.symbol(case_id)
            route = row.get("inner_symbol") or symbol
            known |= {route, symbol, _dtype(route, case_id)}
    return [t for t in focus["routes"] + focus["dtypes"] if t not in known]


def score(
    baseline: list[Path] = typer.Argument(..., help="Baseline bundle dirs (repeat runs pool)."),
    candidate: list[Path] = typer.Option(..., "--candidate", help="Candidate bundle dir (repeatable)."),
    check_path: Optional[Path] = typer.Option(None, "--check", exists=True, dir_okay=False, help="Candidate check JSON report."),
    no_check: bool = typer.Option(False, "--no-check", help="Score without a check, for harness changes."),
    as_json: bool = typer.Option(False, "--json", help="Print the JSON report."),
    floor_pct: Optional[float] = typer.Option(None, "--floor-pct", min=0.0, help="Noise floor in percent (default: per board)."),
    mad_k: Optional[float] = typer.Option(None, "--mad-k", min=0.0, help="MAD multiplier for the band."),
    min_score: float = typer.Option(DEFAULT_MIN_SCORE, "--min-score", help="Score to beat; 0.005 is about 0.5 % gain."),
    focus_terms: list[str] = typer.Option([], "--focus", help="Score only cases on these routes and dtypes."),
) -> None:
    """Score a kernel candidate against a baseline.

    Exit 0 pass, 1 fail, 3 not comparable, 4 no gain. --focus takes
    exact inner or timed symbols and dtypes (s8, s16, ...); a case
    must match one route and one dtype when both are given.
    """
    if (check_path is None) != no_check:
        raise typer.BadParameter("pass exactly one of --check or --no-check", param_hint="--check")
    for flag, value in (("--floor-pct", floor_pct), ("--mad-k", mad_k), ("--min-score", min_score)):
        if value is not None and not math.isfinite(value):
            raise typer.BadParameter(f"{flag} must be finite", param_hint=flag)
    baselines = [load_bundle(path) for path in baseline]
    candidates = [load_bundle(path) for path in candidate]
    scoring = load_scoring(_identity(baselines[0])["board"])
    if floor_pct is not None:
        scoring["floor_pct"] = floor_pct
    if mad_k is not None:
        scoring["mad_k"] = mad_k
    scoring["min_score"] = min_score
    check = None
    if check_path is not None:
        try:
            check = json.loads(check_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"unreadable check report: {exc}", param_hint="--check") from exc
        if not isinstance(check, dict):
            raise typer.BadParameter("check report is not an object", param_hint="--check")
    focus = parse_focus(focus_terms)
    unknown = unknown_focus(focus, baselines)
    if unknown:
        raise typer.BadParameter(f"no baseline case matches {', '.join(unknown)}", param_hint="--focus")
    report = score_bundles(baselines, candidates, scoring, check, focus)
    typer.echo(json.dumps(report, indent=2) if as_json else format_report(report))
    raise typer.Exit(EXITS[report["verdict"]])
