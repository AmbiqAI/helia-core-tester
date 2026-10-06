from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware.score import load_bundle, load_scoring, score_bundles

FIELDS = ["case_id", "kernel_id", "comparison_passed", "median_cycles", "mad_cycles", "timed_symbol", "inner_symbol", "macs",
          "ARM_PMU_INST_RETIRED", "ARM_PMU_MVE_INST_RETIRED", "timing_status"]
CASES = {
    "conv_a": ("arm_convolve_wrapper_s8", 1000.0),
    "dw_a": ("arm_depthwise_conv_wrapper_s8", 2000.0),
    "fc_a": ("arm_fully_connected_s8", 500.0),
    "add_a": ("arm_elementwise_add_s8", 300.0),
}


def _bundle(root: Path, name: str, *, cycles: dict | None = None, rows: dict | None = None, drop: tuple = (),
            target: dict | None = None, build: dict | None = None, digests: dict | None = None) -> Path:
    """Write a minimal result bundle."""
    path = root / name
    path.mkdir(parents=True)
    out = []
    for case_id, (symbol, base) in CASES.items():
        if case_id in drop:
            continue
        row = {"case_id": case_id, "kernel_id": "1", "comparison_passed": "true", "median_cycles": (cycles or {}).get(case_id, base),
               "mad_cycles": 0, "timed_symbol": symbol, "inner_symbol": symbol, "macs": 100,
               "ARM_PMU_INST_RETIRED": 400, "ARM_PMU_MVE_INST_RETIRED": 200, "timing_status": "valid"}
        row.update((rows or {}).get(case_id, {}))
        out.append(row)
    with (path / "case_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    manifest = {
        "session_id": name,
        "target": target or {"board": "apollo510_evb", "placement": {"name": "tcm"}},
        "boot": {"core_clock_hz": 250000000},
        "build": build or {"options": {"cmsis_nn_ref": "v1"}, "neuralspotx_version": "0.8.1",
                           "modules": [{"name": "nsx-cmsis-nn", "commit": "a"}], "kernels": {"tree_hash": None}},
    }
    (path / "session_manifest.json").write_text(json.dumps(manifest))
    (path / "cases.json").write_text(json.dumps([{"case_id": c, "input_digest": d} for c, d in (digests or {}).items()]))
    return path


def _score(baselines: list[Path], candidates: list[Path], **settings) -> dict:
    scoring = load_scoring("apollo510_evb")
    scoring.update(settings)
    return score_bundles([load_bundle(p) for p in baselines], [load_bundle(p) for p in candidates], scoring)


def _kinds(report: dict) -> set[str]:
    return {failure["kind"] for failure in report["failures"]}


def test_identical_bundles_pass(tmp_path):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b")])
    assert report["verdict"] == "pass" and report["score"] == 0.0
    assert set(report["families"]) == {"conv", "depthwise", "fully_connected", "other"}


def test_faster_conv_scores_weighted_log(tmp_path):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b", cycles={"conv_a": 800.0})])
    assert report["verdict"] == "pass"
    assert report["score"] == pytest.approx(0.50 * math.log(1.25))
    assert report["families"]["conv"]["geomean_speedup"] == pytest.approx(1.25)


def test_failed_comparison_fails(tmp_path):
    cand = _bundle(tmp_path, "b", rows={"dw_a": {"comparison_passed": "false"}}, cycles={"dw_a": 10.0})
    report = _score([_bundle(tmp_path, "a")], [cand])
    assert report["verdict"] == "fail" and "comparison_failed" in _kinds(report)
    dw = next(case for case in report["cases"] if case["case_id"] == "dw_a")
    assert not dw["eligible"]


def test_missing_case_fails(tmp_path):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b", drop=("fc_a",))])
    assert report["verdict"] == "fail"
    assert [f["case_id"] for f in report["failures"]] == ["fc_a"]


@pytest.mark.parametrize("change", [
    {"target": {"board": "apollo330mP_evb", "placement": {"name": "tcm"}}},
    {"target": {"board": "apollo510_evb", "placement": {"name": "mram"}}},
    {"target": {"board": "apollo510_evb"}},
    {"build": {"options": {"cmsis_nn_ref": "v1"}, "neuralspotx_version": "0.9.0", "modules": [{"name": "nsx-cmsis-nn", "commit": "a"}]}},
])
def test_other_board_placement_harness_refused(tmp_path, change):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b", **change)])
    assert report["verdict"] == "not_comparable" and report["score"] is None
    assert _kinds(report) == {"not_comparable"}


def test_kernel_change_stays_comparable(tmp_path):
    build = {"options": {"cmsis_nn_ref": "v2"}, "neuralspotx_version": "0.8.1",
             "modules": [{"name": "nsx-cmsis-nn", "commit": "b"}], "kernels": {"tree_hash": "x"}}
    assert _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b", build=build)])["verdict"] == "pass"


def test_candidate_repeats_need_one_kernel_tree(tmp_path):
    builds = [{"options": {}, "kernels": {"tree_hash": tree}} for tree in ("x", "y")]
    report = _score([_bundle(tmp_path, "a", build=builds[0])], [_bundle(tmp_path, f"b{i}", build=b) for i, b in enumerate(builds)])
    assert report["verdict"] == "not_comparable"


def test_input_digest_mismatch_fails(tmp_path):
    base = _bundle(tmp_path, "a", digests={"conv_a": "sha256:1"})
    report = _score([base], [_bundle(tmp_path, "b", digests={"conv_a": "sha256:2"})])
    assert [(f["kind"], f["case_id"]) for f in report["failures"]] == [("input_digest", "conv_a")]
    # One side only: nothing to check.
    assert _score([base], [_bundle(tmp_path, "c")])["verdict"] == "pass"


def test_regression_past_floor_fails(tmp_path):
    base = _bundle(tmp_path, "a")
    assert _score([base], [_bundle(tmp_path, "b", cycles={"add_a": 306.0})])["verdict"] == "pass"
    report = _score([base], [_bundle(tmp_path, "c", cycles={"add_a": 330.0})])
    assert [(f["kind"], f["case_id"]) for f in report["failures"]] == [("regression", "add_a")]
    assert _score([base], [_bundle(tmp_path, "d", cycles={"add_a": 330.0})], floor_pct=12.0)["verdict"] == "pass"


def test_mad_widens_band(tmp_path):
    noisy = {"add_a": {"mad_cycles": 10}}
    report = _score([_bundle(tmp_path, "a", rows=noisy)], [_bundle(tmp_path, "b", cycles={"add_a": 330.0}, rows=noisy)])
    add = next(case for case in report["cases"] if case["case_id"] == "add_a")
    assert add["band_pct"] == pytest.approx(3.0 * 1.4826 * 10 / 300 * 100)
    assert report["verdict"] == "pass"


def test_repeats_pool_median_of_medians(tmp_path):
    bases = [_bundle(tmp_path, f"a{i}", cycles={"conv_a": c}) for i, c in enumerate((1000.0, 1002.0, 1500.0))]
    cands = [_bundle(tmp_path, f"b{i}", cycles={"conv_a": c}) for i, c in enumerate((900.0, 901.0, 899.0))]
    conv = next(case for case in _score(bases, cands)["cases"] if case["case_id"] == "conv_a")
    assert (conv["baseline_cycles"], conv["candidate_cycles"]) == (1002.0, 900.0)


@pytest.mark.parametrize("status", ["below_floor", "degenerate_output", "error_path", "overflow"])
def test_untimed_status_never_gates(tmp_path, status):
    rows = {"add_a": {"timing_status": status}}
    report = _score([_bundle(tmp_path, "a", rows=rows)], [_bundle(tmp_path, "b", rows=rows, cycles={"add_a": 900.0})])
    add = next(case for case in report["cases"] if case["case_id"] == "add_a")
    assert report["verdict"] == "pass" and add["excluded_by"] == status and "other" not in report["families"]


def test_retired_deltas_reported(tmp_path):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b", rows={"conv_a": {"ARM_PMU_INST_RETIRED": 350}})])
    conv = next(case for case in report["cases"] if case["case_id"] == "conv_a")
    assert conv["retired"]["ARM_PMU_INST_RETIRED"]["delta"] == -50
    assert conv["retired"]["ARM_PMU_MVE_INST_RETIRED"]["delta"] == 0
    assert conv["cycles_per_mac_candidate"] == 10.0


@pytest.mark.parametrize(("change", "code", "verdict"), [
    ({}, 0, "pass"),
    ({"drop": ("conv_a",)}, 1, "fail"),
    ({"target": {"board": "apollo3p_evb"}}, 3, "not_comparable"),
])
def test_cli_json_and_exit_codes(tmp_path, change, code, verdict):
    base, cand = _bundle(tmp_path, "a"), _bundle(tmp_path, "b", **change)
    result = CliRunner().invoke(app, ["score", str(base), "--candidate", str(cand), "--json"])
    assert result.exit_code == code, result.output
    report = json.loads(result.output)
    assert (report["schema"], report["schema_version"], report["verdict"]) == ("hct.score", 1, verdict)
    table = CliRunner().invoke(app, ["score", str(base), "--candidate", str(cand)])
    assert f"== {verdict.upper()}" in table.output
