from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware.score import kernel_commit, load_bundle, load_scoring, parse_focus, score_bundles

FIELDS = ["case_id", "kernel_id", "comparison_passed", "median_cycles", "mad_cycles", "timed_symbol", "inner_symbol", "macs",
          "ARM_PMU_INST_RETIRED", "ARM_PMU_MVE_INST_RETIRED", "timing_status", "prepare_cycles", "hidden"]
CASES = {
    "conv_a": ("arm_convolve_wrapper_s8", 1000.0),
    "dw_a": ("arm_depthwise_conv_wrapper_s8", 2000.0),
    "fc_a": ("arm_fully_connected_s8", 500.0),
    "add_a": ("arm_elementwise_add_s8", 300.0),
}


def _bundle(root: Path, name: str, *, cycles: dict | None = None, rows: dict | None = None, drop: tuple = (),
            target: dict | None = None, build: dict | None = None, digests: dict | None = None,
            cases: dict | None = None) -> Path:
    """Write a minimal result bundle."""
    path = root / name
    path.mkdir(parents=True)
    out = []
    cases = cases or CASES
    for case_id, (symbol, base) in cases.items():
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
    digests = {c: f"sha256:{c}" for c in cases} | (digests or {})
    (path / "cases.json").write_text(json.dumps([{"case_id": c, "input_digest": d} for c, d in digests.items() if d]))
    return path


def _score(baselines: list[Path], candidates: list[Path], check: dict | None = None, focus: dict | None = None,
           **settings) -> dict:
    """Score with the gain gate off."""
    scoring = load_scoring("apollo510_evb") | {"min_score": -math.inf}
    scoring.update(settings)
    return score_bundles([load_bundle(p) for p in baselines], [load_bundle(p) for p in candidates], scoring, check, focus)


def _pairs(report: dict) -> list[tuple]:
    return [(f["kind"], f["case_id"]) for f in report["failures"]]


def _kinds(report: dict) -> set[str]:
    return {failure["kind"] for failure in report["failures"]}


def test_identical_bundles_no_gain(tmp_path):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b")], min_score=0.0)
    assert report["verdict"] == "no_gain" and report["score"] == 0.0 and not report["failures"]
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


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_missing_input_digest_fails(tmp_path, side):
    bare = {"conv_a": None}
    base = _bundle(tmp_path, "a", digests=bare if side == "baseline" else None)
    cand = _bundle(tmp_path, "b", digests=bare if side == "candidate" else None)
    report = _score([base], [cand])
    assert [(f["kind"], f["case_id"]) for f in report["failures"]] == [("input_digest", "conv_a")]


def test_regression_past_floor_fails(tmp_path):
    base = _bundle(tmp_path, "a")
    assert _score([base], [_bundle(tmp_path, "b", cycles={"add_a": 306.0})])["verdict"] == "pass"
    report = _score([base], [_bundle(tmp_path, "c", cycles={"add_a": 330.0})])
    # One-case family: its gate fires too.
    assert _pairs(report) == [("regression", "add_a"), ("family_regression", None)]
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
    ({"cycles": {"conv_a": 800.0}}, 0, "pass"),
    ({}, 4, "no_gain"),
    ({"drop": ("conv_a",)}, 1, "fail"),
    ({"target": {"board": "apollo3p_evb"}}, 3, "not_comparable"),
])
def test_cli_json_and_exit_codes(tmp_path, change, code, verdict):
    base, cand = _bundle(tmp_path, "a"), _bundle(tmp_path, "b", **change)
    result = CliRunner().invoke(app, ["score", str(base), "--candidate", str(cand), "--no-check", "--json"])
    assert result.exit_code == code, result.output
    report = json.loads(result.output)
    assert (report["schema"], report["schema_version"], report["verdict"]) == ("hct.score", 2, verdict)
    table = CliRunner().invoke(app, ["score", str(base), "--candidate", str(cand), "--no-check"])
    assert f"== {verdict.upper()}" in table.output


def test_candidate_noise_keeps_band(tmp_path):
    cand = _bundle(tmp_path, "b", cycles={"add_a": 330.0}, rows={"add_a": {"mad_cycles": 30}})
    report = _score([_bundle(tmp_path, "a")], [cand])
    assert _pairs(report)[0] == ("regression", "add_a")


@pytest.mark.parametrize("status", ["overflow", "below_floor", "zero_cycles"])
def test_lost_timing_fails(tmp_path, status):
    cand = _bundle(tmp_path, "b", cycles={"conv_a": 9000.0}, rows={"conv_a": {"timing_status": status}})
    report = _score([_bundle(tmp_path, "a")], [cand])
    assert [(f["kind"], f["case_id"]) for f in report["failures"]] == [("timing_lost", "conv_a")]


def test_min_score_sets_no_gain(tmp_path):
    base, cand = _bundle(tmp_path, "a"), _bundle(tmp_path, "b", cycles={"conv_a": 990.0})
    assert _score([base], [cand], min_score=0.0)["verdict"] == "pass"
    assert _score([base], [cand], min_score=0.01)["verdict"] == "no_gain"
    slower = _bundle(tmp_path, "c", cycles={"conv_a": 1020.0})
    assert _score([base], [slower], min_score=0.0)["verdict"] == "no_gain"


def test_default_min_score_needs_half_percent(tmp_path):
    base = _bundle(tmp_path, "a")
    scoring = load_scoring("apollo510_evb")
    assert scoring["min_score"] == 0.005
    small = _bundle(tmp_path, "b", cycles={"conv_a": 995.0})
    big = _bundle(tmp_path, "c", cycles={"conv_a": 980.0})
    assert score_bundles([load_bundle(base)], [load_bundle(small)], scoring)["verdict"] == "no_gain"
    assert score_bundles([load_bundle(base)], [load_bundle(big)], scoring)["verdict"] == "pass"


def test_repeats_without_kernel_identity_refused(tmp_path):
    builds = [{"options": {}, "kernels": {"commit": c}} for c in ("x", "y")]
    report = _score([_bundle(tmp_path, "a", build=builds[0])], [_bundle(tmp_path, f"b{i}", build=b) for i, b in enumerate(builds)])
    assert report["verdict"] == "not_comparable"
    unknown = [{"options": {}, "kernels": {}} for _ in range(2)]
    report = _score([_bundle(tmp_path, f"c{i}", build=b) for i, b in enumerate(unknown)], [_bundle(tmp_path, "d", build=unknown[0])])
    assert [f["reason"] for f in report["failures"]] == ["baseline repeats lack kernel identity"]


def test_missing_baseline_repeat_excludes_case(tmp_path):
    bases = [_bundle(tmp_path, "a0"), _bundle(tmp_path, "a1", drop=("conv_a",))]
    report = _score(bases, [_bundle(tmp_path, "b", cycles={"conv_a": 500.0})])
    conv = next(case for case in report["cases"] if case["case_id"] == "conv_a")
    assert not conv["eligible"] and conv["excluded_by"] == "missing_baseline_repeat"
    assert "conv" not in report["families"]


@pytest.mark.parametrize("cycles", ["nan", "inf", 0.0])
def test_non_finite_or_zero_candidate_cycles_fail(tmp_path, cycles):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b", cycles={"conv_a": cycles})])
    assert report["verdict"] == "fail"
    assert ("timing_lost", "conv_a") in [(f["kind"], f["case_id"]) for f in report["failures"]]
    conv = next(case for case in report["cases"] if case["case_id"] == "conv_a")
    assert conv["candidate_cycles"] in (None, 0.0) and not conv["eligible"]


def test_repeat_digests_must_agree(tmp_path):
    mixed = [_bundle(tmp_path, "c0", digests={"conv_a": "sha256:1"}), _bundle(tmp_path, "c1", digests={"conv_a": "sha256:2"})]
    report = _score(mixed, [_bundle(tmp_path, "d")])
    assert [(f["kind"], f["reason"]) for f in report["failures"]] == [("input_digest", "inputs differ between repeats")]


@pytest.mark.parametrize("flag", ["--min-score", "--floor-pct", "--mad-k"])
def test_cli_refuses_non_finite_settings(tmp_path, flag):
    base, cand = _bundle(tmp_path, "a"), _bundle(tmp_path, "b")
    result = CliRunner().invoke(app, ["score", str(base), "--candidate", str(cand), "--no-check", flag, "nan"])
    assert result.exit_code == 2 and "must be finite" in result.output


def _with_compare(path, strict, golden=None):
    data = json.loads((path / "session_manifest.json").read_text())
    data["harness_digest"] = "d" * 64
    data["compare"] = {"strict": strict, "golden_session_id": golden}
    (path / "session_manifest.json").write_text(json.dumps(data))
    return path


def test_candidate_may_be_stricter_never_looser(tmp_path):
    loose_base = _with_compare(_bundle(tmp_path, "a"), False)
    # The agent loop: --golden-from the baseline.
    assert _score([loose_base], [_with_compare(_bundle(tmp_path, "b"), True, "a")])["verdict"] == "pass"
    strict_base = _with_compare(_bundle(tmp_path, "c"), True)
    report = _score([strict_base], [_with_compare(_bundle(tmp_path, "d"), False)])
    assert report["verdict"] == "not_comparable" and "looser compare" in report["failures"][0]["reason"]


def test_candidate_goldens_must_come_from_baseline(tmp_path):
    base = _with_compare(_bundle(tmp_path, "a"), False)
    report = _score([base], [_with_compare(_bundle(tmp_path, "b"), True, "elsewhere")])
    assert report["verdict"] == "not_comparable" and "not a baseline" in report["failures"][0]["reason"]



TREE, BASE = "sha256:cand", "c" * 40


def _trusted(root: Path, name: str, tree: str, strict: bool = True, dirty: bool = True, **kwargs) -> Path:
    """A bundle built from a clean or dirty tree."""
    kernels = {"root_head": BASE, "root_dirty": dirty, "tree_hash": tree}
    path = _bundle(root, name, build={"options": {}, "kernels": kernels}, **kwargs)
    return _with_compare(path, strict, "a" if strict else None)


def _check(**change) -> dict:
    return {"schema": "hct.candidate_check", "ok": True, "tree_hash": TREE, "base_commit": BASE} | change


@pytest.mark.parametrize(("check", "reason"), [
    (_check(), None),
    (_check(ok=False), "did not pass"),
    (_check(tree_hash=None), "lacks tree_hash"),
    (_check(tree_hash="sha256:other"), "tree hash"),
    (_check(base_commit="d" * 40), "kernel commit"),
    (_check(schema="other"), "wrong schema"),
])
def test_check_ties_score_to_tree(tmp_path, check, reason):
    base = _trusted(tmp_path, "a", "sha256:base", strict=False, dirty=False)
    cand = _trusted(tmp_path, "b", TREE, cycles={"conv_a": 800.0})
    report = _score([base], [cand], check)
    if reason is None:
        assert report["verdict"] == "pass" and report["settings"]["check"] == {"tree_hash": TREE, "base_commit": BASE}
    else:
        assert report["verdict"] == "not_comparable" and reason in report["failures"][0]["reason"]


def test_check_refuses_dirty_baseline(tmp_path):
    base = _trusted(tmp_path, "a", "sha256:dirty", strict=False)
    cand = _trusted(tmp_path, "b", TREE)
    report = _score([base], [cand], _check())
    assert [f["reason"] for f in report["failures"]] == [f"a: kernel commit None != check base {BASE}"]


def test_check_refuses_tolerant_candidate(tmp_path):
    base = _trusted(tmp_path, "a", "sha256:base", strict=False, dirty=False)
    cand = _trusted(tmp_path, "b", TREE, strict=False)
    report = _score([base], [cand], _check())
    assert report["verdict"] == "not_comparable" and "tolerant compare" in report["failures"][0]["reason"]


def test_cli_needs_check_or_opt_out(tmp_path):
    base = _trusted(tmp_path, "a", "sha256:base", strict=False, dirty=False)
    cand = _trusted(tmp_path, "b", TREE, cycles={"conv_a": 800.0})
    argv = ["score", str(base), "--candidate", str(cand)]
    assert CliRunner().invoke(app, argv).exit_code == 2
    report = tmp_path / "check.json"
    report.write_text(json.dumps(_check()))
    assert CliRunner().invoke(app, [*argv, "--check", str(report), "--no-check"]).exit_code == 2
    assert CliRunner().invoke(app, [*argv, "--check", str(report)]).exit_code == 0
    report.write_text(json.dumps(_check(ok=False)))
    assert CliRunner().invoke(app, [*argv, "--check", str(report)]).exit_code == 3
    report.write_text("[")
    assert CliRunner().invoke(app, [*argv, "--check", str(report)]).exit_code == 2


CONV3 = {
    "conv_a": ("arm_convolve_wrapper_s8", 1000.0),
    "conv_b": ("arm_convolve_wrapper_s8", 1000.0),
    "conv_mlperf_c": ("arm_convolve_wrapper_s8", 1000.0),
}


def _conv3(root: Path, name: str, scale: dict | float = 1.0, **kwargs) -> Path:
    cycles = {c: 1000.0 * (scale.get(c, 1.0) if isinstance(scale, dict) else scale) for c in CONV3}
    return _bundle(root, name, cases=CONV3, cycles=cycles, **kwargs)


def test_focus_scores_only_matching_cases(tmp_path):
    fast = {"conv_a": {"inner_symbol": "arm_convolve_1x1_s8_fast"}}
    base = _bundle(tmp_path, "a", rows=fast)
    cand = _bundle(tmp_path, "b", cycles={"conv_a": 800.0}, rows=fast)
    report = _score([base], [cand], focus=parse_focus(["arm_convolve_1x1_s8_fast"]))
    # Renormalized: conv is the only family hit.
    assert report["verdict"] == "pass" and report["score"] == pytest.approx(math.log(1.25))
    assert [c["case_id"] for c in report["cases"] if c["in_focus"]] == ["conv_a"]
    assert report["settings"]["focus"] == {"routes": ["arm_convolve_1x1_s8_fast"], "dtypes": []}


def test_focus_keeps_gates_everywhere(tmp_path):
    cand = _bundle(tmp_path, "b", cycles={"conv_a": 800.0, "add_a": 400.0}, rows={"fc_a": {"comparison_passed": "false"}})
    report = _score([_bundle(tmp_path, "a")], [cand], focus=parse_focus(["arm_convolve_wrapper_s8"]))
    assert {"comparison_failed", "regression", "family_regression"} <= _kinds(report)


def test_focus_without_hits_fails(tmp_path):
    report = _score([_bundle(tmp_path, "a")], [_bundle(tmp_path, "b")], focus=parse_focus(["s16"]))
    assert _pairs(report) == [("no_eligible_cases", None)]


def test_parse_focus_splits_dtypes():
    assert parse_focus(["s8", "arm_x", "f32"]) == {"routes": ["arm_x"], "dtypes": ["f32", "s8"]}
    assert parse_focus([]) is None


def test_family_gate_catches_spread_slowdown(tmp_path):
    report = _score([_conv3(tmp_path, "a")], [_conv3(tmp_path, "b", 1.025)])
    # Each case sits inside the 3 % floor.
    assert _pairs(report) == [("family_regression", None)]
    assert report["families"]["conv"]["band_pct"] == pytest.approx(3.0 / math.sqrt(3))


def test_mlperf_cases_weigh_more(tmp_path):
    report = _score([_conv3(tmp_path, "a")], [_conv3(tmp_path, "b", {"conv_mlperf_c": 0.5})])
    weight = load_scoring("apollo510_evb")["mlperf_weight"]
    expected = math.exp(weight * math.log(2.0) / (weight + 2))
    assert report["families"]["conv"]["geomean_speedup"] == pytest.approx(expected)


def test_repeat_baselines_set_session_band(tmp_path):
    bases = [_conv3(tmp_path, "a0"), _conv3(tmp_path, "a1", {"conv_a": 1.001})]
    cand = _conv3(tmp_path, "b", {"conv_b": 1.02})
    assert _score(bases[:1], [cand])["verdict"] == "pass"
    report = _score(bases, [cand])
    assert report["settings"]["band_source"] == "session"
    assert ("regression", "conv_b") in _pairs(report)


@pytest.mark.parametrize(("cand", "fails"), [("1200", False), ("3000", True), ("", True)])
def test_prepare_growth_fails(tmp_path, cand, fails):
    base = _bundle(tmp_path, "a", rows={"conv_a": {"prepare_cycles": 1000}})
    report = _score([base], [_bundle(tmp_path, "b", rows={"conv_a": {"prepare_cycles": cand}})])
    assert (("prepare_regression", "conv_a") in _pairs(report)) == fails


def test_case_reports_peak_and_dtype(tmp_path):
    target = {"board": "apollo510_evb", "cpu": "cortex-m55", "placement": {"name": "tcm"}}
    report = _score([_bundle(tmp_path, "a", target=target)], [_bundle(tmp_path, "b", target=target)])
    conv = next(case for case in report["cases"] if case["case_id"] == "conv_a")
    # 0.125 cycles/MAC peak over 10 measured.
    assert conv["pct_of_peak_candidate"] == pytest.approx(1.25)
    assert (conv["dtype"], conv["mlperf"], conv["weight"]) == ("s8", False, 1.0)


def test_focus_wins_cannot_hide_slowdowns(tmp_path):
    fast = {"conv_mlperf_c": {"inner_symbol": "arm_convolve_1x1_s8_fast"}}
    base = _conv3(tmp_path, "a", rows=fast)
    cand = _conv3(tmp_path, "b", {"conv_a": 1.025, "conv_b": 1.025, "conv_mlperf_c": 0.8}, rows=fast)
    assert _score([base], [cand])["verdict"] == "pass"
    report = _score([base], [cand], focus=parse_focus(["arm_convolve_1x1_s8_fast"]))
    assert _pairs(report) == [("family_regression", None)] and report["families"]["conv"]["gated_cases"] == 2


def test_focus_uses_baseline_route(tmp_path):
    cand = _bundle(tmp_path, "b", rows={"conv_a": {"inner_symbol": "arm_convolve_1x1_s8_fast"}})
    report = _score([_bundle(tmp_path, "a")], [cand], focus=parse_focus(["arm_convolve_1x1_s8_fast"]))
    assert _pairs(report) == [("no_eligible_cases", None)]


def test_cli_refuses_unknown_focus(tmp_path):
    base, cand = _bundle(tmp_path, "a"), _bundle(tmp_path, "b")
    argv = ["score", str(base), "--candidate", str(cand), "--no-check", "--focus"]
    result = CliRunner().invoke(app, [*argv, "arm_convolve_typo_s8"])
    assert result.exit_code == 2 and "no baseline case matches" in result.output
    assert CliRunner().invoke(app, [*argv, "s8"]).exit_code == 4


def _hide(root: Path, name: str, cases: tuple = ("conv_a",), commitment: str | None = "c1", **kwargs) -> Path:
    """A bundle with hidden cases and a commitment."""
    rows = {case_id: {"hidden": "true"} for case_id in cases}
    path = _bundle(root, name, rows=rows, **kwargs)
    selection = {"hidden_set": {"seed_commitment": commitment, "cases": len(cases)}} if commitment else {}
    (path / "session_summary.json").write_text(json.dumps({"selection": selection}))
    return path


def test_hidden_cases_score_apart(tmp_path):
    base = _hide(tmp_path, "a")
    report = _score([base], [_hide(tmp_path, "b", cycles={"conv_a": 800.0, "dw_a": 1000.0})])
    hidden = next(case for case in report["cases"] if case["case_id"] == "conv_a")
    assert hidden["hidden"] and report["verdict"] == "pass"
    assert report["subscores"]["hidden"] == {"cases": 1, "score": pytest.approx(math.log(1.25))}
    assert report["subscores"]["public"]["cases"] == 3
    assert _score([_bundle(tmp_path, "c")], [_bundle(tmp_path, "d")])["subscores"] is None


@pytest.mark.parametrize(("cases", "commitment", "reason"), [
    (("conv_a",), "c2", "commitment differs"),
    (("conv_a", "fc_a"), "c1", "case set differs"),
    ((), None, "commitment differs"),
])
def test_hidden_sets_must_match(tmp_path, cases, commitment, reason):
    report = _score([_hide(tmp_path, "a")], [_hide(tmp_path, "b", cases, commitment)])
    assert report["verdict"] == "not_comparable" and reason in report["failures"][0]["reason"]


@pytest.mark.parametrize(("kernels", "commit"), [
    ({"root": "/k", "root_head": None, "root_dirty": None}, None),
    ({"root": "/k", "root_head": BASE, "root_dirty": True}, None),
    ({"root": "/k", "root_head": BASE, "root_dirty": False}, BASE),
    ({"ref": "v1", "root": None}, "a"),
])
def test_kernel_commit_needs_clean_root(tmp_path, kernels, commit):
    build = {"options": {}, "modules": [{"name": "nsx-cmsis-nn", "commit": "a"}], "kernels": kernels}
    assert kernel_commit(load_bundle(_bundle(tmp_path, "a", build=build))) == commit


def test_scoring_files_versioned():
    scoring = load_scoring("apollo510_evb")
    assert scoring["weights_version"] == 2 and scoring["mlperf_weight"] == 4.0


@pytest.mark.parametrize("name", ["family_weights.yaml", "noise_floors.yaml"])
def test_old_scoring_schema_refused(tmp_path, name):
    from helia_core_tester.hardware.score import SCORING_DIR
    for other in ("family_weights.yaml", "noise_floors.yaml"):
        (tmp_path / other).write_text((SCORING_DIR / other).read_text())
    path = tmp_path / name
    path.write_text(path.read_text().replace("schema_version: 2", "schema_version: 1"))
    with pytest.raises(ValueError, match="need schema_version 2"):
        load_scoring("apollo510_evb", tmp_path)
