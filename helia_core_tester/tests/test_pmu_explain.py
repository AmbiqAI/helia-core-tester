import json

import pytest
from typer.testing import CliRunner

from helia_core_tester.cli import app
from helia_core_tester.hardware.measurement import check_pass_count, counter_passes_for_selection
from helia_core_tester.hardware.pmu_explain import (
    AGENT_PMU_SELECTION,
    SCHEMA_VERSION,
    classify_route,
    explain_case,
    load_ceilings,
)

M55 = "cortex-m55"


def _row(**counters):
    """Healthy s8 conv row: no rule fires."""
    base = {
        "INST_RETIRED": 3000, "STALL_FRONTEND": 0, "STALL_BACKEND": 400, "L1D_CACHE_REFILL": 0,
        "BUS_ACCESS": 0, "MVE_INST_RETIRED": 2400, "MVE_INT_MAC_RETIRED": 1000, "MVE_FP_MAC_RETIRED": 0,
        "MVE_PRED": 0, "MVE_STALL_RESOURCE_MEM": 100, "MVE_STALL_DEPENDENCY": 0,
    }
    base.update(counters)
    return {
        "case_id": "c", "timed_symbol": "arm_convolve_wrapper_s8", "inner_symbol": "arm_convolve_s8",
        "median_cycles": 4000, "macs": 16000, "prepare_cycles": 100, "timing_status": "valid",
        "counters": {f"ARM_PMU_{name}": value for name, value in base.items()},
    }


def _rules(row, **kwargs):
    return [finding.rule for finding in explain_case(row, cpu=M55, **kwargs).findings]


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("arm_depthwise_conv_s8_opt", ("depthwise", "s8")),
        ("arm_convolve_1x1_s8_fast", ("conv", "s8")),
        ("arm_fully_connected_per_channel_s8", ("fc", "s8")),
        ("arm_batch_matmul_f16", ("fc", "f16")),
        ("arm_transpose_conv_f32", ("conv", "f32")),
        ("arm_convolve_wrapper_s4", ("conv", "s8")),
        ("arm_depthwise_conv_wrapper_s16", ("depthwise", "s16")),
        ("arm_softmax_s8", (None, "s8")),
    ],
)
def test_classify_route(symbol, expected):
    assert classify_route(symbol) == expected


def test_healthy_row_has_no_bottleneck():
    result = explain_case(_row(), cpu=M55)
    assert [f.rule for f in result.findings] == ["no_bottleneck"]
    assert result.pct_of_peak == pytest.approx(50.0)
    assert "= 50% of peak" in result.lines()[0]
    assert result.metrics["mve_mul_ratio"] == pytest.approx(1.0)
    assert result.missing_counters == []
    assert len(result.lines()) == 4


def test_near_peak():
    row = _row()
    row["median_cycles"] = 2500
    result = explain_case(row, cpu=M55)
    assert result.findings[0].rule == "near_peak"
    assert result.pct_of_peak == pytest.approx(80.0)
    assert result.findings[0].impact == pytest.approx(0.8)
    assert "80% of peak" in result.findings[0].diagnosis


def test_memory_bound_tcm_and_mram():
    row = _row(INST_RETIRED=1000, STALL_BACKEND=3000, L1D_CACHE_REFILL=2560)
    assert _rules(row)[0] == "memory_bound"
    assert "Memory-bound" in explain_case(row, cpu=M55, placement="tcm").diagnosis
    assert explain_case(row, cpu=M55, placement="mram").diagnosis.startswith("MRAM-bound")


def test_backend_stall_without_memory_signal():
    row = _row(INST_RETIRED=1000, STALL_BACKEND=3000, MVE_STALL_RESOURCE_MEM=0)
    assert _rules(row)[0] == "backend_bound"
    assert _rules(_row(INST_RETIRED=1000, STALL_BACKEND=3000, MVE_STALL_RESOURCE_MEM=800))[0] == "memory_bound"


def test_missing_counters_drop_clauses():
    row = _row(INST_RETIRED=8000, MVE_INST_RETIRED=6000, MVE_INT_MAC_RETIRED=2000)
    del row["counters"]["ARM_PMU_MVE_PRED"]
    findings = {f.rule: f.diagnosis for f in explain_case(row, cpu=M55).findings}
    assert "predicated" not in findings["underfilled_vectors"]
    row = _row(INST_RETIRED=8000, MVE_INST_RETIRED=6000)
    del row["counters"]["ARM_PMU_MVE_INST_RETIRED"]
    overhead = next(f for f in explain_case(row, cpu=M55).findings if f.rule == "instruction_overhead")
    assert "scalar" not in overhead.diagnosis


def test_instruction_overhead():
    assert _rules(_row(INST_RETIRED=8000, MVE_INST_RETIRED=6000))[0] == "instruction_overhead"


def test_depthwise_has_own_target():
    conv = explain_case(_row(), cpu=M55)
    dw = explain_case(_row() | {"inner_symbol": "arm_depthwise_conv_s8_opt"}, cpu=M55)
    assert conv.ceiling["target_inst_per_mac_instr"] == 2.6
    assert dw.ceiling["target_inst_per_mac_instr"] == 5.9


def test_underfilled_vectors():
    assert "underfilled_vectors" in _rules(_row(MVE_INT_MAC_RETIRED=2000, INST_RETIRED=5000, MVE_INST_RETIRED=4000))


def test_predicated_share_counts_cycles():
    result = explain_case(_row(MVE_INT_MAC_RETIRED=2000, MVE_PRED=1000), cpu=M55)
    assert result.metrics["pred_cycle_share"] == pytest.approx(1000 / 4000)
    finding = next(f for f in result.findings if f.rule == "underfilled_vectors")
    assert "25% cycles predicated" in finding.diagnosis


def test_depthwise_s16_widens_like_s8():
    depthwise = load_ceilings()["cpus"][M55]["ops"]["depthwise"]
    assert depthwise["s16"]["cycles_per_mac"] == depthwise["s8"]["cycles_per_mac"] == 0.5
    assert depthwise["s16"]["lanes"] == depthwise["s8"]["lanes"] == 4


def test_scalar_macs_beats_overhead():
    rules = _rules(_row(MVE_INT_MAC_RETIRED=100))
    assert "scalar_macs" in rules
    assert "instruction_overhead" not in rules


def test_low_mve_share():
    assert "low_mve_share" in _rules(_row(MVE_INST_RETIRED=500))


def test_dependency_stall():
    assert _rules(_row(MVE_STALL_DEPENDENCY=800)) == ["dependency_stall"]


def test_frontend_stall():
    assert _rules(_row(STALL_FRONTEND=800)) == ["frontend_stall"]


def test_prepare_heavy():
    row = _row()
    row["prepare_cycles"] = 4000
    assert _rules(row) == ["prepare_heavy"]


def test_findings_rank_by_impact():
    row = _row(STALL_FRONTEND=400, MVE_STALL_DEPENDENCY=800)
    assert _rules(row) == ["dependency_stall", "frontend_stall"]


def test_float_uses_fp_mac_counter():
    row = _row(MVE_INT_MAC_RETIRED=0, MVE_FP_MAC_RETIRED=4000)
    row["inner_symbol"] = "arm_convolve_f32"
    assert explain_case(row, cpu=M55).metrics["mve_mul_ratio"] == pytest.approx(1.0)


def test_cycles_only_row_degrades():
    row = {"case_id": "c", "timed_symbol": "arm_convolve_wrapper_s8", "inner_symbol": "arm_convolve_s8",
           "median_cycles": "48000", "macs": "16000", "prepare_cycles": "", "ARM_PMU_CPU_CYCLES": "48000"}
    result = explain_case(row, cpu="cortex-m4")
    assert [f.rule for f in result.findings] == ["no_counters"]
    assert result.pct_of_peak == pytest.approx(25.0)
    assert "ARM_PMU_INST_RETIRED" in result.missing_counters
    json.dumps(result.to_dict())


def test_invalid_timing_skips_rules():
    row = _row(STALL_FRONTEND=800)
    row["timing_status"] = "below_floor"
    assert _rules(row) == ["timing_invalid"]


def test_partial_counters_still_run_rules():
    row = _row(MVE_STALL_DEPENDENCY=800)
    del row["counters"]["ARM_PMU_INST_RETIRED"]
    assert _rules(row) == ["dependency_stall"]


def test_csv_row_matches_nested_row():
    nested = _row(STALL_FRONTEND=800)
    flat = {key: str(value) for key, value in nested.items() if key != "counters"}
    flat.update({name: str(value) for name, value in nested["counters"].items()})
    assert explain_case(flat, cpu=M55).to_dict() == explain_case(nested, cpu=M55).to_dict()


def test_agent_selection_is_small_and_complete():
    passes = counter_passes_for_selection(AGENT_PMU_SELECTION)
    check_pass_count(passes)
    assert len(passes) == 4
    assert explain_case(_row(), cpu=M55).missing_counters == []


def test_ceilings_entries_are_complete():
    for cpu in load_ceilings()["cpus"].values():
        for dtypes in cpu["ops"].values():
            for entry in dtypes.values():
                assert entry["cycles_per_mac"] > 0 and entry["lanes"] >= 0 and entry["basis"]


def test_explain_cli_json(tmp_path):
    bundle = tmp_path / "hardware" / "s1"
    bundle.mkdir(parents=True)
    target = {"board": "apollo510_evb", "cpu": M55, "placement": {"name": "tcm"}}
    (bundle / "session_manifest.json").write_text(json.dumps({"target": target}))
    softmax = {"case_id": "softmax", "timed_symbol": "arm_softmax_s8", "median_cycles": 10, "macs": None}
    (bundle / "cases.json").write_text(json.dumps([_row(), softmax]))
    result = CliRunner().invoke(app, ["explain", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"] == SCHEMA_VERSION
    assert [case["case_id"] for case in data["cases"]] == ["c"]
    assert data["cases"][0]["bundle"] == str(bundle) and "cases" not in data["bundles"][0]
    result = CliRunner().invoke(app, ["explain", str(tmp_path), "--op", "softmax", "--all"])
    assert result.exit_code == 0 and "softmax" in result.output
    result = CliRunner().invoke(app, ["explain", str(tmp_path), "--op", "depthwise"])
    assert result.exit_code == 0 and "c:" not in result.output
