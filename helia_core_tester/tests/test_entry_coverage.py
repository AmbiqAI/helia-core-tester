"""CMSIS-NN entry-point coverage in the hardware bundle."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

from helia_core_tester.hardware.entry_coverage import (
    DEPLOYED_PATH,
    NO_ADAPTER,
    build_coverage,
    coverage_line,
    coverage_totals,
    public_entry_points,
    write_coverage,
)
from helia_core_tester.hardware.kernel_registry import load_kernel_registry
from helia_core_tester.hardware.session_runner import build_generated_test_case_bundles, no_bridgeable_cases_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_HEADER = """
/* Mentions arm_relu6_s8() in a comment only. */
arm_cmsis_nn_status arm_convolve_s8(const cmsis_nn_context *ctx);
int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims);
arm_cmsis_nn_status arm_svdf_s8(const cmsis_nn_context *ctx);
"""


def _case(kernel_id: int, *, rejected: bool = False, samples: int = 1, capabilities=()) -> SimpleNamespace:
    return SimpleNamespace(
        case_bundle=SimpleNamespace(kernel_id=kernel_id, manifest={"required_target_capabilities": list(capabilities)}),
        rejection=object() if rejected else None,
        samples=[object()] * samples,
    )


def _kernel_id(name: str) -> int:
    return next(e.kernel_id for e in load_kernel_registry(PROJECT_ROOT) if e.cmsis_function == name)


def test_public_entry_points_drop_setup_calls_and_comments(tmp_path: Path) -> None:
    (tmp_path / "arm_nnfunctions.h").write_text(_HEADER, encoding="utf-8")
    (tmp_path / "arm_nn_types.h").write_text("void arm_ignored(void);", encoding="utf-8")
    assert public_entry_points(tmp_path) == ["arm_convolve_s8", "arm_svdf_s8"]


def test_deployed_data_file_is_pinned() -> None:
    data = json.loads((PROJECT_ROOT / DEPLOYED_PATH).read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", data["source"]["commit"])
    names = data["entry_points"]
    assert names == sorted(set(names)) and "arm_svdf_s8" in names
    assert not any(n.endswith("_get_buffer_size") for n in names)


def test_build_coverage_counts_only_unrejected_timed_kernels() -> None:
    conv, add = _kernel_id("arm_convolve_wrapper_s16"), _kernel_id("arm_add_s8")
    relu = _kernel_id("arm_relu_s8")
    concat = _kernel_id("arm_concatenation_f32_w")
    cases = [
        _case(conv, capabilities=["convolve_s16"]),
        _case(add, rejected=True, capabilities=["arm_add_s8"]),
        _case(relu, samples=0),
        _case(concat, capabilities=["arm_concatenation_f32_x"]),
    ]
    coverage = build_coverage(PROJECT_ROOT, cases, build_dir=None)

    assert coverage["timed_entry_points"] == [
        "arm_concatenation_f32_w", "arm_concatenation_f32_x", "arm_convolve_wrapper_s16",
    ]
    assert coverage["public"]["total"] is None
    deployed = coverage["deployed"]["entry_points"]
    assert coverage["deployed_entry_points_total"] == len(deployed)
    assert coverage["deployed_entry_points_timed"] == 3
    assert "arm_add_s8" in coverage["deployed"]["missing"]
    assert "arm_convolve_wrapper_s16" not in coverage["deployed"]["missing"]
    assert coverage_line(coverage) == f"Coverage: 3/{len(deployed)} deployed entry points timed"
    assert coverage_totals(None) is None


def test_build_coverage_reads_the_built_headers(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "arm_nnfunctions.h").write_text(_HEADER, encoding="utf-8")
    monkeypatch.setattr("helia_core_tester.hardware.entry_coverage.built_include_dir", lambda _bd: tmp_path)
    coverage = build_coverage(PROJECT_ROOT, [_case(_kernel_id("arm_convolve_s8"))], build_dir=tmp_path)
    assert coverage["public"]["timed"] == 1 and coverage["public"]["total"] == 2
    assert coverage_line(coverage).endswith(", 1/2 public")
    assert coverage_totals(coverage)["public_entry_points_total"] == 2


def test_write_coverage_extends_manifest_and_summary(tmp_path: Path) -> None:
    (tmp_path / "session_manifest.json").write_text(json.dumps({"artifacts": {"cases": "cases.json"}}), encoding="utf-8")
    (tmp_path / "session_summary.json").write_text(json.dumps({"case_count": 1}), encoding="utf-8")
    test = SimpleNamespace(name="lstm_a_s8", family="LSTMFunctions", suite="int")
    write_coverage(tmp_path, {"schema": "hct.hardware.coverage"}, [(test, f"{NO_ADAPTER}: x")])

    manifest = json.loads((tmp_path / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == {"cases": "cases.json", "coverage": "coverage.json"}
    summary = json.loads((tmp_path / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert summary["skipped_cases"] == [
        {"case_id": "lstm_a_s8", "family": "LSTMFunctions", "suite": "int", "reason": f"{NO_ADAPTER}: x"}
    ]
    assert json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))["schema"] == "hct.hardware.coverage"


def _write_case(root: Path, family: str, name: str) -> None:
    case = root / "artifacts" / "generated_tests" / "int" / "cortex-m55" / family / name
    case.mkdir(parents=True)
    (case / "descriptor.yaml").write_text(yaml.safe_dump({"operator": "LSTM", "activation_dtype": "S8"}), encoding="utf-8")


def test_unbridged_families_are_skipped_not_dropped(tmp_path: Path) -> None:
    _write_case(tmp_path, "LSTMFunctions", "lstm_one_s8")
    _write_case(tmp_path, "SVDFunctions", "svdf_one_s8")
    bundles, skipped = build_generated_test_case_bundles(tmp_path, family=None, board_id="b1")

    assert bundles == []
    assert [(t.name, t.board, r) for t, r in skipped] == [
        ("lstm_one_s8", "b1", f"{NO_ADAPTER}: LSTMFunctions has no firmware adapter"),
        ("svdf_one_s8", "b1", f"{NO_ADAPTER}: SVDFunctions has no firmware adapter"),
    ]
    # An explicit family keeps the old path.
    assert build_generated_test_case_bundles(tmp_path, family="ConvolutionFunctions") == ([], [])


def test_adapter_gaps_keep_the_fvp_hint() -> None:
    fvp = (SimpleNamespace(name="conv_a"), "FVP recorded a failure for these artifacts")
    gap = (SimpleNamespace(name="lstm_a"), f"{NO_ADAPTER}: LSTMFunctions has no firmware adapter")
    error = no_bridgeable_cases_error([fvp, gap], cpu="cortex-m55", family=None, name_filter=None, suite="int")
    assert "rejected by the FVP gate" in str(error)
