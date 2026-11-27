"""Record runner: each host run and its capture become one outcome, and only the
library's own host reference build may supply recorded values."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.reference import host_kernels as hk
from helia_core_tester.mutation import host_build
from helia_core_tester.record import parity, runner
from helia_core_tester.record.capture import encode_capture, encode_status, encode_tensor, encode_unrecordable
from helia_core_tester.tests.test_host_reference import _fake_tree
from helia_core_tester.tests.library_stub import NN_STATUS_ENUM

needs_cmake = pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("cc") is None, reason="cmake and a host cc required"
)


def _case() -> runner.CaseRecord:
    return runner.CaseRecord("Fam", "case", "/cases/Fam/case", hk.STATUS_BLOCKING, runner.RECORDED)


def _result(kind=host_build.KIND_PASS, exit_code=0, detail="", stderr="") -> host_build.CaseResult:
    return host_build.CaseResult("case", "Fam", kind == host_build.KIND_PASS, detail, kind=kind, exit_code=exit_code,
                                 stderr=stderr)


def _write(path: Path, *records: bytes) -> Path:
    path.write_bytes(encode_capture(list(records)))
    return path


def _s8(actual, golden) -> bytes:
    return encode_tensor("x_expected_output", np.array(actual, np.int8), np.array(golden, np.int8))


@pytest.mark.parametrize(
    "result, outcome",
    [
        (_result(host_build.KIND_COMPILE_FAILED, None, "error: x"), runner.COMPILE_FAILED),
        (_result(host_build.KIND_TIMEOUT, None, "timeout after 60s"), runner.TIMEOUT),
        (_result(host_build.KIND_NO_SOURCE, None, "no case source found"), runner.NO_SOURCE),
        (_result(host_build.KIND_CASE_FAIL, 3, "", "HCT_RECORD_ERROR: short write"), runner.RECORD_ERROR),
        (_result(host_build.KIND_CASE_FAIL, -11, "last line"), runner.CRASHED),
        (_result(host_build.KIND_CASE_FAIL, 2, "last line"), runner.CRASHED),
    ],
)
def test_runs_that_never_produce_a_capture(result, outcome, tmp_path) -> None:
    case = runner.judge(_case(), result, tmp_path / "none.hctr", "_Float16")
    assert case.outcome == outcome and case.verdict is None and case.detail


def test_record_error_reports_the_writer_message(tmp_path) -> None:
    result = _result(host_build.KIND_CASE_FAIL, 3, "", "HCT_RECORD_ERROR: HCT_RECORD_PATH is not set\n")
    assert runner.judge(_case(), result, tmp_path / "x", "_Float16").detail == "HCT_RECORD_ERROR: HCT_RECORD_PATH is not set"


def test_clean_exit_without_a_capture_is_a_capture_error(tmp_path) -> None:
    case = runner.judge(_case(), _result(), tmp_path / "none.hctr", "_Float16")
    assert case.outcome == runner.CAPTURE_ERROR and "does not exist" in case.detail


def test_truncated_capture_is_a_capture_error(tmp_path) -> None:
    path = tmp_path / "c.hctr"
    path.write_bytes(encode_capture([_s8([1], [1])])[:-3])
    assert runner.judge(_case(), _result(), path, "_Float16").outcome == runner.CAPTURE_ERROR


def test_identical_and_mismatched_recordings(tmp_path) -> None:
    same = runner.judge(_case(), _result(), _write(tmp_path / "a", _s8([1, 2], [1, 2]), encode_status("K", 0)), "_Float16")
    assert (same.outcome, same.verdict) == (runner.RECORDED, parity.IDENTICAL)
    assert same.statuses == [{"label": "K", "status": 0, "expected": 0}]
    wrong = runner.judge(_case(), _result(), _write(tmp_path / "b", _s8([1, 3], [1, 2])), "_Float16")
    assert (wrong.outcome, wrong.verdict) == (runner.RECORDED, parity.MISMATCH)
    assert wrong.tensors[0]["failing"] == 1


def test_status_disagreement_wins_over_the_exit_code(tmp_path) -> None:
    path = _write(tmp_path / "c", encode_status("K", -1, 0))
    case = runner.judge(_case(), _result(host_build.KIND_CASE_FAIL, 1, "Kernel status failure"), path, "_Float16")
    assert case.outcome == runner.STATUS_MISMATCH and "returned -1, the case expects 0" in case.detail


def test_an_expected_decline_is_recorded(tmp_path) -> None:
    case = runner.judge(_case(), _result(), _write(tmp_path / "c", encode_status("K", -2, -2)), "_Float16")
    assert (case.outcome, case.verdict) == (runner.RECORDED, parity.IDENTICAL)


def test_harness_failure_with_a_valid_capture(tmp_path) -> None:
    path = _write(tmp_path / "c", _s8([1], [1]), encode_status("K", 0))
    case = runner.judge(_case(), _result(host_build.KIND_CASE_FAIL, 1, "guard breached"), path, "_Float16")
    assert case.outcome == runner.HARNESS_FAILED and case.detail == "guard breached"
    assert case.tensors and case.verdict is None


def test_per_element_validation_is_not_recordable(tmp_path) -> None:
    path = _write(tmp_path / "c", _s8([1], [1]), encode_unrecordable("x_expected_bits[i]"))
    case = runner.judge(_case(), _result(), path, "_Float16")
    assert case.outcome == runner.NOT_RECORDABLE and case.unrecordable == ["x_expected_bits[i]"]


def test_a_capture_with_nothing_in_it(tmp_path) -> None:
    assert runner.judge(_case(), _result(), _write(tmp_path / "c"), "_Float16").outcome == runner.NO_RECORDS


def _f16(path: Path) -> Path:
    return _write(path, encode_tensor("h_expected_output", np.array([1.5], np.float16), np.array([1.5], np.float16)))


def test_f16_is_refused_unless_the_build_uses_float16(tmp_path) -> None:
    refused = runner.judge(_case(), _result(), _f16(tmp_path / "a"), "__fp16")
    assert refused.outcome == runner.REFUSED_F16 and "h_expected_output" in refused.detail
    accepted = runner.judge(_case(), _result(), _f16(tmp_path / "b"), "_Float16")
    assert (accepted.outcome, accepted.verdict) == (runner.RECORDED, parity.IDENTICAL)


def test_report_verdict_and_summaries() -> None:
    def case(family, verdict=None, outcome=runner.RECORDED, cls=hk.STATUS_BLOCKING):
        return runner.CaseRecord(family, "c", "/c", cls, outcome, verdict=verdict)

    report = runner.RecordReport("/t", {}, "k", "cc", "_Float16", ["/r"])
    assert not report.ok
    report.cases = [case("A", parity.IDENTICAL), case("A", outcome=runner.NOT_RECORDABLE),
                    case("B", parity.MISMATCH, cls=hk.STATUS_ADVISORY)]
    assert report.ok
    assert report.families() == {"A": {"identical": 1, "not_recordable": 1}, "B": {"mismatch": 1}}
    report.cases.append(case("B", parity.WITHIN_TOLERANCE))
    assert not report.ok
    report.cases.append(case("C", outcome=runner.CRASHED))
    assert report.totals()["crashed"] == 1
    assert json.loads(json.dumps(report.to_json()))["schema"] == runner.REPORT_SCHEMA


_KERNEL = "int32_t hct_kernel(void) { return HCT_FAKE_REF * 42; }"
_HEADER = (
    "#ifndef ARM_NNFUNCTIONS_H\n#define ARM_NNFUNCTIONS_H\n"
    + NN_STATUS_ENUM
    + "#endif\n"
)


def _record_case(root: Path, name: str, golden: int, body: str = "") -> Path:
    case = root / "Fam" / name
    (case / "includes").mkdir(parents=True)
    (case / f"{name}.c").write_text(
        "#include <stdlib.h>\n"
        '#include "test_runtime/helia_test_runtime.h"\n'
        "int32_t hct_kernel(void);\n"
        f"static const int32_t {name}_expected_output[] = {{{golden}}};\n"
        "static const uint32_t bits[] = {1u};\n"
        "static int32_t run(void) {\n"
        "    int failures = 0;\n"
        "    int32_t out[1] = {hct_kernel()};\n"
        '    HELIA_VALIDATE_STATUS("K", ARM_CMSIS_NN_SUCCESS);\n'
        f"    HELIA_VALIDATE_OUTPUTS(EXACT_INT, out, {name}_expected_output, 1, 0, 0.0f, 0.0f, 20, failures);\n"
        f"{body}"
        "    HELIA_VALIDATE_RETURN_FAILURES(failures);\n"
        "}\n"
        "int main(void) { helia_test_finish(run()); }\n"
    )
    return case


@needs_cmake
def test_end_to_end_against_a_host_reference_build(tmp_path) -> None:
    tree = _fake_tree(tmp_path, kernel=_KERNEL)
    (tree / "Include" / "arm_nnfunctions.h").write_text(_HEADER)
    cases = tmp_path / "cases"
    _record_case(cases, "case_same", 42)
    _record_case(cases, "case_differs", 41)
    _record_case(cases, "case_bits", 42, "    HELIA_VALIDATE_FLOAT_BITS(bits[0], bits[0], 0x7f800000u, 0, 0, 20, failures);\n")
    _record_case(cases, "case_dies", 42, "    abort();\n")
    report = runner.run_record([cases], tree, cache_root=tmp_path / "hk", work_dir=tmp_path / "work", jobs=2)
    by_name = {c.name: c for c in report.cases}
    assert report.float16_type == "_Float16"
    assert (by_name["case_same"].outcome, by_name["case_same"].verdict) == (runner.RECORDED, parity.IDENTICAL)
    assert by_name["case_differs"].verdict == parity.MISMATCH
    assert by_name["case_differs"].tensors[0]["symbol"] == "case_differs_expected_output"
    assert by_name["case_bits"].outcome == runner.NOT_RECORDABLE
    assert by_name["case_dies"].outcome in (runner.CRASHED, runner.CAPTURE_ERROR)
    assert not report.ok


def test_a_tree_without_a_host_reference_is_refused(tmp_path) -> None:
    tree = _fake_tree(tmp_path, kernel="int32_t hct_kernel(void) { return 42; }")
    (tree / "CMakeLists.txt").unlink()
    cases = tmp_path / "cases"
    _record_case(cases, "case_same", 42)
    with pytest.raises(runner.RecordError, match="no host reference build"):
        runner.run_record([cases], tree, cache_root=tmp_path / "hk", work_dir=tmp_path / "work")


def test_missing_cases_root_is_refused(tmp_path) -> None:
    with pytest.raises(hk.HostCheckError, match="generated cases root not found"):
        runner.run_record([tmp_path / "none"], tmp_path)


def test_step_refuses_a_tree_without_sources(tmp_path) -> None:
    from helia_core_tester.core.config import Config
    from helia_core_tester.core.steps import RecordStep
    from helia_core_tester.core.steps.base import StepStatus

    (tmp_path / "empty").mkdir()
    config = Config(project_root=hk.tester_root(), generated_tests_root=tmp_path / "g", reports_root=tmp_path / "r",
                    cmsis_nn_root=tmp_path / "empty")
    result = RecordStep(config).execute()
    assert result.status == StepStatus.FAILED and "record needs an ns-cmsis-nn checkout" in result.message


@pytest.mark.parametrize("ok", [True, False])
def test_step_writes_one_report_per_target_and_fails_on_differences(tmp_path, monkeypatch, ok) -> None:
    from helia_core_tester.core.config import Config
    from helia_core_tester.core.steps import RecordStep, record as record_step
    from helia_core_tester.core.steps.base import StepStatus

    tree = tmp_path / "tree"
    (tree / "Source").mkdir(parents=True)
    config = Config(project_root=hk.tester_root(), generated_tests_root=tmp_path / "g", reports_root=tmp_path / "r",
                    cmsis_nn_root=tree, cpu="cortex-m0", suite="both")
    seen = []

    def fake_run_record(roots, tree_arg, jobs=None):
        seen.append(Path(roots[0]))
        report = runner.RecordReport(str(tree_arg), {}, "k", "cc", "_Float16", [str(roots[0])])
        verdict = parity.IDENTICAL if ok else parity.MISMATCH
        report.cases = [runner.CaseRecord("Fam", "case_a", "/c", hk.STATUS_BLOCKING, runner.RECORDED, verdict=verdict)]
        return report

    monkeypatch.setattr(runner, "run_record", fake_run_record)
    monkeypatch.setattr(RecordStep, "_report_path", lambda self, cpu, suite: tmp_path / "out" / suite / record_step.REPORT_NAME)
    result = RecordStep(config).execute()
    assert len(seen) == 2
    assert result.status == (StepStatus.SUCCESS if ok else StepStatus.FAILED)
    for suite in ("int", "float"):
        data = json.loads((tmp_path / "out" / suite / record_step.REPORT_NAME).read_text())
        assert data["ok"] is ok and data["families"] == {"Fam": {"identical" if ok else "mismatch": 1}}
