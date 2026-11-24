"""Golden freeze: recorded outputs written into a generated case's goldens, and the
`--golden record|hct_ref` switch that runs it after generation."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.artifact_identity import generated_case_artifact_sha256
from helia_core_tester.generation.reference import host_kernels as hk
from helia_core_tester.generation.reuse import STAMP_FILENAME, case_reusable, case_stamp, read_stamp, write_stamp
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
from helia_core_tester.record import freeze, parity, runner
from helia_core_tester.record.capture import encode_capture, encode_tensor, read_capture
from helia_core_tester.tests.test_host_reference import _fake_tree
from helia_core_tester.tests.test_record_runner import _HEADER, _KERNEL, _record_case, needs_cmake


def _case_dir(root: Path, name: str = "case_a", family: str = "Fam") -> Path:
    case = root / family / name
    (case / "includes").mkdir(parents=True)
    (case / f"{name}.c").write_text(f'#include "includes/{name}.h"\nint main(void) {{ return 0; }}\n')
    return case


def _header(case: Path, ctype: str, values, symbol: str | None = None, dtype=None) -> Path:
    symbol = symbol or f"{case.name}_expected_output"
    array = np.asarray(values, dtype=dtype)
    body = TemplateContextBuilder.format_array_as_c_literal(array)
    path = case / "includes" / f"{case.name}.h"
    path.write_text(
        "#pragma once\n"
        f"static const int32_t {case.name}_shape[] = {{ 1, {array.size} }};\n\n"
        f"static const {ctype} {symbol}[] = {{\n{body}\n}};\n\n"
        "static const int dummy = 1;\n"
    )
    return path


def _tensor(symbol: str, actual, golden, dtype: str | None = None):
    record = encode_tensor(symbol, np.asarray(actual), np.asarray(golden), dtype=dtype) if dtype else encode_tensor(
        symbol, np.asarray(actual), np.asarray(golden)
    )
    return record


def _capture_tensors(tmp_path: Path, *records: bytes, name: str = "cap.hctr"):
    path = tmp_path / name
    path.write_bytes(encode_capture(list(records)))
    return path, read_capture(path).tensors


# --- find_golden -------------------------------------------------------------------------


def test_finds_the_single_definition_in_a_nested_header(tmp_path) -> None:
    case = _case_dir(tmp_path)
    path = _header(case, "int8_t", np.array([1, -2, 3], np.int8))
    site = freeze.find_golden(case, "case_a_expected_output")
    assert (site.path, site.ctype, site.values) == (path, "int8_t", ("1", "-2", "3"))


@pytest.mark.parametrize("symbol", ["", "1abc", "a b", "x[0]", "(x)y"])
def test_a_symbol_that_is_not_an_identifier_is_refused(tmp_path, symbol) -> None:
    with pytest.raises(freeze.FreezeError, match="not a plain C identifier"):
        freeze.find_golden(_case_dir(tmp_path), symbol)


def test_a_missing_golden_is_refused(tmp_path) -> None:
    case = _case_dir(tmp_path)
    _header(case, "int8_t", np.array([1], np.int8))
    with pytest.raises(freeze.FreezeError, match="no initialised definition"):
        freeze.find_golden(case, "case_a_other")


def test_a_declaration_without_initializer_is_not_a_definition(tmp_path) -> None:
    case = _case_dir(tmp_path)
    (case / "includes" / "decl.h").write_text("extern const int8_t case_a_expected_output[3];\n")
    with pytest.raises(freeze.FreezeError, match="no initialised definition"):
        freeze.find_golden(case, "case_a_expected_output")


def test_a_golden_defined_twice_is_refused(tmp_path) -> None:
    case = _case_dir(tmp_path)
    _header(case, "int8_t", np.array([1], np.int8))
    (case / "case_a.c").write_text("static const int8_t case_a_expected_output[1] = { 7 };\n")
    with pytest.raises(freeze.FreezeError, match="defined 2 times"):
        freeze.find_golden(case, "case_a_expected_output")


def test_a_longer_name_sharing_the_prefix_is_not_matched(tmp_path) -> None:
    case = _case_dir(tmp_path)
    _header(case, "int8_t", np.array([1], np.int8), symbol="case_a_expected_output_2")
    with pytest.raises(freeze.FreezeError, match="no initialised definition"):
        freeze.find_golden(case, "case_a_expected_output")


def test_a_struct_initializer_is_not_a_golden(tmp_path) -> None:
    case = _case_dir(tmp_path)
    (case / "includes" / "s.h").write_text("static const params case_a_p[] = { { 1, 2 } };\n")
    with pytest.raises(freeze.FreezeError, match="no initialised definition"):
        freeze.find_golden(case, "case_a_p")


# --- rewrite_golden ----------------------------------------------------------------------


def test_rewrites_only_the_golden_body(tmp_path) -> None:
    case = _case_dir(tmp_path)
    path = _header(case, "int8_t", np.zeros(20, np.int8))
    before = path.read_text()
    actual = np.arange(-10, 10, dtype=np.int8)
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", actual, np.zeros(20, np.int8)))
    assert freeze.rewrite_golden(case, tensor) == path
    after = path.read_text()
    assert after.split("static const int8_t")[0] == before.split("static const int8_t")[0]
    assert after.endswith("};\n\nstatic const int dummy = 1;\n")
    site = freeze.find_golden(case, "case_a_expected_output")
    assert [int(v) for v in site.values] == actual.tolist()


def test_bits_already_in_the_golden_leave_the_file_untouched(tmp_path) -> None:
    case = _case_dir(tmp_path)
    path = _header(case, "int16_t", np.array([5, -5], np.int16))
    before = path.read_bytes()
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", np.array([5, -5], np.int16),
                                                      np.array([5, -5], np.int16)))
    assert freeze.rewrite_golden(case, tensor) is None
    assert path.read_bytes() == before


def test_rewriting_back_restores_the_generated_bytes(tmp_path) -> None:
    case = _case_dir(tmp_path)
    original = np.array([0.5, -1.25e-7, 3.0e9, 0.0], np.float32)
    path = _header(case, "float", original)
    before = path.read_bytes()
    other = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    _, (there,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", other, original), name="a.hctr")
    _, (back,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", original, other), name="b.hctr")
    freeze.rewrite_golden(case, there)
    assert path.read_bytes() != before
    freeze.rewrite_golden(case, back)
    assert path.read_bytes() == before


def test_element_type_must_match_the_recorded_dtype(tmp_path) -> None:
    case = _case_dir(tmp_path)
    _header(case, "int16_t", np.array([1], np.int16))
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", np.array([2], np.int8), np.array([1], np.int8)))
    with pytest.raises(freeze.FreezeError, match="is int16_t but int8 was recorded"):
        freeze.rewrite_golden(case, tensor)


def test_an_unmapped_element_type_is_refused(tmp_path) -> None:
    case = _case_dir(tmp_path)
    _header(case, "my_q7", np.array([1], np.int8))
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", np.array([2], np.int8), np.array([1], np.int8)))
    with pytest.raises(freeze.FreezeError, match="cannot map"):
        freeze.rewrite_golden(case, tensor)


@pytest.mark.parametrize("held, recorded", [(3, 4), (4, 3), (1, 2)])
def test_element_count_must_match(tmp_path, held, recorded) -> None:
    case = _case_dir(tmp_path)
    _header(case, "int32_t", np.zeros(held, np.int32))
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", np.ones(recorded, np.int32),
                                                      np.zeros(recorded, np.int32)))
    with pytest.raises(freeze.FreezeError, match=f"holds {held} element\\(s\\), {recorded} were recorded"):
        freeze.rewrite_golden(case, tensor)


def test_a_zero_length_record_changes_nothing(tmp_path) -> None:
    case = _case_dir(tmp_path)
    path = _header(case, "int8_t", np.array([1], np.int8))
    before = path.read_bytes()
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", np.zeros(0, np.int8), np.zeros(0, np.int8)))
    assert freeze.rewrite_golden(case, tensor) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("dtype, ctype", [(np.uint8, "uint8_t"), (np.int32, "int32_t"), (np.uint32, "uint32_t"),
                                          (np.int64, "int64_t"), (np.uint16, "uint16_t"), (np.int16, "q15_t")])
def test_integer_limits_round_trip(tmp_path, dtype, ctype) -> None:
    info = np.iinfo(dtype)
    case = _case_dir(tmp_path)
    _header(case, ctype, np.zeros(3, dtype))
    actual = np.array([info.min, info.max, 0], dtype)
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", actual, np.zeros(3, dtype)))
    freeze.rewrite_golden(case, tensor)
    values = freeze.find_golden(case, "case_a_expected_output").values
    assert [freeze.parse_literal(v, np.dtype(dtype).name) for v in values] == actual.tolist()


def _f32(bits: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", bits), np.float32)[0]


def test_float32_specials_round_trip(tmp_path) -> None:
    case = _case_dir(tmp_path)
    actual = np.array([np.nan, -np.nan, np.inf, -np.inf, -0.0, _f32(1), _f32(0x00800000), np.finfo(np.float32).max,
                       _f32(0x7FC00123)], np.float32)
    _header(case, "float", np.zeros(actual.size, np.float32))
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", actual, np.zeros(actual.size, np.float32)))
    freeze.rewrite_golden(case, tensor)
    values = freeze.find_golden(case, "case_a_expected_output").values
    assert values[:5] == ("NAN", "-NAN", "INFINITY", "-INFINITY", "-0.0f")
    parsed = np.array([freeze.parse_literal(v, "float32") for v in values], np.float32)
    finite = ~np.isnan(actual)
    assert parsed[finite].tobytes() == actual[finite].tobytes()
    assert np.isnan(parsed[~finite]).all()
    assert np.signbit(parsed[~finite]).tolist() == np.signbit(actual[~finite]).tolist()


def test_float16_keeps_its_cast_and_round_trips(tmp_path) -> None:
    case = _case_dir(tmp_path)
    actual = np.array([1.0, -6.1e-5, 65504.0, np.float16(5.96e-8)], np.float16)
    _header(case, "float16_t", np.zeros(4, np.float16))
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", actual, np.zeros(4, np.float16)))
    freeze.rewrite_golden(case, tensor)
    values = freeze.find_golden(case, "case_a_expected_output").values
    assert all(v.startswith("(float16_t)") for v in values)
    assert np.array([freeze.parse_literal(v, "float16") for v in values], np.float16).tobytes() == actual.tobytes()


def test_bool_goldens_round_trip(tmp_path) -> None:
    case = _case_dir(tmp_path)
    _header(case, "bool", np.zeros(3, np.bool_))
    actual = np.array([1, 0, 1], np.uint8)
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", actual, np.zeros(3, np.uint8), dtype="bool"))
    freeze.rewrite_golden(case, tensor)
    assert freeze.find_golden(case, "case_a_expected_output").values == ("true", "false", "true")


def test_a_bool_byte_other_than_zero_or_one_freezes_as_true(tmp_path) -> None:
    case = _case_dir(tmp_path)
    _header(case, "bool", np.zeros(2, np.bool_))
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", np.array([2, 0], np.uint8),
                                                      np.zeros(2, np.uint8), dtype="bool"))
    freeze.rewrite_golden(case, tensor)
    assert freeze.find_golden(case, "case_a_expected_output").values == ("true", "false")


def test_a_literal_that_does_not_round_trip_is_refused(tmp_path, monkeypatch) -> None:
    case = _case_dir(tmp_path)
    path = _header(case, "int8_t", np.zeros(2, np.int8))
    before = path.read_bytes()
    monkeypatch.setattr(freeze, "format_values", lambda values, dtype: "    1, 1")
    _, (tensor,) = _capture_tensors(tmp_path, _tensor("case_a_expected_output", np.array([1, 2], np.int8), np.zeros(2, np.int8)))
    with pytest.raises(freeze.FreezeError, match="does not round-trip"):
        freeze.rewrite_golden(case, tensor)
    assert path.read_bytes() == before


@pytest.mark.parametrize("token, dtype", [("abc", "int8"), ("1.5", "int32"), ("NANx", "float32"), ("2", "bool"),
                                          ("", "float32")])
def test_unparseable_literals_are_refused(token, dtype) -> None:
    with pytest.raises(freeze.FreezeError):
        freeze.parse_literal(token, dtype)


@pytest.mark.parametrize("token, dtype, value", [("-5", "int8", -5), ("4294967295u", "uint32", 4294967295),
                                                 ("0x10", "int32", 16), ("1.5e-3f", "float32", 1.5e-3),
                                                 ("(float16_t)2.0f", "float16", 2.0), ("false", "bool", False)])
def test_literals_the_generator_writes_parse(token, dtype, value) -> None:
    assert freeze.parse_literal(token, dtype) == value


# --- freeze_tree -------------------------------------------------------------------------


def _record_report(tmp_path: Path, cases) -> runner.RecordReport:
    report = runner.RecordReport(str(tmp_path / "tree"), {"git": "abc"}, "key", "cc", "_Float16", [str(tmp_path)])
    report.cases = list(cases)
    return report


def _stamped(case: Path, stamp: str = "s" * 64) -> str:
    write_stamp(case, stamp)
    return stamp


def _recorded(case: Path, tmp_path: Path, actual, golden, cls=hk.STATUS_BLOCKING, verdict=parity.MISMATCH):
    path, _ = _capture_tensors(tmp_path, _tensor(f"{case.name}_expected_output", actual, golden), name=f"{case.name}.hctr")
    return runner.CaseRecord(case.parent.name, case.name, str(case), cls, runner.RECORDED, verdict=verdict,
                             capture_path=str(path))


def test_freezes_a_blocking_case_and_restamps_it(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1, 2], np.int8))
    stamp = _stamped(case)
    rec = _recorded(case, tmp_path, np.array([1, 3], np.int8), np.array([1, 2], np.int8))
    report = freeze.freeze_tree(_record_report(tmp_path, [rec]))
    assert [c.rewritten for c in report.cases] == [["case_a_expected_output"]]
    assert report.counts() == {"record:rewritten": 1}
    assert freeze.find_golden(case, "case_a_expected_output").values == ("1", "3")
    sidecar = json.loads((case / "case_a.golden.json").read_text())
    assert sidecar["schema"] == freeze.SIDECAR_SCHEMA and sidecar["source"] == freeze.SOURCE_RECORD
    assert sidecar["goldens"]["case_a_expected_output"]["count"] == 2
    assert read_stamp(case)["artifacts_sha256"] == generated_case_artifact_sha256(case)
    assert case_reusable(case, stamp)


def test_an_identical_case_is_unchanged_but_still_marked_recorded(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    path = _header(case, "int8_t", np.array([4], np.int8))
    before = path.read_bytes()
    _stamped(case)
    rec = _recorded(case, tmp_path, np.array([4], np.int8), np.array([4], np.int8), verdict=parity.IDENTICAL)
    report = freeze.freeze_tree(_record_report(tmp_path, [rec]))
    assert report.counts() == {"record:unchanged": 1}
    assert path.read_bytes() == before
    assert json.loads((case / "case_a.golden.json").read_text())["source"] == freeze.SOURCE_RECORD


def test_advisory_and_skipped_cases_keep_the_generated_golden(tmp_path) -> None:
    root = tmp_path / "cases"
    advisory, skipped, absent = _case_dir(root, "case_adv"), _case_dir(root, "case_skip"), _case_dir(root, "case_na")
    for case in (advisory, skipped, absent):
        _header(case, "int8_t", np.array([1], np.int8))
        _stamped(case)
    before = {c: (c / "includes" / f"{c.name}.h").read_bytes() for c in (advisory, skipped, absent)}
    cases = [
        _recorded(advisory, tmp_path, np.array([9], np.int8), np.array([1], np.int8), cls=hk.STATUS_ADVISORY),
        runner.CaseRecord("Fam", "case_skip", str(skipped), hk.STATUS_BLOCKING, runner.NOT_RECORDABLE,
                          detail="validated per element, not as an array: x"),
        runner.CaseRecord("Fam", "case_na", str(absent), hk.STATUS_NOT_APPLICABLE, runner.NOT_APPLICABLE),
    ]
    report = freeze.freeze_tree(_record_report(tmp_path, cases))
    assert {c: (c / "includes" / f"{c.name}.h").read_bytes() for c in before} == before
    assert json.loads((advisory / "case_adv.golden.json").read_text())["source"] == freeze.SOURCE_GENERATED
    assert "not_recordable" in json.loads((skipped / "case_skip.golden.json").read_text())["reason"]
    assert not (absent / "case_na.golden.json").exists()
    assert report.counts() == {"hct_ref": 3}


@pytest.mark.parametrize("outcome", [runner.CRASHED, runner.COMPILE_FAILED, runner.STATUS_MISMATCH, runner.HARNESS_FAILED,
                                     runner.NO_RECORDS, runner.RECORD_ERROR, runner.CAPTURE_ERROR, runner.TIMEOUT,
                                     runner.NO_SOURCE])
def test_a_blocking_case_that_failed_to_record_fails_the_freeze_before_any_write(tmp_path, outcome) -> None:
    root = tmp_path / "cases"
    good, bad = _case_dir(root, "case_good"), _case_dir(root, "case_bad")
    path = _header(good, "int8_t", np.array([1], np.int8))
    before = path.read_bytes()
    cases = [_recorded(good, tmp_path, np.array([2], np.int8), np.array([1], np.int8)),
             runner.CaseRecord("Fam", "case_bad", str(bad), hk.STATUS_BLOCKING, outcome, detail="d")]
    with pytest.raises(freeze.FreezeError, match=f"case_bad: {outcome}"):
        freeze.freeze_tree(_record_report(tmp_path, cases))
    assert path.read_bytes() == before and not (good / "case_good.golden.json").exists()


def test_an_advisory_case_that_failed_to_record_does_not_block(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1], np.int8))
    rec = runner.CaseRecord("Fam", "case_a", str(case), hk.STATUS_ADVISORY, runner.CRASHED, detail="exit -11")
    assert freeze.freeze_tree(_record_report(tmp_path, [rec])).counts() == {"hct_ref": 1}


def test_a_recorded_case_without_its_capture_is_refused(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1], np.int8))
    rec = runner.CaseRecord("Fam", "case_a", str(case), hk.STATUS_BLOCKING, runner.RECORDED, verdict=parity.IDENTICAL)
    with pytest.raises(freeze.FreezeError, match="capture was not kept"):
        freeze.freeze_tree(_record_report(tmp_path, [rec]))


def test_a_corrupt_capture_is_refused(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1], np.int8))
    rec = _recorded(case, tmp_path, np.array([2], np.int8), np.array([1], np.int8))
    raw = bytearray(Path(rec.capture_path).read_bytes())
    raw[-1] ^= 0xFF
    Path(rec.capture_path).write_bytes(bytes(raw))
    with pytest.raises(freeze.FreezeError, match="case_a"):
        freeze.freeze_tree(_record_report(tmp_path, [rec]))


def test_a_golden_recorded_twice_is_refused(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1], np.int8))
    path = tmp_path / "twice.hctr"
    record = _tensor("case_a_expected_output", np.array([2], np.int8), np.array([1], np.int8))
    path.write_bytes(encode_capture([record, record]))
    rec = runner.CaseRecord("Fam", "case_a", str(case), hk.STATUS_BLOCKING, runner.RECORDED, verdict=parity.MISMATCH,
                            capture_path=str(path))
    with pytest.raises(freeze.FreezeError, match="recorded twice"):
        freeze.freeze_tree(_record_report(tmp_path, [rec]))
    assert freeze.find_golden(case, "case_a_expected_output").values == ("1",)


def test_a_freeze_stopped_part_way_leaves_no_record_stamp_without_a_sidecar(tmp_path) -> None:
    root = tmp_path / "cases"
    first, second = _case_dir(root, "case_1"), _case_dir(root, "case_2")
    _header(first, "int8_t", np.array([1], np.int8))
    _header(second, "int8_t", np.array([1], np.int8), symbol="case_2_renamed")
    for case in (first, second):
        _stamped(case)
    cases = [_recorded(first, tmp_path, np.array([2], np.int8), np.array([1], np.int8)),
             _recorded(second, tmp_path, np.array([2], np.int8), np.array([1], np.int8))]
    with pytest.raises(freeze.FreezeError, match="no initialised definition"):
        freeze.freeze_tree(_record_report(tmp_path, cases))
    assert freeze.invalidate_unfrozen(root) == [second]
    assert (first / STAMP_FILENAME).exists() and not (second / STAMP_FILENAME).exists()


def test_a_stale_sidecar_is_dropped_before_its_case_is_refrozen(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1], np.int8), symbol="case_a_other")
    (case / "case_a.golden.json").write_text("{}\n")
    _stamped(case)
    rec = _recorded(case, tmp_path, np.array([2], np.int8), np.array([1], np.int8))
    with pytest.raises(freeze.FreezeError):
        freeze.freeze_tree(_record_report(tmp_path, [rec]))
    assert freeze.invalidate_unfrozen(tmp_path / "cases") == [case]


def test_report_json_names_the_library_and_every_case(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1], np.int8))
    rec = _recorded(case, tmp_path, np.array([2], np.int8), np.array([1], np.int8))
    report = freeze.freeze_tree(_record_report(tmp_path, [rec]))
    data = json.loads(freeze.write_report(report, tmp_path / "r" / freeze.REPORT_NAME).read_text())
    assert data["schema"] == freeze.REPORT_SCHEMA and data["library"]["tree_identity"] == {"git": "abc"}
    assert data["cases"][0]["verdict_against_generated"] == parity.MISMATCH


# --- stamps, config and the generate step --------------------------------------------------


def test_the_default_golden_keeps_existing_stamps() -> None:
    kw = dict(case_name="c", cpu="cortex-m0", suite="int", seed=1, version_hash="v")
    assert case_stamp({"name": "c"}, **kw) == case_stamp({"name": "c"}, golden="hct_ref", **kw)
    assert case_stamp({"name": "c"}, **kw) != case_stamp({"name": "c"}, golden="record", **kw)


def _config(tmp_path, **kw):
    from helia_core_tester.core.config import Config

    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    return Config(project_root=project, generation_dir=hk.tester_root() / "helia_core_tester" / "generation",
                  generated_tests_root=tmp_path / "g", reports_root=tmp_path / "r", **kw)


@pytest.mark.parametrize("value, expected", [("record", "record"), ("RECORD", "record"), (" hct_ref ", "hct_ref")])
def test_golden_config_is_normalised(tmp_path, value, expected) -> None:
    assert _config(tmp_path, golden=value).golden == expected


@pytest.mark.parametrize("value", ["", "fvp", "hct-ref"])
def test_an_unknown_golden_source_is_refused(tmp_path, value) -> None:
    from helia_core_tester.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="golden must be one of"):
        _config(tmp_path, golden=value)


def test_golden_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HELIA_CORE_TESTER_GOLDEN", "record")
    assert _config(tmp_path).golden == "record"


def test_generation_command_carries_golden_only_when_recording(tmp_path) -> None:
    from helia_core_tester.core.steps.generate import GenerateStep

    cmd = GenerateStep(_config(tmp_path, cpu="cortex-m0", seed=1))._build_cmd("cortex-m0", "int")
    assert "--golden" not in cmd
    cmd = GenerateStep(_config(tmp_path, cpu="cortex-m0", seed=1, golden="record"))._build_cmd("cortex-m0", "int")
    assert cmd[cmd.index("--golden") + 1] == "record"


def test_recording_needs_a_library_tree(tmp_path) -> None:
    from helia_core_tester.core.steps.generate import GenerateStep

    (tmp_path / "empty").mkdir()
    step = GenerateStep(_config(tmp_path, golden="record", cmsis_nn_root=tmp_path / "empty"))
    assert step._record_tree() == tmp_path / "empty"
    message = step.validate()
    assert message is not None and ("--golden record needs" in message or "ns-cmsis-nn" in message)


@pytest.mark.parametrize("fault", ["record", "freeze"])
def test_a_failed_freeze_fails_generation_and_drops_unfrozen_stamps(tmp_path, monkeypatch, fault) -> None:
    from helia_core_tester.core.steps import generate as generate_step
    from helia_core_tester.core.steps.base import StepStatus

    tree = tmp_path / "tree"
    (tree / "Source").mkdir(parents=True)
    config = _config(tmp_path, cpu="cortex-m0", suite="int", seed=1, golden="record", cmsis_nn_root=tree)
    step = generate_step.GenerateStep(config)
    case = _case_dir(step._cpu_generated_tests_dir("cortex-m0", "int"))
    _header(case, "int8_t", np.array([1], np.int8))
    _stamped(case)
    monkeypatch.setattr(generate_step, "run_command", lambda *a, **k: None)

    def fake_run_record(roots, tree_arg, jobs=None, work_dir=None):
        if fault == "record":
            raise runner.RecordError("no host reference build")
        return _record_report(tmp_path, [runner.CaseRecord("Fam", "case_a", str(case), hk.STATUS_BLOCKING,
                                                           runner.CRASHED, detail="exit -11")])

    monkeypatch.setattr(runner, "run_record", fake_run_record)
    result = step._do_execute()
    assert result.status == StepStatus.FAILED
    assert not (case / STAMP_FILENAME).exists()


def test_a_successful_freeze_reports_per_target(tmp_path, monkeypatch) -> None:
    from helia_core_tester.core.steps import generate as generate_step
    from helia_core_tester.core.steps.base import StepStatus

    tree = tmp_path / "tree"
    (tree / "Source").mkdir(parents=True)
    config = _config(tmp_path, cpu="cortex-m0", suite="int", seed=1, golden="record", cmsis_nn_root=tree)
    step = generate_step.GenerateStep(config)
    case = _case_dir(step._cpu_generated_tests_dir("cortex-m0", "int"))
    _header(case, "int8_t", np.array([1], np.int8))
    stamp = _stamped(case)
    monkeypatch.setattr(generate_step, "run_command", lambda *a, **k: None)
    monkeypatch.setattr(runner, "run_record", lambda roots, tree_arg, jobs=None, work_dir=None: _record_report(
        tmp_path, [_recorded(case, tmp_path, np.array([5], np.int8), np.array([1], np.int8))]))
    result = step._do_execute()
    assert result.status == StepStatus.SUCCESS and result.details["golden"] == "record"
    report = Path(result.outputs["golden_freeze_cortex-m0_int"])
    assert json.loads(report.read_text())["counts"] == {"record:rewritten": 1}
    assert freeze.find_golden(case, "case_a_expected_output").values == ("5",)
    assert case_reusable(case, stamp)


# --- end to end against a host reference build ---------------------------------------------


@needs_cmake
def test_frozen_goldens_record_identical(tmp_path) -> None:
    tree = _fake_tree(tmp_path, kernel=_KERNEL)
    (tree / "Include" / "arm_nnfunctions.h").write_text(_HEADER)
    cases = tmp_path / "cases"
    differs = _record_case(cases, "case_differs", 41)
    same = _record_case(cases, "case_same", 42)
    stamp = _stamped(differs)
    unchanged = (same / "case_same.c").read_bytes()
    first = runner.run_record([cases], tree, cache_root=tmp_path / "hk", work_dir=tmp_path / "w1", jobs=2)
    report = freeze.freeze_tree(first)
    assert report.counts() == {"record:rewritten": 1, "record:unchanged": 1}
    assert (same / "case_same.c").read_bytes() == unchanged
    assert case_reusable(differs, stamp)
    again = runner.run_record([cases], tree, cache_root=tmp_path / "hk", work_dir=tmp_path / "w2", jobs=2)
    assert again.totals() == {parity.IDENTICAL: 2} and again.ok


# --- golden expressions ------------------------------------------------------------------


@pytest.mark.parametrize("expression, symbol", [("&x[0]", "x"), ("& x [ 0 ]", "x"), ("x", "x"), ("&x[1]", "&x[1]"),
                                                ("x.body", "x.body")])
def test_golden_symbol(expression, symbol) -> None:
    assert freeze.golden_symbol(expression) == symbol


def test_the_address_of_the_first_element_freezes_the_array(tmp_path) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int16_t", np.array([1, 2], np.int16), symbol="lstm_out")
    path, _ = _capture_tensors(tmp_path, _tensor("&lstm_out[0]", np.array([3, 4], np.int16), np.array([1, 2], np.int16)))
    rec = runner.CaseRecord("Fam", "case_a", str(case), hk.STATUS_BLOCKING, runner.RECORDED, verdict=parity.MISMATCH,
                            capture_path=str(path))
    report = freeze.freeze_tree(_record_report(tmp_path, [rec]))
    assert report.cases[0].rewritten == ["lstm_out"]
    assert freeze.find_golden(case, "lstm_out").values == ("3", "4")
    assert list(json.loads((case / "case_a.golden.json").read_text())["goldens"]) == ["lstm_out"]


_GUARD = (
    "static struct __attribute__((aligned(16))) {\n"
    "    uint8_t head[16];\n    int16_t body[2];\n    uint8_t tail[16];\n"
    "} case_a_full_guard;\n"
)


@pytest.mark.parametrize("source, expression", [(_GUARD, "case_a_full_guard.body"),
                                                ("static int16_t case_a_full[2];\n", "case_a_full"),
                                                ("int16_t case_a_full[CASE_A_COUNT];\n", "case_a_full")])
def test_a_buffer_the_case_fills_at_run_time_is_not_frozen(tmp_path, source, expression) -> None:
    case = _case_dir(tmp_path / "cases")
    path = _header(case, "int8_t", np.array([1], np.int8))
    (case / "case_a.c").write_text(source)
    before = path.read_bytes()
    capture, _ = _capture_tensors(tmp_path, _tensor(expression, np.array([3, 4], np.int16), np.array([3, 4], np.int16)))
    rec = runner.CaseRecord("Fam", "case_a", str(case), hk.STATUS_BLOCKING, runner.RECORDED, verdict=parity.IDENTICAL,
                            capture_path=str(capture))
    report = freeze.freeze_tree(_record_report(tmp_path, [rec]))
    assert report.cases[0].runtime == [expression] and report.cases[0].rewritten == []
    assert path.read_bytes() == before
    sidecar = json.loads((case / "case_a.golden.json").read_text())
    assert sidecar["runtime_goldens"] == [expression] and sidecar["goldens"] == {}


@pytest.mark.parametrize("source, expression", [
    ("static const struct { int16_t body[2]; } case_a_full_guard = { { 1, 2 } };\n", "case_a_full_guard.body"),
    ("static const int16_t case_a_full[2];\n", "case_a_full"),
    ("extern int16_t case_a_full[2];\n", "case_a_full"),
    ("", "case_a_full"),
    ("", "case_a_full + 1"),
    ("", "&case_a_full[1]"),
])
def test_an_expression_that_names_no_golden_or_runtime_buffer_is_refused(tmp_path, source, expression) -> None:
    case = _case_dir(tmp_path / "cases")
    _header(case, "int8_t", np.array([1], np.int8))
    (case / "case_a.c").write_text(source)
    assert not freeze.is_runtime_buffer(case, expression)
    capture, _ = _capture_tensors(tmp_path, _tensor(expression, np.array([3, 4], np.int16), np.array([1, 2], np.int16)))
    rec = runner.CaseRecord("Fam", "case_a", str(case), hk.STATUS_BLOCKING, runner.RECORDED, verdict=parity.MISMATCH,
                            capture_path=str(capture))
    with pytest.raises(freeze.FreezeError):
        freeze.freeze_tree(_record_report(tmp_path, [rec]))
