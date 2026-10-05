"""Hardware case selection by op, dtype and id."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from helia_core_tester.generation.test_ops import should_run_test
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.cli import _read_case_ids
from helia_core_tester.hardware.generated_test_bridge import CaseSelection, discover_generated_tests
from helia_core_tester.hardware.hardware_pipeline import StreamOptions, resolved_selection

FAMILY = "ConvolutionFunctions"
DESCRIPTORS = {
    "depthwise_conv_a_s8": {"operator": "DepthwiseConv", "resolved_tensor_dtypes": {"input": "S8", "output": "S8", "weights": "S8"}},
    "depthwise_conv_b_s16": {"operator": "DepthwiseConv", "resolved_tensor_dtypes": {"input": "S16", "output": "S16", "weights": "S8"}},
    "depthwise_conv_c_s4": {"operator": "DepthwiseConv", "resolved_tensor_dtypes": {"input": "S8", "output": "S8", "weights": "S4"}},
    "convolve_d_s16": {"operator": "Convolve", "_source_stem": "convolve", "resolved_tensor_dtypes": {"input": "S16", "output": "S16", "weights": "S8"}},
}


def _tree(root: Path) -> Path:
    for name, desc in DESCRIPTORS.items():
        case = root / "artifacts" / "generated_tests" / "int" / "cortex-m55" / FAMILY / name
        case.mkdir(parents=True)
        (case / "descriptor.yaml").write_text(yaml.safe_dump({"name": name, **desc}))
    return root


def _names(root: Path, select: CaseSelection, **kwargs) -> list[str]:
    return [c.name for c in discover_generated_tests(root, family=FAMILY, select=select, **kwargs)]


def test_ops_and_dtypes_or_within_and_across(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    assert _names(root, CaseSelection(ops=("DepthwiseConv",), dtypes=("S16",))) == ["depthwise_conv_b_s16"]
    assert _names(root, CaseSelection(dtypes=("S4", "s16"))) == ["convolve_d_s16", "depthwise_conv_b_s16", "depthwise_conv_c_s4"]
    assert _names(root, CaseSelection(ops=("convolve", "depthwise_conv_a"))) == ["convolve_d_s16", "depthwise_conv_a_s8"]
    assert _names(root, CaseSelection()) == sorted(DESCRIPTORS)


def test_selection_applies_before_limit(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    assert _names(root, CaseSelection(ops=("DepthwiseConv",)), limit=1) == ["depthwise_conv_a_s8"]
    assert _names(root, CaseSelection(dtypes=("S4",)), limit=1) == ["depthwise_conv_c_s4"]


def test_case_ids_match_name_or_hw_id(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    select = CaseSelection(case_ids=("depthwise_conv_a_s8", "convolve_d_s16_hw_generated", "nope"))

    assert _names(root, select) == ["convolve_d_s16", "depthwise_conv_a_s8"]
    assert select.unmatched_ids(["convolve_d_s16", "depthwise_conv_a_s8"]) == ["nope"]


def test_op_and_dtype_match_generate(tmp_path: Path) -> None:
    for name, desc in DESCRIPTORS.items():
        desc = {"name": name, **desc}
        for op, dtype in [("DepthwiseConv", "S8"), ("convolve", "S16"), ("depthwise_conv_c", None)]:
            filters = {"op": op, "dtype": dtype}
            picked = CaseSelection(ops=(op,), dtypes=(dtype,) if dtype else ()).matches(name, desc)
            assert picked == should_run_test(desc, filters), (name, op, dtype)


def test_read_case_ids_joins_flag_and_file(tmp_path: Path) -> None:
    listing = tmp_path / "ids.txt"
    listing.write_text("# rerun\na_hw_generated\n\n  b  \na_hw_generated\n")

    assert _read_case_ids(["c", "b"], listing) == ("c", "b", "a_hw_generated", "b", "a_hw_generated")
    assert CaseSelection(case_ids=_read_case_ids(["c"], listing)).case_ids == ("c", "a", "b")
    assert _read_case_ids(None, None) == ()


def test_bad_dtype_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="int8"):
        CaseSelection(dtypes=("int8",))


def test_selection_records_case_filters() -> None:
    options = StreamOptions(ops=("DepthwiseConv",), dtypes=("S8",), case_ids=("x",))
    selection = resolved_selection(Path("."), resolve_board("apollo510_evb"), options)

    assert (selection["ops"], selection["dtypes"], selection["case_ids"]) == (["DepthwiseConv"], ["S8"], ["x"])


def test_empty_filter_result_names_the_filters(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", lambda *a, **k: ([], []))
    build_dir = tmp_path / "bd"
    build_dir.mkdir()
    (build_dir / "hct_build_id.txt").write_text("id\n")
    options = StreamOptions(ops=("Depthwise",))
    with pytest.raises(RuntimeError, match="--op/--dtype/--case-id"):
        hardware_pipeline.stream_generated_tests(
            tmp_path, resolve_board("apollo510_evb"), 5, build_dir=build_dir, options=options, echo=lambda _: None,
        )


def test_limit_with_case_ids_refused(capsys) -> None:
    from helia_core_tester.hardware.cli import _stream_options

    with pytest.raises(SystemExit):
        _stream_options(None, "int", None, None, 1, None, None, None, None, None, None, None, ["a"], None)
    assert "--limit cannot combine" in capsys.readouterr().err


def test_empty_cases_from_refused(tmp_path: Path, capsys) -> None:
    listing = tmp_path / "ids.txt"
    listing.write_text("# only a comment\n\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _read_case_ids(None, listing)
    assert "lists no case ids" in capsys.readouterr().err
