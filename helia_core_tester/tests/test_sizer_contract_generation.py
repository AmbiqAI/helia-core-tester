"""Generation contract for the sizer_contract cases (issue #69).

The point of these cases is that the expected value comes from the header's
prose and not from a Python re-derivation of the kernel formula. That is a
property of the generator, not of the FVP run, so it is pinned here: every
probe must carry the header sentence it relies on into the emitted source, and
no probe may expect anything other than the two documented sentinels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.BufferSizeFunctions.sizer_contract import (
    _SIZERS,
    OpSizerContract,
)
from helia_core_tester.generation.test_ops import default_seed_for_case


def _shipped() -> list[dict]:
    return [
        desc
        for desc in load_all_descriptors(str(find_descriptors_dir()))
        if desc.get("operator") == "SizerContract"
    ]


def _emit(desc: dict, out_dir: Path) -> str:
    op = OpSizerContract(desc, default_seed_for_case(desc["name"]), target_cpu="cortex-m55")
    op.generate_c_files(out_dir)
    return (out_dir / f"{desc['name']}_sizer_contract.c").read_text()


def test_every_shipped_descriptor_generates(tmp_path: Path) -> None:
    descriptors = _shipped()
    assert descriptors
    for desc in descriptors:
        case_dir = tmp_path / desc["name"]
        case_dir.mkdir()
        source = _emit(desc, case_dir)
        assert f"int32_t {desc['name']}_test_case_run(void)" in source
        symbol = _SIZERS[desc["sizer"]]["symbol"]
        assert f"{symbol}(" in source
        # One assertion per declared probe, each naming the sizer under test.
        assert source.count("HELIA_VALIDATE_SCALAR_EQ_INT(") == len(desc["probes"])
        assert source.count(f'"sizer contract {symbol}"') == len(desc["probes"])


def test_the_corpus_covers_every_family_the_issue_names(tmp_path: Path) -> None:
    # #69 names the convolution, depthwise, fully-connected and pooling
    # families; #71 names the two SVDF staging queries. A descriptor set that
    # quietly lost one of them would still pass every other test here.
    covered = {desc["sizer"] for desc in _shipped()}
    assert covered == set(_SIZERS)


@pytest.mark.parametrize("kind,expected", [("negative_dim", -1), ("overflow", -1), ("degenerate_zero", 0)])
def test_probe_kinds_expect_only_the_documented_sentinels(kind: str, expected: int) -> None:
    # -1 and 0 are the only two values the headers state. Anything else would
    # be a value derived from the formula, which is what issue #69 rules out.
    for sizer in _SIZERS.values():
        probe = sizer["probes"].get(kind)
        if probe is not None:
            assert probe["expect"] == expected, sizer["symbol"]


def test_every_probe_carries_the_header_sentence_it_relies_on(tmp_path: Path) -> None:
    for desc in _shipped():
        case_dir = tmp_path / desc["name"]
        case_dir.mkdir()
        source = _emit(desc, case_dir)
        for kind in desc["probes"]:
            for line in _SIZERS[desc["sizer"]]["probes"][kind]["doc"]:
                assert f"// {line}" in source, (desc["name"], kind, line)


def test_an_unknown_sizer_or_probe_is_refused(tmp_path: Path) -> None:
    base = dict(_shipped()[0])
    with pytest.raises(ValueError, match="unknown sizer"):
        OpSizerContract({**base, "sizer": "not_a_sizer"}, 1).generate_c_files(tmp_path)
    with pytest.raises(ValueError, match="no 'not_a_probe' probe"):
        OpSizerContract({**base, "probes": ["not_a_probe"]}, 1).generate_c_files(tmp_path)
    with pytest.raises(ValueError, match="lists no probes"):
        OpSizerContract({**base, "probes": []}, 1).generate_c_files(tmp_path)
