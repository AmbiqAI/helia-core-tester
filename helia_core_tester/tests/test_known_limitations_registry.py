"""Phase 4 of the generation/bridge unification plan: verify the declarative
known-limitations registry (helia_core_tester.perf_stream.known_limitations)
correctly gates case bridging centrally in
build_case_bundle_from_generated_test(), rather than via ad hoc
`if generated_test.name == "...":` checks buried in individual per-operator
builder functions.
"""

from pathlib import Path

import pytest

from helia_core_tester.perf_stream import generated_test_bridge as gtb
from helia_core_tester.perf_stream.known_limitations import lookup_known_limitation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_lookup_known_limitation_returns_none_for_unlisted_case():
    assert lookup_known_limitation("add_default_s8") is None


def test_lookup_known_limitation_returns_entry_for_grouped_conv_case():
    entry = lookup_known_limitation("convolve_grouped_conv_case_01_s8")
    assert entry is not None
    assert entry.case_name == "convolve_grouped_conv_case_01_s8"
    assert "grouped-convolution" in entry.reason


def test_bridge_raises_unsupported_for_known_limitation_case(tmp_path):
    """Even before dispatching to any per-operator builder, a case in the
    known-limitations registry must be rejected with a clear reason."""
    case_dir = (
        _PROJECT_ROOT
        / "artifacts/generated_tests/int/cortex-m55/ConvolutionFunctions/convolve_grouped_conv_case_01_s8"
    )
    if not case_dir.is_dir():
        pytest.skip(f"{case_dir} not present in this local artifacts tree (run generation first)")
    generated_test = gtb.discover_generated_tests(
        _PROJECT_ROOT, family="ConvolutionFunctions", name_filter="convolve_grouped_conv_case_01_s8"
    )
    assert len(generated_test) == 1
    with pytest.raises(gtb.UnsupportedGeneratedTestError, match="grouped-convolution"):
        gtb.build_case_bundle_from_generated_test(
            _PROJECT_ROOT,
            generated_test[0],
            output_root=tmp_path,
            require_fvp_pass=False,
        )
