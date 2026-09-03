"""Phase 4 of the generation/bridge unification plan: verify the declarative
known-limitations registry (helia_core_tester.perf_stream.known_limitations)
correctly gates case bridging centrally in
build_case_bundle_from_generated_test(), rather than via ad hoc
`if generated_test.name == "...":` checks buried in individual per-operator
builder functions.

Note: convolve_grouped_conv_case_01_s8 (the case that originally motivated
this registry) was removed once its underlying tolerance-policy gap was
fixed -- see dtypes.py's Convolve override.
"""

from pathlib import Path

import pytest

from helia_core_tester.perf_stream import generated_test_bridge as gtb
from helia_core_tester.perf_stream import known_limitations
from helia_core_tester.perf_stream.known_limitations import KnownLimitation, lookup_known_limitation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_lookup_known_limitation_returns_none_for_unlisted_case():
    assert lookup_known_limitation("add_default_s8") is None


def test_convolve_grouped_conv_case_no_longer_a_known_limitation():
    """Regression guard: must not be silently re-added as a workaround."""
    assert lookup_known_limitation("convolve_grouped_conv_case_01_s8") is None


@pytest.mark.parametrize("case_name", [
    "batch_matmul_float_mve_scalar_both_lhs_stride_f16",
    "batch_matmul_float_mve_scalar_both_rhs_stride_f16",
    "batch_matmul_float_mve_scalar_lhs_stride_f16",
    "batch_matmul_float_mve_scalar_rhs_stride_f16",
])
def test_large_batch_matmul_cases_no_longer_have_arena_limitations(case_name: str):
    assert lookup_known_limitation(case_name) is None


@pytest.mark.parametrize("case_name", [
    "batch_matmul_float_mve_scalar_both_lhs_stride_f16",
    "batch_matmul_float_mve_scalar_both_rhs_stride_f16",
    "batch_matmul_float_mve_scalar_lhs_stride_f16",
    "batch_matmul_float_mve_scalar_rhs_stride_f16",
])
def test_large_batch_matmul_cases_fit_default_workspace(case_name: str, tmp_path: Path):
    cases = gtb.discover_generated_tests(
        _PROJECT_ROOT,
        suite="float",
        family="FullyConnectedFunctions",
        name_filter=case_name,
    )
    if not cases:
        pytest.skip(f"generated case {case_name} is not present")
    bundle = gtb.build_case_bundle_from_generated_test(
        _PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False
    )
    assert 49152 < bundle.workspace_bytes_required <= 114688


def test_bridge_raises_unsupported_for_known_limitation_case(tmp_path, monkeypatch):
    """A case in the known-limitations registry must be rejected before
    dispatching to any per-operator builder. Uses a synthetic entry so this
    test validates the mechanism, not a specific real case."""
    fake_case_name = "add_default_s8"
    monkeypatch.setitem(
        known_limitations._KNOWN_LIMITATIONS,
        fake_case_name,
        KnownLimitation(case_name=fake_case_name, reason="synthetic test-only limitation"),
    )
    case_dir = _PROJECT_ROOT / "artifacts/generated_tests/int/cortex-m55/BasicMathFunctions" / fake_case_name
    if not case_dir.is_dir():
        pytest.skip(f"{case_dir} not present in this local artifacts tree (run generation first)")
    generated_test = gtb.discover_generated_tests(
        _PROJECT_ROOT, cpu="cortex-m55", family="BasicMathFunctions", name_filter=fake_case_name
    )
    assert len(generated_test) == 1
    with pytest.raises(gtb.UnsupportedGeneratedTestError, match="synthetic test-only limitation"):
        gtb.build_case_bundle_from_generated_test(
            _PROJECT_ROOT,
            generated_test[0],
            output_root=tmp_path,
            require_fvp_pass=False,
        )
