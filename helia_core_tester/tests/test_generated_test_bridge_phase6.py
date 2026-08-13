from __future__ import annotations

from pathlib import Path

from helia_core_tester.perf_stream.case_bundle import load_case_bundle
from helia_core_tester.perf_stream.generated_test_bridge import build_case_bundle_from_generated_test, discover_generated_tests
from helia_core_tester.perf_stream.kernel_registry import lookup_kernel_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, family: str, name_filter: str) -> dict[str, object]:
    cases = discover_generated_tests(PROJECT_ROOT, family=family, name_filter=name_filter)
    assert cases, f"expected a discoverable {family} test matching {name_filter!r}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path)
    return load_case_bundle(bundle.manifest_path).manifest


def test_convolve_s4_case_bridges_to_distinct_kernel_id(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ConvolutionFunctions", "convolve_even_mve_s4")
    assert manifest["tensor_dtypes"]["weights"] == "S4"
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT,
        family="ConvolutionFunctions",
        operator="Convolve",
        dtype="S8",
        weight_dtype="S4",
    )


def test_fully_connected_s4_case_bridges_without_scratch(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "FullyConnectedFunctions", "fully_connected_bias_s4")
    assert manifest["tensor_dtypes"]["weights"] == "S4"
    assert manifest["scratch_buffer"]["bytes"] == 0
    assert manifest["kernel_id"] == lookup_kernel_id(
        PROJECT_ROOT,
        family="FullyConnectedFunctions",
        operator="FullyConnected",
        dtype="S8",
        weight_dtype="S4",
    )


def test_prelu_batch_case_bridges_and_serializes_output_n(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ActivationFunctions", "prelu_broadcast_batch_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["output_n"] == 2
    assert scalars["output_h"] == 2
    assert scalars["output_w"] == 2
    assert scalars["output_c"] == 3


def test_prelu_scalar_multi_pixel_case_bridges(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "ActivationFunctions", "prelu_pixel_scalar_input_broadcast_c_s8")
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["block_size"] == 3
    assert scalars["output_h"] == 2
    assert scalars["output_c"] == 3
