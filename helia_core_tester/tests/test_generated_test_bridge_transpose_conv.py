from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.hardware.generated_test_bridge import (
    UnsupportedGeneratedTestError,
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from helia_core_tester.tests.generated_inputs import discover_or_skip

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bridge(tmp_path: Path, test_name: str) -> dict:
    cases = discover_or_skip(PROJECT_ROOT, family="ConvolutionFunctions", name_filter=test_name)
    assert cases, f"expected discoverable generated test {test_name}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    return bundle.manifest


def test_transpose_conv_no_bias_case_omits_bias_tensor_dtype(tmp_path: Path) -> None:
    manifest = _bridge(tmp_path, "transpose_conv_reverse_valid_kernel1x1_stride2x2_no_bias_s8")
    assert manifest["tensor_dtypes"] == {"input": "S8", "weights": "S8", "output": "S8"}
    assert manifest["required_target_capabilities"] == ["arm_transpose_conv_wrapper_s8"]
    assert manifest["scratch_buffer"]["bytes"] > 0


def _generated_bundle(tmp_path: Path, desc: dict) -> dict:
    generate_test(desc, str(tmp_path / "artifacts" / "generated_tests" / "int" / "cortex-m55"), seed=Config.seed)
    (case,) = discover_generated_tests(tmp_path, family="ConvolutionFunctions", name_filter=desc["name"])
    return build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "bundle",
                                                 require_fvp_pass=False).manifest


def test_transpose_conv_bias_case_extracts_padding_offsets(tmp_path: Path) -> None:
    desc = next(d for d in load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors"))
                if d["name"] == "transpose_conv_same_kernel6x6_stride2x2_bias_s8")
    assert desc["input_shape"][0] == 2
    with pytest.raises(UnsupportedGeneratedTestError, match="batch size > 1"):
        _generated_bundle(tmp_path / "b2", desc)
    single = deepcopy(desc)
    single["input_shape"][0] = 1
    manifest = _generated_bundle(tmp_path / "b1", single)
    scalars = manifest["serialized_scalar_parameters"]
    assert scalars["pad_offset_h"] == 0
    assert scalars["pad_offset_w"] == 0
    assert manifest["tensor_dtypes"]["bias"] == "S32"
    assert manifest["correctness_comparison"] == {"mode": "tolerant_int", "tolerance": 1}
