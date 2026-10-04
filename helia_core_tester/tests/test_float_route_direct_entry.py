"""Float per-route entries (`entry:`, ns-cmsis-nn#674): scratch queries and build-dependent declines."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.kernel_dispatch import resolve_direct_entry
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DECLINES = re.compile(
    r"#if (!defined\(\w+\) \|\| defined\(HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE\))\n(.*?)#else(.*?)#endif", re.S
)


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path) -> str:
    generate_test(_descriptor(name), str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


@pytest.mark.parametrize(
    ("entry", "dtype", "sizer", "sizer_takes_layout"),
    [
        ("arm_convolve_1x1_nhwc_packed_f16_acc16", "FP16", "arm_convolve_1x1_f16_get_buffer_size", True),
        ("arm_convolve_1_x_n_nhwc_ohwi_f32", "FP32", "arm_convolve_1_x_n_f32_get_buffer_size", True),
        ("arm_convolve_patch_gemm_nhwc_packed_f32", "FP32", "arm_convolve_patch_gemm_f32_get_buffer_size", False),
        ("arm_convolve_1d_k3_nhwc_ohwi_f16", "FP16", "arm_convolve_f16_get_buffer_size", True),
        ("arm_convolve_small_c_nhwc_f32", "FP32", "arm_convolve_f32_get_buffer_size", True),
    ],
)
def test_convolve_route_entry_takes_its_route_scratch_query(
    entry: str, dtype: str, sizer: str, sizer_takes_layout: bool
) -> None:
    resolved = resolve_direct_entry("Convolve", entry, dtype, dtype)
    assert resolved["kernel_get_buffer_size_fn"] == sizer
    assert resolved["buffer_size_needs_layout"] is sizer_takes_layout
    assert resolved["kernel_needs_layout"] is False


@pytest.mark.parametrize(
    ("name", "mve_flag"),
    [
        ("convolve_float_route_small_c_packed_f16", "ARM_MATH_MVE_FLOAT16"),
        ("convolve_float_route_small_c_ohwi_f32", "ARM_MATH_MVEF"),
        ("depthwise_conv_float_route_cin1_f16", "ARM_MATH_MVE_FLOAT16"),
    ],
)
def test_mve_float_only_entry_declines_without_mve_float(name: str, mve_flag: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)
    blocks = list(_DECLINES.finditer(source))

    # One block arms the output guard, one validates the call.
    assert len(blocks) == 2
    for block in blocks:
        assert block.group(1) == f"!defined({mve_flag}) || defined(HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE)"
    declined, full = blocks[1].group(2), blocks[1].group(3)
    assert "HELIA_GUARD_CHECK_UNTOUCHED(" in declined and "ARM_CMSIS_NN_NO_IMPL_ERROR" in declined
    assert "HELIA_VALIDATE_OUTPUTS(" in full and "HELIA_VALIDATE_OUTPUTS(" not in declined


def test_decline_case_of_an_entry_with_a_scalar_leg_expects_no_impl_on_every_build(tmp_path: Path) -> None:
    source = _source("depthwise_conv_float_route_direct_declines_mult2_f16", tmp_path)

    assert "HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE" not in source
    assert re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_NO_IMPL_ERROR", source)
    assert "HELIA_GUARD_CHECK_UNTOUCHED(" in source and "HELIA_VALIDATE_OUTPUTS(" not in source
