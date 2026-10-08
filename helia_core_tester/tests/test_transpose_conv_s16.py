"""Direct A16W8 arm_transpose_conv_s16: descriptors, bridge, firmware, hidden shapes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation import random_shapes as rs
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import _required_kernel_symbols
from helia_core_tester.generation.utils.temp_sizer_probe import missing_header_symbols
from helia_core_tester.hardware.adapter_specs import generated_test_bridge_scalar_fields, render_generated_adapters_source
from helia_core_tester.hardware.generated_test_bridge import (
    _build_transpose_conv_case,
    _extract_array,
    _extract_null_pointer_decl,
    _find_header_file,
)
from helia_core_tester.hardware.kernel_registry import lookup_kernel_id
from helia_core_tester.hardware.work_count import case_work
from helia_core_tester.tests.generated_inputs import discover_or_skip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TC16 = ("TransposeConv", "S16")
FASTENHANCER = "transpose_conv_fastenhancer_1x8_stride1x4_valid_s16"


def _s16_descriptors() -> list[dict]:
    descriptors = load_all_descriptors(str(PROJECT_ROOT / "assets" / "descriptors"))
    return [d for d in descriptors if d["operator"] == "TransposeConv" and d.get("activation_dtype") == "S16"]


def test_descriptors_hold_the_fastenhancer_tuple() -> None:
    cases = {d["name"]: d for d in _s16_descriptors()}
    fe = cases[FASTENHANCER]
    # TFLite filter [2, 1, 8, 24] is OHWI.
    assert (fe["input_shape"], fe["filter_shape"], fe["strides"], fe["padding"], fe["use_bias"]) == (
        [1, 1, 64, 24], [1, 8, 2, 24], [1, 4], "VALID", True,
    )
    assert {d["padding"] for d in cases.values()} == {"SAME", "VALID"}
    assert {s for d in cases.values() for s in d["strides"]} == {1, 2, 3, 4}
    assert any("activation_max" in d for d in cases.values()) and any(not d["use_bias"] for d in cases.values())
    for desc in cases.values():
        assert _required_kernel_symbols(desc) == rs.TC16_SYMBOLS


def test_symbol_probe_skips_without_kernel(tmp_path: Path) -> None:
    (tmp_path / "Include").mkdir()
    header = tmp_path / "Include" / "arm_nnfunctions.h"
    header.write_text("int32_t arm_transpose_conv_s8_get_buffer_size(void);\n")
    assert missing_header_symbols(rs.TC16_SYMBOLS, tmp_path) == rs.TC16_SYMBOLS
    header.write_text("".join(f"int32_t {name}(void);\n" for name in rs.TC16_SYMBOLS))
    assert missing_header_symbols(rs.TC16_SYMBOLS, tmp_path) == []


def test_registry_maps_s16_to_direct_kernel() -> None:
    assert lookup_kernel_id(PROJECT_ROOT, family="ConvolutionFunctions", operator="TransposeConv",
                            dtype="S16", weight_dtype="S8") == 186
    # The s8 wrapper keeps its id.
    assert lookup_kernel_id(PROJECT_ROOT, family="ConvolutionFunctions", operator="TransposeConv",
                            dtype="S8", weight_dtype="S8") == 79
    header = (PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_adapters.h").read_text()
    assert "#define HCT_KERNEL_ID_TRANSPOSE_CONV_S16 186u" in header


def test_firmware_links_without_the_kernel() -> None:
    source = render_generated_adapters_source()
    assert "int16_t *output_data) __attribute__((weak));" in source
    assert "const cmsis_nn_dims *out_dims) __attribute__((weak));" in source
    body = source[source.index("run_transpose_conv_s16_once(hct_server_session_t"):]
    assert body.index("arm_transpose_conv_s16 == NULL") < body.index("return ARM_CMSIS_NN_NO_IMPL_ERROR")
    assert "#define arm_transpose_conv_s16(...) HCT_TIMED(" in source
    assert "#define arm_transpose_conv_s16_get_buffer_size(...)" not in source


def _reference(header: str, name: str, manifest: dict) -> np.ndarray:
    """TFLite int16x8 TransposeConv, in numpy."""
    dims = {blob["role"]: blob["dimensions"] for blob in manifest["blob_roles"]}
    x = np.array(_extract_array(header, f"{name}_input"), np.int64).reshape(dims["input_0"])
    w = np.array(_extract_array(header, f"{name}_weights"), np.int64).reshape(dims["weights"])
    bias = None if _extract_null_pointer_decl(header, f"{name}_biases") else _extract_array(header, f"{name}_biases")
    mult, shift = _extract_array(header, f"{name}_multiplier"), _extract_array(header, f"{name}_shift")
    p = manifest["serialized_scalar_parameters"]
    acc = np.zeros((p["output_h"], p["output_w"], p["output_c"]), np.int64)
    for iy, ix, fy, fx in np.ndindex(x.shape[1], x.shape[2], w.shape[1], w.shape[2]):
        oy, ox = iy * p["stride_h"] - p["pad_h"] + fy, ix * p["stride_w"] - p["pad_w"] + fx
        if 0 <= oy < acc.shape[0] and 0 <= ox < acc.shape[1]:
            acc[oy, ox] += w[:, fy, fx, :] @ x[0, iy, ix]
    out = np.empty_like(acc)
    for oc in range(acc.shape[2]):
        # Reduced 16-bit multiplier, as TFLite rounds.
        reduced = (mult[oc] + (1 << 15)) >> 16 if mult[oc] < 0x7FFF0000 else 0x7FFF
        total = 15 - shift[oc]
        values = (acc[..., oc] + (bias[oc] if bias else 0)) * reduced + (1 << (total - 1))
        out[..., oc] = np.clip(values >> total, p["activation_min"], p["activation_max"])
    return out[None]


def test_fastenhancer_case_bridges_bit_exact(tmp_path: Path) -> None:
    cases = discover_or_skip(PROJECT_ROOT, family="ConvolutionFunctions", name_filter=FASTENHANCER)
    bundle = _build_transpose_conv_case(PROJECT_ROOT, cases[0], output_root=tmp_path)
    manifest = bundle.manifest
    assert manifest["kernel_id"] == 186 and manifest["required_target_capabilities"] == ["arm_transpose_conv_s16"]
    assert manifest["tensor_dtypes"] == {"input": "S16", "weights": "S8", "bias": "S64", "output": "S16"}
    assert manifest["correctness_comparison"] == {"mode": "exact_int"}
    assert {b.role: b.dimensions for b in bundle.blobs}["expected_output"] == (1, 1, 260, 2)
    assert set(manifest["serialized_scalar_parameters"]) <= set(
        generated_test_bridge_scalar_fields("run_transpose_conv_s16_once")
    )
    # Whole output in int64 bounds scratch.
    assert manifest["scratch_buffer"]["bytes"] >= 260 * 2 * 8
    assert case_work(bundle)["macs"] == 64 * 24 * 2 * 8
    header = _find_header_file(cases[0].directory).read_text()
    expected = np.frombuffer(bundle.expected_output.path.read_bytes(), np.int16).reshape(1, 1, 260, 2)
    assert np.array_equal(_reference(header, cases[0].name, manifest), expected)


@pytest.mark.parametrize("cpu", ["cortex-m55", "cortex-m4"])
@pytest.mark.parametrize("seed", (0, 1, 7, 12345))
def test_tc16_shapes_fit_the_board(cpu: str, seed: int) -> None:
    workspace = rs.min_workspace(cpu)
    for case in rs.sample_cases(30, seed, cpu, (TC16,)):
        kh, kw, cout, cin = case["filter_shape"]
        layer = rs.Layer(*case["input_shape"][1:], cout, kh, kw, *case["strides"], padding=case["padding"],
                         transpose=True)
        assert case["name"].startswith(f"rs{seed}_tc16_") and case["expected_route"] == rs.TC16_ROUTE
        assert case["required_kernel_symbols"] == rs.TC16_SYMBOLS
        assert rs.footprint("TransposeConv", layer, "S16") <= workspace
        assert rs.layer_macs("TransposeConv", layer) <= rs.MAX_MACS
        assert 1 <= cout <= 32 and max(kh, kw) <= 8 and max(case["strides"]) <= 4


def test_tc16_edges_covered() -> None:
    cases = rs.sample_cases(60, 11, ops=(TC16,))
    shapes = [(c["input_shape"], c["filter_shape"], c["strides"]) for c in cases]
    assert {c["padding"] for c in cases} == {"SAME", "VALID"}
    assert {s[0] for _, _, s in shapes} == {1, 2, 3, 4} and {s[1] for _, _, s in shapes} == {1, 2, 3, 4}
    assert any(i[1] == 1 for i, _, _ in shapes) and any(i[1] > 1 and f[0] > 1 for i, f, _ in shapes)
    # Kernel below stride leaves gaps.
    assert any(f[1] < s[1] for _, f, s in shapes)
    odd = Counter(i[3] % 2 for i, _, _ in shapes)
    assert odd[1] > odd[0]
    # Near FastEnhancer: 1xK, stride 4 on W.
    assert any(i[1] == 1 and f[0] == 1 and s == [1, 4] and f[2] <= 4 for i, f, s in shapes)
    assert any(c["use_bias"] for c in cases) and any(not c["use_bias"] for c in cases)
    assert {c["activation"] for c in cases} == {"NONE", "RELU", "RELU6"}
    for case in cases:
        if case["activation"] != "NONE":
            assert case["activation_min"] >= 0


def test_tc16_out_hw_follows_keras() -> None:
    same = rs.Layer(3, 5, 4, 2, 2, 3, 2, 4, padding="SAME", transpose=True)
    valid = rs.Layer(1, 64, 24, 2, 1, 8, 1, 4, transpose=True)
    gaps = rs.Layer(3, 5, 4, 2, 2, 3, 3, 4, transpose=True)
    assert same.out_hw() == (6, 20) and valid.out_hw() == (1, 260) and gaps.out_hw() == (9, 20)
