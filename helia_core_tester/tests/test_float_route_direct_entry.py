"""Float per-route entries (`entry:`, ns-cmsis-nn#674): scratch queries and build-dependent declines."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.contract import render
from helia_core_tester.contract.bind import takes
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ROUTE = re.compile(r"_nhwc_(?:ohwi|packed)_f|_small_c_|^arm_depthwise_conv_(?:1d_k3|2x5|cin1|direct|generic)_nhwc_")
_DECLINES = re.compile(
    r"#if (!defined\(\w+\) \|\| defined\(HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE\))\n(.*?)#else(.*?)#endif", re.S
)


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path) -> str:
    generate_test(_descriptor(name), str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


_DESCRIPTORS = load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors"))
# Every route entry a descriptor calls, with the scratch query its descriptors declare.
_ROUTE_SIZERS: dict[str, set] = {}
for _d in _DESCRIPTORS:
    if _d.get("entry") and _ROUTE.search(_d["entry"]):
        _ROUTE_SIZERS.setdefault(_d["entry"], set()).add(_d.get("entry_sizer"))
_ROUTE_ENTRIES = sorted(_ROUTE_SIZERS)


def _route_query(entry: str) -> tuple[str, bool]:
    """The scratch query a route entry takes: its route's own, or the router's when it needs none."""
    if entry.startswith("arm_depthwise_conv_"):
        return "arm_depthwise_conv_f16_get_buffer_size", True
    bits = re.search(r"_(f16|f32)(?:_acc16)?$", entry).group(1)
    for route, query, takes_layout in (
        ("1x1", f"arm_convolve_1x1_{bits}_get_buffer_size", True),
        ("1_x_n", f"arm_convolve_1_x_n_{bits}_get_buffer_size", True),
        ("patch_gemm", f"arm_convolve_patch_gemm_{bits}_get_buffer_size", False),
    ):
        if entry.startswith(f"arm_convolve_{route}_nhwc_"):
            return query, takes_layout
    return f"arm_convolve_{bits}_get_buffer_size", True


def test_every_float_route_entry_has_a_case_that_runs_it() -> None:
    succeeding = {
        d.get("entry") for d in _DESCRIPTORS if d.get("expected_status", "ARM_CMSIS_NN_SUCCESS") == "ARM_CMSIS_NN_SUCCESS"
    }

    assert len(_ROUTE_ENTRIES) == 46
    assert set(_ROUTE_ENTRIES) <= succeeding


@pytest.mark.parametrize("entry", _ROUTE_ENTRIES)
def test_route_entry_takes_its_route_scratch_query(entry: str) -> None:
    query, takes_layout = _route_query(entry)
    contracts = render.load_current_contracts()

    # Every descriptor of the entry declares the route's query; the contract says whether it takes a layout.
    assert _ROUTE_SIZERS[entry] == {query}
    assert takes(contracts.require(query), "layout") is takes_layout
    assert takes(contracts.require(entry), "layout") is False


@pytest.mark.parametrize(
    ("name", "sizer_call"),
    [
        (
            "convolve_float_route_1x1_ohwi_stride2_f32",
            r"arm_convolve_1x1_f32_get_buffer_size\([^;]*ARM_NN_LAYOUT_NHWC\s*\)",
        ),
        (
            "convolve_float_route_1_x_n_packed_acc16_f16",
            r"arm_convolve_1_x_n_f16_get_buffer_size\([^;]*ARM_NN_LAYOUT_NHWC\s*\)",
        ),
        (
            "convolve_float_route_patch_gemm_packed_f32",
            r"arm_convolve_patch_gemm_f32_get_buffer_size\([^;,]*,[^;,]*,[^;,]*,[^;,]*_output_dims\s*\)",
        ),
        ("convolve_float_route_1d_k3_ohwi_f16", r"arm_convolve_f16_get_buffer_size\([^;]*ARM_NN_LAYOUT_NHWC\s*\)"),
    ],
)
def test_route_case_sizes_scratch_with_its_route_query(name: str, sizer_call: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)
    entry = _descriptor(name)["entry"]

    # The calls are bound from the kernel contract, which names each argument in a comment.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    assert re.search(sizer_call, code)
    # The entry itself takes no layout argument.
    assert re.search(rf"{entry}\((?:[^;]*?,){{9}}[^,;]*?output\s*\)", code)


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


def test_declines_flag_cannot_take_an_expected_status() -> None:
    from helia_core_tester.generation.ops.ConvolutionFunctions.convolve import OpConvolve

    desc = {**_descriptor("convolve_float_route_small_c_ohwi_f16"), "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    with pytest.raises(ValueError, match="cannot be combined with expected_status"):
        OpConvolve(desc).expected_status()


def test_float_coverage_build_flags_the_harness() -> None:
    cmake = (_PROJECT_ROOT / "CMakeLists.txt").read_text()
    block = re.search(r"if\(ENABLE_COVERAGE AND NOT ENABLE_COVERAGE_MVE_FLOAT\)(.*?)endif\(\)", cmake, re.S)

    assert (
        block
        and "target_compile_definitions(helia_test_runtime PUBLIC HELIA_CMSIS_NN_FLOAT_AUTOVECTORIZE)" in block.group(1)
    )


@pytest.mark.parametrize(
    "name",
    ["depthwise_conv_entry_3x3_5x8_c64_s8", "fully_connected_float_entry_nhwc_k24_n17_f16"],
)
def test_declines_flag_is_rejected_where_the_template_does_not_render_it(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="autovectorize_declines is not supported"):
        generate_test({**_descriptor(name), "autovectorize_declines": True}, str(tmp_path))
