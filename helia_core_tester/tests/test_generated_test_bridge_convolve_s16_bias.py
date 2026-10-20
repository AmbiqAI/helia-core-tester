"""S16 Convolve bias width over the hardware bridge (issue #488).

`arm_convolve_wrapper_s16` takes `cmsis_nn_bias_data`, so a generated S16 case may carry an
int64 bias (default) or an int32 one (`bias_dtype: S32`). The bridge streams the width the
header declares, the contract accepts both, and the firmware adapter sets `is_int32_bias`
from the blob's dtype.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.hardware.adapter_specs import render_generated_adapters_source
from helia_core_tester.hardware.generated_test_bridge import (
    UnsupportedGeneratedTestError,
    _extract_array_c_type,
    build_case_bundle_from_generated_test,
)
from helia_core_tester.hardware.kernel_registry import lookup_kernel_id
from helia_core_tester.tests.test_generated_test_bridge_recurrent import _array, _case

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _convolve_s16_header(name: str, bias_c_type: str | None) -> str:
    bias = (
        f"static const int64_t *{name}_biases = NULL;\n"
        if bias_c_type is None
        else _array(bias_c_type, f"{name}_biases", [-5, 7])
    )
    return (
        _array("int16_t", f"{name}_input", range(1, 10))
        + _array("int8_t", f"{name}_weights", [1, -1] * 4)
        + bias
        + _array("int32_t", f"{name}_multiplier", [1073741824, 1073741824])
        + _array("int32_t", f"{name}_shift", [-1, -1])
        + _array("int16_t", f"{name}_expected_output", [3, -3, 5, -5, 7, -7, 9, -9])
        + f"const cmsis_nn_dims {name}_input_dims = {{.n = 1, .h = 3, .w = 3, .c = 1}};\n"
        f"const cmsis_nn_dims {name}_filter_dims = {{.n = 2, .h = 2, .w = 2, .c = 1}};\n"
        f"const cmsis_nn_dims {name}_output_dims = {{.n = 1, .h = 2, .w = 2, .c = 2}};\n"
        f"const cmsis_nn_conv_params {name}_conv_params = {{.input_offset = 0, .output_offset = 0, "
        f".stride = {{.w = 1, .h = 1}}, .padding = {{.w = 0, .h = 0}}, .dilation = {{.w = 1, .h = 1}}, "
        f".activation = {{.min = -32768, .max = 32767}}}};\n"
    )


def _descriptor(name: str, **extra) -> dict:
    return {
        "operator": "Convolve",
        "name": name,
        "activation_dtype": "S16",
        "weight_dtype": "S8",
        "strides": [1, 1],
        "padding": "VALID",
        **extra,
    }


def _bias_blob(bundle):
    return next(blob for blob in bundle.streamable_blobs if blob.role == "bias")


@pytest.mark.parametrize(
    ("bias_c_type", "wire_dtype", "numpy_dtype"),
    [("int64_t", "S64", np.int64), ("int32_t", "S32", np.int32)],
)
def test_bridge_streams_the_bias_width_the_header_declares(tmp_path: Path, bias_c_type, wire_dtype, numpy_dtype) -> None:
    name = f"convolve_bias_{wire_dtype.lower()}_s16"
    case = _case(tmp_path, "ConvolutionFunctions", _descriptor(name), _convolve_s16_header(name, bias_c_type))
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "out", require_fvp_pass=False)
    assert bundle.kernel_id == lookup_kernel_id(PROJECT_ROOT, family="ConvolutionFunctions", operator="Convolve", dtype="S16", weight_dtype="S8")
    bias = _bias_blob(bundle)
    assert bias.dtype == wire_dtype
    assert bias.byte_length == 2 * np.dtype(numpy_dtype).itemsize
    assert np.frombuffer(Path(bias.path).read_bytes(), dtype=numpy_dtype).tolist() == [-5, 7]


def test_bridge_omits_the_bias_blob_for_a_null_bias(tmp_path: Path) -> None:
    name = "convolve_no_bias_s16"
    case = _case(tmp_path, "ConvolutionFunctions", _descriptor(name), _convolve_s16_header(name, None))
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "out", require_fvp_pass=False)
    assert all(blob.role != "bias" for blob in bundle.streamable_blobs)


def test_bridge_rejects_an_unexpected_bias_c_type(tmp_path: Path) -> None:
    name = "convolve_bias_odd_s16"
    case = _case(tmp_path, "ConvolutionFunctions", _descriptor(name), _convolve_s16_header(name, "int16_t"))
    with pytest.raises(UnsupportedGeneratedTestError, match=re.escape("unexpected C type 'int16_t'")):
        build_case_bundle_from_generated_test(PROJECT_ROOT, case, output_root=tmp_path / "out", require_fvp_pass=False)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("static const int32_t x_biases[] = {1, 2};", "int32_t"),
        ("static const int64_t x_biases[2] = {\n 1,\n 2};", "int64_t"),
        ("static const int64_t *x_biases = NULL;", None),
        ("static const int32_t xx_biases[] = {1};", None),
        ("", None),
    ],
)
def test_extract_array_c_type(header: str, expected: str | None) -> None:
    assert _extract_array_c_type(header, "x_biases") == expected


def test_convolve_s16_adapter_derives_the_bias_width_from_the_blob_dtype() -> None:
    source = render_generated_adapters_source()
    block = source[source.index("HCT_KERNEL_ID_CONVOLVE_S16") :]
    block = block[: block.index("HCT_KERNEL_ID_CONVOLVE_S4")]
    assert "bias->dtype == HCT_DTYPE_S32}" in block
    assert "bias_elem_size = sizeof(int32_t);" in block
    assert "bias_elem_size = sizeof(int64_t);" in block
    assert "hct_checked_count_bytes(bias_dims.c, bias_elem_size, bias->byte_length" in block
    # Anything but S32/S64 is refused even if the CASE_META contract check were bypassed.
    assert re.search(r"else\s*\{\s*return ARM_CMSIS_NN_ARG_ERROR;", block)
