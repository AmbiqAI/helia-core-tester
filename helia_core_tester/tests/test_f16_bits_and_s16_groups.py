"""Cases for arm_dequantize_f16_bits_f32 (ns-cmsis-nn#719) and the s16 whole-group rule (#725)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _sources(name: str, tmp_path: Path, **overrides) -> tuple[str, str]:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p for p in tmp_path.rglob(name) if p.is_dir())
    header = "".join(p.read_text() for p in case_dir.rglob("*.h"))
    source = "".join(p.read_text() for p in case_dir.glob("*.c"))
    return header, source


def _array(header: str, suffix: str) -> list[int]:
    body = re.search(rf"_{suffix}\[\] = \{{(.*?)\}};", header, re.S).group(1)
    return [int(v, 16) for v in re.findall(r"0x([0-9A-F]+)u", body)]


@pytest.mark.parametrize("name", ["dequantize_float_f16_bits_vec7_f32", "dequantize_float_f16_bits_storage_vec75_f32"])
def test_bits_case_carries_both_nan_rules_and_compares_bits(name: str, tmp_path: Path) -> None:
    header, source = _sources(name, tmp_path)
    halves = _array(header, "input")
    vector = dict(zip(halves, _array(header, "expected_vector_bits")))
    scalar = dict(zip(halves, _array(header, "expected_scalar_bits")))

    # A signalling NaN: the vector conversion gives the default NaN, the scalar one keeps the payload, quiet.
    assert (vector[0x7C01], scalar[0x7C01]) == (0x7FC00000, 0x7FC02000)
    # A negative quiet NaN keeps its sign only on the scalar rule; finite values and Inf widen alike.
    assert (vector[0xFE00], scalar[0xFE00]) == (0x7FC00000, 0xFFC00000)
    assert vector[0x0001] == scalar[0x0001] == 0x33800000
    assert vector[0x7C00] == scalar[0x7C00] == 0x7F800000
    assert "arm_dequantize_f16_bits_f32(" in source and "HELIA_VALIDATE_FLOAT_BITS(" in source
    assert re.search(
        r"#if ARM_NN_ENABLE_F16 && defined\(ARM_MATH_MVE_FLOAT16\).*?ARM_NN_VCVT_F16_SCALAR_FORM", source, re.S
    )


def test_bits_case_rejects_keys_it_would_ignore(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="takes none of"):
        _sources("dequantize_float_f16_bits_vec75_f32", tmp_path, activation="RELU")


def test_u16_is_storage_for_the_dequantize_entry_only() -> None:
    from helia_core_tester.generation.ops.QuantizationFunctions.dequantize import OpDequantize

    desc = {**_descriptor("dequantize_float_f16_bits_storage_vec75_f32")}
    OpDequantize(desc)
    with pytest.raises(ValueError, match="U16"):
        OpDequantize({k: v for k, v in desc.items() if k != "entry"})


@pytest.mark.parametrize(
    ("name", "line"),
    [
        ("convolve_fault_zero_filter_depth_s16", "filter_dims.c = 0;"),
        ("convolve_fault_filter_deeper_than_input_s16", "filter_dims.c = 2 * input_dims.c;"),
        ("convolve_fault_partial_filter_group_s16", "input_dims.c = filter_dims.c + 1;"),
        ("convolve_fault_negative_output_depth_1x1_s16", "output_dims.c = -output_dims.c;"),
    ],
)
def test_group_fault_reaches_the_wrapper_with_the_broken_dims(name: str, line: str, tmp_path: Path) -> None:
    _, source = _sources(name, tmp_path)

    assert line in source
    assert re.search(r"arm_convolve_wrapper_s16\([^;]*&filter_dims,[^;]*&output_dims,", source, re.S)


@pytest.mark.parametrize("kernel_case", ["convolve_float_default_f32", "convolve_default_s8"])
def test_group_faults_are_refused_on_kernels_without_the_s16_rule(kernel_case: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="s16 whole-group rule"):
        _sources(kernel_case, tmp_path, fault="zero_filter_depth", expected_status="ARM_CMSIS_NN_ARG_ERROR")
