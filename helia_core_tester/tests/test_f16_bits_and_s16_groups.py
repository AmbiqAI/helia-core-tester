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
    desc = {**_descriptor(name), **overrides}
    if "tensor_dtypes" in overrides:
        desc.pop("resolved_tensor_dtypes", None)
    generate_test(desc, str(tmp_path))
    case_dir = next(p for p in tmp_path.rglob(name) if p.is_dir())
    header = "".join(p.read_text() for p in case_dir.rglob("*.h"))
    source = "".join(p.read_text() for p in case_dir.glob("*.c"))
    return header, source


def _array(header: str, suffix: str) -> list[int]:
    body = re.search(rf"_{suffix}\[\] = \{{(.*?)\}};", header, re.S).group(1)
    return [int(v, 16) for v in re.findall(r"0x([0-9A-F]+)u", body)]


@pytest.mark.parametrize("name", ["dequantize_float_f16_bits_vec75_f32", "dequantize_float_f16_bits_storage_vec75_f32"])
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


@pytest.mark.parametrize(
    "name",
    [
        "dequantize_float_f16_bits_vec7_f32",
        "dequantize_float_f16_bits_vec75_f32",
        "dequantize_float_f16_bits_storage_vec75_f32",
        "dequantize_float_f16_bits_vec1001_f32",
    ],
)
def test_bits_case_ends_on_nans_so_every_tail_meets_the_rule(name: str, tmp_path: Path) -> None:
    header, _ = _sources(name, tmp_path)
    halves = _array(header, "input")

    assert halves[-3:] == [0x7D55, 0xFC01, 0x7E01]
    assert all((h >> 10) & 0x1F == 0x1F and h & 0x3FF for h in halves[-3:])


@pytest.mark.parametrize(
    "overrides",
    [
        {"activation": "RELU"},
        {"input_mode": "nonfinite_sweep"},
        {"expected_status": "ARM_CMSIS_NN_ARG_ERROR"},
        {"comparison": {"atol": 1.0e-3, "rtol": 1.0e-3}},
        {"tensor_dtypes": {"input": "FP16", "output": "FP16"}},
        {"tensor_dtypes": {"input": "FP16", "output": "FP32", "bias": "U16"}},
        {"input_shape": [1, 70000]},
        {"input_shape": [-1, -75]},
        {"input_shape": [1, 7.9]},
        {"input_shape": "75"},
        {"scale": 0.5},
    ],
)
def test_bits_case_rejects_keys_it_would_ignore(overrides: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bit-pattern case"):
        _sources("dequantize_float_f16_bits_vec75_f32", tmp_path, **overrides)


def test_u16_is_storage_for_the_dequantize_entry_only() -> None:
    from helia_core_tester.generation.ops.QuantizationFunctions.dequantize import OpDequantize

    desc = {**_descriptor("dequantize_float_f16_bits_storage_vec75_f32")}
    OpDequantize(desc)
    with pytest.raises(ValueError, match="U16"):
        OpDequantize({k: v for k, v in desc.items() if k != "entry"})


@pytest.mark.parametrize(
    ("name", "param", "depth"),
    [
        ("convolve_fault_zero_filter_depth_s16", "filter_dims", lambda i, f, o: 0),
        ("convolve_fault_filter_deeper_than_input_s16", "filter_dims", lambda i, f, o: 2 * i),
        ("convolve_fault_partial_filter_group_s16", "input_dims", lambda i, f, o: f + 1),
        ("convolve_fault_negative_output_depth_1x1_s16", "output_dims", lambda i, f, o: -o),
        ("convolve_fault_output_not_whole_groups_s16", "output_dims", lambda i, f, o: o + 1),
        ("convolve_fault_negative_input_depth_s16", "input_dims", lambda i, f, o: -i),
        ("convolve_fault_negative_filter_depth_s16", "filter_dims", lambda i, f, o: -f),
    ],
)
def test_group_fault_reaches_the_wrapper_with_the_broken_dims(name: str, param: str, depth, tmp_path: Path) -> None:
    desc = _descriptor(name)
    input_c, filter_c, output_c = desc["input_shape"][3], desc["filter_shape"][2], desc["filter_shape"][3]
    _, source = _sources(name, tmp_path)
    # The kernel gets a copy of the dims struct with its channel count changed, just before the call.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    edit = code.find(f"{name}_fault_{param}.c = {depth(input_c, filter_c, output_c)};")
    call = re.search(rf"arm_convolve_wrapper_s16\([^;]*&{name}_fault_{param},[^;]*\)", code, re.S)

    assert call
    assert -1 < edit < call.start()
    # The copy is initialised from the case's own dims, so only the channel count differs.
    assert f"{name}_fault_{param} = {name}_{param};" in code[:edit]


@pytest.mark.parametrize("kernel_case", ["convolve_float_default_f32", "convolve_default_s8"])
def test_group_faults_are_refused_on_kernels_without_the_s16_rule(kernel_case: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="s16 whole-group rule"):
        _sources(kernel_case, tmp_path, fault="zero_filter_depth", expected_status="ARM_CMSIS_NN_ARG_ERROR")
