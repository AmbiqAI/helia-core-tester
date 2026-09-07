"""The float copy-class kernels of ns-cmsis-nn#475: arm_dequantize_f16_f32, arm_split_f32,
any-rank arm_concatenation_f32/f16, arm_pack_f32/f16, arm_unpack_f32/f16 and
arm_nn_fill_f32/f16.

Every descriptor for them carries required_kernel_symbols so a checkout that predates the
kernels skips the case instead of failing the build; the goldens are numpy, the kernels are
bit copies, and the comparison is zero-tolerance. These tests pin the descriptor contract,
the LiteRT topologies, and the shape of the emitted C for one case per kernel.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

import helia_core_tester.generation.test_ops as generation_module
from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.BasicMathFunctions.fill import parse_fill_value
from helia_core_tester.generation.ops.catalog import get_operator_spec
from helia_core_tester.generation.utils.litert_builder import build_fill_op, build_pack_op, build_unpack_op
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


COPY_CLASS_SYMBOLS = {
    "arm_dequantize_f16_f32",
    "arm_split_f32",
    "arm_split_f16",
    "arm_concatenation_f32",
    "arm_concatenation_f16",
    "arm_pack_f32",
    "arm_pack_f16",
    "arm_unpack_f32",
    "arm_unpack_f16",
    "arm_nn_fill_f32",
    "arm_nn_fill_f16",
}

# One representative per kernel, with the C the template must emit for it, and
# whether the op writes the structured generation sidecar. Pack, unpack, fill and
# dequantize go through OperationBase._write_op_outputs, which emits it; split and
# concatenation still write their .h/.c/CMakeLists by hand and predate that
# helper, so they emit no sidecar. That is pre-existing and unchanged here -- the
# flag records which is which instead of asserting a file that never existed.
REPRESENTATIVE_CASES = [
    ("pack_float_rank0_n8_f16", "pack", ["arm_pack_f16(", "NULL,", "0,   // input_dims"], True),
    ("unpack_float_rank5_axis4_f32", "unpack", ["arm_unpack_f32(", "5,   // input_dims", "_out_1_output,"], True),
    ("split_float_zero_slice_v_f32", "split", ["arm_split_f32(", "_out_1_output[1];", "OUT_1_OUTPUT_SIZE (0)"], False),
    ("concatenation_any_rank_rank1_f16", "concatenation", ["arm_concatenation_f16(", "1,                    // output_dims"], False),
    ("fill_float_nan_block17_f16", "fill", ["arm_nn_fill_f16(", "_fill_value[0],"], True),
    ("dequantize_float_f16_widen_nonfinite_f32", "dequantize", ["arm_dequantize_f16_f32(", "40         // block_size"], True),
]


@pytest.fixture(scope="module")
def descriptors() -> dict[str, dict]:
    return {desc["name"]: desc for desc in load_all_descriptors(str(find_descriptors_dir()))}


@pytest.fixture(scope="module")
def copy_class_descriptors(descriptors: dict[str, dict]) -> list[dict]:
    selected = [
        desc
        for desc in descriptors.values()
        if COPY_CLASS_SYMBOLS & set(desc.get("required_kernel_symbols") or [])
    ]
    assert len(selected) == 134
    return selected


def test_copy_class_descriptors_pin_symbols_and_zero_tolerance(copy_class_descriptors: list[dict]) -> None:
    for desc in copy_class_descriptors:
        symbols = set(desc["required_kernel_symbols"])
        assert symbols <= COPY_CLASS_SYMBOLS, desc["name"]
        assert desc["suite"] == "float", desc["name"]
        comparison = desc["resolved_comparison"]
        assert comparison["mode"] == "float", desc["name"]
        assert comparison["atol"] == 0.0 and comparison["rtol"] == 0.0, desc["name"]
        family = get_operator_spec(desc["operator"]).family
        assert desc["_family"] == family


def test_split_f16_cases_pin_the_f32_sibling(copy_class_descriptors: list[dict]) -> None:
    # The rank-1/2/5, zero-extent and unequal-length f16 shapes are the #475 contract
    # (the shared axis-copy core), not the older standalone arm_split_f16, so they pin
    # the f32 sibling that arrived in the same change.
    for desc in copy_class_descriptors:
        if desc["operator"] == "Split" and desc["resolved_tensor_dtypes"]["input"] == "FP16":
            assert set(desc["required_kernel_symbols"]) == {"arm_split_f16", "arm_split_f32"}, desc["name"]


def test_fill_is_a_cmsis_parity_operator() -> None:
    spec = get_operator_spec("Fill")
    assert spec.parity_kind == "cmsis"
    assert spec.family == "BasicMathFunctions"
    assert spec.descriptor_relpaths == ("BasicMathFunctions/fill_float.yaml",)


@pytest.mark.parametrize(
    "precision, expected",
    [("f16", True), ("f32", False), ("both", True)],
)
def test_f16_widening_case_runs_on_the_f16_leg_only(descriptors: dict[str, dict], precision: str, expected: bool) -> None:
    # arm_dequantize_f16_f32 compiles under ARM_NN_ENABLE_F16, which the f32-only leg
    # switches off, so a case touching FP16 anywhere belongs to the f16 leg.
    desc = descriptors["dequantize_float_f16_widen_vec17_f32"]
    assert desc["resolved_tensor_dtypes"] == {"input": "FP16", "output": "FP32"}
    assert generation_module.should_run_test(desc, {"suite": "float", "float_precision": precision}) is expected


def test_pure_precision_cases_keep_their_legs(descriptors: dict[str, dict]) -> None:
    f32_case = descriptors["split_float_rank1_v_pair_f32"]
    f16_case = descriptors["split_float_rank1_v_pair_f16"]
    assert generation_module.should_run_test(f32_case, {"suite": "float", "float_precision": "f32"})
    assert not generation_module.should_run_test(f32_case, {"suite": "float", "float_precision": "f16"})
    assert generation_module.should_run_test(f16_case, {"suite": "float", "float_precision": "f16"})
    assert not generation_module.should_run_test(f16_case, {"suite": "float", "float_precision": "f32"})


def test_any_rank_dims_helper_keeps_the_element_count() -> None:
    helper = TemplateContextBuilder.shape_to_cmsis_dims_any_rank
    assert helper(()) == {"n": 1, "h": 1, "w": 1, "c": 1}
    assert helper((13,)) == TemplateContextBuilder.nhwc_to_cmsis_dims((13,))
    assert helper((1, 4, 6, 3)) == TemplateContextBuilder.nhwc_to_cmsis_dims((1, 4, 6, 3))
    dims = helper((2, 2, 9, 3, 2))
    assert dims == {"n": 4, "h": 9, "w": 3, "c": 2}
    assert dims["n"] * dims["h"] * dims["w"] * dims["c"] == 2 * 2 * 9 * 3 * 2


@pytest.mark.parametrize(
    "raw, expected_repr",
    [
        (1.5, "1.5"),
        (-2, "-2.0"),
        ("-0.0", "-0.0"),
        ("inf", "inf"),
        ("-inf", "-inf"),
        ("2.5e-3", "0.0025"),
    ],
)
def test_parse_fill_value_tokens(raw, expected_repr: str) -> None:
    value = parse_fill_value(raw)
    assert repr(value) == expected_repr
    if expected_repr == "-0.0":
        assert math.copysign(1.0, value) < 0


def test_parse_fill_value_nan_and_rejects_junk() -> None:
    assert math.isnan(parse_fill_value("nan"))
    with pytest.raises(ValueError):
        parse_fill_value("seven")
    with pytest.raises(ValueError):
        parse_fill_value(True)


def _output_shapes(model_bytes: bytes) -> list[tuple[int, ...]]:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    return [tuple(int(v) for v in detail["shape"]) for detail in interpreter.get_output_details()]


def test_litert_topologies_match_numpy_shapes() -> None:
    # Cross-checks the builders' own shape arithmetic against LiteRT's.
    assert _output_shapes(build_pack_op(input_shape=[], num_inputs=8, axis=0)) == [(8,)]
    assert _output_shapes(build_pack_op(input_shape=[2, 3], num_inputs=3, axis=2)) == [(2, 3, 3)]
    assert _output_shapes(build_unpack_op(input_shape=[2, 5, 3], axis=1)) == [(2, 3)] * 5
    assert _output_shapes(build_unpack_op(input_shape=[2, 5, 3], axis=1, dtype="float16")) == [(2, 3)] * 5
    assert _output_shapes(build_fill_op(output_shape=[17], dtype="float16")) == [(17,)]
    with pytest.raises(ValueError):
        build_pack_op(input_shape=[2, 3], num_inputs=2, axis=3)
    with pytest.raises(ValueError):
        build_unpack_op(input_shape=[2, 0], axis=1)


def test_float16_pack_model_is_provenance_only() -> None:
    # LiteRT's reference PACK has no FLOAT16 registration, so the f16 pack model
    # cannot be prepared. It does not have to be: the .tflite is a provenance
    # artifact and nothing in the pipeline interprets it -- OpPack's golden is
    # numpy.stack over its own emitted operands (asserted below). UNPACK and FILL
    # do register FLOAT16, which is why only pack is called out here. If LiteRT
    # ever gains the registration this test fails and the cross-check above can
    # widen to cover it.
    model = build_pack_op(input_shape=[], num_inputs=8, axis=0, dtype="float16")
    assert model
    with pytest.raises(RuntimeError, match="FLOAT16"):
        _output_shapes(model)


@pytest.fixture(scope="module")
def emitted_cases(descriptors: dict[str, dict], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out_dir = tmp_path_factory.mktemp("copy_class")
    emitted: dict[str, Path] = {}
    for case_name, _suffix, _needles, _sidecar in REPRESENTATIVE_CASES:
        desc = descriptors[case_name]
        generation_module.generate_test(desc, str(out_dir), cpu="cortex-m55")
        emitted[case_name] = out_dir / desc["_family"] / case_name
    return emitted


@pytest.mark.parametrize("case_name, op_suffix, needles, emits_sidecar", REPRESENTATIVE_CASES)
def test_representative_cases_emit_the_kernel_call(
    emitted_cases: dict[str, Path], case_name: str, op_suffix: str, needles: list[str], emits_sidecar: bool
) -> None:
    test_dir = emitted_cases[case_name]
    source = (test_dir / f"{case_name}_{op_suffix}.c").read_text()
    for needle in needles:
        assert needle in source, (case_name, needle)
    assert "HELIA_VALIDATE_OUTPUTS(" in source
    assert "        0.0f,\n        0.0f,\n" in source, "zero-tolerance comparison"
    assert (test_dir / f"{case_name}.tflite").exists()
    assert (test_dir / f"{case_name}_{op_suffix}.sidecar.json").exists() is emits_sidecar


def _c_array(header: str, name: str) -> np.ndarray:
    match = re.search(rf"{re.escape(name)}\[\]\s*=\s*\{{(.*?)\}};", header, re.DOTALL)
    assert match, name
    tokens = [tok.strip() for tok in match.group(1).replace("\n", " ").split(",") if tok.strip()]
    values = []
    for tok in tokens:
        tok = tok.replace("(float16_t)", "")
        if tok == "NAN":
            values.append(math.nan)
        elif tok == "INFINITY":
            values.append(math.inf)
        elif tok == "-INFINITY":
            values.append(-math.inf)
        else:
            values.append(float(tok.rstrip("f")))
    return np.array(values, dtype=np.float32)


def test_pack_golden_is_numpy_stack_of_the_emitted_inputs(emitted_cases: dict[str, Path]) -> None:
    case = "pack_float_rank0_n8_f16"
    header = (emitted_cases[case] / "includes" / f"{case}_pack.h").read_text()
    inputs = [_c_array(header, f"{case}_input{i + 1}") for i in range(8)]
    expected = _c_array(header, f"{case}_expected_output")
    assert len(set(float(v[0]) for v in inputs)) == 8, "operands must differ"
    np.testing.assert_array_equal(np.stack(inputs, axis=0).reshape(-1), expected)


def test_unpack_golden_is_numpy_take_of_the_emitted_input(emitted_cases: dict[str, Path]) -> None:
    case = "unpack_float_rank5_axis4_f32"
    header = (emitted_cases[case] / "includes" / f"{case}_unpack.h").read_text()
    data = _c_array(header, f"{case}_input").reshape(2, 2, 8, 3, 2)
    for index in range(2):
        expected = _c_array(header, f"{case}_out_{index}_expected_output")
        np.testing.assert_array_equal(np.take(data, index, axis=4).reshape(-1), expected)


def test_dequantize_widening_golden_carries_the_tokens(emitted_cases: dict[str, Path]) -> None:
    case = "dequantize_float_f16_widen_nonfinite_f32"
    header = (emitted_cases[case] / "includes" / f"{case}_dequantize.h").read_text()
    data = _c_array(header, f"{case}_input")
    expected = _c_array(header, f"{case}_expected_output")
    assert math.isnan(data[0]) and math.isnan(expected[0])
    assert data[1] == math.inf == expected[1]
    assert data[2] == -math.inf == expected[2]
    np.testing.assert_array_equal(data[3:], expected[3:])
    # The float16 literals widen exactly: every finite lane is a float16 value.
    assert np.array_equal(data[3:].astype(np.float16).astype(np.float32), data[3:])
