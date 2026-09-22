"""Float squared difference: arm_elementwise_squared_difference_f16 (ns-cmsis-nn#490).

Covers the float path added to OpSquaredDifference: kernel selection, the
single-rounding IEEE binary16 golden model, pinned operands for the overflow /
underflow / equal-operand boundary cases, the masked non-finite sweep, and the
status-only fault harness for the kernel's one argument guard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.squared_difference import OpSquaredDifference
from helia_core_tester.generation.ops.catalog import get_operator_spec
from helia_core_tester.generation.utils.litert_builder import LITERT_AVAILABLE

TESTER_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR_PATH = TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "squared_difference_float.yaml"
KERNEL = "arm_elementwise_squared_difference_f16"
CPU = "cortex-m55"

# (name, flat block size) -- the sizes are chosen against the 8-lane MVE loop.
RANDOM_CASES = (
    ("squared_difference_float_default_f16", 128),
    ("squared_difference_float_single_f16", 1),
    ("squared_difference_float_tail_seven_f16", 7),
    ("squared_difference_float_vector_f16", 8),
    ("squared_difference_float_tail_f16", 9),
    ("squared_difference_float_tail_fifteen_f16", 15),
    ("squared_difference_float_two_vectors_f16", 16),
    ("squared_difference_float_odd_block_f16", 45),
    ("squared_difference_float_large_f16", 512),
)
PINNED_CASES = (
    "squared_difference_float_overflow_f16",
    "squared_difference_float_underflow_f16",
    "squared_difference_float_equal_operands_f16",
)
NONFINITE_CASES = (
    ("squared_difference_float_nonfinite_f16", (0, 1, 2)),
    ("squared_difference_float_nonfinite_tail_f16", (42, 43, 44)),
)
# (name, fault kind, source marker the fault must plant)
FAULT_CASES = (
    ("squared_difference_float_fault_null_input_1_f16", "null_input_1", "*input1_arg = NULL"),
    ("squared_difference_float_fault_null_input_2_f16", "null_input_2", "*input2_arg = NULL"),
    ("squared_difference_float_fault_null_output_f16", "null_output", "*output_arg = NULL"),
    ("squared_difference_float_fault_zero_block_f16", "zero_block", "block_size = 0;"),
    ("squared_difference_float_fault_negative_block_f16", "negative_block", "block_size = -1;"),
)


def _descriptors() -> dict[str, dict]:
    return {desc["name"]: desc for desc in load_descriptor(str(DESCRIPTOR_PATH))}


def _float_desc(
    name: str,
    *,
    dtype: str = "FP16",
    shape: tuple[int, ...] = (1, 1, 1, 9),
    shape_2: tuple[int, ...] | None = None,
    **extra,
) -> dict:
    desc = {
        "operator": "SquaredDifference",
        "name": name,
        "suite": "float",
        "_descriptor_suite": "float",
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_1_shape": list(shape),
        "input_2_shape": list(shape_2 or shape),
    }
    desc.update(extra)
    return desc


def _generate(desc: dict, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, dict]:
    """Run convert + C generation; return (c source, header, sidecar)."""
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for squared difference LiteRT generation")
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    op = OpSquaredDifference(desc, seed=1, target_cpu=CPU)
    tflite_path = out_dir / f"{desc['name']}.tflite"
    op.convert_to_tflite(None, str(tflite_path), 1)
    op.generate_c_files(out_dir)
    op.assert_input_mode_consumed()
    name = desc["name"]
    c_text = (out_dir / f"{name}_squared_difference.c").read_text()
    h_text = (out_dir / "includes" / f"{name}_squared_difference.h").read_text()
    sidecar = json.loads((out_dir / f"{name}_squared_difference.sidecar.json").read_text())
    return c_text, h_text, sidecar


def _golden(h_text: str, name: str) -> list[str]:
    body = h_text.split(f"{name}_expected_output[] = {{", 1)[1].split("};", 1)[0]
    return [token.strip() for token in body.replace("\n", " ").split(",") if token.strip()]


def _f16(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float16)


# --- descriptor contract -------------------------------------------------------


def test_catalog_registers_float_descriptor_file() -> None:
    spec = get_operator_spec("SquaredDifference")
    assert "BasicMathFunctions/squared_difference_float.yaml" in spec.descriptor_relpaths
    assert "BasicMathFunctions/squared_difference.yaml" in spec.descriptor_relpaths


def test_float_descriptors_are_f16_gated_on_the_pr490_symbol() -> None:
    descriptors = _descriptors()
    expected_names = (
        [name for name, _ in RANDOM_CASES]
        + list(PINNED_CASES)
        + [name for name, _ in NONFINITE_CASES]
        + [name for name, _, _ in FAULT_CASES]
    )
    assert sorted(descriptors) == sorted(expected_names)
    for desc in descriptors.values():
        assert desc["operator"] == "SquaredDifference"
        assert desc["suite"] == "float"
        assert desc["tensor_dtypes"] == {"input": "FP16", "output": "FP16"}
        assert tuple(desc["input_1_shape"]) == tuple(desc["input_2_shape"])
        # A checkout without ns-cmsis-nn#490 must skip every case instead of
        # failing the build on an undeclared symbol.
        assert desc["required_kernel_symbols"] == [KERNEL]


def test_random_cases_cover_every_predication_shape_of_the_8_lane_loop() -> None:
    descriptors = _descriptors()
    sizes = {name: int(np.prod(descriptors[name]["input_1_shape"])) for name, _ in RANDOM_CASES}
    assert sizes == dict(RANDOM_CASES)
    remainders = {size % 8 for size in sizes.values()}
    # a lone partial vector (1, 7), full vectors only (8, 16), full + tail (9, 15, 45, 128, 512)
    assert {1, 7} <= {size for size in sizes.values() if size < 8}
    assert {8, 16} <= {size for size in sizes.values() if size % 8 == 0}
    assert {1, 7, 5, 0} <= remainders


def test_pinned_and_fault_cases_share_the_nine_element_shape() -> None:
    descriptors = _descriptors()
    for name in PINNED_CASES + tuple(name for name, _, _ in FAULT_CASES):
        assert descriptors[name]["input_1_shape"] == [1, 1, 1, 9], name
    for name in PINNED_CASES:
        assert descriptors[name]["comparison"] == {"atol": 0.0, "rtol": 0.0}, name
        extras = descriptors[name]["hint"]["extras"]
        assert len(extras["input_1_values"]) == 9 and len(extras["input_2_values"]) == 9


# --- kernel selection ----------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "kernel_fn", "c_type"),
    [
        ("FP16", "arm_elementwise_squared_difference_f16", "float16_t"),
        ("FP32", "arm_elementwise_squared_difference_f32", "float"),
    ],
)
def test_float_dtypes_select_the_flat_float_kernel(dtype: str, kernel_fn: str, c_type: str) -> None:
    op = OpSquaredDifference(_float_desc("k", dtype=dtype), seed=1, target_cpu=CPU)
    info = op._select_cmsis_squared_difference_kernel()
    assert info == {"kernel_fn": kernel_fn, "input_c_type": c_type, "output_c_type": c_type, "float_kernel": True}


def test_int_kernel_selection_is_unchanged() -> None:
    desc = {"operator": "SquaredDifference", "name": "i", "activation_dtype": "S8", "weight_dtype": "S8",
            "input_1_shape": [1, 2, 2, 3], "input_2_shape": [1, 2, 2, 3]}
    info = OpSquaredDifference(desc, seed=1, target_cpu=CPU)._select_cmsis_squared_difference_kernel()
    assert info["kernel_fn"] == "arm_squared_difference_s8"
    assert info["float_kernel"] is False


# --- golden model --------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (256.0, 0.0, np.inf),          # 65536 overflows
        (0.0, 256.0, np.inf),          # sign of the difference is irrelevant
        (255.875, 0.0, 65472.0),       # largest finite square of a half
        (-255.875, 0.0, 65472.0),
        (300.0, 44.0625, np.inf),      # 255.9375 ties up to 256 before squaring
        (300.0, 44.1875, 65408.0),     # 255.8125 ties down to 255.75
        (65504.0, -65504.0, np.inf),   # the difference itself overflows
        (181.0, 0.0, 32768.0),         # 32761 rounds up across the 2^15 binade
        (0.00390625, 0.0, 2.0 ** -16),  # subnormal square
        (0.0078125, 0.0, 2.0 ** -14),   # smallest normal
        (0.000244140625, 0.0, 2.0 ** -24),  # smallest subnormal
        (0.0001220703125, 0.0, 0.0),    # 2^-26 rounds to zero
        (1.0, 0.99951171875, 2.0 ** -22),
        (-0.0, 0.0, 0.0),
    ],
)
def test_binary16_reference_rounds_once_per_operation(a: float, b: float, expected: float) -> None:
    reference = OpSquaredDifference._float_reference(np.float16)
    result = reference([_f16(a), _f16(b)])
    assert result.dtype == np.float16
    if np.isinf(expected):
        assert np.isposinf(result[0])
    else:
        assert float(result[0]) == expected
        assert not np.signbit(result[0])


def test_binary16_reference_is_exact_across_the_full_exponent_span() -> None:
    # 65504 - 2^-24 needs 40 significand bits: exact in the float64 model, so
    # the difference rounds once, to 65504, and its square overflows. NumPy's
    # own half subtraction would first round the operands' difference in
    # float32, a second rounding the kernel never performs.
    a = _f16(65504.0)
    b = _f16(2.0 ** -24)
    single = OpSquaredDifference._float_reference(np.float16)([a, b])
    assert np.isposinf(single[0])
    small = OpSquaredDifference._float_reference(np.float16)([_f16(2.0 ** -24), _f16(-(2.0 ** -24))])
    assert float(small[0]) == 0.0  # (2^-23)^2 = 2^-46 underflows to +0


def test_binary32_reference_computes_in_float32() -> None:
    reference = OpSquaredDifference._float_reference(np.float32)
    result = reference([np.asarray([3.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32)])
    assert result.dtype == np.float32 and float(result[0]) == 4.0
    big = reference([np.asarray([3.0e19], dtype=np.float32), np.asarray([-3.0e19], dtype=np.float32)])
    assert np.isposinf(big[0])


# --- generated harness ---------------------------------------------------------


def test_float_case_renders_flat_call_and_float_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _descriptors()["squared_difference_float_tail_f16"]
    c_text, h_text, sidecar = _generate(desc, tmp_path, monkeypatch)

    call = re.search(r"arm_elementwise_squared_difference_f16\((.*?)\);", c_text, re.DOTALL)
    assert call is not None
    args = [line.split("//")[0].strip().rstrip(",") for line in call.group(1).strip().splitlines()]
    assert args == ["input1", "input2", "output", "9"]
    assert "input1_offset" not in c_text
    assert "HELIA_VALIDATE_OUTPUTS(\n        FLOAT," in c_text
    assert "0.001f,\n        0.001f," in c_text  # FP16 suite default tolerance
    assert "HELIA_GUARD_CHECK(" in c_text
    assert "#include <math.h>" in h_text
    assert "static const float16_t squared_difference_float_tail_f16_input1[]" in h_text
    assert sidecar["kernel_fn"] == KERNEL
    assert sidecar["comparison"] == {"mode": "float", "atol": 0.001, "rtol": 0.001}
    assert sidecar["scalars"]["block_size"] == 9
    assert sidecar["scalars"]["float_kernel"] is True


def test_random_golden_matches_the_binary16_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _descriptors()["squared_difference_float_odd_block_f16"]
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required")
    op = OpSquaredDifference(desc, seed=1, target_cpu=CPU)
    a, b = op._float_operands((1, 3, 5, 3), (1, 3, 5, 3), np.float16)
    assert a.dtype == np.float16 and b.dtype == np.float16 and a.shape == (1, 3, 5, 3)
    # Both operands come from one RNG stream: a == b would make the golden zero.
    assert not np.array_equal(a, b)
    golden = op._float_reference(np.float16)([a, b])
    assert np.all(golden >= 0) and np.all(np.isfinite(golden))
    assert np.array_equal(golden, op._float_reference(np.float16)([b, a]))  # symmetric


def test_pinned_operands_are_emitted_verbatim_with_exact_goldens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _descriptors()["squared_difference_float_overflow_f16"]
    c_text, h_text, sidecar = _generate(desc, tmp_path, monkeypatch)

    assert "(float16_t)256.0f, (float16_t)0.0f, (float16_t)255.875f, (float16_t)-255.875f" in h_text
    assert "(float16_t)44.0625f, (float16_t)44.1875f" in h_text
    assert _golden(h_text, desc["name"]) == [
        "(float16_t)INFINITY", "(float16_t)INFINITY", "(float16_t)65472.0f", "(float16_t)65472.0f",
        "(float16_t)INFINITY", "(float16_t)65408.0f", "(float16_t)INFINITY", "(float16_t)INFINITY",
        "(float16_t)32768.0f",
    ]
    assert "0.0f,\n        0.0f," in c_text  # zero tolerance
    assert sidecar["comparison"] == {"mode": "float", "atol": 0.0, "rtol": 0.0}


def test_underflow_case_pins_subnormal_squares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _descriptors()["squared_difference_float_underflow_f16"]
    _, h_text, _ = _generate(desc, tmp_path, monkeypatch)
    golden = _golden(h_text, desc["name"])
    # The literal carries float32 round-trip digits and is cast to float16_t by
    # the C compiler, so compare after that same narrowing.
    values = [
        float(np.float16(np.float32(token.replace("(float16_t)", "").rstrip("f"))))
        for token in golden
    ]
    assert values == [2.0 ** -16, 2.0 ** -14, 2.0 ** -24, 0.0, 0.0, 2.0 ** -22, 2.0 ** -20, 0.0, 0.0]
    assert "(float16_t)-0.0f" in h_text  # the -0 - 0 pair is emitted as written


def test_equal_operands_case_is_all_positive_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _descriptors()["squared_difference_float_equal_operands_f16"]
    _, h_text, _ = _generate(desc, tmp_path, monkeypatch)
    assert _golden(h_text, desc["name"]) == ["(float16_t)0.0f"] * 9


@pytest.mark.parametrize(
    ("extras", "match"),
    [
        ({"input_1_values": [1.0] * 8}, "has 8 entries, expected 9"),
        ({"input_2_values": [0.1] + [0.0] * 8}, "not exactly representable in float16"),
        ({"input_1_values": [70000.0] + [0.0] * 8}, "not exactly representable in float16"),
    ],
)
def test_pinned_operands_are_validated(extras: dict, match: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _float_desc("pinned_bad", hint={"extras": extras})
    with pytest.raises(ValueError, match=match):
        _generate(desc, tmp_path, monkeypatch)


def test_pinned_left_operand_cannot_be_combined_with_a_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _float_desc(
        "pinned_sweep",
        hint={"extras": {"input_1_values": [1.0] * 9}},
        input_mode="nonfinite_sweep",
        nonfinite_policy="mask",
    )
    with pytest.raises(ValueError, match="pins input_1_values and requests input_mode 'nonfinite_sweep'"):
        _generate(desc, tmp_path, monkeypatch)


def test_pinned_right_operand_alone_leaves_the_left_drawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _float_desc("pinned_right", hint={"extras": {"input_2_values": [0.5] * 9}})
    _, h_text, _ = _generate(desc, tmp_path, monkeypatch)
    assert h_text.count("(float16_t)0.5f") >= 9
    left = h_text.split("_input1[] = {", 1)[1].split("};", 1)[0]
    assert "(float16_t)0.5f, (float16_t)0.5f, (float16_t)0.5f" not in left


def test_mismatched_shapes_are_rejected_for_the_flat_float_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desc = _float_desc("bcast", shape=(1, 2, 3, 4), shape_2=(1, 1, 1, 4))
    with pytest.raises(NotImplementedError, match="no broadcast entry point"):
        _generate(desc, tmp_path, monkeypatch)


@pytest.mark.parametrize(("name", "token_lanes"), NONFINITE_CASES)
def test_nonfinite_sweep_masks_exactly_the_token_lanes(
    name: str, token_lanes: tuple[int, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desc = _descriptors()[name]
    assert desc["nonfinite_policy"] == "mask"
    c_text, h_text, sidecar = _generate(desc, tmp_path, monkeypatch)

    assert "HELIA_VALIDATE_FLOATS_MASKED(" in c_text
    mask_body = c_text.split(f"{name}_expected_mask[] = {{", 1)[1].split("};", 1)[0]
    mask = [int(tok) for tok in mask_body.replace("\n", " ").split(",") if tok.strip()]
    assert len(mask) == int(np.prod(desc["input_1_shape"]))
    assert [i for i, bit in enumerate(mask) if bit] == list(token_lanes)
    assert sidecar["scalars"]["nonfinite_masked_lanes"] == len(token_lanes)
    # Tokens are in the left operand only; the golden is finite everywhere.
    left = h_text.split("_input1[] = {", 1)[1].split("};", 1)[0]
    assert left.count("NAN") == 1 and left.count("INFINITY") == 2
    assert "NAN" not in "".join(_golden(h_text, name)) and "INFINITY" not in "".join(_golden(h_text, name))


@pytest.mark.parametrize(("name", "kind", "marker"), FAULT_CASES)
def test_fault_cases_render_a_status_only_harness(
    name: str, kind: str, marker: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desc = _descriptors()[name]
    assert desc["fault"] == kind and desc["expected_status"] == "ARM_CMSIS_NN_ARG_ERROR"
    c_text, _, sidecar = _generate(desc, tmp_path, monkeypatch)

    assert marker in c_text
    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in c_text
    assert "ARM_CMSIS_NN_ARG_ERROR" in c_text
    assert "HELIA_VALIDATE_OUTPUTS(" not in c_text and "HELIA_VALIDATE_FLOATS" not in c_text
    assert c_text.count(f"{KERNEL}(") == 1
    if kind == "null_output":
        # No output buffer is passed, so none is declared (-Wunused-variable).
        assert "_output_guard" not in c_text and "HELIA_GUARD_CHECK_UNTOUCHED(" not in c_text
    else:
        assert "HELIA_GUARD_ARM(" in c_text and "true /* poison" in c_text
        assert "HELIA_GUARD_CHECK_UNTOUCHED(" in c_text
    assert sidecar["scalars"]["fault"] == kind
    assert sidecar["scalars"]["expected_status"] == "ARM_CMSIS_NN_ARG_ERROR"


def test_fault_kinds_are_rejected_on_the_int_kernels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required")
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = {"operator": "SquaredDifference", "name": "int_fault", "activation_dtype": "S8", "weight_dtype": "S8",
            "input_1_shape": [1, 2, 2, 3], "input_2_shape": [1, 2, 2, 3],
            "fault": "null_input_1", "expected_status": "ARM_CMSIS_NN_ARG_ERROR"}
    op = OpSquaredDifference(desc, seed=1, target_cpu=CPU)
    op.convert_to_tflite(None, str(tmp_path / "int_fault.tflite"), 1)
    with pytest.raises(ValueError, match="not covered by the float fault template"):
        op.generate_c_files(tmp_path)


def test_hardware_bridge_skips_the_float_kernel_instead_of_extracting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The benchmark firmware dispatches arm_squared_difference_s8/s16 only; an
    # FP16 case must be reported unsupported, not fed to the int extraction
    # (which used to raise KeyError on the missing quantization scalars).
    from helia_core_tester.hardware import generated_test_bridge as bridge

    desc = _descriptors()["squared_difference_float_tail_f16"]
    _generate(desc, tmp_path, monkeypatch)
    case = bridge.GeneratedTestCase(
        name=desc["name"],
        cpu=CPU,
        family="BasicMathFunctions",
        directory=tmp_path,
        descriptor={**desc, "activation_dtype": "FP16"},
        suite="float",
    )
    with pytest.raises(bridge.UnsupportedGeneratedTestError, match="arm_squared_difference_s8/s16"):
        bridge._build_squared_difference_case(TESTER_ROOT, case, output_root=tmp_path / "bundle")


def test_unknown_fault_kind_is_rejected() -> None:
    op = OpSquaredDifference(_float_desc("bad_fault", fault="null_ctx_buf"), seed=1, target_cpu=CPU)
    with pytest.raises(ValueError, match="Unsupported fault 'null_ctx_buf'"):
        op.fault_kind()
