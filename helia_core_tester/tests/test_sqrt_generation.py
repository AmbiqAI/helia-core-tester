from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.sqrt import (
    OpSqrt,
    _quant_param_to_scalar,
    make_sqrt_lut,
    make_sqrt_lut_s16,
    make_sqrt_lut_s8,
)
from helia_core_tester.generation.utils.litert_builder import LITERT_AVAILABLE
from helia_core_tester.generation.utils.litert_utils import (
    get_operator_tensors_from_litert,
    load_litert_model,
)

TESTER_ROOT = Path(__file__).resolve().parents[2]
SQRT_DESCRIPTOR_PATH = TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "sqrt.yaml"
SQRT_PARITY_CASES = (
    ("sqrt_small_tensor_s8", "S8", (1, 2, 3, 4), "arm_sqrt_s8", np.int8),
    ("sqrt_small_tensor_s16", "S16", (1, 2, 3, 4), "arm_sqrt_s16", np.int16),
    ("sqrt_long_row_s8", "S8", (1, 1, 64, 1), "arm_sqrt_s8", np.int8),
    ("sqrt_long_row_s16", "S16", (1, 1, 64, 1), "arm_sqrt_s16", np.int16),
    ("sqrt_multi_batch_s8", "S8", (2, 3, 5, 3), "arm_sqrt_s8", np.int8),
    ("sqrt_multi_batch_s16", "S16", (2, 3, 5, 3), "arm_sqrt_s16", np.int16),
)

SQRT_S16_REGRESSION_INPUTS = np.array([696, 1601, 6447, 21618], dtype=np.int16)
SQRT_S16_REGRESSION_EXPECTED = np.array([4775, 7243, 14534, 26615], dtype=np.int16)


def _sqrt_descriptor_map() -> dict[str, dict]:
    return {desc["name"]: desc for desc in load_descriptor(str(SQRT_DESCRIPTOR_PATH))}


def _simulate_arm_sqrt_s16(input_values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    output = np.zeros_like(input_values, dtype=np.int16)
    for idx, value in enumerate(input_values.astype(np.int16)):
        lut_index = 256 + (int(value) >> 7)
        offset = int(value) & 0x7F
        base = int(lut[lut_index])
        slope = int(lut[lut_index + 1]) - base
        output[idx] = np.int16(base + ((slope * offset + 64) >> 7))
    return output


def test_sqrt_descriptors_match_unit_test_parity() -> None:
    descriptors = load_descriptor(str(SQRT_DESCRIPTOR_PATH))

    assert [desc["name"] for desc in descriptors] == [name for name, *_ in SQRT_PARITY_CASES]
    assert len(descriptors) == 6

    for desc, (_, dtype, shape, _, _) in zip(descriptors, SQRT_PARITY_CASES):
        assert desc["operator"] == "Sqrt"
        assert desc["activation_dtype"] == dtype
        assert desc["weight_dtype"] == "S8"
        assert tuple(desc["input_shape"]) == shape
        if dtype == "S16":
            assert desc["resolved_comparison"] == {"mode": "tolerant_int", "tolerance": 1}
        else:
            assert desc["resolved_comparison"] == {"mode": "exact_int"}


def test_quant_param_to_scalar_rejects_non_scalar_arrays() -> None:
    with pytest.raises(ValueError, match="Sqrt expects scalar quantization"):
        _quant_param_to_scalar(np.array([1.0, 2.0], dtype=np.float32), "input_scale", float)


def test_make_sqrt_lut_s8_clamps_negative_domain_to_output_zero_point() -> None:
    lut = make_sqrt_lut_s8(
        input_scale=np.array([0.25], dtype=np.float32),
        input_zp=np.array([0], dtype=np.int64),
        output_scale=np.array([0.125], dtype=np.float32),
        output_zp=np.array([-5], dtype=np.int64),
    )

    assert lut.dtype == np.int8
    assert int(lut[0]) == -5
    assert int(lut[255]) == -5


def test_make_sqrt_lut_s16_uses_513_entry_int16_contract() -> None:
    lut = make_sqrt_lut_s16(
        input_scale=np.array([0.25], dtype=np.float32),
        input_zp=np.array([0], dtype=np.int64),
        output_scale=np.array([0.125], dtype=np.float32),
        output_zp=np.array([-5], dtype=np.int64),
    )

    assert lut.dtype == np.int16
    assert lut.shape == (513,)
    assert int(lut[0]) == -5
    assert int(lut[256]) == -5
    assert int(lut[-1]) >= -5


def test_make_sqrt_lut_s16_tracks_regression_inputs_within_one_lsb() -> None:
    lut = make_sqrt_lut_s16(
        input_scale=np.array([1.0 / 32768.0], dtype=np.float32),
        input_zp=np.array([0], dtype=np.int64),
        output_scale=np.array([1.0 / 32768.0], dtype=np.float32),
        output_zp=np.array([0], dtype=np.int64),
    )

    actual = _simulate_arm_sqrt_s16(SQRT_S16_REGRESSION_INPUTS, lut)
    diffs = np.abs(actual.astype(np.int32) - SQRT_S16_REGRESSION_EXPECTED.astype(np.int32))

    assert int(diffs.max()) <= 1


@pytest.mark.parametrize(
    ("dtype", "expected_dtype", "expected_len"),
    (("S8", np.int8, 256), ("S16", np.int16, 513)),
)
def test_make_sqrt_lut_dispatches_per_dtype(dtype: str, expected_dtype, expected_len: int) -> None:
    lut = make_sqrt_lut(
        input_scale=np.array([0.25], dtype=np.float32),
        input_zp=np.array([0], dtype=np.int64),
        output_scale=np.array([0.125], dtype=np.float32),
        output_zp=np.array([-5], dtype=np.int64),
        activation_dtype=dtype,
    )

    assert lut.dtype == expected_dtype
    assert lut.shape == (expected_len,)


@pytest.mark.parametrize(("name", "dtype", "shape", "expected_kernel", "output_dtype"), SQRT_PARITY_CASES)
def test_sqrt_parity_descriptors_generate_litert_and_c(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    expected_kernel: str,
    output_dtype,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for sqrt LiteRT generation")

    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqrt_descriptor_map()[name]
    assert desc["activation_dtype"] == dtype
    op = OpSqrt(desc, seed=1, target_cpu="cortex-m55")
    tflite_path = tmp_path / f"{name}.tflite"

    assert op.needs_keras_model() is False
    with pytest.raises(NotImplementedError, match="LiteRT-only"):
        op.build_keras_model()

    op.convert_to_tflite(None, str(tflite_path), 1)

    model, subgraph = load_litert_model(str(tflite_path))
    op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
    assert tuple(op_tensors["inputs"][0]["shape"]) == shape
    assert tuple(op_tensors["outputs"][0]["shape"]) == shape

    fake_output = np.zeros(shape, dtype=output_dtype)
    monkeypatch.setattr(op, "run_inference", lambda *_args, **_kwargs: fake_output)
    op.generate_c_files(tmp_path)

    c_path = tmp_path / f"{name}_sqrt.c"
    h_path = tmp_path / "includes" / f"{name}_sqrt.h"
    assert c_path.exists()
    assert h_path.exists()

    content = c_path.read_text()
    header = h_path.read_text()
    assert expected_kernel in content
    if dtype == "S8":
        assert "static int8_t sqrt_lut[256]" in header
        assert "HELIA_VALIDATE_OUTPUTS_EXACT_INT" in content or "EXACT_INT" in content
    else:
        assert "static int16_t sqrt_lut[513]" in header
        assert "static int8_t sqrt_lut[513]" not in header
        assert "TOLERANT_INT" in content


def test_sqrt_generator_rejects_unsupported_dtype() -> None:
    op = OpSqrt(
        {
            "name": "sqrt_bad_dtype_s32",
            "operator": "Sqrt",
            "activation_dtype": "S32",
            "weight_dtype": "S8",
            "input_shape": [1, 1, 4, 1],
        },
        seed=1,
        target_cpu="cortex-m55",
    )

    with pytest.raises(NotImplementedError, match="Unsupported Sqrt dtype"):
        op._select_cmsis_sqrt_kernel()
