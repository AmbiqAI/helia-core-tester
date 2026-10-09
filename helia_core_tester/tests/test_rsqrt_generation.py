from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.rsqrt import (
    OpRsqrt,
    _quant_param_to_scalar,
    derive_rsqrt_universal_quant_params,
    make_rsqrt_per_op_lut,
    make_rsqrt_universal_lut,
)

TESTER_ROOT = Path(__file__).resolve().parents[2]
RSQRT_DESCRIPTOR_PATH = TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "rsqrt.yaml"
RSQRT_PARITY_CASES = (
    ("rsqrt_small_tensor_per_op_s16", "per_op", (1, 2, 3, 4), "arm_rsqrt_s16_per_op"),
    ("rsqrt_small_tensor_universal_s16", "universal", (1, 2, 3, 4), "arm_rsqrt_s16_universal"),
    ("rsqrt_long_row_per_op_s16", "per_op", (1, 1, 64, 1), "arm_rsqrt_s16_per_op"),
    ("rsqrt_long_row_universal_s16", "universal", (1, 1, 64, 1), "arm_rsqrt_s16_universal"),
    ("rsqrt_multi_batch_per_op_s16", "per_op", (2, 3, 5, 3), "arm_rsqrt_s16_per_op"),
    ("rsqrt_multi_batch_universal_s16", "universal", (2, 3, 5, 3), "arm_rsqrt_s16_universal"),
    ("rsqrt_small_input_per_op_s16", "per_op", (1, 1, 64, 1), "arm_rsqrt_s16_per_op"),
    ("rsqrt_small_input_universal_s16", "universal", (1, 1, 64, 1), "arm_rsqrt_s16_universal"),
    ("rsqrt_mid_input_per_op_s16", "per_op", (1, 1, 64, 1), "arm_rsqrt_s16_per_op"),
    ("rsqrt_mid_input_universal_s16", "universal", (1, 1, 64, 1), "arm_rsqrt_s16_universal"),
)


def _rsqrt_descriptor_map() -> dict[str, dict]:
    return {desc["name"]: desc for desc in load_descriptor(str(RSQRT_DESCRIPTOR_PATH))}


def test_rsqrt_descriptors_match_unit_test_parity() -> None:
    descriptors = load_descriptor(str(RSQRT_DESCRIPTOR_PATH))

    assert [desc["name"] for desc in descriptors] == [name for name, *_ in RSQRT_PARITY_CASES]
    assert len(descriptors) == 10

    for desc, (_, call_style, shape, _) in zip(descriptors, RSQRT_PARITY_CASES):
        assert desc["operator"] == "Rsqrt"
        assert desc["activation_dtype"] == "S16"
        assert desc["weight_dtype"] == "S8"
        assert desc["hint"]["call_style"] == call_style
        assert tuple(desc["input_shape"]) == shape


def test_rsqrt_quant_param_to_scalar_rejects_non_scalar_arrays() -> None:
    with pytest.raises(ValueError, match="Rsqrt expects scalar quantization"):
        _quant_param_to_scalar(np.array([1.0, 2.0], dtype=np.float32), "input_scale", float)


def test_rsqrt_per_op_lut_has_expected_shape_and_monotonicity() -> None:
    lut = make_rsqrt_per_op_lut(1.0 / 32768.0, 1.0 / 32768.0, 0)

    assert lut.shape == (513,)
    assert lut.dtype == np.int16
    assert int(lut[256]) == 32767
    assert int(lut[512]) > 0
    assert int(lut[512]) <= int(lut[256])
    assert int(lut[300]) >= int(lut[400]) >= int(lut[500])


def test_rsqrt_universal_lut_has_expected_shape_and_monotonicity() -> None:
    lut = make_rsqrt_universal_lut(1.0 / 32768.0)

    assert lut.shape == (513,)
    assert lut.dtype == np.int32
    assert int(lut[256]) == 32767
    assert int(lut[512]) > 0
    assert int(lut[300]) >= int(lut[400]) >= int(lut[500])


def test_rsqrt_universal_quant_params_default_to_no_rescale() -> None:
    params = derive_rsqrt_universal_quant_params(1.0 / 32768.0)

    assert params["needs_rescale"] == 0
    assert params["out_mult"] > 0


def test_rsqrt_universal_quant_params_derive_rescale_for_custom_scale() -> None:
    params = derive_rsqrt_universal_quant_params(1.0 / 16384.0)

    assert params["needs_rescale"] == 1
    assert params["out_mult"] > 0


def test_rsqrt_positive_input_generation_stays_in_valid_domain() -> None:
    op = OpRsqrt(
        {
            "name": "rsqrt_positive_domain_s16",
            "operator": "Rsqrt",
            "activation_dtype": "S16",
            "weight_dtype": "S8",
            "input_shape": [1, 2, 2, 3],
        },
        seed=1,
        target_cpu="cortex-m55",
    )

    input_data = op._generate_positive_float_input((1, 2, 2, 3), 1.0 / 32768.0)
    input_q = op._quantize_input(input_data, 1.0 / 32768.0, 0)

    op._ensure_positive_domain_input(input_q, 0)
    assert np.all(input_q.astype(np.int32) >= 0)


def test_rsqrt_positive_domain_validator_rejects_negative_inputs() -> None:
    op = OpRsqrt(
        {
            "name": "rsqrt_negative_domain_s16",
            "operator": "Rsqrt",
            "activation_dtype": "S16",
            "weight_dtype": "S8",
            "input_shape": [1, 2, 2, 3],
        },
        seed=1,
        target_cpu="cortex-m55",
    )

    with pytest.raises(ValueError, match="non-negative post-offset domain"):
        op._ensure_positive_domain_input(np.array([-1, 0, 1], dtype=np.int16), 0)


def test_rsqrt_generator_rejects_unsupported_dtype() -> None:
    op = OpRsqrt(
        {
            "name": "rsqrt_bad_dtype_s8",
            "operator": "Rsqrt",
            "activation_dtype": "S8",
            "weight_dtype": "S8",
            "input_shape": [1, 1, 4, 1],
        },
        seed=1,
        target_cpu="cortex-m55",
    )

    with pytest.raises(NotImplementedError, match="Unsupported Rsqrt dtype"):
        op._select_cmsis_rsqrt_kernel()


def test_rsqrt_generator_rejects_unknown_call_style() -> None:
    op = OpRsqrt(
        {
            "name": "rsqrt_bad_call_style_s16",
            "operator": "Rsqrt",
            "activation_dtype": "S16",
            "weight_dtype": "S8",
            "input_shape": [1, 1, 4, 1],
            "hint": {"call_style": "bogus"},
        },
        seed=1,
        target_cpu="cortex-m55",
    )

    with pytest.raises(ValueError, match="Unsupported Rsqrt call_style"):
        op._variant()


@pytest.mark.parametrize(("name", "call_style", "shape", "expected_kernel"), RSQRT_PARITY_CASES)
def test_rsqrt_parity_descriptors_generate_litert_and_c(
    name: str,
    call_style: str,
    shape: tuple[int, ...],
    expected_kernel: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _rsqrt_descriptor_map()[name]
    assert desc["hint"]["call_style"] == call_style

    op = OpRsqrt(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)

    c_path = tmp_path / f"{name}_rsqrt.c"
    h_path = tmp_path / "includes" / f"{name}_rsqrt.h"
    assert c_path.exists()
    assert h_path.exists()

    c_content = c_path.read_text()
    h_content = h_path.read_text()
    assert expected_kernel in c_content
    if call_style == "universal":
        assert "int32_t" in h_content
        assert "out_mult" in c_content
        assert "needs_rescale" in c_content
    else:
        assert "int16_t" in h_content
        assert "out_mult" not in c_content
        assert "needs_rescale" not in c_content


@pytest.mark.parametrize(("call_style", "expected_kernel"), (("per_op", "arm_rsqrt_s16_per_op"), ("universal", "arm_rsqrt_s16_universal")))
def test_rsqrt_negative_input_case_generates_expected_status_contract(
    call_style: str,
    expected_kernel: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = {
        "name": f"rsqrt_negative_case_{call_style}_s16",
        "operator": "Rsqrt",
        "activation_dtype": "S16",
        "weight_dtype": "S8",
        "input_shape": [1, 1, 4, 1],
        "hint": {"call_style": call_style, "force_negative_input_case": True},
    }
    op = OpRsqrt(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)

    c_content = (tmp_path / f"{desc['name']}_rsqrt.c").read_text()
    assert expected_kernel in c_content
    assert "HELIA_VALIDATE_EXPECTED_STATUS(" in c_content
    assert "ARM_CMSIS_NN_ARG_ERROR" in c_content
