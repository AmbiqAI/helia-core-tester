from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.squared_difference import (
    OpSquaredDifference,
    squared_difference_quant_preset,
)
from helia_core_tester.generation.utils.tflite_utils import (
    elementwise_squared_difference_quant_params,
)

TESTER_ROOT = Path(__file__).resolve().parents[2]
SQDIFF_DESCRIPTOR_PATH = (
    TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "squared_difference.yaml"
)
SQDIFF_PARITY_CASES = (
    ("squared_difference_ident_s8", "S8", (1, 2, 3, 4), (1, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_scalar_input1_s8", "S8", (1, 1, 1, 1), (1, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_scalar_input2_s8", "S8", (1, 2, 3, 4), (1, 1, 1, 1), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_n_s8", "S8", (1, 2, 3, 4), (2, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_h_s8", "S8", (1, 1, 2, 3), (1, 4, 2, 3), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_w_s8", "S8", (1, 2, 3, 4), (1, 2, 1, 4), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_c_s8", "S8", (1, 2, 3, 4), (1, 2, 3, 1), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_hc_s8", "S8", (1, 2, 3, 1), (1, 1, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_hc_w2_s8", "S8", (1, 1, 2, 1), (1, 2, 2, 4), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_hc_w2_rev_s8", "S8", (1, 2, 2, 4), (1, 1, 2, 1), "arm_squared_difference_s8"),
    ("squared_difference_row_scalar_input1_s8", "S8", (1, 2, 1, 1), (1, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_row_scalar_input2_s8", "S8", (1, 2, 3, 4), (1, 2, 1, 1), "arm_squared_difference_s8"),
    ("squared_difference_batch_broadcast_input1_s8", "S8", (1, 2, 3, 4), (2, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_batch_broadcast_input2_s8", "S8", (2, 2, 3, 4), (1, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_ident_s16", "S16", (1, 2, 3, 4), (1, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_scalar_input1_s16", "S16", (1, 1, 1, 1), (1, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_scalar_input2_s16", "S16", (1, 2, 3, 4), (1, 1, 1, 1), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_n_s16", "S16", (1, 2, 3, 4), (2, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_h_s16", "S16", (1, 1, 2, 3), (1, 4, 2, 3), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_w_s16", "S16", (1, 2, 3, 4), (1, 2, 1, 4), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_c_s16", "S16", (1, 2, 3, 4), (1, 2, 3, 1), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_hc_s16", "S16", (1, 2, 3, 1), (1, 1, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_hc_w2_s16", "S16", (1, 1, 2, 1), (1, 2, 2, 4), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_hc_w2_rev_s16", "S16", (1, 2, 2, 4), (1, 1, 2, 1), "arm_squared_difference_s16"),
    ("squared_difference_row_scalar_input1_s16", "S16", (1, 2, 1, 1), (1, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_row_scalar_input2_s16", "S16", (1, 2, 3, 4), (1, 2, 1, 1), "arm_squared_difference_s16"),
    ("squared_difference_batch_broadcast_input1_s16", "S16", (1, 2, 3, 4), (2, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_batch_broadcast_input2_s16", "S16", (2, 2, 3, 4), (1, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_row_scalar_channel_s8", "S8", (1, 3, 1, 1), (1, 3, 1, 4), "arm_squared_difference_s8"),
    ("squared_difference_row_scalar_channel_reverse_s8", "S8", (1, 3, 1, 4), (1, 3, 1, 1), "arm_squared_difference_s8"),
    ("squared_difference_row_scalar_channel_s16", "S16", (1, 3, 1, 1), (1, 3, 1, 4), "arm_squared_difference_s16"),
    ("squared_difference_row_scalar_channel_reverse_s16", "S16", (1, 3, 1, 4), (1, 3, 1, 1), "arm_squared_difference_s16"),
)


def _sqdiff_desc(
    name: str,
    dtype: str,
    *,
    input_1_shape: tuple[int, ...] = (1, 2, 2, 3),
    input_2_shape: tuple[int, ...] = (1, 2, 2, 3),
    call_style: str | None = None,
) -> dict:
    desc = {
        "operator": "SquaredDifference",
        "name": name,
        "activation_dtype": dtype,
        "weight_dtype": "S8",
        "input_1_shape": list(input_1_shape),
        "input_2_shape": list(input_2_shape),
    }
    if call_style:
        desc["hint"] = {"call_style": call_style}
    return desc


def _sqdiff_descriptor_map() -> dict[str, dict]:
    return {desc["name"]: desc for desc in load_descriptor(str(SQDIFF_DESCRIPTOR_PATH))}


def test_squared_difference_descriptors_match_unit_test_parity() -> None:
    descriptors = load_descriptor(str(SQDIFF_DESCRIPTOR_PATH))

    assert [desc["name"] for desc in descriptors] == [case[0] for case in SQDIFF_PARITY_CASES]
    assert len(descriptors) == len(SQDIFF_PARITY_CASES)

    for desc, (name, dtype, input_1_shape, input_2_shape, _) in zip(descriptors, SQDIFF_PARITY_CASES):
        assert desc["name"] == name
        assert desc["operator"] == "SquaredDifference"
        assert desc["activation_dtype"] == dtype
        assert desc["weight_dtype"] == "S8"
        assert tuple(desc["input_1_shape"]) == input_1_shape
        assert tuple(desc["input_2_shape"]) == input_2_shape
        assert desc.get("hint", {}) == {}


def test_squared_difference_quant_params_s8_match_expected_shape() -> None:
    params = elementwise_squared_difference_quant_params(
        input1_scale=1.0 / 128.0,
        input2_scale=1.0 / 256.0,
        output_scale=1.0 / 64.0,
        activation_dtype="S8",
    )

    assert params["left_shift"] == 7
    assert params["input1_shift"] == 0
    assert params["input2_shift"] == -1
    assert params["out_shift"] == -19


def test_squared_difference_quant_params_s16_match_expected_shape() -> None:
    params = elementwise_squared_difference_quant_params(
        input1_scale=1.0 / 32768.0,
        input2_scale=1.0 / 65536.0,
        output_scale=1.0 / 32768.0,
        activation_dtype="S16",
    )

    assert params["left_shift"] == 0
    assert params["input1_shift"] == 0
    assert params["input2_shift"] == -1
    assert params["out_shift"] == -12


def test_squared_difference_uses_preset_quantization() -> None:
    op = OpSquaredDifference(_sqdiff_desc("sqdiff_s8", "S8"), seed=1, target_cpu="cortex-m55")
    (s1, zp1), (s2, zp2), (so, zpo) = op._preset_quant()
    # A moderate asymmetric input zero point: -128 would pin every lane to a
    # non-negative post-offset value and hide the sign-dependent kernel paths,
    # 0 would leave the input offset term dead in every s8 case. Only the
    # output, non-negative by definition, keeps -128 (hct#81).
    assert (s1, zp1) == (pytest.approx(1.0 / 128.0), -40)
    assert (s2, zp2) == (pytest.approx(1.0 / 256.0), -40)
    assert (so, zpo) == (pytest.approx(1.0 / 64.0), -128)


def test_squared_difference_golden_is_the_tflite_formula() -> None:
    a = np.array([-128, -40, 0, 127], dtype=np.int8)
    b = np.array([127, -40, 5, -128], dtype=np.int8)
    params = elementwise_squared_difference_quant_params(
        input1_scale=1.0 / 128.0, input2_scale=1.0 / 256.0, output_scale=1.0 / 64.0, activation_dtype="S8"
    )
    out = OpSquaredDifference._simulate_squared_difference_quantized(
        a, b, input1_offset=40, input2_offset=40, input1_mult=params["input1_mult"],
        input1_shift=params["input1_shift"], input2_mult=params["input2_mult"], input2_shift=params["input2_shift"],
        left_shift=params["left_shift"], out_offset=-128, out_mult=params["out_mult"], out_shift=params["out_shift"],
        out_activation_min=-128, out_activation_max=127, out_dtype=np.int8,
    )
    real = ((a.astype(float) + 40) / 128.0 - (b.astype(float) + 40) / 256.0) ** 2
    assert out.dtype == np.int8
    assert out[1] == -128
    np.testing.assert_allclose(out.astype(float), np.clip(np.round(real * 64) - 128, -128, 127), atol=1)


def test_squared_difference_s8_generates_expected_c_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqdiff_desc("sqdiff_s8", "S8")
    op = OpSquaredDifference(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)

    c_path = tmp_path / "sqdiff_s8_squared_difference.c"
    assert c_path.exists()
    content = c_path.read_text()

    assert "arm_squared_difference_s8" in content
    # input*_offset is -zero_point; only the non-negative output keeps the
    # -128 zero point (hct#81).
    assert "40, /* input1_offset */" in content
    assert "40, /* input2_offset */" in content
    assert "-128, /* out_offset */" in content
    assert "0, /* input1_shift */" in content
    assert "-1, /* input2_shift */" in content
    assert "7, /* left_shift */" in content


@pytest.mark.parametrize(
    ("name", "dtype", "input_1_shape", "input_2_shape", "expected_kernel"),
    SQDIFF_PARITY_CASES,
)
def test_squared_difference_parity_descriptors_generate_wrapper_c(
    name: str,
    dtype: str,
    input_1_shape: tuple[int, ...],
    input_2_shape: tuple[int, ...],
    expected_kernel: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqdiff_descriptor_map()[name]
    assert desc["activation_dtype"] == dtype
    assert tuple(desc["input_1_shape"]) == input_1_shape
    assert tuple(desc["input_2_shape"]) == input_2_shape

    op = OpSquaredDifference(desc, seed=1, target_cpu="cortex-m55")

    op.generate_c_files(tmp_path)

    c_path = tmp_path / f"{name}_squared_difference.c"
    h_path = tmp_path / "includes" / f"{name}_squared_difference.h"
    assert c_path.exists()
    assert h_path.exists()

    content = c_path.read_text()
    assert expected_kernel in content
    assert "arm_elementwise_squared_difference_s16" not in content or expected_kernel == "arm_elementwise_squared_difference_s16"


def test_squared_difference_s16_elementwise_generates_expected_c_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqdiff_desc("sqdiff_s16", "S16", call_style="elementwise")
    op = OpSquaredDifference(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)

    c_path = tmp_path / "sqdiff_s16_squared_difference.c"
    assert c_path.exists()
    content = c_path.read_text()

    assert "arm_elementwise_squared_difference_s16" in content
    assert "0, /* input_1_offset */" in content
    assert "0, /* input_2_offset */" in content
    assert "0, /* out_offset */" in content
    assert "0, /* input_1_shift */" in content
    assert "-1, /* input_2_shift */" in content
    assert "0, /* left_shift */" in content
    assert "-12, /* out_shift */" in content


def test_relu_range_preset_is_kept_on_exactly_two_s8_cases() -> None:
    # The regime the upstream reference vectors for this kernel were captured
    # in: zero point at the bottom of the domain, every post-offset lane
    # non-negative. Two cases hold it; the rest carry a sign-spanning offset.
    from helia_core_tester.generation.io.descriptors import load_all_descriptors

    descriptors = load_all_descriptors(str(TESTER_ROOT / "assets" / "descriptors"))
    sq = [d for d in descriptors if d["operator"] == "SquaredDifference"]
    relu_range = [d for d in sq if d.get("quant_preset") == "relu_range"]
    assert len(relu_range) == 2
    for desc in relu_range:
        assert desc["activation_dtype"] == "S8"
        # The rule cannot steer past a zero point that pins the domain, so the
        # waiver is what keeps these cases generatable.
        assert set(desc["operand_sign_span_exempt"]) == {"input_1", "input_2"}
    preset = squared_difference_quant_preset("int8", "relu_range")
    assert preset["input_1_quant"][1] == [-128]
    assert preset["input_2_quant"][1] == [-128]


def test_unknown_quant_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="no 'wat' quantization preset"):
        squared_difference_quant_preset("int8", "wat")
