from pathlib import Path

import numpy as np
import pytest

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.BasicMathFunctions.squared_difference import (
    OpSquaredDifference,
    build_squared_difference_op,
)
from helia_core_tester.generation.utils.litert_builder import LITERT_AVAILABLE
from helia_core_tester.generation.utils.litert_utils import (
    get_operator_tensors_from_litert,
    load_litert_model,
)
from helia_core_tester.generation.utils.tflite_utils import (
    elementwise_squared_difference_quant_params,
)

TESTER_ROOT = Path(__file__).resolve().parents[2]
SQDIFF_DESCRIPTOR_PATH = (
    TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "squared_difference.yaml"
)
SQDIFF_PARITY_CASES = (
    ("squared_difference_scalar_s8", "S8", (1, 2, 3, 4), (1, 1, 1, 1), "arm_squared_difference_s8"),
    ("squared_difference_ident_s8", "S8", (1, 2, 3, 4), (1, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_n_s8", "S8", (1, 2, 3, 4), (2, 2, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_h_s8", "S8", (1, 1, 2, 3), (1, 4, 2, 3), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_w_s8", "S8", (1, 2, 3, 4), (1, 2, 1, 4), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_c_s8", "S8", (1, 2, 3, 4), (1, 2, 3, 1), "arm_squared_difference_s8"),
    ("squared_difference_broadcast_hc_s8", "S8", (1, 2, 3, 1), (1, 1, 3, 4), "arm_squared_difference_s8"),
    ("squared_difference_scalar_s16", "S16", (1, 2, 3, 4), (1, 1, 1, 1), "arm_squared_difference_s16"),
    ("squared_difference_ident_s16", "S16", (1, 2, 3, 4), (1, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_n_s16", "S16", (1, 2, 3, 4), (2, 2, 3, 4), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_h_s16", "S16", (1, 1, 2, 3), (1, 4, 2, 3), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_w_s16", "S16", (1, 2, 3, 4), (1, 2, 1, 4), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_c_s16", "S16", (1, 2, 3, 4), (1, 2, 3, 1), "arm_squared_difference_s16"),
    ("squared_difference_broadcast_hc_s16", "S16", (1, 2, 3, 1), (1, 1, 3, 4), "arm_squared_difference_s16"),
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
    assert len(descriptors) == 14

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


def test_squared_difference_builder_uses_explicit_quantization(tmp_path: Path) -> None:
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for squared difference LiteRT generation")

    model_bytes = build_squared_difference_op(
        input_1_shape=(1, 2, 2, 3),
        input_2_shape=(1, 2, 2, 3),
        dtype="int8",
    )
    tflite_path = tmp_path / "sqdiff_s8.tflite"
    tflite_path.write_bytes(model_bytes)

    model, subgraph = load_litert_model(str(tflite_path))
    op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)

    input1_quant = op_tensors["inputs"][0]["quantization"]
    input2_quant = op_tensors["inputs"][1]["quantization"]
    output_quant = op_tensors["outputs"][0]["quantization"]

    assert input1_quant["scale"] == pytest.approx(1.0 / 128.0)
    assert input1_quant["zero_point"] == -128
    assert input2_quant["scale"] == pytest.approx(1.0 / 256.0)
    assert input2_quant["zero_point"] == -128
    assert output_quant["scale"] == pytest.approx(1.0 / 64.0)
    assert output_quant["zero_point"] == -128


def test_squared_difference_s8_generates_expected_c_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for squared difference LiteRT generation")

    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqdiff_desc("sqdiff_s8", "S8")
    op = OpSquaredDifference(desc, seed=1, target_cpu="cortex-m55")
    tflite_path = tmp_path / "sqdiff_s8.tflite"
    op.convert_to_tflite(None, str(tflite_path), 1)
    op.generate_c_files(tmp_path)

    c_path = tmp_path / "sqdiff_s8_squared_difference.c"
    assert c_path.exists()
    content = c_path.read_text()

    assert "arm_squared_difference_s8" in content
    assert "128,       // input1_offset" in content
    assert "128,       // input2_offset" in content
    assert "-128,          // out_offset" in content
    assert "0,        // input1_shift" in content
    assert "-1,        // input2_shift" in content
    assert "7,          // left_shift" in content


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
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for squared difference LiteRT generation")

    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqdiff_descriptor_map()[name]
    assert desc["activation_dtype"] == dtype
    assert tuple(desc["input_1_shape"]) == input_1_shape
    assert tuple(desc["input_2_shape"]) == input_2_shape

    op = OpSquaredDifference(desc, seed=1, target_cpu="cortex-m55")
    tflite_path = tmp_path / f"{name}.tflite"
    op.convert_to_tflite(None, str(tflite_path), 1)

    if dtype == "S16" and input_1_shape == input_2_shape:
        fake_output = np.zeros(input_1_shape, dtype=np.int16)

        class _FakeInterpreter:
            def get_input_details(self):
                return [{"index": 0}, {"index": 1}]

            def get_output_details(self):
                return [{"index": 0}]

            def set_tensor(self, index, value):
                del index, value

            def invoke(self):
                return None

            def get_tensor(self, index):
                del index
                return fake_output

        monkeypatch.setattr(op, "load_litert_interpreter", lambda _: _FakeInterpreter())

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
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for squared difference LiteRT generation")

    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(TESTER_ROOT))
    desc = _sqdiff_desc("sqdiff_s16", "S16", call_style="elementwise")
    op = OpSquaredDifference(desc, seed=1, target_cpu="cortex-m55")
    tflite_path = tmp_path / "sqdiff_s16.tflite"
    op.convert_to_tflite(None, str(tflite_path), 1)

    fake_output = np.zeros((1, 2, 2, 3), dtype=np.int16)

    class _FakeInterpreter:
        def get_input_details(self):
            return [{"index": 0}, {"index": 1}]

        def get_output_details(self):
            return [{"index": 0}]

        def set_tensor(self, index, value):
            del index, value

        def invoke(self):
            return None

        def get_tensor(self, index):
            del index
            return fake_output

    monkeypatch.setattr(op, "load_litert_interpreter", lambda _: _FakeInterpreter())
    op.generate_c_files(tmp_path)

    c_path = tmp_path / "sqdiff_s16_squared_difference.c"
    assert c_path.exists()
    content = c_path.read_text()

    assert "arm_elementwise_squared_difference_s16" in content
    assert "0,       // input1_offset" in content
    assert "0,       // input2_offset" in content
    assert "0,          // out_offset" in content
    assert "0,        // input1_shift" in content
    assert "-1,        // input2_shift" in content
    assert "0,          // left_shift" in content
    assert "-12,           // out_shift" in content
