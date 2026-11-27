"""Exercise actual LiteRT resize models and their coordinate options."""

from pathlib import Path

import numpy as np
import pytest
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.catalog import get_operator_spec
from helia_core_tester.generation.utils.litert_builder import (
    build_resize_bilinear_op,
    build_resize_nearest_neighbor_op,
)
from helia_core_tester.generation.ops.ReshapeFunctions.resize_bilinear import (
    OpResizeBilinear,
)


@pytest.mark.parametrize(
    "builder,align,half,expected",
    [
        (build_resize_bilinear_op, False, False, [0, 6, 12, 12]),
        (build_resize_bilinear_op, False, True, [0, 3, 9, 12]),
        (build_resize_bilinear_op, True, False, [0, 4, 8, 12]),
        (build_resize_nearest_neighbor_op, False, False, [0, 0, 12, 12]),
    ],
)
def test_model_operator_and_coordinate_options(builder, align, half, expected):
    model = builder(
        input_shape=[1, 1, 2, 1],
        new_size=[1, 4],
        align_corners=align,
        half_pixel_centers=half,
    )
    interpreter = Interpreter(
        model_content=model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF
    )
    interpreter.allocate_tensors()
    interpreter.set_tensor(
        interpreter.get_input_details()[0]["index"],
        np.array([0, 12], dtype=np.int8).reshape(1, 1, 2, 1),
    )
    interpreter.invoke()
    result = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    np.testing.assert_array_equal(result.reshape(-1), expected)


def test_bilinear_rejects_unsupported_dtype_before_generating(tmp_path):
    op = OpResizeBilinear.__new__(OpResizeBilinear)
    op.desc = {"activation_dtype": "S16", "input_shape": [1, 1, 2, 1], "size": [1, 4]}
    with pytest.raises(ValueError, match="require S8"):
        op.convert_to_tflite(None, str(tmp_path / "model.tflite"), rep_seed=0)


@pytest.mark.parametrize("bits", [16, 32])
def test_float_descriptor_generates_direct_call_and_reference(tmp_path, monkeypatch, bits):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("CMSIS_NN_REPO_ROOT", str(root))
    relpath = "ReshapeFunctions/resize_bilinear_float.yaml"
    assert relpath in get_operator_spec("ResizeBilinear").descriptor_relpaths
    name = f"resize_bilinear_aligned_odd_f{bits}"
    desc = next(d for d in load_descriptor(str(root / "assets/descriptors" / relpath)) if d["name"] == name)
    symbol = f"arm_resize_bilinear_f{bits}"
    assert desc["required_kernel_symbols"] == [symbol]
    op = OpResizeBilinear(desc, seed=1, target_cpu="cortex-m55")
    model_path = tmp_path / f"{name}.tflite"
    op.convert_to_tflite(None, str(model_path), rep_seed=1)
    interpreter = Interpreter(model_path=str(model_path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    assert interpreter.get_input_details()[0]["dtype"] == np.float32
    op.generate_c_files(tmp_path)
    source = (tmp_path / f"{name}_resize_bilinear.c").read_text()
    header = (tmp_path / "includes" / f"{name}_resize_bilinear.h").read_text()
    assert f"{symbol}(" in source
    storage = "float16_t" if bits == 16 else "float"
    assert f"{storage} {name}_expected_output[]" in header
