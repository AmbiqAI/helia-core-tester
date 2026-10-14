"""Exercise actual LiteRT resize models and their coordinate options."""

import numpy as np
import pytest
from ai_edge_litert.interpreter import Interpreter, OpResolverType

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
