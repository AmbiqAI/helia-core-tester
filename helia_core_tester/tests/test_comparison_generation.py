import pytest

from helia_core_tester.generation.ops.comparison import OpComparison
from helia_core_tester.generation.utils.litert_builder import LITERT_AVAILABLE


def test_comparison_equal_generates(tmp_path):
    if not LITERT_AVAILABLE:
        pytest.skip("ai_edge_litert is required for comparison LiteRT generation")

    desc = {
        "operator": "Comparison",
        "name": "equal_test",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "activation": "NONE",
        "operation": "equal",
        "input_1_shape": [1, 2, 2, 1],
        "input_2_shape": [1, 2, 2, 1],
    }
    op = OpComparison(desc, seed=1, target_cpu="cortex-m55")
    assert op.needs_keras_model() is False
    with pytest.raises(NotImplementedError, match="LiteRT-only"):
        op.build_keras_model()
    tflite_path = tmp_path / "equal_test.tflite"
    op.convert_to_tflite(None, str(tflite_path), 1)
    op.generate_c_files(tmp_path)

    c_path = tmp_path / "equal_test_comparison.c"
    h_path = tmp_path / "includes" / "equal_test_comparison.h"
    assert c_path.exists()
    assert h_path.exists()
    content = c_path.read_text()
    assert "arm_equal_s8" in content
