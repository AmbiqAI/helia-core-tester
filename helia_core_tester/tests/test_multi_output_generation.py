import pytest

from helia_core_tester.generation.ops.split import OpSplit
from helia_core_tester.generation.ops.unpack import OpUnpack


def test_split_generates_multiple_outputs(tmp_path):
    tf = pytest.importorskip("tensorflow")
    desc = {
        "operator": "Split",
        "name": "split_test",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "activation": "NONE",
        "input_shape": [1, 2, 2, 4],
        "axis": -1,
        "num_splits": 2,
    }
    op = OpSplit(desc, seed=1, target_cpu="cortex-m55")
    model = op.build_keras_model()
    tflite_path = tmp_path / "split_test.tflite"
    op.convert_to_tflite(model, str(tflite_path), 1)
    op.generate_c_files(tmp_path)

    c_path = tmp_path / "split_test_split.c"
    assert c_path.exists()
    content = c_path.read_text()
    assert "split_test_out_0" in content
    assert "split_test_out_1" in content


def test_unpack_generates_multiple_outputs(tmp_path):
    tf = pytest.importorskip("tensorflow")
    desc = {
        "operator": "Unpack",
        "name": "unpack_test",
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "activation": "NONE",
        "input_shape": [1, 2, 2, 2],
        "axis": -1,
        "num_tensors": 2,
    }
    op = OpUnpack(desc, seed=1, target_cpu="cortex-m55")
    model = op.build_keras_model()
    tflite_path = tmp_path / "unpack_test.tflite"
    op.convert_to_tflite(model, str(tflite_path), 1)
    op.generate_c_files(tmp_path)

    c_path = tmp_path / "unpack_test_unpack.c"
    assert c_path.exists()
    content = c_path.read_text()
    assert "unpack_test_out_0" in content
    assert "unpack_test_out_1" in content
