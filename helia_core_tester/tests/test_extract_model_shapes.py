"""Tests for the MLPerf layer-shape script."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "extract_model_shapes.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("extract_model_shapes", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_activation_splits_layer_key():
    script = _load_script()
    relu = {"activation": "RELU", "input_shape": [1, 32, 32, 16]}
    plain = {**relu, "activation": "NONE"}
    assert script.layer_key(relu) != script.layer_key(plain)
