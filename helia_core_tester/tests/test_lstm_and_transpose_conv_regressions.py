from __future__ import annotations

from pathlib import Path

import numpy as np

from helia_core_tester.generation.utils import lstm_data
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


def test_transpose_conv_s8_non_reverse_buffer_covers_mve_stride_h_width_term() -> None:
    buffer_size = TemplateContextBuilder.calculate_transpose_conv_buffer_size_max(
        input_dims={"n": 1, "h": 3, "w": 3, "c": 32},
        filter_dims={"n": 32, "h": 2, "w": 2, "c": 8},
        output_dims={"n": 1, "h": 8, "w": 4, "c": 32},
        output_dtype="S8",
        stride_h=3,
        stride_w=1,
    )

    assert buffer_size == 1920


class _FakeWeight:
    def assign(self, _value) -> None:
        pass


class _FakeLayerWithWeights:
    def __init__(self, *args, **kwargs) -> None:
        self.weights = [_FakeWeight(), _FakeWeight(), _FakeWeight()]

    def __call__(self, _value):
        return object()


class _FakeLambdaLayer:
    def __init__(self, fn, *args, **kwargs) -> None:
        self._fn = fn

    def __call__(self, value):
        return self._fn(value)


class _FakeLayers:
    LSTM = _FakeLayerWithWeights
    Lambda = _FakeLambdaLayer

    @staticmethod
    def Input(*args, **kwargs):
        return object()


class _FakeModel:
    def __init__(self, *args, **kwargs) -> None:
        self.layers = [object(), _FakeLayerWithWeights(), _FakeLayerWithWeights()]


class _FakeKeras:
    layers = _FakeLayers
    Model = _FakeModel


class _FakeTf:
    keras = _FakeKeras
    float32 = "float32"

    @staticmethod
    def convert_to_tensor(value, dtype=None):
        return value

    @staticmethod
    def transpose(value, perm):
        return value


def _fake_lstm_data(batch_size: int, time_steps: int, input_size: int, hidden_size: int) -> lstm_data.LstmGeneratedData:
    tensors = {
        "input_tensor": np.zeros(time_steps * batch_size * input_size, dtype=np.int8),
        "output": np.zeros(batch_size * time_steps * hidden_size, dtype=np.int8),
        "input_gate_input_weights": np.zeros(input_size * hidden_size, dtype=np.int8),
        "forget_gate_input_weights": np.zeros(input_size * hidden_size, dtype=np.int8),
        "cell_gate_input_weights": np.zeros(input_size * hidden_size, dtype=np.int8),
        "output_gate_input_weights": np.zeros(input_size * hidden_size, dtype=np.int8),
        "input_gate_hidden_weights": np.zeros(hidden_size * hidden_size, dtype=np.int8),
        "forget_gate_hidden_weights": np.zeros(hidden_size * hidden_size, dtype=np.int8),
        "cell_gate_hidden_weights": np.zeros(hidden_size * hidden_size, dtype=np.int8),
        "output_gate_hidden_weights": np.zeros(hidden_size * hidden_size, dtype=np.int8),
        "input_gate_bias": np.zeros(hidden_size, dtype=np.int32),
        "forget_gate_bias": np.zeros(hidden_size, dtype=np.int32),
        "cell_gate_bias": np.zeros(hidden_size, dtype=np.int32),
        "output_gate_bias": np.zeros(hidden_size, dtype=np.int32),
    }
    params = {
        "input_zero_point": 128,
        "output_zero_point": 1,
        "cell_scale_power": -14,
        "cell_clip": 32767,
    }
    for key in lstm_data._MULTIPLIER_KEYS:
        params[f"{key}_multiplier"] = 1
        params[f"{key}_shift"] = 0

    return lstm_data.LstmGeneratedData(params=params, tensors=tensors, scales={}, effective_scales={})


def test_lstm_s8_conversion_failure_falls_back_to_generation_ready_data(tmp_path: Path, monkeypatch) -> None:
    fallback_data = _fake_lstm_data(batch_size=1, time_steps=10, input_size=22, hidden_size=11)

    def _raise_conversion_error(*args, **kwargs):
        raise TypeError("'NoneType' object is not callable")

    monkeypatch.setattr(lstm_data, "tf", _FakeTf)
    monkeypatch.setattr(lstm_data, "_convert_keras_to_tflite", _raise_conversion_error)
    monkeypatch.setattr(lstm_data, "_load_unit_test_data", lambda dataset: fallback_data)

    data = lstm_data.generate_lstm_data(
        rng=np.random.default_rng(123),
        activation_dtype="S8",
        batch_size=1,
        time_steps=10,
        input_size=22,
        hidden_size=11,
        time_major=True,
        templates_dir=tmp_path,
        schema_path=tmp_path / "schema.fbs",
        work_dir=tmp_path,
        dataset="lstm_1",
    )

    context = lstm_data.build_lstm_context(
        name="lstm_unidirectional_dataset_1_time_major_s8",
        dataset="lstm_1",
        activation_dtype="S8",
        batch_size=1,
        time_steps=10,
        input_size=22,
        hidden_size=11,
        time_major=True,
        data=data,
    )

    assert data is fallback_data
    assert context["tensors"]["input_tensor"].size == 220
    assert context["mult_shift"]["output_multiplier"] == 1
