"""Regression guards for the GRU/LSTM coverage backlog (issue #56).

Locks in the flag-combination, time_steps == 0, and null-buffers descriptors
added to close #56, so a future accidental edit that silently drops one of
these combinations is caught here rather than by nothing at all -- the exact
"untestable blind spot" shape the issue is about.
"""
from __future__ import annotations

from pathlib import Path

from helia_core_tester.generation.io.descriptors import load_descriptor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_descriptor(relpath: str, name: str) -> dict:
    desc_path = _repo_root() / "assets" / "descriptors" / relpath
    for desc in load_descriptor(str(desc_path)):
        if desc.get("name") == name:
            return desc
    raise AssertionError(f"descriptor {name!r} not found in {desc_path}")


def _load_gru(name: str) -> dict:
    return _load_descriptor("LSTMFunctions/gru_unidirectional_float.yaml", name)


def _load_lstm(name: str) -> dict:
    return _load_descriptor("LSTMFunctions/lstm_unidirectional_float.yaml", name)


def test_gru_pre_reset_combines_with_time_major() -> None:
    desc = _load_gru("gru_unidirectional_float_pre_reset_time_major_f32")
    assert desc["reset_after"] is False
    assert desc["time_major"] is True


def test_gru_pre_reset_combines_with_streaming() -> None:
    desc = _load_gru("gru_unidirectional_float_pre_reset_stream_f32")
    assert desc["reset_after"] is False
    assert desc["hint"]["stream"] is True
    assert desc["batch_size"] == 1  # streaming is batch_size == 1 only


def test_gru_time_major_combines_with_streaming() -> None:
    desc = _load_gru("gru_unidirectional_float_time_major_stream_f32")
    assert desc["time_major"] is True
    assert desc["hint"]["stream"] is True


def test_gru_zero_time_steps_is_a_legal_no_op() -> None:
    desc = _load_gru("gru_unidirectional_float_zero_time_steps_f32")
    assert desc["time_steps"] == 0
    # Not a fault case: a 0-length streaming chunk must return SUCCESS.
    assert desc.get("fault") is None
    assert desc.get("expected_status") is None


def test_lstm_zero_time_steps_is_a_legal_no_op() -> None:
    desc = _load_lstm("lstm_unidirectional_float_zero_time_steps_f32")
    assert desc["time_steps"] == 0
    assert desc.get("fault") is None


def test_gru_null_buffers_is_legal_only_because_reset_after_true() -> None:
    desc = _load_gru("gru_unidirectional_float_null_buffers_reset_after_f32")
    assert desc["fault"] == "null_buffers"
    assert desc["reset_after"] is True
    # expected_status defaults to SUCCESS (see gru_unidirectional_fault.c.j2) --
    # this is a legal-path guard, not an error-injection case.
    assert desc.get("expected_status") is None


def test_gru_no_bias_passes_null_bias_pointers() -> None:
    desc = _load_gru("gru_unidirectional_float_no_bias_f32")
    assert desc["use_bias"] is False


def test_lstm_no_bias_passes_null_bias_pointers() -> None:
    desc = _load_lstm("lstm_unidirectional_float_no_bias_f32")
    assert desc["use_bias"] is False


def test_lstm_fault_mechanism_ported_from_gru() -> None:
    # LSTM previously had no `fault:` support at all (see gru_unidirectional.py
    # for the pattern this mirrors). buffers/cell_state are unconditionally
    # required for LSTM (no reset_after-style legal NULL path, unlike GRU),
    # so null_buffers is a real rejection case here, not a legal-path guard.
    for name, expected_fault in [
        ("lstm_unidirectional_error_null_input_f32", "null_input"),
        ("lstm_unidirectional_error_null_output_f32", "null_output"),
        ("lstm_unidirectional_error_null_params_f32", "null_params"),
        ("lstm_unidirectional_error_null_buffers_f32", "null_buffers"),
    ]:
        desc = _load_lstm(name)
        assert desc["fault"] == expected_fault
        assert desc["expected_status"] == "ARM_CMSIS_NN_ARG_ERROR"


def test_gru_f16_twins_exist_for_all_new_issue_56_cases() -> None:
    for f32_name in [
        "gru_unidirectional_float_pre_reset_time_major_f32",
        "gru_unidirectional_float_pre_reset_stream_f32",
        "gru_unidirectional_float_time_major_stream_f32",
        "gru_unidirectional_float_zero_time_steps_f32",
        "gru_unidirectional_float_null_buffers_reset_after_f32",
        "gru_unidirectional_float_no_bias_f32",
    ]:
        f16_name = f32_name[: -len("_f32")] + "_f16"
        _load_gru(f32_name)
        _load_gru(f16_name)  # raises AssertionError if missing
