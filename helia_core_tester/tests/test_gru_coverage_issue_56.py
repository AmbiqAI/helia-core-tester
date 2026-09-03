"""Regression guards for the GRU/LSTM coverage backlog (issue #56).

Locks in the flag-combination, time_steps == 0, and null-buffers descriptors
added to close #56, so a future accidental edit that silently drops one of
these combinations is caught here rather than by nothing at all -- the exact
"untestable blind spot" shape the issue is about.
"""
from __future__ import annotations

from pathlib import Path

from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops.LSTMFunctions.gru_unidirectional import OpGRUUnidirectional
from helia_core_tester.generation.ops.LSTMFunctions.lstm_unidirectional import OpLSTMUnidirectional


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


def test_gru_fault_and_no_bias_compose(tmp_path: Path) -> None:
    # Copilot review on the PR that introduced fault: and use_bias support
    # (#56): the fault template referenced `{{name}}_*_bias` unconditionally
    # while the header only declares those arrays when use_bias is true, so
    # a descriptor combining both would generate C that references an
    # undeclared identifier. This exercises that exact combination
    # end-to-end and inspects the rendered C, not just descriptor fields.
    desc = _load_gru("gru_unidirectional_error_null_input_no_bias_f32")
    assert desc["fault"] == "null_input"
    assert desc["use_bias"] is False

    op = OpGRUUnidirectional(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    c_path = next(tmp_path.rglob("*.c"))
    rendered = c_path.read_text()

    # Every bias field must be NULL, not a reference to an array the header
    # never declares.
    assert rendered.count(".input_bias = NULL,") == 3
    assert rendered.count(".hidden_bias = NULL,") == 3
    assert "_bias," not in rendered  # no leftover bare bias-array references


def test_lstm_fault_and_no_bias_compose(tmp_path: Path) -> None:
    desc = _load_lstm("lstm_unidirectional_error_null_input_no_bias_f32")
    assert desc["fault"] == "null_input"
    assert desc["use_bias"] is False

    op = OpLSTMUnidirectional(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    c_path = next(tmp_path.rglob("*.c"))
    rendered = c_path.read_text()

    assert rendered.count(".bias = NULL,") == 4
    assert "_bias," not in rendered


def test_gru_no_bias_combines_with_pre_reset_time_major_and_streaming() -> None:
    # Follow-up to #56: use_bias: false was only ever tested with the
    # baseline flag set. This is the completion of that gap -- every other
    # flag axis now has a no_bias combination.
    pre_reset = _load_gru("gru_unidirectional_float_no_bias_pre_reset_f32")
    assert pre_reset["use_bias"] is False
    assert pre_reset["reset_after"] is False

    time_major = _load_gru("gru_unidirectional_float_no_bias_time_major_f32")
    assert time_major["use_bias"] is False
    assert time_major["time_major"] is True

    stream = _load_gru("gru_unidirectional_float_no_bias_stream_f32")
    assert stream["use_bias"] is False
    assert stream["hint"]["stream"] is True


def test_lstm_no_bias_combines_with_time_major() -> None:
    desc = _load_lstm("lstm_unidirectional_float_no_bias_time_major_f32")
    assert desc["use_bias"] is False
    assert desc["time_major"] is True


def test_gru_no_bias_stream_composes_end_to_end(tmp_path: Path) -> None:
    # The stream template (gru_unidirectional_stream.c.j2) had the same
    # unconditional bias-symbol reference Copilot flagged in the fault
    # template -- it wasn't exercised by the original PR because no
    # descriptor combined use_bias: false with streaming. Fixed alongside
    # this case; verify end-to-end like the fault+no_bias tests above.
    desc = _load_gru("gru_unidirectional_float_no_bias_stream_f32")
    op = OpGRUUnidirectional(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    c_path = next(tmp_path.rglob("*.c"))
    rendered = c_path.read_text()

    assert rendered.count(".input_bias = NULL,") == 3
    assert rendered.count(".hidden_bias = NULL,") == 3
    assert "_bias," not in rendered


def test_lstm_streaming_carries_both_hidden_and_cell_state(tmp_path: Path) -> None:
    # LSTM previously had zero hidden_state/cell_state streaming coverage at
    # all (unlike GRU). Unlike GRU, LSTM's cell_state is also caller-owned
    # state once hidden_state != NULL (arm_lstm_unidirectional_f32.c only
    # auto-zeroes cell_state in single-shot/NULL-hidden_state mode) -- the
    # new stream template must seed AND preserve cell_state across chunks,
    # not just hidden_state. Verify the rendered C actually does this.
    desc = _load_lstm("lstm_unidirectional_float_stream_f32")
    assert desc["hint"]["stream"] is True
    assert desc["batch_size"] == 1

    op = OpLSTMUnidirectional(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    c_path = next(tmp_path.rglob("*.c"))
    rendered = c_path.read_text()

    # Both state buffers seeded to zero once, and both passed (by name, not
    # NULL) into every chunk call -- not re-zeroed between chunks.
    assert rendered.count("hidden_state[h] = (float)0;") == 1
    assert rendered.count("cell_state[h] = (float)0;") == 1
    assert f".hidden_state = {desc['name']}_hidden_state," in rendered
    assert f".cell_state = {desc['name']}_cell_state," in rendered
    assert rendered.count("run_lstm_chunk(") >= 3  # 1 definition + >=2 call sites (chunk_lengths: [2, 2])


def test_lstm_streaming_combines_with_time_major_and_no_bias() -> None:
    time_major = _load_lstm("lstm_unidirectional_float_stream_time_major_f32")
    assert time_major["hint"]["stream"] is True
    assert time_major["time_major"] is True

    no_bias = _load_lstm("lstm_unidirectional_float_no_bias_stream_f32")
    assert no_bias["hint"]["stream"] is True
    assert no_bias["use_bias"] is False


def test_lstm_no_bias_stream_composes_end_to_end(tmp_path: Path) -> None:
    desc = _load_lstm("lstm_unidirectional_float_no_bias_stream_f32")
    op = OpLSTMUnidirectional(desc, seed=1, target_cpu="cortex-m55")
    op.generate_c_files(tmp_path)
    c_path = next(tmp_path.rglob("*.c"))
    rendered = c_path.read_text()

    assert rendered.count(".bias = NULL,") == 4
    assert "_bias," not in rendered


def test_gru_f16_twins_exist_for_all_new_issue_56_cases() -> None:
    for f32_name in [
        "gru_unidirectional_float_pre_reset_time_major_f32",
        "gru_unidirectional_float_pre_reset_stream_f32",
        "gru_unidirectional_float_time_major_stream_f32",
        "gru_unidirectional_float_zero_time_steps_f32",
        "gru_unidirectional_float_null_buffers_reset_after_f32",
        "gru_unidirectional_float_no_bias_f32",
        "gru_unidirectional_float_no_bias_pre_reset_f32",
        "gru_unidirectional_float_no_bias_time_major_f32",
        "gru_unidirectional_float_no_bias_stream_f32",
    ]:
        f16_name = f32_name[: -len("_f32")] + "_f16"
        _load_gru(f32_name)
        _load_gru(f16_name)  # raises AssertionError if missing


def test_lstm_f16_twins_exist_for_all_new_follow_up_cases() -> None:
    for f32_name in [
        "lstm_unidirectional_float_no_bias_time_major_f32",
        "lstm_unidirectional_float_stream_f32",
        "lstm_unidirectional_float_stream_time_major_f32",
        "lstm_unidirectional_float_no_bias_stream_f32",
    ]:
        f16_name = f32_name[: -len("_f32")] + "_f16"
        _load_lstm(f32_name)
        _load_lstm(f16_name)  # raises AssertionError if missing
