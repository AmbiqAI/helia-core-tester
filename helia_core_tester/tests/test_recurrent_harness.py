"""LSTM and GRU render through the generic harness (iteration 1b, G16c): their bodies stay Jinja
fragments and the kernel call goes through `_run(input, output, params, buffers)`, whose struct
types come from the kernel contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.harness import HarnessError
from helia_core_tester.generation.ops._shared.recurrent_pool import recurrent_argument_pool
from helia_core_tester.tests.test_lstm_gru_temp_sizer_generation import (
    _float_lstm_context,
    _gru_context,
    _int_lstm_context,
    _render,
)

LSTM = "LSTMFunctions/lstm_unidirectional"
GRU = "LSTMFunctions/gru_unidirectional"
TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates"


def _signature(source: str) -> list[str]:
    text = re.search(r"int32_t case_x_run\((.*?)\) \{", source, flags=re.S).group(1)
    return [" ".join(part.split()) for part in text.split(",")]


def _bound(source: str, kernel: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", "", re.search(rf"return {kernel}\((.*?)\);", source, flags=re.S).group(1))
    return [a.strip() for a in body.split(",")]


@pytest.mark.parametrize("key, context, kernel, signature", [
    (f"{LSTM}/lstm_unidirectional.c.j2", _int_lstm_context(), "arm_lstm_unidirectional_s8",
     ["const int8_t* __restrict input", "int8_t* __restrict output", "const cmsis_nn_lstm_params * params",
      "cmsis_nn_lstm_context * buffers"]),
    (f"{LSTM}/lstm_unidirectional_f32.c.j2", _float_lstm_context(), "arm_lstm_unidirectional_f32",
     ["const float* __restrict input", "float* __restrict output", "const cmsis_nn_lstm_params_f32 * params",
      "cmsis_nn_lstm_context_f32 * buffers"]),
    (f"{GRU}/gru_unidirectional.c.j2", _gru_context(), "arm_gru_unidirectional_f32",
     ["const float32_t* __restrict input", "float32_t* __restrict output", "const cmsis_nn_gru_params_f32 * params",
      "cmsis_nn_gru_context_f32 * buffers"]),
])
def test_run_takes_the_params_and_buffers_structs_with_their_contract_types(key, context, kernel, signature) -> None:
    source = _render(key, context)
    assert _signature(source) == signature
    assert _bound(source, kernel) == ["input", "output", "params", "buffers"]
    # The fragment's own run function calls the bound wrapper, which is defined ahead of it.
    helper = "run_gru" if "gru" in key else "run_lstm"
    assert source.index("int32_t case_x_run(") < source.index(f"static int32_t {helper}(void)")
    assert re.search(r"case_x_run\([^;]*&params, &buffers\);", source)
    assert "_Static_assert(__builtin_types_compatible_p(__typeof__(" + kernel in source


def test_the_pool_refuses_a_kernel_without_params_and_buffers() -> None:
    with pytest.raises(HarnessError, match="arm_relu_s8 takes no 'params', so it is not a recurrent kernel"):
        recurrent_argument_pool({"name": "x", "kernel_fn": "arm_relu_s8"}, body=f"{GRU}/gru_unidirectional.fragment.j2",
                                header=f"{GRU}/gru_unidirectional.fragment.j2", ctype="float")


@pytest.mark.parametrize("fragment_path", sorted(str(p.relative_to(TEMPLATES)) for p in TEMPLATES.glob("**/*.fragment.j2")))
def test_every_fragment_defines_the_macros_its_pool_imports_and_calls_run(fragment_path: str) -> None:
    text = (TEMPLATES / fragment_path).read_text()
    assert "{% macro file_scope() %}" in text and "{% macro test_body() %}" in text
    assert "{{ name }}_run(" in text, "the kernel call must go through the harness's bound wrapper"
    assert "{{ kernel_fn }}(" not in text and not re.search(r"\barm_(lstm|gru|svdf)\w*?_(s8|s16|f32|f16)\(", text)
    # A macro's `set` variables are its own: a switch the test body reads must be set there too.
    scope, body = text.split("{% macro test_body() %}")
    for name in re.findall(r"\{% set (\w+) =", scope):
        if re.search(rf"\b{name}\b", body):
            assert f"{{% set {name} =" in body, f"{fragment_path}: {name} is set only in file_scope"
