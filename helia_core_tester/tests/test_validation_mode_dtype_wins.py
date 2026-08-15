"""Regression lock for issue #54: float outputs must get float validation.

Before the fix, ``infer_validation_mode`` consulted template-path allowlists
before the output dtype, so float descriptors routed through shared int
templates (convolve, pooling, softmax, lstm, ...) emitted an INTEGER
comparison: ``(long long)`` truncation made every output with ``|v| < 1``
match unconditionally. 209 of 373 float cases performed no float comparison
at all; a 5% error in every Convolve output and a swapped LSTM gate pair both
passed 100% of their suites.

These tests pin the contract: output dtype wins over the allowlists, and a
float-dtype context can never resolve (or be explicitly coerced) to an
integer validation mode.
"""

from __future__ import annotations

import pytest

from helia_core_tester.generation.utils.template_context import TemplateContextBuilder


_INT_ALLOWLISTED_TEMPLATES = sorted(
    TemplateContextBuilder._EXACT_INT_VALIDATION_TEMPLATES
    | TemplateContextBuilder._TOLERANT_INT_VALIDATION_TEMPLATES
)

_FLOAT_OUTPUT_DTYPES = ("float32_t", "float16_t", "float")


@pytest.mark.parametrize("template_path", _INT_ALLOWLISTED_TEMPLATES)
@pytest.mark.parametrize("output_dtype", _FLOAT_OUTPUT_DTYPES)
def test_float_dtype_wins_over_int_template_allowlists(template_path, output_dtype):
    """Every int-allowlisted template must yield FLOAT mode for float outputs."""
    mode = TemplateContextBuilder.infer_validation_mode(
        template_path, {"output_dtype": output_dtype}
    )
    assert mode == "float", (
        f"{template_path} resolved '{mode}' for output_dtype={output_dtype}; "
        "float outputs must use the float validator (issue #54)"
    )


@pytest.mark.parametrize("template_path", _INT_ALLOWLISTED_TEMPLATES)
def test_int_dtype_still_uses_template_allowlists(template_path):
    """Int descriptors keep their historical int comparison modes."""
    expected = (
        "exact_int"
        if template_path in TemplateContextBuilder._EXACT_INT_VALIDATION_TEMPLATES
        else "tolerant_int"
    )
    mode = TemplateContextBuilder.infer_validation_mode(
        template_path, {"output_dtype": "int8_t"}
    )
    assert mode == expected


@pytest.mark.parametrize("coerced_mode", ("exact_int", "tolerant_int"))
def test_explicit_int_coercion_of_float_output_is_rejected(coerced_mode):
    """An explicit validation_mode override cannot force int compare on floats."""
    with pytest.raises(ValueError, match="coercion"):
        TemplateContextBuilder.build_validation_context(
            "ConvolutionFunctions/convolve/convolve.c.j2",
            {"output_dtype": "float32_t", "validation_mode": coerced_mode},
        )


@pytest.mark.parametrize(
    "template_path",
    (
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional.c.j2",
        "SVDFunctions/svdf/svdf.c.j2",
    ),
)
def test_data_dtype_fallback_covers_generators_without_output_dtype(template_path):
    """LSTM/SVDF historically set only data_dtype; the resolver must fall back
    to it rather than silently dropping to the int allowlists (issue #54)."""
    mode = TemplateContextBuilder.infer_validation_mode(
        template_path, {"data_dtype": "float16_t"}
    )
    assert mode == "float"


def test_output_dtype_beats_data_dtype_when_both_present():
    """Quantize-style ops (float data in, int out) must keep int validation."""
    mode = TemplateContextBuilder.infer_validation_mode(
        "QuantizationFunctions/quantize/quantize.c.j2",
        {"data_dtype": "float", "output_dtype": "int8_t"},
    )
    assert mode == "tolerant_int"


def test_build_validation_context_emits_float_for_float_output():
    """End-to-end: a float context through an int-allowlisted template renders
    FLOAT mode, float helpers, and the dtype-default tolerances."""
    resolved = TemplateContextBuilder.build_validation_context(
        "ConvolutionFunctions/convolve/convolve.c.j2",
        {"output_dtype": "float32_t"},
    )
    assert resolved["validation_mode"] == "float"
    assert resolved["validation_mode_token"] == "FLOAT"
    assert "float" in resolved["validation_helpers"]
    assert resolved["validation_atol"] > 0.0
