"""Generation-time feature detection for the ns-cmsis-nn temp-buffer sizers.

ns-cmsis-nn#377 / helia-core-tester#71: the LSTM/GRU templates must call the
``*_temp*_get_buffer_size`` queries added by ns-cmsis-nn#381 when the
configured ns-cmsis-nn checkout has them, and must fall back to byte-identical
legacy output when it does not (ns-cmsis-nn main). These tests pin:

1. the header-probe itself (both detection branches, via synthetic Include/
   fixtures);
2. the expected-constant derivations that the templates bake into the
   generated asserts;
3. fallback byte-identity: rendering with the probe reporting "absent" is
   byte-for-byte what rendering without any probe key at all produces, and
   contains none of the sizer machinery.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.generation.utils import temp_sizer_probe as probe

SIZER_SYMBOLS = (
    "arm_lstm_unidirectional_s8_temp1_get_buffer_size",
    "arm_lstm_unidirectional_s8_temp2_get_buffer_size",
    "arm_lstm_unidirectional_s16_temp1_get_buffer_size",
    "arm_lstm_unidirectional_s16_temp2_get_buffer_size",
    "arm_lstm_unidirectional_f32_temp1_get_buffer_size",
    "arm_lstm_unidirectional_f32_temp2_get_buffer_size",
    "arm_lstm_unidirectional_f16_temp1_get_buffer_size",
    "arm_lstm_unidirectional_f16_temp2_get_buffer_size",
    "arm_gru_unidirectional_f32_temp1_get_buffer_size",
    "arm_gru_unidirectional_f16_temp1_get_buffer_size",
)


def _make_checkout(tmp_path: Path, *, with_sizers: bool) -> Path:
    """Synthesize a minimal ns-cmsis-nn checkout Include/ layout."""
    root = tmp_path / ("nn-with-sizers" if with_sizers else "nn-without-sizers")
    include = root / "Include"
    include.mkdir(parents=True)
    int_decls = ["arm_cmsis_nn_status arm_lstm_unidirectional_s8(void);"]
    flt_decls = ["arm_cmsis_nn_status arm_lstm_unidirectional_f32(void);"]
    if with_sizers:
        for sym in SIZER_SYMBOLS:
            target = flt_decls if ("_f32_" in sym or "_f16_" in sym) else int_decls
            target.append(f"int32_t {sym}(const void *params);")
    (include / "arm_nnfunctions.h").write_text("\n".join(int_decls) + "\n")
    (include / "arm_nnfunctions_flt.h").write_text("\n".join(flt_decls) + "\n")
    return root


# ---------------------------------------------------------------------------
# 1. The probe
# ---------------------------------------------------------------------------


def test_probe_detects_symbols_in_synthetic_headers(tmp_path: Path) -> None:
    root = _make_checkout(tmp_path, with_sizers=True)
    assert probe.probe_header_symbols(SIZER_SYMBOLS, cmsis_nn_root=root) is True


def test_probe_reports_absent_for_main_like_headers(tmp_path: Path) -> None:
    root = _make_checkout(tmp_path, with_sizers=False)
    for sym in SIZER_SYMBOLS:
        assert probe.probe_header_symbols([sym], cmsis_nn_root=root) is False


def test_probe_requires_every_symbol(tmp_path: Path) -> None:
    root = _make_checkout(tmp_path, with_sizers=True)
    assert probe.probe_header_symbols(
        [SIZER_SYMBOLS[0], "arm_lstm_unidirectional_s8_temp3_get_buffer_size"],
        cmsis_nn_root=root,
    ) is False


def test_probe_missing_checkout_is_absent(tmp_path: Path) -> None:
    assert probe.probe_header_symbols(SIZER_SYMBOLS, cmsis_nn_root=tmp_path / "nope") is False


def test_probe_ignores_comment_only_mentions(tmp_path: Path) -> None:
    """A header that merely *mentions* every sizer symbol in prose (doxygen
    comments, changelog notes) must NOT flip the probe: the match requires
    declaration shape (``symbol(``), not a substring."""
    root = _make_checkout(tmp_path, with_sizers=False)
    include = root / "Include"
    comment_lines = ["/*", " * Changelog: ns-cmsis-nn#381 will add these queries:"]
    comment_lines += [f" *   {sym} -- see the sizer contract." for sym in SIZER_SYMBOLS]
    comment_lines += [" */", ""]
    for name in ("arm_nnfunctions.h", "arm_nnfunctions_flt.h"):
        header = include / name
        header.write_text("\n".join(comment_lines) + header.read_text())
    for sym in SIZER_SYMBOLS:
        assert probe.probe_header_symbols([sym], cmsis_nn_root=root) is False
    assert probe.probe_header_symbols(SIZER_SYMBOLS, cmsis_nn_root=root) is False


def test_probe_ignores_code_form_comment_mentions(tmp_path: Path) -> None:
    """A doxygen mention in CODE form -- the symbol WITH parens inside a
    block or line comment -- must not read as a declaration either: the
    probe strips comments before matching (independently flagged by review
    round 2 and Copilot)."""
    root = _make_checkout(tmp_path, with_sizers=False)
    include = root / "Include"
    lines = ["/* Call " + SIZER_SYMBOLS[0] + "(&params) before allocating. */"]
    lines += ["// " + sym + "(NULL) returns -1." for sym in SIZER_SYMBOLS]
    lines += [""]
    for name in ("arm_nnfunctions.h", "arm_nnfunctions_flt.h"):
        header = include / name
        header.write_text("\n".join(lines) + header.read_text())
    for sym in SIZER_SYMBOLS:
        assert probe.probe_header_symbols([sym], cmsis_nn_root=root) is False


def test_probe_accepts_declaration_with_whitespace_before_paren(tmp_path: Path) -> None:
    """Declarations split as ``int32_t sym (args)`` or across lines still count."""
    root = _make_checkout(tmp_path, with_sizers=False)
    include = root / "Include"
    decls = [f"int32_t {sym}\n    (const void *params);" for sym in SIZER_SYMBOLS]
    (include / "arm_nnfunctions.h").write_text("\n".join(decls) + "\n")
    assert probe.probe_header_symbols(SIZER_SYMBOLS, cmsis_nn_root=root) is True


def test_probe_rejects_prefix_extended_symbol(tmp_path: Path) -> None:
    """A longer symbol containing a probed name as a prefix-plus-suffix (e.g.
    ``<sym>_v2(``) must not satisfy the probe for ``<sym>``... but note the
    word boundary sits at the *front*; a declared ``x_<sym>(`` must not match
    either."""
    root = _make_checkout(tmp_path, with_sizers=False)
    include = root / "Include"
    sym = SIZER_SYMBOLS[0]
    (include / "arm_nnfunctions.h").write_text(
        f"int32_t {sym}_v2(const void *params);\nint32_t x_{sym}(const void *params);\n"
    )
    assert probe.probe_header_symbols([sym], cmsis_nn_root=root) is False


def test_resolve_root_honors_cmsis_nn_root_env(tmp_path: Path, monkeypatch) -> None:
    root = _make_checkout(tmp_path, with_sizers=True)
    monkeypatch.setenv("CMSIS_NN_ROOT", str(root))
    assert probe.resolve_cmsis_nn_root() == root.resolve()
    assert probe.detect_temp_sizers(SIZER_SYMBOLS, "test") is True


def test_resolve_root_rejects_non_checkout_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CMSIS_NN_ROOT", str(tmp_path))  # no Include/
    assert probe.resolve_cmsis_nn_root() is None
    assert probe.detect_temp_sizers(SIZER_SYMBOLS, "test") is False


# ---------------------------------------------------------------------------
# 2. Expected-constant derivation (mirrors the ns-cmsis-nn#381 sizers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_major", "batch", "hidden", "expected"),
    [
        (False, 1, 8, 16),   # batch-major: gate batch is always 1
        (False, 3, 8, 16),   # batch does not multiply in when batch-major
        (True, 3, 8, 48),    # time-major: batch multiplies in
        (True, 1, 10, 20),
    ],
)
def test_lstm_int_temp_expected_bytes(time_major, batch, hidden, expected) -> None:
    assert (
        probe.lstm_int_temp_expected_bytes(
            time_major=time_major, batch_size=batch, hidden_size=hidden
        )
        == expected
    )


@pytest.mark.parametrize(
    ("reset_after", "hidden", "elem", "expected"),
    [
        (True, 7, 4, 0),    # reset-after: temp1 unused
        (True, 7, 2, 0),
        (False, 7, 4, 28),  # pre-reset f32
        (False, 7, 2, 14),  # pre-reset f16
    ],
)
def test_gru_temp1_expected_bytes(reset_after, hidden, elem, expected) -> None:
    assert (
        probe.gru_temp1_expected_bytes(
            reset_after=reset_after, hidden_size=hidden, elem_bytes=elem
        )
        == expected
    )


def test_lstm_float_temp_expected_bytes_is_zero() -> None:
    assert probe.lstm_float_temp_expected_bytes() == 0


# ---------------------------------------------------------------------------
# 3. Detection branches + fallback byte-identity at the template level
# ---------------------------------------------------------------------------

_TEMPLATES = "LSTMFunctions/lstm_unidirectional/lstm_unidirectional{suffix}.c.j2"


def _render(template_relpath: str, context: dict) -> str:
    from helia_core_tester.generation.ops._shared.base import OperationBase

    class _Renderer(OperationBase):
        def build_keras_model(self):  # pragma: no cover - never called
            raise NotImplementedError

    renderer = _Renderer.__new__(_Renderer)
    is_float = "gru" in template_relpath or "_f32" in template_relpath
    renderer.desc = {
        "operator": "GRUUnidirectional" if "gru" in template_relpath else "LSTMUnidirectional",
        "tensor_dtypes": (
            {"input": "FP32", "output": "FP32"} if is_float else {"input": "S8", "output": "S8"}
        ),
    }
    return renderer.render_template(template_relpath, context)


def _int_lstm_context(**overrides) -> dict:
    context = {
        "name": "case_x",
        "dataset": "lstm_x",
        "macro_prefix": "LSTM_X_",
        "data_prefix": "lstm_x_",
        "dtype": "s8",
        "output_dtype": "int8_t",
        "input_dtype": "int8_t",
        "bias_dtype": "int32_t",
        "weight_dtype": "int8_t",
        "validation_atol": 0.0,
        "validation_rtol": 0.0,
    }
    context.update(overrides)
    return context


def _float_lstm_context(**overrides) -> dict:
    context = {
        "name": "case_x",
        "data_dtype": "float",
        "kernel_fn": "arm_lstm_unidirectional_f32",
        "lstm_params_type": "cmsis_nn_lstm_params_f32",
        "lstm_context_type": "cmsis_nn_lstm_context_f32",
        "time_major_literal": "0",
        "batch_size": 1,
        "time_steps": 4,
        "input_size": 3,
        "hidden_size": 5,
        "cell_clip_literal": "0.0",
        "cell_state_size": 5,
        "dst_size": 20,
        "use_bias": True,
        "validation_atol": 0.001,
        "validation_rtol": 0.001,
    }
    context.update(overrides)
    return context


def _gru_context(**overrides) -> dict:
    context = {
        "name": "case_x",
        "data_dtype": "float32_t",
        "kernel_fn": "arm_gru_unidirectional_f32",
        "gru_params_type": "cmsis_nn_gru_params_f32",
        "gru_context_type": "cmsis_nn_gru_context_f32",
        "output_dtype": "float32_t",
        "time_major_literal": "0",
        "reset_after_literal": "1",
        "reset_after": True,
        "batch_size": 2,
        "time_steps": 3,
        "input_size": 3,
        "hidden_size": 4,
        "hidden_state_size": 8,
        "dst_size": 24,
        "expected_status": "ARM_CMSIS_NN_SUCCESS",
        "use_bias": True,
        "validation_atol": 0.001,
        "validation_rtol": 0.001,
        "gru_temp1_expected_bytes": 0,
        "gru_temp1_expected_bytes_flipped": 16,
    }
    context.update(overrides)
    return context


_CASES = [
    (
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional.c.j2",
        _int_lstm_context,
        {
            "lstm_temp_expected_bytes": 16,
            "lstm_temp_expected_bytes_flipped": 32,
        },
        "arm_lstm_unidirectional_s8_temp1_get_buffer_size",
        # Buffers are guard-wrapped (issue #68); the array itself is the
        # struct's "body" field, so a legacy-sized buffer1 is pinned by its
        # element type/count rather than a plain "static T name[N];" line.
        "int8_t body[LSTM_BUFFER_SIZE];",
    ),
    (
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional_f32.c.j2",
        _float_lstm_context,
        {},
        "arm_lstm_unidirectional_f32_temp1_get_buffer_size",
        "float body[5];",
    ),
    (
        "LSTMFunctions/gru_unidirectional/gru_unidirectional.c.j2",
        _gru_context,
        {},
        "arm_gru_unidirectional_f32_temp1_get_buffer_size",
        "float32_t body[8];",
    ),
]


@pytest.mark.parametrize(
    ("template", "context_factory", "extras", "sizer_symbol", "legacy_line"),
    _CASES,
    ids=["lstm-int", "lstm-float", "gru-float"],
)
def test_fallback_render_is_byte_identical_to_legacy(
    template, context_factory, extras, sizer_symbol, legacy_line
) -> None:
    # No probe key at all == exactly what the pre-detection pipeline rendered;
    # probe reporting False must be byte-identical to it. The probe machinery
    # must leave no trace: no sizer symbol, no detection comment.
    legacy = _render(template, context_factory())
    fallback = _render(template, context_factory(temp_sizers_available=False, **extras))
    assert fallback == legacy
    assert sizer_symbol not in fallback
    assert "temp-buffer sizer detection" not in fallback
    assert legacy_line in fallback


@pytest.mark.parametrize(
    ("template", "context_factory", "extras", "sizer_symbol", "legacy_line"),
    _CASES,
    ids=["lstm-int", "lstm-float", "gru-float"],
)
def test_detected_render_emits_sizer_variant(
    template, context_factory, extras, sizer_symbol, legacy_line
) -> None:
    rendered = _render(template, context_factory(temp_sizers_available=True, **extras))
    assert sizer_symbol in rendered
    assert "temp-buffer sizer detection: DETECTED" in rendered
    assert "check_temp_sizers" in rendered
    # Every sizer variant asserts the NULL-params -1 contract edge.
    assert "temp1_null_params" in rendered


def test_detected_int_lstm_sizes_buffers_from_expected_constant() -> None:
    rendered = _render(
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional.c.j2",
        _int_lstm_context(
            temp_sizers_available=True,
            lstm_temp_expected_bytes=16,
            lstm_temp_expected_bytes_flipped=32,
        ),
    )
    assert "#define LSTM_TEMP_EXPECTED_BYTES 16" in rendered
    assert "#define LSTM_TEMP_FLIPPED_EXPECTED_BYTES 32" in rendered
    # Buffers are guard-wrapped (issue #68): buffer1 is `buffer1_guard.body`,
    # sized by the block immediately preceding its `#define buffer1 (...)`.
    assert (
        "int16_t body[LSTM_TEMP_EXPECTED_BYTES / 2];\n"
        "    uint8_t tail[HELIA_GUARD_BYTES];\n"
        "} buffer1_guard;\n"
        "#define buffer1 (buffer1_guard.body)"
    ) in rendered
    assert (
        "int8_t body[LSTM_BUFFER_SIZE];\n"
        "    uint8_t tail[HELIA_GUARD_BYTES];\n"
        "} buffer1_guard;\n"
        "#define buffer1 (buffer1_guard.body)"
    ) not in rendered
    # cell_state (buffer3) stays on the legacy allocation: out of the sizers' scope.
    assert (
        "int8_t body[LSTM_BUFFER_SIZE];\n"
        "    uint8_t tail[HELIA_GUARD_BYTES];\n"
        "} buffer3_guard;\n"
        "#define buffer3 (buffer3_guard.body)"
    ) in rendered
    assert "temp1_bytes_flipped_time_major" in rendered
    assert "temp1_capacity_bytes" in rendered


def test_detected_float_lstm_passes_null_temp_buffers() -> None:
    rendered = _render(
        "LSTMFunctions/lstm_unidirectional/lstm_unidirectional_f32.c.j2",
        _float_lstm_context(temp_sizers_available=True),
    )
    assert ".temp1 = NULL," in rendered
    assert ".temp2 = NULL," in rendered
    assert "case_x_temp1" not in rendered
    assert "arm_lstm_unidirectional_f32_temp2_get_buffer_size(NULL)" in rendered


def test_detected_gru_reset_after_passes_null_temp1() -> None:
    rendered = _render(
        "LSTMFunctions/gru_unidirectional/gru_unidirectional.c.j2",
        _gru_context(temp_sizers_available=True),
    )
    assert ".temp1 = NULL," in rendered
    assert "static float32_t case_x_temp1" not in rendered
    assert "temp1_bytes_flipped_reset_after" in rendered
    assert "temp1_negative_hidden" in rendered


def test_detected_gru_pre_reset_sizes_temp1_from_sizer_contract() -> None:
    rendered = _render(
        "LSTMFunctions/gru_unidirectional/gru_unidirectional.c.j2",
        _gru_context(
            temp_sizers_available=True,
            reset_after=False,
            reset_after_literal="0",
            gru_temp1_expected_bytes=16,
            gru_temp1_expected_bytes_flipped=0,
        ),
    )
    # hidden_size elements, not the legacy batch*hidden heuristic. Guard-wrapped
    # (issue #68): temp1 is case_x_temp1_guard.body.
    assert (
        "float32_t body[4];\n"
        "    uint8_t tail[HELIA_GUARD_BYTES];\n"
        "} case_x_temp1_guard;\n"
        "#define case_x_temp1 (case_x_temp1_guard.body)"
    ) in rendered
    assert ".temp1 = case_x_temp1," in rendered
    assert "temp1_capacity_bytes" in rendered
