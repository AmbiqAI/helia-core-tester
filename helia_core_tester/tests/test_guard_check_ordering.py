"""The guard check has to run before any validator that can return early.

HELIA_VALIDATE_STATUS, HELIA_VALIDATE_EXPECTED_STATUS and
HELIA_VALIDATE_SCALAR_EQ_INT all `return` on a failure, so a HELIA_GUARD_CHECK
placed after them is skipped exactly when a kernel returned an error after
scribbling out of bounds -- the case the guard exists to catch. Both the
template text and a rendered sample covering every distinct call-site shape
(single call, expected-error, multi-output, streamed chunks, fault injection,
malloc'd contexts) are held to: first guard check before first returning
validator, counted from the first guard arm. Anchoring on the arm skips the
temp-sizer scalar checks the recurrent templates run before touching any
buffer, and lands before every kernel call, since a buffer is armed before
it is handed to the kernel.

Participation is enforced by inventory, not by the templates that already opt
in: every template must arm a guard, must not declare a writable static array
outside helia_guard_declare, and must arm and check every buffer it declares
that way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_ROOT = PROJECT_ROOT / "assets" / "templates"

RETURNING_VALIDATOR_RE = re.compile(
    r"HELIA_VALIDATE_STATUS\(|HELIA_VALIDATE_EXPECTED_STATUS\(|HELIA_VALIDATE_SCALAR_EQ_INT\("
)
GUARD_CHECK_RE = re.compile(r"HELIA_GUARD_CHECK(?:_SLACK)?\(|helia_guard_check\(")
GUARD_ARM_RE = re.compile(r"HELIA_GUARD_ARM\(|helia_guard_arm\(")
# A returning validator whose subject is a sizer query rather than a kernel result.
SIZER_PROBE_RE = re.compile(r"ctx size|_get_buffer_size")
# A static array declaration, optionally behind Jinja control tags on the same line.
# Const tables and pointer tables are not kernel write targets.
STATIC_ARRAY_RE = re.compile(
    r"^[ \t]*(?:\{%-?[^%]*-?%\}[ \t]*)*static\s+(?!const\b)"
    r"(?P<type>[^;=(){}\[\]]*?(?:\{\{[^}]*\}\}[^;=(){}\[\]]*?)*)\s*"
    r"(?P<ident>(?:\{\{[^}]*\}\})?\w*)\s*\[",
    re.MULTILINE,
)
GUARD_DECLARE_RE = re.compile(r"helia_guard_declare\(\s*[^,]+,\s*(?P<ident>[^,]+?)\s*,")
# A canary check. HELIA_GUARD_CHECK_UNTOUCHED verifies only the body, not the canaries.
GUARD_CANARY_CHECK_RE = r"HELIA_GUARD_CHECK(?:_SLACK)?\(\s*"
LITERAL_FOR_RE = re.compile(r"\{%-?\s*for\s+(?P<var>\w+)\s+in\s+\[(?P<items>[^\]]*)\]\s*-?%\}")

RENDERED_CASES = [
    "convolve_default_s8",
    "fully_connected_float_default_f32",
    "svdf_bias_rank1_s8",
    "svdf_float_default_f32",
    "where_2d_s8",
    "split_float_channels_pairs_f16",
    "chunked_equivalence_add_offcut_s8",
    "prelu_float_per_channel_f32",
    "rsqrt_small_tensor_per_op_s16",
    "quantize_relu6_vec_s8",
    "lstm_unidirectional_dataset_1_s16_time_major_s16",
    "lstm_unidirectional_float_stream_f32",
    "gru_unidirectional_float_stream_f32",
    "lstm_unidirectional_error_null_input_f32",
    "gru_unidirectional_error_stateful_batch_gt1_f32",
    "gru_unidirectional_error_missing_temp1_prereset_f32",
    "fill_float_block16_f32",
    "fill_float_block0_noop_f32",
    "pack_float_rank1_n2_axis0_f32",
    "unpack_float_rank2_axis1_f32",
    "split_float_zero_slice_v_f32",
    "rsqrt_float_special_inplace_f32",
    "reduce_max_all_boundary_f32",
    "convolve_fault_null_ctx_buf_s8",
    "max_pool_fault_zero_dim_s8",
    "svdf_fault_null_output_f32",
]
RENDERED_CPU = "cortex-m55"


def _first_guard_precedes_first_returning_validator(text: str, label: str) -> None:
    arm = GUARD_ARM_RE.search(text)
    assert arm is not None, f"{label}: no guarded buffer"
    guard = GUARD_CHECK_RE.search(text, arm.end())
    # Sizer probes call a *_get_buffer_size() function and return on a mismatch, but they run
    # before the kernel is handed any buffer, so a guard check after them is not the defect
    # this pins. They are skipped by starting the search at the first guard check when the
    # validators before it are probes; anything else before the check still fails below.
    start = arm.end()
    if guard is not None:
        probes_only = all(
            SIZER_PROBE_RE.search(text, m.start(), guard.start())
            for m in RETURNING_VALIDATOR_RE.finditer(text, arm.end(), guard.start())
        )
        if probes_only:
            start = guard.start()
    validator = RETURNING_VALIDATOR_RE.search(text, start)
    if validator is None:
        return
    assert guard is not None, f"{label}: guarded buffers but no guard check"
    assert guard.start() < validator.start(), (
        f"{label}: first guard check at {guard.start()} comes after the first "
        f"returning validator at {validator.start()}"
    )


def _all_templates() -> list[Path]:
    return sorted(TEMPLATES_ROOT.glob("**/*.c.j2"))


def _unguarded_writable_arrays(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in STATIC_ARRAY_RE.finditer(text)
        if "*" not in match.group("type")
    ]


def _declared_guard_idents(text: str) -> list[tuple[str, str]]:
    """(expanded, as-written) spellings of each helia_guard_declare identifier.

    name ~ "_output" is referenced as {{ name }}_output. A loop variable over a
    literal list ({% for gate in ["input", ...] %}) is expanded to each value,
    and a site may use either the expanded name or the loop form.
    """
    literal_loops = {
        match.group("var"): re.findall(r"[\"']([^\"']*)[\"']", match.group("items"))
        for match in LITERAL_FOR_RE.finditer(text)
    }
    idents = []
    for match in GUARD_DECLARE_RE.finditer(text):
        parts = [part.strip() for part in match.group("ident").split("~")]
        written = "".join(part.strip("\"'") if part[:1] in "\"'" else "{{ " + part + " }}" for part in parts)
        spellings = [""]
        for part in parts:
            if part[:1] in "\"'":
                values = [part.strip("\"'")]
            elif part in literal_loops:
                values = literal_loops[part]
            else:
                values = ["{{ " + part + " }}"]
            spellings = [prefix + value for prefix in spellings for value in values]
        idents.extend((spelling, written) for spelling in spellings)
    return idents


def _template_id(path: Path) -> str:
    return str(path.relative_to(TEMPLATES_ROOT))


def test_inventory_covers_every_template() -> None:
    assert len(_all_templates()) > 80


@pytest.mark.parametrize("template", _all_templates(), ids=_template_id)
def test_template_guards_every_writable_buffer(template: Path) -> None:
    text = template.read_text()
    assert GUARD_ARM_RE.search(text), f"{_template_id(template)}: no guarded buffer"
    raw = _unguarded_writable_arrays(text)
    assert not raw, f"{_template_id(template)}: writable static arrays outside helia_guard_declare: {raw}"


@pytest.mark.parametrize("template", _all_templates(), ids=_template_id)
def test_template_arms_and_checks_every_declared_guard(template: Path) -> None:
    text = template.read_text()
    for ident, written in _declared_guard_idents(text):
        names = "(?:" + re.escape(ident) + "|" + re.escape(written) + ")"
        assert re.search(r"HELIA_GUARD_ARM\(\s*" + names + r"\s*,", text), (
            f"{_template_id(template)}: {ident} is declared guarded but never armed"
        )
        assert re.search(GUARD_CANARY_CHECK_RE + names + r"\s*,", text), (
            f"{_template_id(template)}: {ident} is declared guarded but its canaries are never checked"
        )


@pytest.mark.parametrize("template", _all_templates(), ids=_template_id)
def test_template_guard_check_precedes_returning_validators(template: Path) -> None:
    _first_guard_precedes_first_returning_validator(template.read_text(), str(template))


@pytest.fixture(scope="module")
def _descriptors_by_name() -> dict[str, dict]:
    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.generation.io.descriptors import load_all_descriptors

    return {desc["name"]: desc for desc in load_all_descriptors(str(find_descriptors_dir()))}


@pytest.fixture
def rendered_source(request, tmp_path, _descriptors_by_name) -> str:
    """Render one case. Per-case rather than a single batch, because a few descriptors read
    data from an ns-cmsis-nn checkout that the pure-Python job does not have; those skip
    while the rest still assert. The same templates are covered textually above."""
    import helia_core_tester.generation.test_ops as generation_module

    case_name = request.param
    desc = _descriptors_by_name[case_name]
    try:
        generation_module.generate_test(desc, str(tmp_path), cpu=RENDERED_CPU)
    except FileNotFoundError as exc:
        pytest.skip(f"{case_name} needs generation inputs this environment lacks: {exc}")
    case_dir = tmp_path / desc["_family"] / case_name
    (c_file,) = case_dir.glob("*.c")
    return c_file.read_text()


@pytest.mark.parametrize("rendered_source", RENDERED_CASES, indirect=True)
def test_guard_check_precedes_returning_validators(rendered_source: str, request) -> None:
    case_name = request.node.callspec.params["rendered_source"]
    _first_guard_precedes_first_returning_validator(rendered_source, case_name)


@pytest.mark.parametrize(
    ("rendered_source", "output"),
    [
        ("fill_float_block0_noop_f32", "fill_float_block0_noop_f32_output"),
        ("split_float_zero_slice_v_f32", "split_float_zero_slice_v_f32_out_1_output"),
    ],
    indirect=["rendered_source"],
)
def test_zero_extent_output_must_stay_untouched(rendered_source: str, output: str) -> None:
    """A write into 1-element placeholder storage lands inside the body, where the
    canaries cannot see it; only the poison check does."""
    assert f"HELIA_GUARD_ARM({output}, true" in rendered_source
    assert f"HELIA_GUARD_CHECK_UNTOUCHED({output}," in rendered_source
