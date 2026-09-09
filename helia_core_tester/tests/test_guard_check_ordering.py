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


def _guarded_templates() -> list[Path]:
    return sorted(
        path
        for path in TEMPLATES_ROOT.glob("**/*.c.j2")
        if "HELIA_GUARD_ARM(" in path.read_text() or "helia_guard_arm(" in path.read_text()
    )


@pytest.mark.parametrize(
    "template", _guarded_templates(), ids=lambda p: str(p.relative_to(TEMPLATES_ROOT))
)
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
