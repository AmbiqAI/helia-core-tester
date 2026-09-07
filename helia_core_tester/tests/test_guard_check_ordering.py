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
    validator = RETURNING_VALIDATOR_RE.search(text, arm.end())
    if validator is None:
        return
    guard = GUARD_CHECK_RE.search(text, arm.end())
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
def rendered_sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.generation.io.descriptors import load_all_descriptors
    import helia_core_tester.generation.test_ops as generation_module

    descriptors = {
        desc["name"]: desc for desc in load_all_descriptors(str(find_descriptors_dir()))
    }
    out_dir = tmp_path_factory.mktemp("guard_ordering")
    sources: dict[str, str] = {}
    for case_name in RENDERED_CASES:
        desc = descriptors[case_name]
        generation_module.generate_test(desc, str(out_dir), cpu=RENDERED_CPU)
        case_dir = out_dir / desc["_family"] / case_name
        (c_file,) = case_dir.glob("*.c")
        sources[case_name] = c_file.read_text()
    return sources


@pytest.mark.parametrize("case_name", RENDERED_CASES)
def test_guard_check_precedes_returning_validators(
    rendered_sources: dict[str, str], case_name: str
) -> None:
    _first_guard_precedes_first_returning_validator(rendered_sources[case_name], case_name)
