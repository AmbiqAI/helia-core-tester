"""Regression test for the ground-truth output-dims/padding bridge fix.

Guards against a real hardware bug hit in practice: firmware's `compute_convolve_output_dims()`
and `run_convolve_once()` used to *re-derive* output h/w/c and pad_h/pad_w from a simplified
SAME/VALID formula (`session->padding` flag + stride/filter/input math with a `pad/2`
symmetric-split assumption) instead of using the real generator's true values. This diverged
from the actual generated-test reference data for real cases -- e.g. a case whose true output
was (1,6,6,2)=72 bytes came back as a formula-guessed 128 bytes -- causing
`ValueError: cannot reshape array of size 128 into shape (1,6,6,2)` on the host.

The fix: `generated_test_bridge.py` now extracts the exact `output_dims` (already done
previously) plus the exact `conv_params.padding.h`/`.w` ("before" pad, read from the nested
`.padding = {...}` sub-struct in the generated header) and sends them explicitly as new
CASE_META scalar parameters (`output_h`, `output_w`, `output_c`, `pad_h`, `pad_w`), which
firmware now uses directly instead of recomputing anything.

This test does not touch real hardware; it bridges real generated-test artifacts already
checked into `artifacts/generated_tests/` and asserts the extracted scalars are present and
correct against known-good header values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.perf_stream.generated_test_bridge import (
    build_case_bundle_from_generated_test,
    discover_generated_tests,
)
from helia_core_tester.perf_stream.case_bundle import load_case_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "artifacts" / "generated_tests").is_dir(),
    reason="no generated-test artifacts under artifacts/generated_tests/ "
    "(artifacts/ is gitignored -- run `helia_core_tester generate` first)",
)


def _bridge_scalars(tmp_path: Path, name_filter: str) -> dict[str, object]:
    cases = discover_generated_tests(PROJECT_ROOT, name_filter=name_filter)
    assert cases, f"expected a discoverable generated test matching {name_filter!r}"
    bundle = build_case_bundle_from_generated_test(PROJECT_ROOT, cases[0], output_root=tmp_path, require_fvp_pass=False)
    loaded = load_case_bundle(bundle.manifest_path)
    return loaded.manifest["serialized_scalar_parameters"]


def test_valid_padding_case_sends_zero_pad_and_true_output_dims(tmp_path: Path) -> None:
    scalars = _bridge_scalars(tmp_path, "convolve_2x2_dilation_s8")
    assert scalars["padding"] == "VALID"
    assert scalars["pad_h"] == 0
    assert scalars["pad_w"] == 0
    assert (scalars["output_h"], scalars["output_w"], scalars["output_c"]) == (6, 6, 2)


def test_same_padding_case_sends_true_nonzero_pad_from_header(tmp_path: Path) -> None:
    # These "before" pad values come from the generated header's nested
    # `conv_params.padding = {.w, .h}` struct, not a re-derived SAME formula.
    scalars = _bridge_scalars(tmp_path, "convolve_case_02_s8")
    assert scalars["padding"] == "SAME"
    assert scalars["pad_h"] == 1
    assert scalars["pad_w"] == 1
    assert (scalars["output_h"], scalars["output_w"], scalars["output_c"]) == (3, 6, 4)


def test_dilated_case_sends_true_nonunit_dilation_from_header(tmp_path: Path) -> None:
    # Regression test for a real hardware failure: firmware used to hardcode
    # conv_params.dilation.w/h=1 unconditionally, silently producing wrong output for
    # any real generated test with dilation != 1 (this was NOT caught by FVP/reference
    # runs of the original CMSIS-NN test suite, which correctly apply dilation directly --
    # only the perf_stream bridge/firmware path had this gap).
    scalars = _bridge_scalars(tmp_path, "convolve_2x2_dilation_s8")
    assert scalars["dilation_h"] == 2
    assert scalars["dilation_w"] == 2

    scalars = _bridge_scalars(tmp_path, "convolve_grouped_conv_case_04_s8")
    assert scalars["dilation_h"] == 3
    assert scalars["dilation_w"] == 3

    # Non-dilated cases must still report dilation=1 explicitly (not omitted).
    scalars = _bridge_scalars(tmp_path, "convolve_default_s8")
    assert scalars["dilation_h"] == 1
    assert scalars["dilation_w"] == 1
