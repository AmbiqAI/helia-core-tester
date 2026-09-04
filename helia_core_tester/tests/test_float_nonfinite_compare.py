"""Executed contract for HELIA_VALIDATE_FLOATS on non-finite operands (issue #75).

The validation logic is a macro that expands in the caller's translation unit,
so reading the C is not evidence: this compiles the shipped runtime with the
host compiler at -Ofast (generated tests build at -Ofast or -O3 -ffast-math,
both of which imply -ffinite-math-only) and drives every operand pairing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
C_HOST_DIR = Path(__file__).resolve().parent / "c_host"

# actual/expected pairing -> expected failure count for a single-element tensor.
EXPECTED_FAILURES = {
    "nan_vs_finite": 1,
    "finite_vs_nan": 1,
    "finite_vs_posinf": 1,
    "neginf_vs_posinf": 1,
    "nan_vs_nan": 0,
    "posinf_vs_posinf": 0,
    "neginf_vs_neginf": 0,
    "posinf_vs_finite": 1,
    "nan_vs_posinf": 1,
    "posinf_vs_fltmax": 1,
    "finite_match": 0,
    "finite_mismatch": 1,
}

NONFINITE_CASES = {
    name
    for name in EXPECTED_FAILURES
    if name not in {"finite_match", "finite_mismatch"}
}


@pytest.fixture(scope="module")
def host_sanity_output(tmp_path_factory: pytest.TempPathFactory) -> str:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")

    binary = tmp_path_factory.mktemp("float_nonfinite") / "helia_float_compare_host_sanity"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Ofast",
            "-Wall",
            "-Wextra",
            "-I",
            str(C_HOST_DIR),
            "-I",
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.c"),
            str(C_HOST_DIR / "helia_float_compare_host_sanity.c"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "HOST_SANITY_DONE" in completed.stdout
    return completed.stdout


def _failure_counts(output: str) -> dict[str, int]:
    return {
        name: int(count)
        for name, count in re.findall(r"RESULT (\S+) failures=(\d+)", output)
    }


def _headroom(block: str) -> tuple[float, float]:
    match = re.search(
        r"HELIA_FLOAT_MAXDIFF maxdiff=(\S+) maxfrac=(\S+) n=\d+", block
    )
    assert match is not None
    return float(match.group(1)), float(match.group(2))


def _case_block(output: str, case_id: str) -> str:
    # subprocess text mode has already folded the harness's CRLF line endings.
    start = output.index(f"CASE {case_id}\n")
    end = output.index(f"RESULT {case_id} failures=", start)
    return output[start:end]


@pytest.mark.parametrize("case_name", sorted(EXPECTED_FAILURES))
@pytest.mark.parametrize("tag", ["rtol", "zerortol"])
def test_nonfinite_pairings_produce_expected_failure_counts(
    host_sanity_output: str, case_name: str, tag: str
) -> None:
    counts = _failure_counts(host_sanity_output)
    assert counts[f"{case_name}.{tag}"] == EXPECTED_FAILURES[case_name]


@pytest.mark.parametrize(
    "case_name,expected_token,actual_token",
    [
        ("nan_vs_finite", "1", "nan"),
        ("finite_vs_nan", "nan", "1"),
        ("finite_vs_posinf", "+inf", "1"),
        ("neginf_vs_posinf", "+inf", "-inf"),
        ("posinf_vs_finite", "1", "+inf"),
        ("nan_vs_posinf", "+inf", "nan"),
        # %.6f would render FLT_MAX as 39 integer digits and truncate.
        ("posinf_vs_fltmax", "3.40282e+38", "+inf"),
    ],
)
def test_nonfinite_mismatch_is_reported_with_symbolic_operands(
    host_sanity_output: str, case_name: str, expected_token: str, actual_token: str
) -> None:
    block = _case_block(host_sanity_output, f"{case_name}.rtol")
    assert f"HELIA_NONFINITE_MISMATCH[0]: exp={expected_token} got={actual_token}" in block


@pytest.mark.parametrize("case_name", sorted(NONFINITE_CASES))
def test_nonfinite_element_emits_unmeasurable_headroom_sentinel(
    host_sanity_output: str, case_name: str
) -> None:
    # Every single-element non-finite pairing is unmeasurable either way: it
    # mismatched, or it matched and left no finite element to measure.
    block = _case_block(host_sanity_output, f"{case_name}.rtol")
    assert "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=1" in block


def test_matched_nonfinite_lane_keeps_finite_headroom(host_sanity_output: str) -> None:
    block = _case_block(host_sanity_output, "tensor_matched_nan_with_finite")
    assert _failure_counts(host_sanity_output)["tensor_matched_nan_with_finite"] == 0
    max_diff, max_frac = _headroom(block)
    assert max_diff == pytest.approx(1.0013580322265625e-05, rel=1e-6)
    assert max_frac == pytest.approx(0.333786, rel=1e-5)


def test_nonfinite_mismatch_voids_headroom_for_the_tensor(host_sanity_output: str) -> None:
    block = _case_block(host_sanity_output, "tensor_nonfinite_mismatch")
    assert _failure_counts(host_sanity_output)["tensor_nonfinite_mismatch"] == 1
    assert _headroom(block) == (-1.0, -2.0)


def test_all_nonfinite_matched_tensor_passes_without_headroom(host_sanity_output: str) -> None:
    block = _case_block(host_sanity_output, "tensor_all_nan_matched")
    assert _failure_counts(host_sanity_output)["tensor_all_nan_matched"] == 0
    assert "HELIA_NONFINITE_MISMATCH" not in block
    assert _headroom(block) == (-1.0, -2.0)


def test_finite_cases_keep_measurable_headroom(host_sanity_output: str) -> None:
    matched = _case_block(host_sanity_output, "finite_match.rtol")
    assert "HELIA_FLOAT_MAXDIFF maxdiff=0.00000000e+00 maxfrac=0.000000 n=1" in matched
    assert "HELIA_NONFINITE_MISMATCH" not in matched

    mismatched = _case_block(host_sanity_output, "finite_mismatch.rtol")
    assert "Mismatch[0]: exp=1.000000 got=2.000000" in mismatched
    assert "HELIA_NONFINITE_MISMATCH" not in mismatched


def test_float_validator_classifies_without_library_predicates() -> None:
    # -Ofast implies -ffinite-math-only, under which isnan/isinf fold to a
    # constant false; the macro must keep using the bit-pattern classifier.
    header = (PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.h").read_text()
    validator = header[header.index("#define HELIA_VALIDATE_FLOATS") :]
    validator = validator[: validator.index("#define HELIA_VALIDATE_BOOLEANS")]
    assert "helia_test_float_class(" in validator
    assert "isnan" not in validator
    assert "isinf" not in validator
    assert "isfinite" not in validator
