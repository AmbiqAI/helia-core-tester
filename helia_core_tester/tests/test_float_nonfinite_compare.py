"""Executed contract for HELIA_VALIDATE_FLOATS on non-finite operands (issue #75).

The validation logic is a macro that expands in the caller's translation unit,
so reading the C is not evidence: this compiles the shipped runtime with the
host compiler and drives every operand pairing, for float and for float16.

Two things about the build matter. It runs at both flag sets the generated
tests use (-Ofast for the armclang leg, -O3 -ffast-math for the GCC and ATfE
legs), because it is the -ffinite-math-only they both imply that the
classification has to survive. And the operands come from a second translation
unit, so no result here depends on the compiler failing to notice where a NaN
came from -- except for the `visible_*` rows, which deliberately do the reverse
and put the non-finite bit pattern in the driver's own TU, because that is the
shape a classify-after-widening validator actually folds away.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
C_HOST_DIR = Path(__file__).resolve().parent / "c_host"

FLAG_SETS = [["-Ofast"], ["-O3", "-ffast-math"]]

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

# The largest finite value differs by width: FLT_MAX is +Inf in binary16, so
# the driver pairs +Inf against 65504 there.
MISMATCH_TOKENS = {
    "nan_vs_finite": ("1", "nan"),
    "finite_vs_nan": ("nan", "1"),
    "finite_vs_posinf": ("+inf", "1"),
    "neginf_vs_posinf": ("+inf", "-inf"),
    "posinf_vs_finite": ("1", "+inf"),
    "nan_vs_posinf": ("+inf", "nan"),
    # %.6f would render the finite operand as 39 integer digits and truncate.
    "posinf_vs_fltmax": ("3.40282e+38", "+inf"),
}
F16_MISMATCH_TOKENS = dict(MISMATCH_TOKENS, posinf_vs_fltmax=("65504", "+inf"))

# Rows whose non-finite operand is a literal bit pattern in the driver's own
# translation unit: expected/actual tokens on the HELIA_NONFINITE_MISMATCH line.
VISIBLE_MISMATCH_TOKENS = {
    "visible_nan_vs_finite": ("1", "nan"),
    "visible_finite_vs_nan": ("nan", "1"),
    "visible_neginf_vs_finite": ("1", "-inf"),
    "visible_posinf_vs_neginf": ("-inf", "+inf"),
}


@functools.lru_cache(maxsize=None)
def _compiler_accepts(flags: tuple[str, ...]) -> bool:
    cc = shutil.which("cc")
    if cc is None:
        return False
    probe = subprocess.run(
        [cc, *flags, "-xc", "-c", "-o", os.devnull, "-"],
        input="int main(void) { return 0; }\n",
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0


@pytest.fixture(scope="module", params=FLAG_SETS, ids=lambda flags: " ".join(flags))
def host_sanity_output(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> str:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")

    flags = request.param
    # clang already warns that -Ofast is deprecated; when it becomes an error
    # the other flag set still covers -ffinite-math-only, so skip rather than
    # collapse the module.
    if not _compiler_accepts(tuple(flags)):
        pytest.skip(f"host compiler rejects {' '.join(flags)}")

    workdir = tmp_path_factory.mktemp("float_nonfinite")
    binary = workdir / "helia_float_compare_host_sanity"
    subprocess.run(
        [
            cc,
            "-std=c11",
            *flags,
            "-Wall",
            "-Wextra",
            "-I",
            str(C_HOST_DIR),
            "-I",
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.c"),
            str(C_HOST_DIR / "helia_float_compare_host_producer.c"),
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


def _require_f16(output: str) -> None:
    if "F16_SUPPORTED 1" not in output:
        pytest.skip(
            "host compiler does not define __FLT16_MAX__, so the driver has no "
            "float16 rows to check"
        )


@pytest.mark.parametrize("case_name", sorted(EXPECTED_FAILURES))
@pytest.mark.parametrize("tag", ["rtol", "zerortol"])
def test_nonfinite_pairings_produce_expected_failure_counts(
    host_sanity_output: str, case_name: str, tag: str
) -> None:
    counts = _failure_counts(host_sanity_output)
    assert counts[f"{case_name}.{tag}"] == EXPECTED_FAILURES[case_name]


@pytest.mark.parametrize("case_name", sorted(EXPECTED_FAILURES))
@pytest.mark.parametrize("tag", ["rtol", "zerortol"])
def test_float16_pairings_produce_expected_failure_counts(
    host_sanity_output: str, case_name: str, tag: str
) -> None:
    # A half-width element is the case the widening conversion would have
    # erased first, and float16_t is what the f16 kernels validate.
    _require_f16(host_sanity_output)
    counts = _failure_counts(host_sanity_output)
    assert counts[f"f16_{case_name}.{tag}"] == EXPECTED_FAILURES[case_name]


@pytest.mark.parametrize("case_name", sorted(MISMATCH_TOKENS))
def test_nonfinite_mismatch_is_reported_with_symbolic_operands(
    host_sanity_output: str, case_name: str
) -> None:
    expected_token, actual_token = MISMATCH_TOKENS[case_name]
    block = _case_block(host_sanity_output, f"{case_name}.rtol")
    assert f"HELIA_NONFINITE_MISMATCH[0]: exp={expected_token} got={actual_token}" in block


@pytest.mark.parametrize("case_name", sorted(F16_MISMATCH_TOKENS))
def test_float16_nonfinite_mismatch_is_reported_with_symbolic_operands(
    host_sanity_output: str, case_name: str
) -> None:
    _require_f16(host_sanity_output)
    expected_token, actual_token = F16_MISMATCH_TOKENS[case_name]
    block = _case_block(host_sanity_output, f"f16_{case_name}.rtol")
    assert f"HELIA_NONFINITE_MISMATCH[0]: exp={expected_token} got={actual_token}" in block


@pytest.mark.parametrize("case_name", sorted(MISMATCH_TOKENS))
def test_mismatched_case_emits_the_per_tensor_count_line(
    host_sanity_output: str, case_name: str
) -> None:
    block = _case_block(host_sanity_output, f"{case_name}.rtol")
    assert "HELIA_NONFINITE_MISMATCHES n=1" in block


@pytest.mark.parametrize("case_name", ["nan_vs_nan", "posinf_vs_posinf", "finite_mismatch"])
def test_matched_and_finite_cases_emit_no_count_line(
    host_sanity_output: str, case_name: str
) -> None:
    # The line is the parser's classifier, so it must not appear for a matched
    # non-finite pair or for a plain tolerance overrun.
    block = _case_block(host_sanity_output, f"{case_name}.rtol")
    assert "HELIA_NONFINITE_MISMATCHES" not in block


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
    assert "HELIA_NONFINITE_MISMATCHES n=1" in block
    assert _headroom(block) == (-1.0, -2.0)


def test_all_nonfinite_matched_tensor_passes_without_headroom(host_sanity_output: str) -> None:
    block = _case_block(host_sanity_output, "tensor_all_nan_matched")
    assert _failure_counts(host_sanity_output)["tensor_all_nan_matched"] == 0
    assert "HELIA_NONFINITE_MISMATCH" not in block
    assert _headroom(block) == (-1.0, -2.0)


def test_zero_length_tensor_passes_and_records_the_sentinel(host_sanity_output: str) -> None:
    block = _case_block(host_sanity_output, "tensor_empty")
    assert _failure_counts(host_sanity_output)["tensor_empty"] == 0
    assert "HELIA_NONFINITE_MISMATCH" not in block
    assert "HELIA_FLOAT_MAXDIFF maxdiff=-1.00000000e+00 maxfrac=-2.000000 n=0" in block


def test_float16_nan_lane_is_caught_beside_finite_lanes(host_sanity_output: str) -> None:
    _require_f16(host_sanity_output)
    block = _case_block(host_sanity_output, "f16_tensor_nan_lane")
    assert _failure_counts(host_sanity_output)["f16_tensor_nan_lane"] == 1
    assert "HELIA_NONFINITE_MISMATCH[0]: exp=1 got=nan" in block
    assert "HELIA_NONFINITE_MISMATCHES n=1" in block
    assert _headroom(block) == (-1.0, -2.0)


def test_finite_cases_keep_measurable_headroom(host_sanity_output: str) -> None:
    matched = _case_block(host_sanity_output, "finite_match.rtol")
    assert "HELIA_FLOAT_MAXDIFF maxdiff=0.00000000e+00 maxfrac=0.000000 n=1" in matched
    assert "HELIA_NONFINITE_MISMATCH" not in matched

    mismatched = _case_block(host_sanity_output, "finite_mismatch.rtol")
    assert "Mismatch[0]: exp=1.000000 got=2.000000" in mismatched
    assert "HELIA_NONFINITE_MISMATCH" not in mismatched


@pytest.mark.parametrize("case_name", sorted(VISIBLE_MISMATCH_TOKENS))
@pytest.mark.parametrize("prefix", ["", "f16_"])
def test_locally_visible_nonfinite_operand_is_still_classified(
    host_sanity_output: str, case_name: str, prefix: str
) -> None:
    # The bit pattern is a literal in the driver's own TU, which is the shape
    # most exposed to the fold: an operand whose class the optimizer knows
    # before the comparison. The token assertion is the load-bearing half --
    # a -Inf lane that fell through to the finite path would still count one
    # failure, but would print a Mismatch line instead of this one.
    if prefix:
        _require_f16(host_sanity_output)
    case_id = f"{prefix}{case_name}"
    expected_token, actual_token = VISIBLE_MISMATCH_TOKENS[case_name]
    block = _case_block(host_sanity_output, case_id)
    assert _failure_counts(host_sanity_output)[case_id] == 1
    assert f"HELIA_NONFINITE_MISMATCH[0]: exp={expected_token} got={actual_token}" in block
    assert "HELIA_NONFINITE_MISMATCHES n=1" in block


def test_float_validator_classifies_before_any_conversion() -> None:
    # -ffinite-math-only licenses the compiler to fold isnan/isinf to a
    # constant false, and to treat the result of a widening conversion as
    # finite, so the macro must classify from the element's own storage first.
    header = (PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.h").read_text()
    validator = header[header.index("#define HELIA_VALIDATE_FLOATS") :]
    validator = validator[: validator.index("#define HELIA_VALIDATE_BOOLEANS")]
    assert "isnan" not in validator
    assert "isinf" not in validator
    assert "isfinite" not in validator
    assert validator.index("HELIA_FLOAT_CLASS_OF((actual)[helia_i])") < validator.index(
        "(double)((actual)[helia_i])"
    )


def test_float_class_dispatch_covers_every_float16_spelling() -> None:
    # float16_t is __fp16 on some toolchains and _Float16 on others
    # (ns-cmsis-nn Include/arm_nn_math_types_flt.h), and the two are distinct
    # types, so a _Generic missing either row would not compile the f16 tests.
    header = (PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.h").read_text()
    rows = header[header.index("#define HELIA_FLOAT16_GENERIC_ROWS") :]
    rows = rows[: rows.index("_Static_assert(sizeof(float)")]
    assert "__fp16: handler" in rows
    assert "_Float16: handler" in rows
