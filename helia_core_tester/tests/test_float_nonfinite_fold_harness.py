"""Fold regression for HELIA_VALIDATE_FLOATS in the generated-harness shape (issue #75).

The sibling module test_float_nonfinite_compare.py drives the operand pairings
one probe at a time. That is not enough to catch the defect this module exists
for: under -ffinite-math-only the widening conversion to double carries nnan
and ninf, so a validator that classifies after widening loses non-finite lanes,
but only where there is enough surrounding validator code for the optimizer to
act on those flags. Cut-down probes stay correct at every optimization level on
the same compiler that drops lanes here.

So this module compiles the real shape: a file-scope `static const` golden with
bare NAN/INFINITY literals, the kernel output produced in a second translation
unit, and the validation macro expanded in a *_test_case_run() that main calls.
Measured against the classify-after-widening runtime, this shape reports 3 of 4
planted failures and loses 11 of the 14 must-fail lane pairings; the shipped
runtime reports all of them.

The build runs at both flag sets the generated tests use: -Ofast for the
armclang leg and -O3 -ffast-math for the GCC and ATfE legs. What is asserted is
correctness, not the miss, because whether a given toolchain folds depends on
the toolchain.
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
RUNTIME_C = PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.c"

FLAG_SETS = [["-Ofast"], ["-O3", "-ffast-math"]]

PLANTED_FAILURES = 4

# Planted non-finite lane -> the operand tokens its evidence line must carry.
PLANTED_NONFINITE_LANES = {
    1: ("nan", "3.5"),
    3: ("+inf", "-inf"),
    5: ("0.5", "nan"),
}
# The one planted failure that belongs on the finite path.
PLANTED_FINITE_LANE = 6

# Lane table of helia_fold_sweep.h: pairing name -> must this lane fail?
SWEEP_LANES = {
    "gNAN__aNaN": 0,
    "gNAN__aPInf": 1,
    "gNAN__aMInf": 1,
    "gNAN__aFin": 1,
    "gMNAN_aNaN": 0,
    "gMNAN_aFin": 1,
    "gPInf_aPInf": 0,
    "gPInf_aMInf": 1,
    "gPInf_aNaN": 1,
    "gPInf_aFin": 1,
    "gMInf_aMInf": 0,
    "gMInf_aPInf": 1,
    "gMInf_aNaN": 1,
    "gMInf_aFin": 1,
    "gFin__aNaN": 1,
    "gFin__aPInf": 1,
    "gFin__aMInf": 1,
    "gFin__aFin": 0,
    "gFin__aFinBad": 1,
}
MUST_FAIL_LANES = sorted(name for name, fails in SWEEP_LANES.items() if fails)
FINITE_ONLY_LANES = {"gFin__aFin", "gFin__aFinBad"}
SWEEP_TENSOR_FAILURES = sum(SWEEP_LANES.values())

FoldOutput = dict[str, "subprocess.CompletedProcess[str]"]

NONFINITE_TOKEN = re.compile(r"\b(?:[-+]?inf(?:inity)?|-?nan)\b", re.IGNORECASE)


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


def _build_and_run(
    cc: str, flags: list[str], workdir: Path, stem: str
) -> subprocess.CompletedProcess[str]:
    binary = workdir / stem
    subprocess.run(
        [
            cc,
            "-std=c11",
            *flags,
            "-I",
            str(C_HOST_DIR),
            "-I",
            str(PROJECT_ROOT / "src"),
            str(RUNTIME_C),
            str(C_HOST_DIR / f"{stem}_kernel.c"),
            str(C_HOST_DIR / f"{stem}_main.c"),
            "-o",
            str(binary),
            "-lm",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    # The exit status is asserted by a test rather than here: a harness that
    # under-reports still produces output worth checking lane by lane, and that
    # is the diagnosis this module exists to give.
    return subprocess.run([str(binary)], capture_output=True, text=True)


@pytest.fixture(scope="module", params=FLAG_SETS, ids=lambda flags: " ".join(flags))
def fold_output(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> FoldOutput:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")

    flags = request.param
    # clang already warns that -Ofast is deprecated; when it becomes an error
    # the other flag set still covers -ffinite-math-only.
    if not _compiler_accepts(tuple(flags)):
        pytest.skip(f"host compiler rejects {' '.join(flags)}")

    workdir = tmp_path_factory.mktemp("float_fold")
    return {
        "harness": _build_and_run(cc, flags, workdir, "helia_fold_harness"),
        "sweep": _build_and_run(cc, flags, workdir, "helia_fold_sweep"),
    }


def _stdout(fold_output: FoldOutput, name: str) -> str:
    return fold_output[name].stdout


def _mismatch_indices(output: str) -> set[int]:
    return {
        int(index)
        for index in re.findall(r"^Mismatch\[(\d+)\]:", output, re.MULTILINE)
    }


def _maxdiff_lines(output: str) -> list[str]:
    lines = [line for line in output.splitlines() if "HELIA_FLOAT_MAXDIFF" in line]
    assert lines
    return lines


def _lane_block(sweep_output: str, lane: str) -> str:
    start = sweep_output.index(f"LANE {lane}\n")
    end = sweep_output.index(f"LANEVERDICT {lane} ", start)
    return sweep_output[start:end]


@pytest.mark.parametrize("name", ["harness", "sweep"])
def test_harness_exits_clean(fold_output: FoldOutput, name: str) -> None:
    completed = fold_output[name]
    assert completed.returncode == 0, completed.stdout


def test_generated_shape_reports_every_planted_failure(fold_output: FoldOutput) -> None:
    assert (
        f"HELIA_FOLD_RESULT failures={PLANTED_FAILURES} expected={PLANTED_FAILURES}"
        in _stdout(fold_output, "harness")
    )


@pytest.mark.parametrize("lane", sorted(PLANTED_NONFINITE_LANES))
def test_planted_nonfinite_lane_keeps_its_evidence_line(
    fold_output: FoldOutput, lane: int
) -> None:
    expected_token, actual_token = PLANTED_NONFINITE_LANES[lane]
    assert (
        f"HELIA_NONFINITE_MISMATCH[{lane}]: exp={expected_token} got={actual_token}"
        in _stdout(fold_output, "harness")
    )


def test_planted_nonfinite_lanes_are_counted_as_such(fold_output: FoldOutput) -> None:
    assert (
        f"HELIA_NONFINITE_MISMATCHES n={len(PLANTED_NONFINITE_LANES)}"
        in _stdout(fold_output, "harness")
    )


def test_no_nonfinite_lane_falls_through_to_the_finite_path(
    fold_output: FoldOutput,
) -> None:
    # A lane that lost its class still counts a failure when the tolerance
    # happens to catch it, so the discriminating evidence is which report line
    # it produced, not the count.
    assert _mismatch_indices(_stdout(fold_output, "harness")) == {PLANTED_FINITE_LANE}


def test_headroom_instrumentation_stays_finite(fold_output: FoldOutput) -> None:
    # Inf reaching the headroom fields means a non-finite element was measured
    # on the finite path instead of being classified out of it.
    for name in ("harness", "sweep"):
        for line in _maxdiff_lines(_stdout(fold_output, name)):
            assert not NONFINITE_TOKEN.search(line), line


def test_every_lane_pairing_gets_the_right_verdict(fold_output: FoldOutput) -> None:
    assert (
        f"HELIA_FOLD_SWEEP_RESULT tensor_failures={SWEEP_TENSOR_FAILURES} "
        f"expected={SWEEP_TENSOR_FAILURES} lanes_wrong=0" in _stdout(fold_output, "sweep")
    )


@pytest.mark.parametrize("lane", MUST_FAIL_LANES)
def test_must_fail_lane_pairing_is_reported(fold_output: FoldOutput, lane: str) -> None:
    assert f"LANEVERDICT {lane} want=1 got=1 OK" in _stdout(fold_output, "sweep")


@pytest.mark.parametrize("lane", sorted(set(SWEEP_LANES) - FINITE_ONLY_LANES))
def test_nonfinite_lane_pairing_never_uses_the_finite_report(
    fold_output: FoldOutput, lane: str
) -> None:
    assert "Mismatch[" not in _lane_block(_stdout(fold_output, "sweep"), lane)
