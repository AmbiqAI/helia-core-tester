"""Which ns-cmsis-nn sources keep ARM_MATH_AUTOVECTORIZE in each coverage mode.

CMakeLists.txt applies cmake/coverage_autovectorize.cmake per source when
ENABLE_COVERAGE_MVE_FLOAT or ENABLE_COVERAGE_MVE_INT is on. The define compiles out MVE
paths, so a source that keeps it by mistake is silently missing from that lane's coverage.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _PROJECT_ROOT / "cmake" / "coverage_autovectorize.cmake"

_INT = [
    "Source/ConvolutionFunctions/arm_convolve_s8.c",
    "Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
    "Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c",
    # A float token in a directory name does not make a source float.
    "Source/fp_f32_helpers/arm_add_s8.c",
]
_ALWAYS = ["Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c"]
_FLOAT = [
    "Source/ConvolutionFunctions/arm_convolve_f16.c",
    "Source/SoftmaxFunctions/arm_softmax_f32.c",
    "Source/QuantizationFunctions/arm_dequantize_s8_f32.c",
    # The float token sits mid-name; these compile MVE float bodies.
    "Source/QuantizationFunctions/arm_quantize_f32_s8.c",
    "Source/QuantizationFunctions/arm_quantize_f32_s16.c",
    "Source/QuantizationFunctions/arm_dequantize_half_bits.c",
]


def _autovectorized(tmp_path: Path, *, mve_float: bool, mve_int: bool) -> set[str]:
    script = tmp_path / "probe.cmake"
    sources = " ".join(f'"{s}"' for s in _INT + _ALWAYS + _FLOAT)
    script.write_text(
        "cmake_minimum_required(VERSION 3.15)\n"
        f'include("{_MODULE.as_posix()}")\n'
        f"set(ENABLE_COVERAGE_MVE_FLOAT {'ON' if mve_float else 'OFF'})\n"
        f"set(ENABLE_COVERAGE_MVE_INT {'ON' if mve_int else 'OFF'})\n"
        f"helia_coverage_autovectorize_sources(selected {sources})\n"
        'foreach(s IN LISTS selected)\n  message(STATUS "AUTOVECTORIZE:${s}")\nendforeach()\n'
    )
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, check=True)
    return {
        line.split("AUTOVECTORIZE:", 1)[1]
        for line in (result.stdout + result.stderr).splitlines()
        if "AUTOVECTORIZE:" in line
    }


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake is not installed")
@pytest.mark.parametrize(
    ("mve_float", "mve_int", "expected"),
    [
        (False, False, set(_INT + _ALWAYS + _FLOAT)),
        (True, False, set(_INT + _ALWAYS)),
        (False, True, set(_ALWAYS + _FLOAT)),
        (True, True, set(_ALWAYS)),
    ],
    ids=["neither", "mve-float", "mve-int", "both"],
)
def test_autovectorize_selection(tmp_path: Path, mve_float: bool, mve_int: bool, expected: set[str]) -> None:
    assert _autovectorized(tmp_path, mve_float=mve_float, mve_int=mve_int) == expected
