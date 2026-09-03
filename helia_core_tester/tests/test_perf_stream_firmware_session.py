from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CMSIS_NN_ROOT = PROJECT_ROOT.parent.parent



def test_c_firmware_session_loop_executes_abs_correctness_flow(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")
    if not (CMSIS_NN_ROOT / "Include" / "arm_nnfunctions.h").is_file():
        pytest.skip(f"no real ns-cmsis-nn checkout found at {CMSIS_NN_ROOT}")

    binary = tmp_path / "session_harness"
    subprocess.run(
        [
            cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DHCT_HOST_ABS_ONLY",
            "-I",
            str(PROJECT_ROOT / "cmake" / "perf_stream"),
            "-I",
            str(CMSIS_NN_ROOT / "Include"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "hctp_protocol.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_catalog.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_messages.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_adapter.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_session.c"),
            str(PROJECT_ROOT / "cmake" / "perf_stream" / "benchmark_server_session_host_main.c"),
            str(CMSIS_NN_ROOT / "Source" / "BasicMathFunctions" / "arm_abs_s8.c"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "chunks=" in result.stdout
    assert "bytes=12" in result.stdout


def test_shared_c_validation_rejects_range_and_shape_overflow(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")
    source = tmp_path / "validation.c"
    binary = tmp_path / "validation"
    source.write_text(
        """
#include <stdint.h>
#include "benchmark_server_validation.h"
int main(void) {
    uint32_t begin = 0, end = 0, bytes = 0;
    const int32_t valid[4] = {1, 2, 3, 4};
    const int32_t invalid[2] = {INT32_MAX, INT32_MAX};
    if (!hct_checked_aligned_range(3u, 16u, 20u, 64u, &begin, &end)) return 1;
    if (begin != 16u || end != 36u) return 2;
    if (hct_checked_aligned_range(UINT32_MAX, 16u, UINT32_MAX, UINT32_MAX, &begin, &end)) return 3;
    if (!hct_checked_shape_bytes(valid, 4, 2u, 48u, &bytes) || bytes != 48u) return 4;
    if (hct_checked_shape_bytes(invalid, 2, 4u, UINT32_MAX, &bytes)) return 5;
    if (hct_checked_shape_bytes(valid, 4, 2u, 47u, &bytes)) return 6;
    return 0;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            cc, "-std=c99", "-Wall", "-Wextra", "-Werror",
            "-I", str(PROJECT_ROOT / "cmake" / "perf_stream"),
            str(source), "-o", str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
