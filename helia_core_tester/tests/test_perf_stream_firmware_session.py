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
