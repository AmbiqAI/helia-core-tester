from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The ns-cmsis-nn checkout: $CMSIS_NN_ROOT (what `generate` and the firmware build use),
# else the tester's conventional location two levels up.
CMSIS_NN_ROOT = Path(os.environ.get("CMSIS_NN_ROOT") or PROJECT_ROOT.parent.parent)



# Stubbed nsx-pmu-armv8m API and PMU registers.
PMU_STUB_DIR = PROJECT_ROOT / "helia_core_tester" / "tests" / "fixtures" / "pmu_stub"
PMU_STUB_FLAGS = [
    "-DHCT_HOST_PMU_STUB",
    "-D__PMU_PRESENT=1",
    "-D__PMU_NUM_EVENTCNT=8",
    "-I",
    str(PMU_STUB_DIR),
    str(PMU_STUB_DIR / "pmu_stub.c"),
]


@pytest.mark.parametrize("pmu", [False, True], ids=["dwt-only", "pmu-stub"])
def test_c_firmware_session_loop_executes_abs_correctness_flow(tmp_path: Path, pmu: bool) -> None:
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
            *(PMU_STUB_FLAGS if pmu else []),
            "-I",
            str(PROJECT_ROOT / "cmake" / "hardware"),
            "-I",
            str(CMSIS_NN_ROOT / "Include"),
            str(PROJECT_ROOT / "cmake" / "hardware" / "hctp_protocol.c"),
            str(PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_catalog.c"),
            str(PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_messages.c"),
            str(PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_adapter.c"),
            str(PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_session.c"),
            str(PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_session_host_main.c"),
            str(CMSIS_NN_ROOT / "Source" / "BasicMathFunctions" / "arm_abs_s8.c"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, f"harness exit {result.returncode}"
    assert "chunks=" in result.stdout
    assert "bytes=12" in result.stdout
    # v2: the harness continues through CORRECTNESS_ACK/RUN_PERFORMANCE with two PMU
    # passes (3 samples each) and checks every SAMPLE_RESULT leads with the CCNTR entry.
    # DWT-only: event counters come back unsupported. PMU stub: the harness also checks
    # an unmapped event id fails SESSION_PLAN, chained vs 16-bit setup, CCNTR/OVS read
    # before the module's read resets them, and each counter's overflow slot.
    assert "samples=6 passes=2" in result.stdout
    # Refusals end one case; the next runs.
    for line in ("rejected correctness samples_dropped=0", "rejected warmup samples_dropped=0", "rejected sampling samples_dropped=3"):
        assert line in result.stdout


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
            "-I", str(PROJECT_ROOT / "cmake" / "hardware"),
            str(source), "-o", str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
