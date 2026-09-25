"""A benchmark whose kernel call fails must fail the case, not report cycles (#146).

Both benchmark backends are rendered from the real templates and compiled on the
host against the real test runtime, with only the DWT/PMU registers stubbed. The op
under test fails on a chosen call; the output and return value must show the run
stopped there.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import jinja2
import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "assets" / "templates"
WARMUP = 3
MEASURED = 4

HOST_STUBS = """\
#pragma once
#include <stdint.h>
#define __CORTEX_M 55
typedef struct { uint32_t DEMCR; } helia_host_coredebug;
typedef struct { uint32_t CTRL; uint32_t CYCCNT; } helia_host_dwt;
static helia_host_coredebug helia_host_coredebug_regs;
static helia_host_dwt helia_host_dwt_regs;
#define CoreDebug (&helia_host_coredebug_regs)
#define DWT (&helia_host_dwt_regs)
#define CoreDebug_DEMCR_TRCENA_Msk 1u
#define DWT_CTRL_CYCCNTENA_Msk 1u
"""

PMU_STUBS = """\
#pragma once
#include <stdint.h>
#define ARM_PMU_INST_RETIRED 0x0008u
#define ARM_PMU_L1D_CACHE_REFILL 0x0003u
#define ARM_PMU_STALL 0x0024u
#define ARM_PMU_BR_MIS_PRED 0x0010u
static inline void ARM_PMU_Enable(void) {}
static inline void ARM_PMU_Disable(void) {}
static inline void ARM_PMU_CYCCNT_Reset(void) {}
static inline void ARM_PMU_EVCNTR_ALL_Reset(void) {}
static inline void ARM_PMU_Set_EVTYPER(uint32_t num, uint32_t type) { (void)num; (void)type; }
static inline void ARM_PMU_CNTR_Enable(uint32_t mask) { (void)mask; }
static inline void ARM_PMU_CNTR_Disable(uint32_t mask) { (void)mask; }
static inline uint32_t ARM_PMU_Get_EVCNTR(uint32_t num) { (void)num; return 0u; }
"""

NNFUNCTIONS_STUB = """\
#pragma once
#include <stdint.h>
typedef _Float16 float16_t;
typedef int arm_cmsis_nn_status;
#define ARM_CMSIS_NN_SUCCESS 0
#define ARM_CMSIS_NN_ARG_ERROR -1
"""

PROBE_MAIN = """\
#include <stdlib.h>
#include "host_stubs.h"
#include "test_runtime/helia_test_runtime.h"
{benchmark}
static int helia_probe_calls;
static int helia_probe_fail_at;
static int32_t helia_probe_op(void)
{{
    return ++helia_probe_calls == helia_probe_fail_at ? ARM_CMSIS_NN_ARG_ERROR : ARM_CMSIS_NN_SUCCESS;
}}
int main(int argc, char **argv)
{{
    helia_probe_fail_at = argc > 1 ? atoi(argv[1]) : 0;
    int32_t failures = helia_benchmark_run("probe", helia_probe_op);
    printf("RESULT %d calls=%d\\n", (int)failures, helia_probe_calls);
    return failures != 0;
}}
"""


def _render_benchmark(target: str) -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)))
    return env.get_template("common/standalone/benchmark.j2").render(benchmark_target=target)


@pytest.fixture(params=["fvp", "hardware"])
def probe(request, tmp_path: Path) -> Path:
    cc = shutil.which("gcc-12") or shutil.which("gcc")
    if cc is None:
        pytest.skip("host GCC required for the benchmark status probe")
    (tmp_path / "host_stubs.h").write_text(HOST_STUBS)
    (tmp_path / "pmu_armv8.h").write_text(PMU_STUBS)
    (tmp_path / "arm_nnfunctions.h").write_text(NNFUNCTIONS_STUB)
    (tmp_path / "probe.c").write_text(PROBE_MAIN.format(benchmark=_render_benchmark(request.param)))
    # The real runtime, except that finishing exits instead of signalling the FVP and spinning.
    (tmp_path / "runtime.c").write_text(
        "#define helia_test_finish helia_test_finish_on_target\n"
        f'#include "{ROOT / "src" / "test_runtime" / "helia_test_runtime.c"}"\n'
        "#undef helia_test_finish\n"
        "#include <stdlib.h>\n"
        "void helia_test_finish(int32_t n) { exit(n != 0); }\n"
    )
    binary = tmp_path / "probe"
    subprocess.run(
        [
            cc,
            "-std=gnu11",
            "-O2",
            "-DHELIA_BENCHMARK_MODE",
            f"-DHELIA_BENCHMARK_WARMUP_RUNS={WARMUP}",
            f"-DHELIA_BENCHMARK_MEASURED_RUNS={MEASURED}",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src"),
            str(tmp_path / "probe.c"),
            str(tmp_path / "runtime.c"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    return binary


def _run(binary: Path, fail_at: int) -> tuple[int, str]:
    result = subprocess.run([str(binary), str(fail_at)], capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout


def test_successful_calls_report_every_measured_run(probe: Path) -> None:
    code, out = _run(probe, 0)
    assert code == 0, out
    assert out.count("[BENCH] probe") == 1
    assert len(re.findall(r"^\[PERF\] probe: \d+ cycles\r?$", out, re.MULTILINE)) == MEASURED
    assert "Failures" not in out
    assert f"RESULT 0 calls={WARMUP + MEASURED}" in out


def test_failed_warmup_call_fails_before_any_measurement(probe: Path) -> None:
    code, out = _run(probe, 1)
    assert code == 1, out
    assert "[BENCH]" not in out and "[PERF]" not in out and "[PMU]" not in out
    assert "probe failed with status -1" in out
    assert re.search(r"^1 Failures\r?$", out, re.MULTILINE)
    assert "RESULT 1 calls=1" in out


def test_failed_measured_call_is_not_reported_as_a_cycle_count(probe: Path) -> None:
    code, out = _run(probe, WARMUP + 2)
    assert code == 1, out
    assert len(re.findall(r"^\[PERF\] probe:", out, re.MULTILINE)) == 1
    assert out.count("[PMU] probe:") in (0, 1)
    assert "probe failed with status -1" in out
    assert re.search(r"^1 Failures\r?$", out, re.MULTILINE)
    assert f"RESULT 1 calls={WARMUP + 2}" in out


def test_failed_benchmark_output_is_classified_as_a_failure(probe: Path, tmp_path: Path) -> None:
    from helia_core_tester.reporting.models import TestStatus
    from helia_core_tester.reporting.parser import TestResultParser

    _code, out = _run(probe, WARMUP + 2)
    result = TestResultParser().parse_fvp_output(out, tmp_path / "probe.elf", "cortex-m55", 0.0)
    assert result.status == TestStatus.FAIL
