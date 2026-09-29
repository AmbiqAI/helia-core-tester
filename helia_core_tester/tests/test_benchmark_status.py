"""A benchmark whose kernel call fails must fail the case, not report cycles (#146).

Both benchmark backends are rendered from the real templates and compiled on the
host against the real test runtime, with only the DWT/PMU registers stubbed. The op
under test fails on a chosen call; the output and return value must show the run
stopped there.
"""

from __future__ import annotations

import os
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

# HELIA_HARDWARE_BUILD routes output through am_util_stdio_printf because plain printf is
# a no-op on the NSX bring-up; tagging that channel shows which one each line used.
AMBIQ_STUBS = """\
#pragma once
#include <stdarg.h>
#include <stdio.h>
static inline int am_util_stdio_printf(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    fputs("[am] ", stdout);
    int n = vprintf(fmt, args);
    va_end(args);
    return n;
}
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


def _host_cc() -> str:
    cc = shutil.which("gcc-12") or shutil.which("gcc")
    if cc is None:
        pytest.skip("host GCC required for the benchmark status probe")
    return cc


def _write_runtime(directory: Path) -> Path:
    """The real runtime, except that finishing exits with the verdict instead of
    signalling the FVP and spinning."""
    runtime = directory / "runtime.c"
    runtime.write_text(
        "#define helia_test_finish helia_test_finish_on_target\n"
        f'#include "{ROOT / "src" / "test_runtime" / "helia_test_runtime.c"}"\n'
        "#undef helia_test_finish\n"
        "#include <stdlib.h>\n"
        "void helia_test_finish(int32_t n) { exit(n != 0); }\n"
    )
    return runtime


def _build_probe(tmp_path: Path, target: str, extra_flags: list[str]) -> Path:
    cc = _host_cc()
    (tmp_path / "host_stubs.h").write_text(HOST_STUBS)
    (tmp_path / "pmu_armv8.h").write_text(PMU_STUBS)
    (tmp_path / "arm_nnfunctions.h").write_text(NNFUNCTIONS_STUB)
    (tmp_path / "am_mcu_apollo.h").write_text("#pragma once\n")
    (tmp_path / "am_util_stdio.h").write_text(AMBIQ_STUBS)
    (tmp_path / "probe.c").write_text(PROBE_MAIN.format(benchmark=_render_benchmark(target)))
    binary = tmp_path / "probe"
    subprocess.run(
        [
            cc,
            "-std=gnu11",
            "-O2",
            "-DHELIA_BENCHMARK_MODE",
            f"-DHELIA_BENCHMARK_WARMUP_RUNS={WARMUP}",
            f"-DHELIA_BENCHMARK_MEASURED_RUNS={MEASURED}",
            *extra_flags,
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src"),
            str(tmp_path / "probe.c"),
            str(_write_runtime(tmp_path)),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    return binary


@pytest.fixture(params=["fvp", "hardware"])
def probe(request, tmp_path: Path) -> tuple[Path, str]:
    return _build_probe(tmp_path, request.param, []), request.param


def _run(binary: Path, fail_at: int) -> tuple[int, str]:
    result = subprocess.run([str(binary), str(fail_at)], capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout


def _pmu_lines(target: str, measured_runs_reported: int) -> int:
    return measured_runs_reported if target == "hardware" else 0


def test_successful_calls_report_every_measured_run(probe: tuple[Path, str]) -> None:
    binary, target = probe
    code, out = _run(binary, 0)
    assert code == 0, out
    assert out.count("[BENCH] probe") == 1
    assert len(re.findall(r"^\[PERF\] probe: \d+ cycles\r?$", out, re.MULTILINE)) == MEASURED
    assert out.count("[PMU] probe:") == _pmu_lines(target, MEASURED)
    assert "Failures" not in out
    assert f"RESULT 0 calls={WARMUP + MEASURED}" in out


def test_failed_warmup_call_fails_before_any_measurement(probe: tuple[Path, str]) -> None:
    binary, _target = probe
    code, out = _run(binary, 1)
    assert code == 1, out
    assert "[BENCH]" not in out and "[PERF]" not in out and "[PMU]" not in out
    assert "probe failed with status -1" in out
    assert re.search(r"^1 Failures\r?$", out, re.MULTILINE)
    assert "RESULT 1 calls=1" in out


def test_failed_measured_call_is_not_reported_as_a_cycle_count(probe: tuple[Path, str]) -> None:
    binary, target = probe
    code, out = _run(binary, WARMUP + 2)
    assert code == 1, out
    assert len(re.findall(r"^\[PERF\] probe:", out, re.MULTILINE)) == 1
    assert out.count("[PMU] probe:") == _pmu_lines(target, 1)
    assert "probe failed with status -1" in out
    assert re.search(r"^1 Failures\r?$", out, re.MULTILINE)
    assert f"RESULT 1 calls={WARMUP + 2}" in out


def test_failed_benchmark_output_is_classified_as_a_failure(probe: tuple[Path, str], tmp_path: Path) -> None:
    from helia_core_tester.reporting.models import TestStatus
    from helia_core_tester.reporting.parser import TestResultParser

    binary, _target = probe
    _code, out = _run(binary, WARMUP + 2)
    result = TestResultParser().parse_fvp_output(out, tmp_path / "probe.elf", "cortex-m55", 0.0)
    assert result.status == TestStatus.FAIL


@pytest.mark.parametrize("fail_at", [1, WARMUP + 2], ids=["warmup", "measured"])
def test_hardware_build_reports_failures_on_the_board_console(tmp_path: Path, fail_at: int) -> None:
    """Plain printf is swallowed on the NSX bring-up, so a failure printed through it
    would be invisible on the board."""
    binary = _build_probe(tmp_path, "hardware", ["-DHELIA_HARDWARE_BUILD"])
    code, out = _run(binary, fail_at)
    assert code == 1, out
    assert re.search(r"^\[am\] probe failed with status -1\r?$", out, re.MULTILINE), out
    assert re.search(r"^\[am\] 1 Failures\r?$", out, re.MULTILINE), out
    assert "probe failed with status -1" not in out.replace("[am] probe failed", ""), out


CONVOLVE_KERNEL_STUB = """\
#include <stdlib.h>
#include "arm_nnfunctions.h"
static int helia_stub_env(const char *name)
{
    const char *value = getenv(name);
    return value ? atoi(value) : 0;
}
int32_t arm_convolve_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims,
                                         arm_nn_tensor_layout layout)
{
    (void)conv_params; (void)input_dims; (void)filter_dims; (void)output_dims; (void)layout;
    return helia_stub_env("STUB_SIZER");
}
arm_cmsis_nn_status arm_convolve_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f32 *conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const float32_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const float32_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *output_data,
                                     arm_nn_tensor_layout layout)
{
    static int calls;
    (void)ctx; (void)conv_params; (void)input_dims; (void)input_data; (void)filter_dims;
    (void)filter_data; (void)bias_dims; (void)bias_data; (void)output_dims; (void)output_data;
    (void)layout;
    return ++calls == helia_stub_env("STUB_FAIL_AT") ? ARM_CMSIS_NN_ARG_ERROR : ARM_CMSIS_NN_SUCCESS;
}
"""


@pytest.fixture(scope="module", params=["fvp", "hardware"])
def convolve_benchmark(request, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """The generated convolution harness (entry point and main included) in benchmark mode,
    built for the host against the real ns-cmsis-nn headers and a scripted kernel. The
    hardware variant is rendered for that backend and built as a hardware build, so its
    console output is tagged by the am_util_stdio_printf stub."""
    target = request.param
    cmsis_nn_root = os.environ.get("CMSIS_NN_ROOT")
    if not cmsis_nn_root or not (Path(cmsis_nn_root) / "Include" / "arm_nnfunctions.h").exists():
        pytest.skip("CMSIS_NN_ROOT with ns-cmsis-nn headers required to build the generated harness")
    cc = _host_cc()
    from helia_core_tester.core.discovery import find_descriptors_dir
    from helia_core_tester.generation.io.descriptors import load_all_descriptors
    import helia_core_tester.generation.test_ops as generation_module

    case = "convolve_float_default_f32"
    desc = next(d for d in load_all_descriptors(str(find_descriptors_dir())) if d["name"] == case)
    out_dir = tmp_path_factory.mktemp(f"convolve_benchmark_{target}")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HELIA_BENCH_TARGET", target)
        generation_module.generate_test(desc, str(out_dir), cpu="cortex-m55")
    case_dir = out_dir / desc["_family"] / case
    (out_dir / "host_stubs.h").write_text(HOST_STUBS)
    (out_dir / "pmu_armv8.h").write_text(PMU_STUBS)
    (out_dir / "am_mcu_apollo.h").write_text("#pragma once\n")
    (out_dir / "am_util_stdio.h").write_text(AMBIQ_STUBS)
    (out_dir / "kernel.c").write_text(CONVOLVE_KERNEL_STUB)
    binary = out_dir / "convolve_benchmark"
    subprocess.run(
        [
            cc,
            "-std=gnu11",
            "-O0",
            "-DARM_NN_ENABLE_F32=1",
            "-DHELIA_BENCHMARK_MODE",
            f"-DHELIA_BENCHMARK_WARMUP_RUNS={WARMUP}",
            f"-DHELIA_BENCHMARK_MEASURED_RUNS={MEASURED}",
            *(["-DHELIA_HARDWARE_BUILD"] if target == "hardware" else []),
            "-include",
            str(out_dir / "host_stubs.h"),
            "-I",
            str(out_dir),
            "-I",
            str(case_dir / "includes"),
            "-I",
            str(Path(cmsis_nn_root) / "Include"),
            "-I",
            str(ROOT / "src"),
            str(case_dir / f"{case}_convolve.c"),
            str(out_dir / "kernel.c"),
            str(_write_runtime(out_dir)),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    return binary, target


def _run_convolve(harness: tuple[Path, str], sizer: int, fail_at: int) -> tuple[int, str, str]:
    binary, target = harness
    env = dict(os.environ, STUB_SIZER=str(sizer), STUB_FAIL_AT=str(fail_at))
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30, env=env)
    # Lines the board console would show carry the stub's tag on a hardware build.
    console = "[am] " if target == "hardware" else ""
    return result.returncode, result.stdout, console


def test_generated_harness_succeeds_with_every_measured_run(convolve_benchmark: tuple[Path, str]) -> None:
    code, out, console = _run_convolve(convolve_benchmark, sizer=0, fail_at=0)
    assert code == 0, out
    perf = rf"^{re.escape(console)}\[PERF\] convolve_float_default_f32: \d+ cycles\r?$"
    assert len(re.findall(perf, out, re.MULTILINE)) == MEASURED
    assert "Failures" not in out


def test_generated_harness_fails_when_init_fails(convolve_benchmark: tuple[Path, str]) -> None:
    code, out, console = _run_convolve(convolve_benchmark, sizer=-1, fail_at=0)
    assert code == 1, out
    assert "HELIA_SIZER_INVALID[arm_convolve_f32_get_buffer_size]" in out
    assert f"{console}[BENCH] convolve_float_default_f32 skipped" in out
    assert f"{console}convolve_float_default_f32 benchmark init failed with status -1" in out
    assert re.search(rf"^{re.escape(console)}1 Failures\r?$", out, re.MULTILINE)
    assert "[PERF]" not in out


def test_generated_harness_fails_when_a_measured_call_fails(convolve_benchmark: tuple[Path, str]) -> None:
    code, out, console = _run_convolve(convolve_benchmark, sizer=0, fail_at=WARMUP + 2)
    assert code == 1, out
    assert len(re.findall(r"\[PERF\] convolve_float_default_f32:", out)) == 1
    assert f"{console}convolve_float_default_f32 failed with status -1" in out
