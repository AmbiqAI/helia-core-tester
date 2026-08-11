#!/bin/bash
# Benchmarks every S4 Convolve test variant -- both the plain (no weight-sum
# context) route and the "_kernel_sum" route that exercises
# arm_convolve_wrapper_s4_with_weight_sum with a precomputed weight-sum buffer
# (see ns-cmsis-nn PR #208) -- on cortex-m55 (MVE, the Apollo510 default
# build) and cortex-m55-dsp (same Apollo510/Cortex-M55 silicon, but compiled
# with `-mcpu=cortex-m55+nomve` to force the DSP-only fallback code path --
# see helia_core_tester/core/cpu_targets.py's target_cpu_cmake_value()) --
# and reports a before/after DWT cycle-count comparison for each pair.
#
# cortex-m0 is intentionally excluded: it has no DWT cycle counter, so
# HELIA_BENCHMARK_MODE refuses to compile there (see
# assets/templates/common/standalone/benchmark_fvp.j2 /
# assets/templates/common/standalone/benchmark_hw.j2).
#
# This drives CMake/the FVP directly instead of going through
# `helia_core_tester full`/`build`, because this tree's CLI has no
# --benchmark/--hardware flags (those only exist on the unpushed
# feature/hardware-benchmark-flags branch, which also requires real Apollo
# silicon and is not usable in this sandbox). Cycle counts reported here are
# from the Corstone-300 FVP's DWT model, which is a *functional* model, not
# cycle-accurate silicon -- treat these numbers as directional/relative
# (same op, same simulated core, with vs. without weight-sum), not as
# absolute real-hardware timing. For real Apollo510 hardware timing (PMU +
# DWT), see the hardware benchmark flow (benchmark_hw.j2).
#
# Usage:
#   scripts/test_s4_conv_benchmark.sh [--cpu CPU_LIST] [--warmup N] [--runs N] [--name NAME_FILTER]
#   scripts/test_s4_conv_benchmark.sh --hardware --cpu-freq HZ --swo-freq HZ [--pf-addr ADDR] \
#       [--jlink-device DEVICE] [--jlink-speed KHZ]
#
# Examples:
#   scripts/test_s4_conv_benchmark.sh
#   scripts/test_s4_conv_benchmark.sh --cpu cortex-m55-dsp
#   scripts/test_s4_conv_benchmark.sh --name convolve_generic_s4
#   scripts/test_s4_conv_benchmark.sh --hardware --cpu-freq <HZ> --swo-freq <HZ>
#       (--jlink-device defaults to AP510NFA-CBR, the confirmed Apollo510 device name;
#        --cpu-freq/--swo-freq are Apollo510-specific and not guessed here -- see
#        Ambiq's clock config docs, e.g. HFRC2-derived core clock and the SWO
#        trace clock configured in your firmware's ITM/TPIU setup)
#
# --hardware mode (real Apollo510 silicon, not FVP):
#   Builds against the vendored NSX Apollo510 board bring-up
#   (modules/nsx-ambiq-sdk + boards/apollo510_evb) via the new
#   HELIA_HARDWARE_BUILD CMake path (CMakeLists.txt's nsx_bootstrap_app()/
#   nsx_finalize_app() wiring), flashes each generated test ELF via SEGGER
#   J-Link (`<target>_flash` CMake target, using
#   cmake/nsx/segger/templates/*.jlink.in), and attempts to capture
#   `[BENCH]`/`[PERF]`/`[PMU]` lines (assets/templates/common/standalone/
#   benchmark_hw.j2, output via am_util_stdio_printf -> ITM/SWO) using
#   JLinkSWOViewerCL.
#
#   The CMake configure/cross-compile-build path (steps 1-2 below) HAS been
#   verified in a sandbox (arm-none-eabi-gcc 14.3.1, no real board attached):
#   both cortex-m55 and cortex-m55-dsp variants of convolve_generic_s4
#   configure, compile, link, and produce a .bin via nsx_finalize_app().
#   Flashing (step 3, via JLinkExe) and SWO capture (step 4, via
#   JLinkSWOViewerCL) have NOT been tested -- no real hardware/JLink was
#   available in that sandbox. Please run this and paste back any errors
#   from the flash/capture steps so they can be fixed. Known unknowns:
#     1. JLinkSWOViewerCL's exact non-interactive/logging CLI flags -- if
#        capture doesn't work, run `JLinkSWOViewerCL -h` and report back.
#     2. NSX_SEGGER_DEVICE/IF_SPEED/CPUFREQ/SWOFREQ have no vendored default
#        for apollo510_evb -- pass them via --jlink-device/--jlink-speed/
#        --cpu-freq/--swo-freq (all required with --hardware except
#        --jlink-speed, which defaults to 4000).
#     3. PF_ADDR (--pf-addr, default 0x00410000) is the Apollo510 MCU_MRAM
#        ORIGIN from the vendored SBL linker script -- correct for a raw
#        dev image, but Apollo5-series parts may require an SBL-signed
#        image instead; see cmake/nsx/segger/templates/flash_cmds.jlink.in.
#
set -euo pipefail

# Get the repository root directory (Tests/helia-core-tester).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Apollo510/Cortex-M55 only: cortex-m55 = MVE path (default build), cortex-m55-dsp
# = same silicon with MVE forced off (DSP-only fallback path). No cortex-m0 (no DWT).
CPU="cortex-m55,cortex-m55-dsp"
WARMUP_RUNS=3
MEASURED_RUNS=10
NAME_FILTER=""
RUN_TIMEOUT=20
JOBS=4
HARDWARE=0
# Confirmed real J-Link device name for Apollo510 (user-verified, 2026-08-05).
JLINK_DEVICE="AP510NFA-CBR"
JLINK_SPEED=4000
# Apollo510 MCU_MRAM ORIGIN from the vendored SBL linker script (factual,
# not guessed -- see modules/nsx-ambiq-sdk/modules/nsx-core/src/apollo510/
# gcc/linker_script_sbl.ld). Only valid for a non-SBL-signed dev image; see
# the header comment above and cmake/nsx/segger/templates/flash_cmds.jlink.in.
PF_ADDR="0x00410000"
CPU_FREQ=""
SWO_FREQ=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --runs)
            MEASURED_RUNS="$2"
            shift 2
            ;;
        --name)
            NAME_FILTER="$2"
            shift 2
            ;;
        --timeout)
            RUN_TIMEOUT="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --hardware)
            HARDWARE=1
            shift
            ;;
        --jlink-device)
            JLINK_DEVICE="$2"
            shift 2
            ;;
        --jlink-speed)
            JLINK_SPEED="$2"
            shift 2
            ;;
        --pf-addr)
            PF_ADDR="$2"
            shift 2
            ;;
        --cpu-freq)
            CPU_FREQ="$2"
            shift 2
            ;;
        --swo-freq)
            SWO_FREQ="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--cpu CPU_LIST] [--warmup N] [--runs N] [--name NAME_FILTER] [--timeout SECONDS] [--jobs N]"
            echo "       $0 --hardware --jlink-device DEVICE [--jlink-speed KHZ] [--pf-addr ADDR] [--cpu-freq HZ] [--swo-freq HZ]"
            echo ""
            echo "Options:"
            echo "  --cpu CPU_LIST   Target CPU(s), comma-separated (default: cortex-m55,cortex-m55-dsp; Apollo510 only, no cortex-m0/no DWT)"
            echo "  --warmup N       Untimed warmup iterations per op (default: 3)"
            echo "  --runs N         Timed measured iterations per op (default: 10)"
            echo "  --name FILTER    Only benchmark tests whose name contains FILTER (e.g. convolve_generic_s4)"
            echo "  --timeout SEC    Per-ELF FVP run timeout in seconds, or per-flash capture window in --hardware mode (default: 20)"
            echo "  --jobs N         Parallel build jobs (default: 4)"
            echo "  --hardware       Run on real Apollo510 silicon via SEGGER J-Link instead of the FVP (see header comment: flash/capture steps are still unverified)"
            echo "  --jlink-device D SEGGER J-Link device name (default: AP510NFA-CBR, the confirmed Apollo510 device name)"
            echo "  --jlink-speed K  J-Link SWD interface speed in kHz (default: 4000)"
            echo "  --pf-addr ADDR   Flash/MRAM load address (default: 0x00410000, Apollo510 MCU_MRAM origin; verify against your image type -- see header comment)"
            echo "  --cpu-freq HZ    Target core clock in Hz, required for SWO capture (--hardware only)"
            echo "  --swo-freq HZ    SWO trace clock in Hz, required for SWO capture (--hardware only)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

IFS=',' read -ra CPU_LIST <<< "${CPU}"
DOWNLOADS_DIR="${REPO_ROOT}/artifacts/downloads"

if [[ "${HARDWARE}" -eq 1 ]]; then
    # --- Real Apollo510 hardware path -----------------------------------
    # See the header comment for context. The CMake configure/cross-compile
    # build has been verified (see CMakeLists.txt comments); flashing/SWO
    # capture below have not -- please run and paste back any errors/output
    # so they can be corrected.
    if [[ -z "${JLINK_DEVICE}" ]]; then
        echo "ERROR: --hardware requires --jlink-device DEVICE." >&2
        exit 1
    fi
    if [[ -z "${CPU_FREQ}" || -z "${SWO_FREQ}" ]]; then
        echo "ERROR: --hardware requires --cpu-freq and --swo-freq (needed for" >&2
        echo "SWO capture, since that's how printf/am_util_stdio_printf output" >&2
        echo "reaches the host on this NSX Apollo510 bring-up -- see" >&2
        echo "modules/nsx-ambiq-sdk/.../apollo510/nsx_system_platform.c, which" >&2
        echo "wires am_util_stdio_printf to am_hal_itm_print)." >&2
        exit 1
    fi

    JLINK_EXE="$(command -v JLinkExe || true)"
    if [[ -z "${JLINK_EXE}" ]]; then
        echo "ERROR: JLinkExe not found on PATH -- SEGGER J-Link tools are" >&2
        echo "required to flash real hardware and are not installed in this" >&2
        echo "environment." >&2
        exit 1
    fi
    JLINK_SWO_EXE="$(command -v JLinkSWOViewerCL || true)"
    if [[ -z "${JLINK_SWO_EXE}" ]]; then
        echo "WARNING: JLinkSWOViewerCL not found on PATH -- flashing will" >&2
        echo "still work, but this script has no way to automatically capture" >&2
        echo "[BENCH]/[PERF]/[PMU] SWO output without it. You can flash with" >&2
        echo "this script and read output manually with SEGGER's GUI SWO" >&2
        echo "viewer instead." >&2
    fi

    for cpu in "${CPU_LIST[@]}"; do
        echo "================================================================================"
        echo "Generating + building S4 Convolve HARDWARE benchmark harness for ${cpu}"
        echo "(HELIA_BENCHMARK_MODE=ON, HELIA_HARDWARE_BUILD=ON, warmup=${WARMUP_RUNS}, measured=${MEASURED_RUNS})"
        echo "================================================================================"

        GEN_ARGS=(--op Convolve --dtype S4 --cpu "${cpu}" --suite int -v 1)
        if [[ -n "${NAME_FILTER}" ]]; then
            GEN_ARGS+=(--name "${NAME_FILTER}")
        fi
        # HELIA_BENCH_TARGET=hardware selects benchmark_hw.j2 (PMU harness,
        # am_util_stdio_printf/ITM output) instead of the FVP DWT-only
        # backend -- see convolve.py's context['benchmark_target'] wiring.
        HELIA_BENCH_TARGET=hardware uv run helia_core_tester generate "${GEN_ARGS[@]}"

        uv run python -m helia_core_tester.fvp.build_and_run_fvp \
            --no-setup \
            --downloads-dir "${DOWNLOADS_DIR}" \
            --ethos-path "${DOWNLOADS_DIR}/ethos-u-core-platform" \
            --cmsis5-path "${DOWNLOADS_DIR}/CMSIS_5" \
            --no-gcc-from-download \
            --cpu "${cpu}" --suite int \
            --cmake-def CMSIS_OPTIMIZATION_LEVEL=-Ofast \
            --cmake-def ARM_NN_ENABLE_F32=OFF --cmake-def ARM_NN_ENABLE_F16=OFF \
            --cmake-def HELIA_BENCHMARK_MODE=ON \
            --cmake-def "HELIA_BENCHMARK_WARMUP_RUNS=${WARMUP_RUNS}" \
            --cmake-def "HELIA_BENCHMARK_MEASURED_RUNS=${MEASURED_RUNS}" \
            --cmake-def HELIA_HARDWARE_BUILD=ON \
            --cmake-def "NSX_SEGGER_DEVICE=${JLINK_DEVICE}" \
            --cmake-def "NSX_SEGGER_IF_SPEED=${JLINK_SPEED}" \
            --cmake-def "NSX_SEGGER_PF_ADDR=${PF_ADDR}" \
            --cmake-def "NSX_SEGGER_CPUFREQ=${CPU_FREQ}" \
            --cmake-def "NSX_SEGGER_SWOFREQ=${SWO_FREQ}" \
            --no-run --no-report -j "${JOBS}" --verbosity 1

        BUILD_DIR="${REPO_ROOT}/artifacts/build-int-${cpu}-gcc"
        TESTS_BUILD_DIR="${BUILD_DIR}/tests/ConvolutionFunctions"
        if [[ ! -d "${TESTS_BUILD_DIR}" ]]; then
            echo "ERROR: expected build dir not found: ${TESTS_BUILD_DIR}" >&2
            exit 1
        fi

        echo "================================================================================"
        echo "Flashing + capturing S4 Convolve benchmarks on real hardware (${cpu})"
        echo "================================================================================"

        shopt -s nullglob
        for elf in "${TESTS_BUILD_DIR}"/convolve_*.elf; do
            name="$(basename "${elf}" .elf)"
            if [[ -n "${NAME_FILTER}" && "${name}" != *"${NAME_FILTER}"* ]]; then
                continue
            fi
            echo "  flashing ${name}..." >&2
            if ! cmake --build "${BUILD_DIR}" --target "${name}_flash"; then
                echo "  WARNING: flash failed/unavailable for ${name}, skipping" >&2
                continue
            fi

            if [[ -n "${JLINK_SWO_EXE}" ]]; then
                # BEST-EFFORT / UNVERIFIED: exact JLinkSWOViewerCL non-interactive
                # logging flags are my best understanding, not confirmed against
                # real hardware. If this doesn't produce a readable log file,
                # run `JLinkSWOViewerCL -h` on your machine, paste the output
                # back, and this will be corrected.
                log_file="$(mktemp)"
                timeout -s KILL "${RUN_TIMEOUT}" "${JLINK_SWO_EXE}" \
                    -device "${JLINK_DEVICE}" -cpufreq "${CPU_FREQ}" -swofreq "${SWO_FREQ}" \
                    -itmport 0 -f "${log_file}" || true
                grep -E "^\[BENCH\]|^\[PERF\]|^\[PMU\]" "${log_file}" || echo "  (no [BENCH]/[PERF]/[PMU] lines captured for ${name})"
                rm -f "${log_file}"
            fi
        done
        shopt -u nullglob
    done
    exit 0
fi

FVP_BIN="${DOWNLOADS_DIR}/corstone300_download/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55"
if [[ ! -x "${FVP_BIN}" ]]; then
    echo "ERROR: FVP binary not found at ${FVP_BIN}" >&2
    echo "Run 'uv run helia_core_tester full --op Convolve --dtype S4 --cpu cortex-m4' once first" >&2
    echo "(or otherwise let helia_core_tester bootstrap artifacts/downloads/), then retry." >&2
    exit 1
fi

run_fvp_and_parse() {
    local elf="$1"
    timeout -s KILL "${RUN_TIMEOUT}" "${FVP_BIN}" \
        -C mps3_board.visualisation.disable-visualisation=1 \
        -C mps3_board.telnetterminal0.start_telnet=0 \
        -C mps3_board.uart0.out_file="-" -C mps3_board.uart0.unbuffered_output=1 \
        "${elf}" 2>&1 | grep "^\[PERF\]" | awk -F': ' '{print $2}' | awk '{print $1}'
}

for cpu in "${CPU_LIST[@]}"; do
    echo "================================================================================"
    echo "Generating + building S4 Convolve benchmark harness for ${cpu}"
    echo "(HELIA_BENCHMARK_MODE=ON, warmup=${WARMUP_RUNS}, measured=${MEASURED_RUNS})"
    echo "================================================================================"

    GEN_ARGS=(--op Convolve --dtype S4 --cpu "${cpu}" --suite int -v 1)
    if [[ -n "${NAME_FILTER}" ]]; then
        GEN_ARGS+=(--name "${NAME_FILTER}")
    fi
    uv run helia_core_tester generate "${GEN_ARGS[@]}"

    # Drive the low-level FVP build script directly with extra -D cmake defines
    # for HELIA_BENCHMARK_MODE: `helia_core_tester build` doesn't expose a
    # --benchmark flag in this tree (see script header comment above).
    uv run python -m helia_core_tester.fvp.build_and_run_fvp \
        --no-setup \
        --downloads-dir "${DOWNLOADS_DIR}" \
        --ethos-path "${DOWNLOADS_DIR}/ethos-u-core-platform" \
        --cmsis5-path "${DOWNLOADS_DIR}/CMSIS_5" \
        --no-gcc-from-download \
        --cpu "${cpu}" --suite int \
        --cmake-def CMSIS_OPTIMIZATION_LEVEL=-Ofast \
        --cmake-def ARM_NN_ENABLE_F32=OFF --cmake-def ARM_NN_ENABLE_F16=OFF \
        --cmake-def HELIA_BENCHMARK_MODE=ON \
        --cmake-def "HELIA_BENCHMARK_WARMUP_RUNS=${WARMUP_RUNS}" \
        --cmake-def "HELIA_BENCHMARK_MEASURED_RUNS=${MEASURED_RUNS}" \
        --no-run --no-report -j "${JOBS}" --verbosity 1

    BUILD_DIR="${REPO_ROOT}/artifacts/build-int-${cpu}-gcc/tests/ConvolutionFunctions"
    if [[ ! -d "${BUILD_DIR}" ]]; then
        echo "ERROR: expected build dir not found: ${BUILD_DIR}" >&2
        exit 1
    fi

    echo "================================================================================"
    echo "Running S4 Convolve benchmarks on FVP (${cpu}) -- DWT cycle counts are a"
    echo "directional/relative FVP-model measurement, not real-silicon timing."
    echo "================================================================================"

    declare -A CYCLES
    shopt -s nullglob
    for elf in "${BUILD_DIR}"/convolve_*.elf; do
        name="$(basename "${elf}" .elf)"
        if [[ -n "${NAME_FILTER}" && "${name}" != *"${NAME_FILTER}"* ]]; then
            continue
        fi
        echo "  benchmarking ${name}..." >&2
        readings="$(run_fvp_and_parse "${elf}")"
        avg="$(echo "${readings}" | awk '{sum+=$1; n++} END {if (n>0) printf "%.1f", sum/n; else print "NA"}')"
        CYCLES["${name}"]="${avg}"
    done
    shopt -u nullglob

    printf "\n%-50s %12s %12s %10s %8s\n" "Test (${cpu})" "Plain" "WeightSum" "Delta" "Delta%"
    printf '%s\n' "--------------------------------------------------------------------------------------------------"
    total_plain=0
    total_ws=0
    for name in "${!CYCLES[@]}"; do
        [[ "${name}" == *_kernel_sum ]] && continue
        ws_name="${name}_kernel_sum"
        [[ -z "${CYCLES[${ws_name}]:-}" ]] && continue
        plain="${CYCLES[${name}]}"
        ws="${CYCLES[${ws_name}]}"
        [[ "${plain}" == "NA" || "${ws}" == "NA" ]] && continue
        delta="$(awk -v a="${plain}" -v b="${ws}" 'BEGIN{printf "%.1f", b-a}')"
        pct="$(awk -v a="${plain}" -v d="${delta}" 'BEGIN{if (a>0) printf "%+.2f", (d/a)*100; else print "NA"}')"
        printf "%-50s %12s %12s %+10s %7s%%\n" "${name}" "${plain}" "${ws}" "${delta}" "${pct}"
        total_plain="$(awk -v a="${total_plain}" -v b="${plain}" 'BEGIN{print a+b}')"
        total_ws="$(awk -v a="${total_ws}" -v b="${ws}" 'BEGIN{print a+b}')"
    done
    total_delta="$(awk -v a="${total_plain}" -v b="${total_ws}" 'BEGIN{printf "%.1f", b-a}')"
    total_pct="$(awk -v a="${total_plain}" -v d="${total_delta}" 'BEGIN{if (a>0) printf "%+.2f", (d/a)*100; else print "NA"}')"
    printf '%s\n' "--------------------------------------------------------------------------------------------------"
    printf "%-50s %12s %12s %+10s %7s%%\n" "TOTAL" "${total_plain}" "${total_ws}" "${total_delta}" "${total_pct}"
    echo ""
    unset CYCLES
done
