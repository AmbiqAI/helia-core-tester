#!/bin/bash
# Runs the full generated test suite on real Apollo510/Cortex-M55 hardware
# and reports performance numbers (DWT cycle counts + PMU counters), via the
# `helia_core_tester perf-stream` CLI:
#
#   1. uv run helia_core_tester generate ...
#   2. uv run helia_core_tester perf-stream flash --cpu <CPU>
#   3. uv run helia_core_tester perf-stream run-generated --serial-no <SERIAL> ...
#
# By default (no --family given) this runs EVERY operator family with real
# firmware dispatch support in one session -- see the `_BUILDERS` dispatch
# table in helia_core_tester/perf_stream/generated_test_bridge.py (or call
# `bridged_families()` at runtime) for the up-to-date list. Pass --family to
# restrict to one family. Kernels without a registered builder are reported
# as skipped (with the reason) rather than silently omitted.
#
# If --serial-no isn't given, this auto-detects the connected J-Link probe's
# serial number via `JLinkExe -CommanderScript ... ShowEmuList` (requires
# SEGGER J-Link tools on PATH and exactly one probe connected).
#
# Usage:
#   scripts/run_hardware_perf_suite.sh [--serial-no SERIAL] [--cpu CPU] \
#       [--family FAMILY] [--test-name FILTER] [--limit N] \
#       [--session-id NAME] [--skip-generate] [--skip-flash]
#
# Examples:
#   scripts/run_hardware_perf_suite.sh                     # run all bridged families
#   scripts/run_hardware_perf_suite.sh --serial-no 1160002276
#   scripts/run_hardware_perf_suite.sh --family ConvolutionFunctions --test-name convolve_generic_s4
#
# Not tested against real hardware (no board/J-Link probe available in this
# sandbox) -- please run and report back any errors from the flash/serial
# auto-detect/RTT-capture steps so they can be fixed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SERIAL_NO=""
CPU="cortex-m55"
FAMILY=""
TEST_NAME=""
LIMIT=""
SESSION_ID="apollo510-full-suite-$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_GENERATE=0
SKIP_FLASH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial-no) SERIAL_NO="$2"; shift 2 ;;
        --cpu) CPU="$2"; shift 2 ;;
        --family) FAMILY="$2"; shift 2 ;;
        --test-name) TEST_NAME="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --session-id) SESSION_ID="$2"; shift 2 ;;
        --skip-generate) SKIP_GENERATE=1; shift ;;
        --skip-flash) SKIP_FLASH=1; shift ;;
        -h|--help)
            sed -n '2,34p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --- Step 0: find the J-Link probe serial number if not given -------------
if [[ -z "${SERIAL_NO}" ]]; then
    JLINK_EXE="$(command -v JLinkExe || true)"
    if [[ -z "${JLINK_EXE}" ]]; then
        echo "ERROR: --serial-no not given and JLinkExe not found on PATH." >&2
        echo "       Install SEGGER J-Link tools, or pass --serial-no explicitly" >&2
        echo "       (find it via \`JLinkExe\` -> ShowEmuList)." >&2
        exit 1
    fi

    echo "[run_hardware_perf_suite] No --serial-no given; auto-detecting connected J-Link probe..."
    JLINK_SCRIPT="$(mktemp)"
    SERIAL_LIST_FILE="$(mktemp)"
    trap 'rm -f "${JLINK_SCRIPT}" "${SERIAL_LIST_FILE}"' EXIT
    echo "ShowEmuList" > "${JLINK_SCRIPT}"

    JLINK_OUT_FILE="$(mktemp)"
    "${JLINK_EXE}" -CommanderScript "${JLINK_SCRIPT}" < /dev/null > "${JLINK_OUT_FILE}" 2>&1 || true
    grep -oE 'Serial [Nn]umber:? ?[0-9]+' "${JLINK_OUT_FILE}" | grep -oE '[0-9]+' > "${SERIAL_LIST_FILE}"
    SERIAL_COUNT="$(wc -l < "${SERIAL_LIST_FILE}" | tr -d '[:space:]')"

    if [[ "${SERIAL_COUNT}" -eq 0 ]]; then
        echo "ERROR: No connected J-Link probes detected. Full JLinkExe output:" >&2
        cat "${JLINK_OUT_FILE}" >&2
        rm -f "${JLINK_OUT_FILE}"
        echo "Pass --serial-no explicitly once you know it." >&2
        exit 1
    elif [[ "${SERIAL_COUNT}" -gt 1 ]]; then
        echo "ERROR: Multiple J-Link probes detected:" >&2
        cat "${SERIAL_LIST_FILE}" >&2
        rm -f "${JLINK_OUT_FILE}"
        echo "Pass --serial-no explicitly to select one." >&2
        exit 1
    fi

    SERIAL_NO="$(head -n 1 "${SERIAL_LIST_FILE}")"
    rm -f "${JLINK_OUT_FILE}"
    echo "[run_hardware_perf_suite] Using detected J-Link serial: ${SERIAL_NO}"
fi

echo
echo "[run_hardware_perf_suite] Config:"
echo "  serial_no=${SERIAL_NO} cpu=${CPU} family=${FAMILY:-<all bridged families>} test_name=${TEST_NAME:-<all>} limit=${LIMIT:-<all>}"
echo "  session_id=${SESSION_ID}"
echo

# --- Step 1: generate the test suite ---------------------------------------
if [[ "${SKIP_GENERATE}" -eq 0 ]]; then
    echo "[run_hardware_perf_suite] Generating tests (cpu=${CPU})..."
    uv run helia_core_tester generate --cpu "${CPU}"
else
    echo "[run_hardware_perf_suite] --skip-generate set; reusing existing artifacts/generated_tests."
fi

# --- Step 2: flash the real benchmark-server firmware ----------------------
# Note: `perf-stream flash` has no --serial-no option -- J-Link flashing
# (via nsx_add_segger_targets()'s CMake target) assumes a single connected
# probe. The detected/given SERIAL_NO is only used below for the RTT
# session in `perf-stream run-generated`.
if [[ "${SKIP_FLASH}" -eq 0 ]]; then
    echo "[run_hardware_perf_suite] Flashing hct_benchmark_server firmware..."
    uv run helia_core_tester perf-stream flash --cpu "${CPU}"
else
    echo "[run_hardware_perf_suite] --skip-flash set; reusing firmware already running on the board."
fi

# --- Step 3: stream the full generated suite and collect performance data --
echo "[run_hardware_perf_suite] Running full generated test suite over HCTP/RTT..."
RUN_GENERATED_ARGS=(
    perf-stream run-generated
    --serial-no "${SERIAL_NO}"
    --session-id "${SESSION_ID}"
    --cpu "${CPU}"
)
[[ -n "${FAMILY}" ]] && RUN_GENERATED_ARGS+=(--family "${FAMILY}")
[[ -n "${TEST_NAME}" ]] && RUN_GENERATED_ARGS+=(--test-name "${TEST_NAME}")
[[ -n "${LIMIT}" ]] && RUN_GENERATED_ARGS+=(--limit "${LIMIT}")

uv run helia_core_tester "${RUN_GENERATED_ARGS[@]}"

BUNDLE_DIR="artifacts/reports/performance_stream/${SESSION_ID}"
CASE_SUMMARY_CSV="${BUNDLE_DIR}/case_summary.csv"

echo
echo "=================================================================="
echo "[run_hardware_perf_suite] Done. Reports written to:"
echo "  ${CASE_SUMMARY_CSV}"
echo "  ${BUNDLE_DIR}/raw_samples.csv"
echo "  ${BUNDLE_DIR}/junit.xml"
echo "=================================================================="

if [[ -f "${CASE_SUMMARY_CSV}" ]]; then
    # Colorized pass/fail summary: counts + a highlighted list of any failing
    # cases, so the important signal isn't buried in a full-suite CSV dump.
    awk -F',' -v red="$(tput setaf 1 2>/dev/null || true)" \
        -v green="$(tput setaf 2 2>/dev/null || true)" \
        -v bold="$(tput bold 2>/dev/null || true)" \
        -v reset="$(tput sgr0 2>/dev/null || true)" '
        NR == 1 {
            for (i = 1; i <= NF; i++) { col[$i] = i }
            next
        }
        {
            total++
            if ($col["comparison_passed"] == "true") {
                passed++
            } else {
                failed++
                fail_list[failed] = $col["case_id"]
            }
        }
        END {
            printf "\n%sSummary: %d/%d passed", bold, passed, total
            if (failed > 0) { printf ", %d failed%s\n", failed, reset } else { printf "%s\n", reset }
            if (failed > 0) {
                printf "\n%s%sFailed cases:%s\n", bold, red, reset
                for (i = 1; i <= failed; i++) { printf "  %s- %s%s\n", red, fail_list[i], reset }
            } else {
                printf "%s%sAll cases passed.%s\n", bold, green, reset
            }
        }
    ' "${CASE_SUMMARY_CSV}"
fi
