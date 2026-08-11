#!/bin/bash
# Runs the full generate -> build -> run pipeline for every S4 Convolve test
# variant, i.e. both the plain (no weight-sum context) route and the
# "_kernel_sum" route that exercises arm_convolve_s4/arm_convolve_wrapper_s4
# with a precomputed weight-sum buffer (see ns-cmsis-nn PR #208), across all
# three CPU profiles (pure-C, DSP, and MVE), then merges and reports code
# coverage across those CPUs.
#
# Usage:
#   scripts/test_s4_conv_weight_sum.sh [--cpu CPU_LIST] [--no-coverage] [-- <extra helia_core_tester args>]
#
# Examples:
#   scripts/test_s4_conv_weight_sum.sh
#   scripts/test_s4_conv_weight_sum.sh --cpu cortex-m4,cortex-m55
#   scripts/test_s4_conv_weight_sum.sh --no-coverage
#
set -euo pipefail

# Get the repository root directory (Tests/helia-core-tester).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# cortex-m0: pure C fallback (no DSP, no MVE)
# cortex-m4: DSP path (no MVE)
# cortex-m55: MVE path
CPU="cortex-m0,cortex-m4,cortex-m55"
COVERAGE=true
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        --help)
            echo "Usage: $0 [--cpu CPU_LIST] [--no-coverage] [-- <extra helia_core_tester args>]"
            echo ""
            echo "Options:"
            echo "  --cpu CPU_LIST   Target CPU(s), comma-separated (default: cortex-m0,cortex-m4,cortex-m55)"
            echo "  --no-coverage    Skip coverage instrumentation and the coverage-merge report"
            echo "  --               Pass remaining args through to 'helia_core_tester full'"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "================================================================================"
echo "Running S4 Convolve weight-sum test matrix (both routes: with and without"
echo "the precomputed weight-sum buffer) on cpu(s): ${CPU}"
[[ "${COVERAGE}" == true ]] && echo "Coverage instrumentation: enabled"
echo "================================================================================"

FULL_ARGS=(
    --op Convolve
    --dtype S4
    --cpu "${CPU}"
    --suite int
    --run-jobs 0
    -v 3
)
if [[ "${COVERAGE}" == true ]]; then
    FULL_ARGS+=(--coverage)
fi

# --op/--dtype narrow generation+build+run to the S4 Convolve descriptors, which
# already include both the default (no weight-sum) and "_kernel_sum" (with
# weight-sum) variants via expand_kernel_sum_variant() in
# helia_core_tester/generation/io/descriptors.py.
uv run helia_core_tester full "${FULL_ARGS[@]}" "${EXTRA_ARGS[@]}"

if [[ "${COVERAGE}" == true ]]; then
    echo "================================================================================"
    echo "Merging coverage across cpu(s): ${CPU}"
    echo "================================================================================"
    uv run helia_core_tester coverage-merge --cpu "${CPU}" --suite int
fi


uv run helia_core_tester coverage-merge --cpu "${CPU}"  --suite int