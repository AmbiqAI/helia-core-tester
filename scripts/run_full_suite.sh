#!/bin/bash
# Full Test Suite Runner for Helia-Core-Tester
#
# Runs the complete generate -> build -> run pipeline in a single command for
# all target CPUs (cortex-m0, cortex-m4, cortex-m55) and both the int and
# float suites (both f16/f32 precisions), with coverage instrumentation
# enabled, using all available CPU cores for build and FVP execution. Once
# the run completes, merges the per-CPU coverage.info files into a single
# LCOV report with JSON/MD/HTML summaries.
#
# Note: --coverage-mve-float (Cortex-M55 float MVE coverage paths) is
# intentionally NOT enabled here, because the CLI only allows it when
# --cpu is restricted to cortex-m55 alone (ConfigurationError otherwise).
# Since this script targets all CPUs in one command, MVE float coverage is
# out of scope; run a separate cortex-m55-only pass with --coverage-mve-float
# if you need that data.
#
# Intended to be run inside the Linux dev container (the build/FVP pipeline
# is not supported on macOS).

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (override via flags below)
CPUS="cortex-m0,cortex-m4,cortex-m55"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
RUN_JOBS=0 # 0 = auto/use all host cores
SKIP_GENERATION=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cpu CPUS             Comma-separated target CPUs (default: ${CPUS})"
    echo "  --jobs N               Parallel build jobs (default: nproc = ${JOBS})"
    echo "  --run-jobs N           Parallel FVP run jobs, 0 = auto (default: ${RUN_JOBS})"
    echo "  --skip-generation      Reuse already-generated TFLite/C test artifacts"
    echo "  --help                 Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            CPUS="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --run-jobs)
            RUN_JOBS="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            usage
            ;;
    esac
done

cd "${REPO_ROOT}"

echo -e "${GREEN}=== Helia-Core-Tester Full Suite Run ===${NC}"
echo "Repository root: ${REPO_ROOT}"
echo "CPUs:             ${CPUS}"
echo "Build jobs:       ${JOBS}"
echo "Run jobs:         ${RUN_JOBS}"
echo ""

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if ! command_exists uv; then
    echo -e "${RED}uv is required but not found. Run scripts/setup_ci.sh first.${NC}" >&2
    exit 1
fi

FULL_ARGS=(
    full
    --cpu "${CPUS}"
    --suite both
    --float-precision both
    --coverage
    --run-jobs "${RUN_JOBS}"
    # --no-fail-fast
    -v 3
)

if [ "${SKIP_GENERATION}" = true ]; then
    FULL_ARGS+=(--skip-generation)
fi

echo -e "${GREEN}=== Running full pipeline (generate -> build -> run) ===${NC}"
uv run helia_core_tester "${FULL_ARGS[@]}"

echo ""
echo -e "${GREEN}=== Merging coverage reports (all CPUs, both suites) ===${NC}"
uv run helia_core_tester coverage-merge \
    --cpu "${CPUS}" \
    --suite both

echo ""
echo -e "${GREEN}=== Full suite + coverage merge complete ===${NC}"