#!/bin/bash
# Python Unit Test Runner for Helia-Core-Tester
#
# Runs the helia_core_tester/tests/ pytest suite (unit tests for the
# generation/build/report pipeline itself), as opposed to run_full_suite.sh
# which generates and executes the CMSIS-NN C tests on the FVP.
#
# Ensures a uv-managed virtual environment with dev dependencies is present,
# then runs pytest. Any extra arguments passed to this script are forwarded
# to pytest (e.g. -k <expr>, -x, a specific test path, etc).

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: $0 [PYTEST_ARGS...]"
    echo ""
    echo "Runs the helia_core_tester/tests/ pytest suite via uv."
    echo "Any arguments are forwarded to pytest, e.g.:"
    echo "  $0 -k squared_difference"
    echo "  $0 helia_core_tester/tests/test_descriptor_naming_contract.py -x"
    exit 0
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
fi

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}Error: 'uv' is not installed or not on PATH.${NC}"
    echo "Install it from https://docs.astral.sh/uv/ and re-run this script."
    exit 1
fi

if [[ ! -x ".venv/bin/python" ]] && [[ ! -x ".venv/bin/python.exe" ]]; then
    echo "No valid .venv found, creating one with dev dependencies..."
    uv venv --clear
fi

uv pip install -q -e ".[dev]"

echo -e "${GREEN}Running pytest suite...${NC}"
uv run pytest helia_core_tester/tests/ "$@"
