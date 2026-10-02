#!/usr/bin/env bash
# One nightly leg on this runner's board.
# Env: LEG_SESSION, LEG_JSON, LEG_BUILD_DIR, optional LEG_REF.
set -euo pipefail

args=(
  hardware run
  --board "${HPX_BOARD}"
  --suite "${HCT_NIGHTLY_SUITE}"
  --session-id "${LEG_SESSION}"
  --build-dir "${LEG_BUILD_DIR}"
  --json
)
if [[ -n "${LEG_REF:-}" ]]; then
  args+=(--cmsis-nn-ref "${LEG_REF}")
fi
if [[ -n "${HCT_NIGHTLY_LIMIT}" ]]; then
  args+=(--limit "${HCT_NIGHTLY_LIMIT}")
fi
# DWT-only boards refuse PMU counters.
tier="$(uv run python -c 'import sys; from helia_core_tester.hardware.boards import resolve_board; print(resolve_board(sys.argv[1]).pmu_tier)' "${HPX_BOARD}")"
if [[ "${tier}" != "dwt" ]]; then
  for counters in ${HCT_NIGHTLY_PMU}; do
    args+=(--pmu-counters "${counters}")
  done
fi
mkdir -p "$(dirname "${LEG_JSON}")"
# --json: document on stdout, log on stderr.
uv run helia_core_tester "${args[@]}" > "${LEG_JSON}"
if [[ "$(jq -r '.totals.ran' "${LEG_JSON}")" == "0" ]]; then
  echo "::error::selection ran no cases on ${HPX_BOARD}" >&2
  exit 2
fi
