#!/usr/bin/env bash
# One hardware-nightly leg on HPX_BOARD.
# Usage: nightly_leg.sh SESSION BUILD_DIR OUT_JSON [hardware run args...]
set -euo pipefail
session="$1" build_dir="$2" out_json="$3"
shift 3
args=(
  hardware run
  --board "${HPX_BOARD}"
  --suite "${HCT_NIGHTLY_SUITE}"
  --session-id "${session}"
  --build-dir "${build_dir}"
  --json
  "$@"
)
if [[ -n "${HCT_NIGHTLY_LIMIT:-}" ]]; then
  args+=(--limit "${HCT_NIGHTLY_LIMIT}")
fi
# DWT-only boards refuse PMU counters.
tier="$(uv run python -c 'import sys; from helia_core_tester.hardware.boards import resolve_board; print(resolve_board(sys.argv[1]).pmu_tier)' "${HPX_BOARD}")"
if [[ "${tier}" != "dwt" ]]; then
  for counters in ${HCT_NIGHTLY_PMU}; do
    args+=(--pmu-counters "${counters}")
  done
fi
mkdir -p "$(dirname "${out_json}")"
# --json: document on stdout, log on stderr.
uv run helia_core_tester "${args[@]}" > "${out_json}"
if [[ "$(jq -r '.totals.ran' "${out_json}")" == "0" ]]; then
  echo "::error::${session} ran no cases on ${HPX_BOARD}" >&2
  exit 2
fi
