#!/usr/bin/env bash
# Stage one leg: <upload dir> <session id> <run json>.
set -euo pipefail
out="$1" session="$2" run_json="$3"

if [[ -d "artifacts/reports/hardware/${session}" ]]; then
  mkdir -p "${out}/hardware"
  cp -r "artifacts/reports/hardware/${session}" "${out}/hardware/"
fi
# Keep the marker until results exist.
if [[ -s "${run_json}" ]]; then
  cp "${run_json}" "${out}/"
  rm -f "${out}/NO_RESULT"
fi
