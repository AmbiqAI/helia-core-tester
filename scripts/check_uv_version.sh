#!/usr/bin/env bash
# Fail closed when the installed uv is older than the floor pinned in
# pyproject.toml `[tool.uv] required-version`. uv releases that predate that
# setting only warn about it and keep going, so the pin alone does not stop a
# pre-fix trust store from downloading dependencies (emSign CA advisory).
#
# Usage: check_uv_version.sh [pyproject.toml]
set -euo pipefail

PYPROJECT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pyproject.toml}"

if [[ ! -f "${PYPROJECT}" ]]; then
    echo "check_uv_version: pyproject not found: ${PYPROJECT}" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "check_uv_version: uv is not installed or not on PATH." >&2
    exit 1
fi

# Only the ">=X.Y.Z" form is supported; anything else is a config error.
min=$(sed -n 's/^required-version *= *">=\([0-9][0-9]*\(\.[0-9][0-9]*\)*\)".*/\1/p' "${PYPROJECT}" | head -n1)
if [[ -z "${min}" ]]; then
    echo "check_uv_version: no 'required-version = \">=X.Y.Z\"' under [tool.uv] in ${PYPROJECT}" >&2
    exit 1
fi

have=$(uv --version 2>/dev/null | sed -n 's/^uv \([0-9][0-9]*\(\.[0-9][0-9]*\)*\).*/\1/p')
if [[ -z "${have}" ]]; then
    echo "check_uv_version: could not parse 'uv --version' output: $(uv --version 2>&1 || true)" >&2
    exit 1
fi

version_lt() {
    # True when $1 < $2. Pure bash so a missing/odd `sort -V` cannot turn a
    # failed comparison into a pass.
    local -a a b
    IFS=. read -r -a a <<< "$1"
    IFS=. read -r -a b <<< "$2"
    local i n=$(( ${#a[@]} > ${#b[@]} ? ${#a[@]} : ${#b[@]} ))
    for (( i = 0; i < n; i++ )); do
        local x=${a[i]:-0} y=${b[i]:-0}
        (( 10#$x < 10#$y )) && return 0
        (( 10#$x > 10#$y )) && return 1
    done
    return 1
}

if version_lt "${have}" "${min}"; then
    echo "check_uv_version: uv ${have} is older than the required >=${min}." >&2
    echo "Upgrade with 'uv self update' or your package manager and re-run." >&2
    exit 1
fi

echo "uv ${have} satisfies >=${min}"
