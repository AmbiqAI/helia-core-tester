#!/usr/bin/env bash
# Resolve HCT_NIGHTLY_REF to one ns-cmsis-nn commit.
set -euo pipefail

# Trim the ends; inner space is invalid.
ref="${HCT_NIGHTLY_REF#"${HCT_NIGHTLY_REF%%[![:space:]]*}"}"
ref="${ref%"${ref##*[![:space:]]}"}"
if [[ -z "${ref}" || "${ref}" =~ [[:space:]] ]]; then
  sha=""
elif [[ "${ref}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  sha="${ref,,}"
elif [[ "${ref}" =~ [*?\[] ]]; then
  # ls-remote would glob-match these.
  sha=""
else
  refs="$(git ls-remote https://github.com/AmbiqAI/ns-cmsis-nn.git \
    "refs/heads/${ref}" "refs/tags/${ref}" "refs/tags/${ref}^{}")"
  # Annotated tags: take the peeled commit.
  sha="$(awk -v t="refs/tags/${ref}^{}" '$2 == t { print $1 }' <<< "${refs}")"
  sha="${sha:-$(awk -v h="refs/heads/${ref}" -v t="refs/tags/${ref}" '$2 == h || $2 == t { print $1; exit }' <<< "${refs}")}"
fi
if [[ -z "${sha}" ]]; then
  echo "::error::ns-cmsis-nn has no ref '${ref}'" >&2
  exit 2
fi
# Only main feeds the main series.
leg=ref
if [[ "${ref}" == main ]]; then
  leg=main
fi
echo "::notice::kernel leg ${leg}: ns-cmsis-nn ${ref} at ${sha}"
{
  echo "leg=${leg}"
  echo "sha=${sha}"
} >> "${GITHUB_OUTPUT}"
