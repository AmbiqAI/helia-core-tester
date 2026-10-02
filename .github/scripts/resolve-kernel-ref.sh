#!/usr/bin/env bash
# Resolve HCT_NIGHTLY_REF to one ns-cmsis-nn commit.
set -euo pipefail

ref="${HCT_NIGHTLY_REF//[[:space:]]/}"
if [[ "${ref}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  sha="${ref,,}"
else
  refs="$(git ls-remote https://github.com/AmbiqAI/ns-cmsis-nn.git \
    "refs/heads/${ref}" "refs/tags/${ref}" "refs/tags/${ref}^{}")"
  # Annotated tags: take the peeled commit.
  sha="$(awk -v t="refs/tags/${ref}^{}" '$2 == t { print $1 }' <<< "${refs}")"
  sha="${sha:-$(awk 'NR == 1 { print $1 }' <<< "${refs}")}"
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
echo "::notice::second leg ${leg}: ns-cmsis-nn ${ref} at ${sha}"
{
  echo "leg=${leg}"
  echo "sha=${sha}"
} >> "${GITHUB_OUTPUT}"
