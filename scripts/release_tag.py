#!/usr/bin/env python3
"""Decide which version tag, if any, a merged pull request gets.

Run by .github/workflows/tag-release.yml on every closed pull request into main.
Only a merged release-please release pull request is tagged; the version is the
one release-please wrote to .release-please-manifest.json in that pull request,
as `v<version>`. Any other pull request -- a feature, fix, docs or CI merge --
produces no tag, so tags follow release decisions rather than merge order.

Usage: release_tag.py <github-event.json> <.release-please-manifest.json>
Prints `tag=v<version>` for a release pull request and `tag=` otherwise, in the
`$GITHUB_OUTPUT` format.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

RELEASE_LABEL = "autorelease: pending"
RELEASE_BRANCH_PREFIX = "release-please--branches--main"
RELEASE_BASE = "main"
MANIFEST_PACKAGE = "."
_VERSION = re.compile(r"\d+\.\d+\.\d+")


def is_release_pull_request(event: dict[str, Any]) -> bool:
    """A release-please release PR merged into main: pending label, release branch."""
    pull_request = event.get("pull_request") or {}
    if event.get("action") != "closed" or pull_request.get("merged") is not True:
        return False
    labels = {label.get("name") for label in pull_request.get("labels") or []}
    head = (pull_request.get("head") or {}).get("ref", "")
    base = (pull_request.get("base") or {}).get("ref", "")
    release_branch = head == RELEASE_BRANCH_PREFIX or head.startswith(RELEASE_BRANCH_PREFIX + "--")
    return RELEASE_LABEL in labels and release_branch and base == RELEASE_BASE


def release_tag(event: dict[str, Any], manifest: dict[str, Any]) -> Optional[str]:
    """`v<version>` from the manifest for a merged release PR, else None."""
    if not is_release_pull_request(event):
        return None
    version = manifest.get(MANIFEST_PACKAGE)
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(f"release-please manifest has no valid version for {MANIFEST_PACKAGE!r}: {version!r}")
    return f"v{version}"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    event = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    print(f"tag={release_tag(event, manifest) or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
