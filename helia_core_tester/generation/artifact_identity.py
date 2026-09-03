"""Stable identity for one generated test case.

The digest deliberately covers only deterministic generation outputs. Build products,
logs, and other transient files are excluded so an FVP result remains tied to the exact
inputs/goldens it executed without becoming machine-dependent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_ROOT_NAMES = {"descriptor.yaml", "CMakeLists.txt"}
_SUFFIXES = {".c", ".h", ".json", ".tflite"}


def generated_case_artifact_sha256(case_dir: Path) -> str:
    """Return a canonical SHA-256 for deterministic files under ``case_dir``."""
    entries: list[dict[str, str]] = []
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(case_dir)
        if any(part.startswith(".") or part in {"build", "CMakeFiles"} for part in relative.parts):
            continue
        if path.name not in _ROOT_NAMES and path.suffix not in _SUFFIXES:
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not entries:
        raise ValueError(f"No deterministic generated artifacts found under {case_dir}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
