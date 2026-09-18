"""The pinned dependency set a hardware build is qualified against.

`assets/dependency_baseline.json` names one immutable commit per project the
firmware is built from. Those pins are what the rendered `nsx.yml`'s
`module_registry` block asserts, so `nsx lock` resolves exactly them instead of
whatever the packaged NSX registry's mutable tags point at today.

The file shape is deliberately a subset of hpx's `hpx.compatibility-baseline`
(`helia_profiler/data/compatibility-baseline-v1.json`), so `--baseline <that file>`
builds the tester against hpx's qualified pins with no code dependency in either
direction: only `baseline_id` and `projects.{url,ref}` are read, and hpx's extra
sections (its own modules, engines, the neuralspotx package identity) are carried
through untouched in the recorded document.

Refs are full 40-character commit SHAs and nothing else. A tag or a branch is a
mutable name, so a baseline built on one would make "this build is qualified" depend
on remote repository state (hpx `compatibility.py::_immutable_ref` says the same).

The `fingerprint` is the sha256 of the canonical JSON of the document as loaded --
the same hash hpx computes over its own baseline -- so the same file identifies the
same baseline in both tools, and a bundle's `baseline_fingerprint` is directly
comparable with an hpx run's.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

BASELINE_SCHEMA = "hct.dependency-baseline"
BASELINE_SCHEMA_VERSION = 1
HPX_BASELINE_SCHEMA = "hpx.compatibility-baseline"
BASELINE_ASSET = "dependency_baseline.json"

# The NSX projects the rendered app's module list resolves to. ns-cmsis-nn is the
# kernels under test; the other three own every module the apollo510_evb starter
# profile declares (nsx-ambiq-sdk the SoC/BSP/HAL stack, neuralspotx the board and
# tooling modules, nsx-pmu-armv8m the PMU). Each gets a `revision` pin in the
# rendered nsx.yml's module_registry block, so a baseline that names none of them
# would leave NSX free to resolve the packaged registry's own (mutable) tags.
CMSIS_NN_PROJECT = "ns-cmsis-nn"
SDK_PROJECT = "nsx-ambiq-sdk"
NEURALSPOTX_PROJECT = "neuralspotx"
PMU_PROJECT = "nsx-pmu-armv8m"
CMSIS5_PROJECT = "CMSIS_5"
REQUIRED_PROJECTS: Tuple[str, ...] = (
    CMSIS_NN_PROJECT,
    SDK_PROJECT,
    NEURALSPOTX_PROJECT,
    PMU_PROJECT,
)
# Projects that are plain checkouts under artifacts/downloads/ rather than NSX
# modules: `nsx lock` never sees them, so their pin has to be applied by the
# tester itself (firmware_build.pin_optional_checkouts) and is not enforced by
# the lock. Mapped to the directory name each lives under.
#
# CMSIS_5 is the only one. It is not an NSX project and is not part of an hpx
# baseline, so a baseline that omits it leaves that checkout unpinned rather than
# failing to load. The firmware takes only `pmu_armv8.h` from it; the FVP path
# takes the device startup/system sources.
NON_NSX_CHECKOUTS: Dict[str, str] = {CMSIS5_PROJECT: "CMSIS_5"}
OPTIONAL_PROJECTS: Tuple[str, ...] = tuple(NON_NSX_CHECKOUTS)

_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


class BaselineError(RuntimeError):
    """A dependency baseline file is missing, malformed, or not immutably pinned."""


@dataclass(frozen=True)
class BaselineProject:
    name: str
    url: str
    ref: str


@dataclass(frozen=True)
class DependencyBaseline:
    """One loaded baseline: where it came from, what it pins, and its identity."""

    path: Optional[Path]
    baseline_id: str
    schema: str
    projects: Dict[str, BaselineProject]
    document: Dict[str, Any]
    """The file as parsed, kept verbatim so the fingerprint and the recorded block
    describe the whole baseline and not just the part this repo reads."""

    def project(self, name: str) -> BaselineProject:
        try:
            return self.projects[name]
        except KeyError:
            raise BaselineError(
                f"Dependency baseline {self.describe()} does not pin project '{name}'."
            ) from None

    def pin(self, name: str) -> Optional[str]:
        """The pinned commit for `name`, or None when this baseline leaves it unpinned."""
        entry = self.projects.get(name)
        return entry.ref if entry else None

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def describe(self) -> str:
        return f"{self.baseline_id} ({self.path})" if self.path else self.baseline_id

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.document)


def default_baseline_path(repo_root: Path) -> Path:
    return repo_root / "assets" / BASELINE_ASSET


def load_baseline(path: Path) -> DependencyBaseline:
    """Load `path` as an `hct.dependency-baseline` or an `hpx.compatibility-baseline`."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BaselineError(f"Cannot read dependency baseline {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineError(f"Dependency baseline {path} is not valid JSON: {exc}") from exc
    return parse_baseline(raw, path=path)


def parse_baseline(raw: Any, *, path: Optional[Path] = None) -> DependencyBaseline:
    where = str(path) if path is not None else "<in-memory baseline>"
    if not isinstance(raw, dict):
        raise BaselineError(f"Dependency baseline {where} must contain a JSON object.")
    schema = raw.get("schema")
    if schema not in (BASELINE_SCHEMA, HPX_BASELINE_SCHEMA):
        raise BaselineError(
            f"Dependency baseline {where} has unsupported schema {schema!r}; expected "
            f"{BASELINE_SCHEMA!r} or {HPX_BASELINE_SCHEMA!r}."
        )
    if raw.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise BaselineError(
            f"Dependency baseline {where} has unsupported schema_version "
            f"{raw.get('schema_version')!r}; this tester reads v{BASELINE_SCHEMA_VERSION}."
        )
    baseline_id = raw.get("baseline_id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        raise BaselineError(f"Dependency baseline {where} needs a non-empty baseline_id.")
    projects_raw = raw.get("projects")
    if not isinstance(projects_raw, Mapping) or not projects_raw:
        raise BaselineError(f"Dependency baseline {where} needs a non-empty 'projects' object.")

    projects: Dict[str, BaselineProject] = {}
    for name, entry in projects_raw.items():
        if not isinstance(entry, Mapping):
            raise BaselineError(f"Dependency baseline {where} project '{name}' must be an object.")
        url = entry.get("url")
        ref = entry.get("ref")
        if not isinstance(url, str) or not url.strip():
            raise BaselineError(f"Dependency baseline {where} project '{name}' needs a url.")
        projects[str(name)] = BaselineProject(str(name), url, immutable_ref(ref, f"project '{name}'", where))

    missing = [name for name in REQUIRED_PROJECTS if name not in projects]
    if missing:
        raise BaselineError(
            f"Dependency baseline {where} is missing required project(s): {', '.join(missing)}."
        )
    return DependencyBaseline(
        path=path, baseline_id=baseline_id, schema=str(schema), projects=projects, document=raw
    )


def immutable_ref(ref: Any, owner: str, where: str) -> str:
    """A baseline ref is a full 40-hex commit SHA; a tag or branch is not qualifiable.

    `fullmatch` rather than `match` with `$`: `$` also matches before a trailing
    newline, which would let `"<sha>\\n"` through (hpx makes the same point).
    """
    if not isinstance(ref, str) or not _COMMIT_SHA_RE.fullmatch(ref):
        raise BaselineError(
            f"Dependency baseline {where} {owner} must use a full 40-character commit SHA, not {ref!r}."
        )
    return ref


def is_commit_sha(value: str) -> bool:
    return bool(_COMMIT_SHA_RE.fullmatch(value))


def resolve_baseline(repo_root: Path, override: Optional[Path] = None) -> DependencyBaseline:
    """The baseline in force: `--baseline FILE` if given, else the repo's own asset."""
    path = Path(override).expanduser() if override is not None else default_baseline_path(repo_root)
    return load_baseline(path)
