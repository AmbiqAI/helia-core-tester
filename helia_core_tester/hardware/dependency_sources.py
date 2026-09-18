"""Where the ns-cmsis-nn checkout comes from.

The hardware firmware compiles ns-cmsis-nn from source (CMake's `CMSIS_NN_ROOT`)
and the generate step reads the same checkout (LSTM unit-test data, header probes,
the sigmoid table). Both must point at one checkout, resolved once, in this order:

1. `--cmsis-nn-root PATH` on the hardware command;
2. the `CMSIS_NN_ROOT` environment variable;
3. the nested layout, where this repo is the `Tests/helia-core-tester` submodule
   of an ns-cmsis-nn checkout (CMakeLists.txt's historical `../..` default).

Anything else is an error that names the flag and the variable. A resolved path
must look like an ns-cmsis-nn checkout (`Include/` and `Source/`).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

ENV_VAR = "CMSIS_NN_ROOT"
CLI_FLAG = "--cmsis-nn-root"

SELECTOR_CLI_ROOT = f"cli.{CLI_FLAG}"
SELECTOR_ENV = f"env.{ENV_VAR}"
SELECTOR_NESTED = "layout.nested"


class CmsisNnSourceError(RuntimeError):
    """No usable ns-cmsis-nn checkout could be resolved."""


@dataclass(frozen=True)
class CmsisNnSelection:
    """What the user asked for. `ref` is reserved for a pinned-commit selection and
    is rejected until the dependency baseline lands."""

    root: Optional[Path] = None
    ref: Optional[str] = None


@dataclass(frozen=True)
class ResolvedCmsisNn:
    root: Path
    selector: str
    """One of SELECTOR_CLI_ROOT / SELECTOR_ENV / SELECTOR_NESTED."""
    requested: Optional[str]
    """The value as the user gave it (before expansion), or None for the nested layout."""


def nested_layout_root(repo_root: Path) -> Path:
    """The ns-cmsis-nn root when this repo sits at `<ns-cmsis-nn>/Tests/helia-core-tester`."""
    return repo_root.resolve().parent.parent


def looks_like_checkout(path: Path) -> bool:
    return (path / "Include").is_dir() and (path / "Source").is_dir()


def validate_checkout(path: Path, *, origin: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise CmsisNnSourceError(f"ns-cmsis-nn checkout from {origin} does not exist: {resolved}")
    for sub in ("Include", "Source"):
        if not (resolved / sub).is_dir():
            raise CmsisNnSourceError(
                f"ns-cmsis-nn checkout from {origin} is missing '{sub}/': {resolved} "
                f"-- expected an ns-cmsis-nn repository with Include/ and Source/."
            )
    return resolved


def resolve_cmsis_nn(
    repo_root: Path,
    selection: Optional[CmsisNnSelection] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ResolvedCmsisNn:
    """Resolve the ns-cmsis-nn checkout: flag > env > nested layout, else raise."""
    selection = selection or CmsisNnSelection()
    env = os.environ if env is None else env
    if selection.ref is not None:
        raise CmsisNnSourceError("--cmsis-nn-ref is not supported yet; pass --cmsis-nn-root PATH or set CMSIS_NN_ROOT.")
    if selection.root is not None:
        root = validate_checkout(selection.root, origin=CLI_FLAG)
        return ResolvedCmsisNn(root, SELECTOR_CLI_ROOT, str(selection.root))
    env_value = env.get(ENV_VAR)
    if env_value:
        root = validate_checkout(Path(env_value), origin=f"${ENV_VAR}")
        return ResolvedCmsisNn(root, SELECTOR_ENV, env_value)
    nested = nested_layout_root(repo_root)
    if looks_like_checkout(nested):
        return ResolvedCmsisNn(nested, SELECTOR_NESTED, None)
    raise CmsisNnSourceError(
        f"No ns-cmsis-nn checkout found: pass {CLI_FLAG} PATH or set {ENV_VAR} to an ns-cmsis-nn "
        f"checkout (a directory with Include/ and Source/). The nested "
        f"<ns-cmsis-nn>/Tests/helia-core-tester layout was not detected at {nested}."
    )


def describe(resolved: ResolvedCmsisNn) -> str:
    return f"{resolved.root} ({resolved.selector})"


# --- provenance -------------------------------------------------------------------
#
# What a build dir was configured against, written by firmware_build.configure() as
# <build_dir>/hct_dependencies.json and copied into every result bundle's
# session_manifest.json / session_summary.json. The `modules` entries follow the
# shape hpx writes into its summary.json (`dependencies.modules[]`: name, project,
# kind, requested_ref, requested_tag, peeled_commit, content_hash, url, vendored_at),
# so the hpx dashboard's existing per-project lookup reads them unchanged.

DEPENDENCIES_SCHEMA = "hct.hardware.dependencies"
DEPENDENCIES_SCHEMA_VERSION = 1
DEPENDENCIES_FILENAME = "hct_dependencies.json"

CMSIS_NN_PROJECT = "ns-cmsis-nn"
CMSIS_NN_MODULE = "nsx-cmsis-nn"  # the NSX registry module name, as hpx records it

PROJECT_URLS = {
    CMSIS_NN_PROJECT: "https://github.com/AmbiqAI/ns-cmsis-nn.git",
    "nsx-ambiq-sdk": "https://github.com/AmbiqAI/nsx-ambiq-sdk.git",
    "neuralspotx": "https://github.com/AmbiqAI/neuralspotx.git",
    "CMSIS_5": "https://github.com/ARM-software/CMSIS_5.git",
}


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def git_tree_identity(root: Path) -> Dict[str, str]:
    """`{state, commit}` for a managed checkout (the SDK, neuralspotx, CMSIS_5): the
    commit when `root` is itself the top of a git working tree, `git-dirty` when it
    has local changes, `content` for a non-git tree, `absent` when missing. No content
    digest: unlike ns-cmsis-nn these are not generation inputs."""
    if not root.is_dir():
        return {"state": "absent"}
    toplevel = _git(root, "rev-parse", "--show-toplevel")
    try:
        is_toplevel = bool(toplevel and toplevel.strip()) and Path(toplevel.strip()).resolve() == root.resolve()
    except OSError:
        is_toplevel = False
    head = _git(root, "rev-parse", "HEAD") if is_toplevel else None
    if head is None:
        return {"state": "content"}
    status = _git(root, "status", "--porcelain")
    if status is None or status.strip():
        return {"state": "git-dirty", "commit": head.strip()}
    return {"state": "git-clean", "commit": head.strip()}


def _vendored_at(root: Path, repo_root: Path) -> str:
    try:
        return root.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return root.resolve().as_posix()


def module_record(
    name: str,
    project: str,
    root: Path,
    *,
    repo_root: Path,
    identity: Dict[str, str],
    requested_ref: Optional[str] = None,
    requested_tag: Optional[str] = None,
) -> Dict[str, object]:
    """One `dependencies.modules[]` entry. `kind` is `git` when the tree is a git
    checkout (clean or dirty), `local` for a plain tree, `absent` when missing."""
    state = identity.get("state", "absent")
    kind = "git" if state in ("git-clean", "git-dirty") else ("local" if state == "content" else "absent")
    content = identity.get("content")
    return {
        "name": name,
        "project": project,
        "kind": kind,
        "requested_ref": requested_ref,
        "requested_tag": requested_tag,
        "peeled_commit": identity.get("commit"),
        "content_hash": {"algorithm": "sha256", "value": content} if content else None,
        "url": PROJECT_URLS.get(project),
        "vendored_at": _vendored_at(root, repo_root) if state != "absent" else None,
        "state": state,
    }


def cmsis_nn_override(resolved: ResolvedCmsisNn) -> Optional[Dict[str, object]]:
    """The `overrides[]` entry for an explicit kernel selection (flag or environment);
    the nested layout is the repo's own default, not an override."""
    if resolved.selector == SELECTOR_NESTED:
        return None
    return {
        "scope": "module",
        "name": CMSIS_NN_MODULE,
        "mode": "path",
        "requested": resolved.requested,
        "selector": resolved.selector,
    }


def build_dependencies_document(
    repo_root: Path,
    resolved: ResolvedCmsisNn,
    *,
    cmake_defines: Dict[str, str],
    kernel_target: str,
    build_profile: str,
    kernel_compile_flags: Optional[Dict[str, str]],
    toolchain: Dict[str, Optional[str]],
) -> Dict[str, object]:
    """Everything a result bundle needs to say which kernels, SDK and toolchain the
    firmware was built from. `kernel_compile_flags` (the kernel target's C_FLAGS /
    C_DEFINES as CMake wrote them) is what later proves the build matches shipping
    firmware; `build_profile` names the flag contract in force."""
    from ..generation.reuse import cmsis_nn_checkout_identity_for
    from ..scripts.setup_dependencies import nsx_ambiq_sdk_dir
    from .toolchain import DOWNLOADS_DIR

    downloads = repo_root / DOWNLOADS_DIR
    sdk_dir = nsx_ambiq_sdk_dir(repo_root, downloads)
    modules = [
        module_record(
            CMSIS_NN_MODULE, CMSIS_NN_PROJECT, resolved.root, repo_root=repo_root,
            identity=cmsis_nn_checkout_identity_for(resolved.root),
        ),
        module_record("nsx-ambiq-sdk", "nsx-ambiq-sdk", sdk_dir, repo_root=repo_root, identity=git_tree_identity(sdk_dir)),
        module_record(
            "neuralspotx", "neuralspotx", downloads / "neuralspotx", repo_root=repo_root,
            identity=git_tree_identity(downloads / "neuralspotx"),
        ),
        module_record(
            "CMSIS_5", "CMSIS_5", downloads / "CMSIS_5", repo_root=repo_root, identity=git_tree_identity(downloads / "CMSIS_5")
        ),
    ]
    override = cmsis_nn_override(resolved)
    return {
        "schema": DEPENDENCIES_SCHEMA,
        "schema_version": DEPENDENCIES_SCHEMA_VERSION,
        "modules": modules,
        "overrides": [override] if override else [],
        "build": {
            "kernel_target": kernel_target,
            "build_profile": build_profile,
            "cmake_defines": dict(cmake_defines),
            "kernel_compile_flags": kernel_compile_flags,
        },
        "toolchain": dict(toolchain),
    }


def dependencies_path(build_dir: Path) -> Path:
    return build_dir / DEPENDENCIES_FILENAME


def write_build_dependencies(build_dir: Path, document: Dict[str, object]) -> Path:
    from .pathutil import write_text_lf

    path = dependencies_path(build_dir)
    write_text_lf(path, json.dumps(document, indent=2, sort_keys=False))
    return path


def read_build_dependencies(build_dir: Path) -> Optional[Dict[str, object]]:
    """The document `configure()` wrote for this build dir, or None when the build dir
    predates it (the bundle then carries no provenance rather than a guess)."""
    path = dependencies_path(build_dir)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != DEPENDENCIES_SCHEMA:
        raise CmsisNnSourceError(f"{path} is not a {DEPENDENCIES_SCHEMA} document; rebuild with `hardware build`.")
    return document


def summarize_kernels(document: Optional[Dict[str, object]]) -> str:
    """One line for logs: the ns-cmsis-nn commit (or content digest) the firmware was built from."""
    if not document:
        return "unknown"
    for module in document.get("modules", []):  # type: ignore[union-attr]
        if isinstance(module, dict) and module.get("project") == CMSIS_NN_PROJECT:
            commit = module.get("peeled_commit")
            state = module.get("state")
            if commit:
                return f"{commit} ({state})"
            content = module.get("content_hash") or {}
            return f"content:{(content.get('value') or 'unknown')[:16]} ({state})"
    return "unknown"
