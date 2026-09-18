"""Dependency provenance for one firmware build, derived from `nsx.lock`.

Every hardware result bundle has to answer "what exactly was this measurement
built from". The answer is taken from the artefacts the build already produced
rather than from the working tree it was launched in: NSX's own `nsx.lock` (the
receipt: peeled commit and content hash per module), the render state the app
was written with (`nsx_app.RENDER_STATE`), the dependency baseline in force, and
the `compile_commands.json` CMake left behind.

The document is written in heliaPROFILER's shape --
`helia_profiler/results/dependencies.py::DependencyProvenance.to_dict()` -- so a
tester bundle and an hpx run describe their dependencies with one vocabulary and
the hpx dashboard's per-project lookup
(`hpx_dashboard/dataset.py::_summary_dependency`, which reads
`dependencies.modules[].{project,requested_ref,peeled_commit,url}`) reads a
tester bundle unchanged. There is no code dependency in either direction; the
contract is pinned by a test that re-implements that lookup here.

Three fields are the tester's own, appended rather than substituted:

- `toolchain`: hpx records it a level up in its manifest (`provenance.toolchain`);
  the tester has one provenance document, so it carries it.
- `build_images[]`: same shape as hpx's `BuildImage`, plus the firmware's build
  id -- the string the board itself reports in TARGET_INFO, which is what ties a
  bundle to the image that produced its numbers.
- `overrides[].local_checkout`: the git HEAD and dirty flag of a
  `--cmsis-nn-root` tree. hpx does not record it (a path override there is
  already disqualifying and it hashes the content). The tester's optimise loop
  runs against a checkout being edited, so "which commit was I on, and was it
  dirty" is the question every such run asks afterwards. The content hash from
  the lock stays the authoritative identity; this is a human-legible companion.

Qualification follows hpx's rule (`deps/compatibility.py::QualificationState`):
a build is `qualified` only when every baseline-pinned project resolved in the
lock to exactly its pin and nothing was overridden by path. A path override, a
dirty override tree, or a lock that resolved a pinned project elsewhere is
`development-overrides` -- a real, runnable build whose numbers are not
qualified evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .dependency_baseline import DependencyBaseline
from .nsx_app import CMSIS_NN_MODULE, AppRender, synced_kernel_dir
from .pathutil import write_text_lf

PROVENANCE_SCHEMA = "hct.hardware.dependencies"
PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_FILENAME = "hct_provenance.json"
WORKSPACE_SCHEMA_VERSION = 1

#: hpx's `QualificationState` members this tool can produce. (Its third member,
#: `qualified-with-engine-override`, is about inference engines the tester has none of.)
QUALIFIED = "qualified"
DEVELOPMENT_OVERRIDES = "development-overrides"

#: Lock modes, matching hpx's `DependencyLockMode`.
LOCK_REUSED = "reused"
LOCK_RESOLVED = "resolved"
LOCK_UPDATED = "updated"

#: The flags that decide the instruction set, same set hpx counts
#: (`helia_profiler/firmware/image.py`). A build that changes any of them is a
#: different measurement, whatever directory it was built in.
_ARCH_FLAG = re.compile(r"(?<![\w=])-m(?:cpu|arch|fpu|float-abi)=[^\s\"']+")

_SHA256_PREFIX = "sha256:"


class ProvenanceError(RuntimeError):
    """A build's dependency provenance cannot be derived or read."""


# --- small helpers ------------------------------------------------------------------


def _digest(value: Optional[str]) -> Optional[Dict[str, str]]:
    """One `{algorithm, value}` content digest, hpx's `ContentDigest` shape."""
    if not value:
        return None
    return {"algorithm": "sha256", "value": str(value).removeprefix(_SHA256_PREFIX)}


def _text(value: Any) -> Optional[str]:
    """A lock scalar as a string, or None when it is absent."""
    return None if value is None else str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(document: Any) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def registry_digest() -> Optional[Dict[str, str]]:
    """sha256 over the canonical JSON of the packaged NSX registry.

    Computed exactly the way hpx computes its `workspace.registry_hash`, so the
    same packaged neuralspotx yields the same value in both tools. Best effort:
    a registry that cannot be loaded costs this field, not the build record.
    """
    try:
        from neuralspotx import api as nsx_api

        return _digest(_canonical_sha256(nsx_api.load_registry()))
    except Exception:
        return None


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def local_checkout_state(root: Path) -> Dict[str, Any]:
    """`{path, commit, dirty}` for a local kernel checkout, best effort.

    `commit` is None and `dirty` True for a tree git cannot describe: an
    unknowable tree must not read as a clean one.
    """
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return {"path": str(root), "commit": None, "dirty": True}
    status = _git(root, "status", "--porcelain")
    return {
        "path": str(root),
        "commit": head.strip(),
        "dirty": status is None or bool(status.strip()),
    }


# --- the lock -----------------------------------------------------------------------


def read_lock(app_dir: Path, nsx_board: str):
    """NSX's own reader for `<app>/nsx.lock`, raising this module's error type."""
    from neuralspotx.nsx_lock import read_lock as nsx_read_lock

    try:
        lock = nsx_read_lock(app_dir, nsx_board)
    except Exception as exc:  # NSX raises on an incompatible on-disk schema
        raise ProvenanceError(f"{app_dir / 'nsx.lock'} is unreadable: {exc}") from exc
    if lock is None:
        raise ProvenanceError(
            f"{app_dir / 'nsx.lock'} has no target section for board '{nsx_board}'."
        )
    return lock


def lock_modules(lock) -> List[Dict[str, Any]]:
    """`modules[]` in hpx's shape: one entry per module the lock resolved.

    Sorted by module name so two builds of the same lock produce byte-identical
    documents; `kind` is NSX's own (`git`, `packaged`, `local`, `vendored`,
    `unresolved`) rather than a re-spelling, because that word is what says how
    the copy on disk is reproduced.
    """
    modules = []
    for name, module in sorted(lock.modules.items()):
        modules.append(
            {
                "name": name,
                "project": module.project,
                "kind": str(module.kind),
                # Stringified: YAML reads an all-digit commit or constraint as an
                # integer, and a bundle whose peeled_commit is a number nothing can
                # compare against a ref is worse than a slow-burning bug.
                "requested_ref": _text(module.constraint),
                "requested_tag": _text(module.tag),
                "peeled_commit": _text(module.commit),
                "content_hash": _digest(module.content_hash),
                "url": module.url,
                "vendored_at": module.vendored_at,
            }
        )
    return modules


def baseline_mismatches(
    modules: Sequence[Mapping[str, Any]],
    baseline: DependencyBaseline,
    *,
    skip_projects: Sequence[str] = (),
) -> List[str]:
    """Baseline-pinned projects the lock resolved somewhere other than their pin.

    The rendered `nsx.yml` *asserts* the pins; the lock is the *outcome*, and NSX
    gives a packaged registry's module-level revision precedence over an app's
    project-level override -- the exact mechanism that let eight hpx hardware
    runs build an unpinned module while every artifact claimed the baseline.
    So the claim "qualified" is only made after the two are compared.
    """
    skipped = set(skip_projects)
    mismatches = []
    for module in modules:
        project = str(module.get("project") or "")
        pin = baseline.pin(project)
        commit = module.get("peeled_commit")
        if pin is None or project in skipped or module.get("kind") != "git" or not commit:
            continue
        if str(commit) != pin:
            mismatches.append(
                f"{module.get('name')} ({project}) resolved to {commit}, baseline pins {pin}"
            )
    return sorted(mismatches)


# --- build images -------------------------------------------------------------------


def _command_text(entry: Mapping[str, Any]) -> str:
    command = entry.get("command")
    if isinstance(command, str):
        return command
    arguments = entry.get("arguments")
    if isinstance(arguments, list):
        return " ".join(str(argument) for argument in arguments)
    return ""


def architecture_flags(
    compile_commands: Path, *, kernel_dir: Path, server_dir: Path
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """ISA flags the compiler was actually given, counted per translation unit.

    Counts rather than a set, for hpx's reason: one flag over every unit is a
    uniform build, while two spellings of `-mcpu` state a genuinely mixed one
    instead of letting whichever appeared first speak for the image.

    Scope is the kernel module's units plus the benchmark server's, not the
    whole tree: those are the two sources of the code that produces the numbers,
    and a `-mcpu` difference confined to, say, a UART driver says nothing about
    a kernel measurement while adding noise to the one field that must stay
    comparable. The two counts are reported separately for the same reason.
    """
    try:
        entries = json.loads(compile_commands.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, {"kernel": 0, "server": 0}
    if not isinstance(entries, list):
        return {}, {"kernel": 0, "server": 0}

    counts: Counter[str] = Counter()
    units = {"kernel": 0, "server": 0}
    kernel = str(kernel_dir.resolve())
    server = str(server_dir.resolve())
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = str(entry.get("file") or "")
        if source.startswith(kernel):
            units["kernel"] += 1
        elif source.startswith(server):
            units["server"] += 1
        else:
            continue
        # Deduplicate within a unit so a flag repeated on one command line
        # cannot outvote a flag that genuinely differs between units.
        counts.update(set(_ARCH_FLAG.findall(_command_text(entry))))
    return dict(sorted(counts.items())), units


def build_image_record(
    *,
    role: str,
    target_name: str,
    binary: Path,
    build_id: Optional[str],
    compile_commands: Path,
    kernel_dir: Path,
    server_dir: Path,
) -> Optional[Dict[str, Any]]:
    """One `build_images[]` entry: what was linked, and what it was compiled with.

    Returns None only when the binary itself cannot be read -- an image with no
    digest states nothing. A missing compile database costs the flags, not the
    record.
    """
    try:
        digest = _sha256_file(binary)
        size_bytes = binary.stat().st_size
    except OSError:
        return None
    flags, units = architecture_flags(compile_commands, kernel_dir=kernel_dir, server_dir=server_dir)
    return {
        "role": role,
        "target_name": target_name,
        "binary_name": binary.name,
        "build_id": build_id,
        "sha256": digest,
        "size_bytes": size_bytes,
        "architecture_flags": flags,
        "translation_units": units["kernel"] + units["server"],
        "kernel_translation_units": units["kernel"],
        "server_translation_units": units["server"],
    }


# --- toolchain ----------------------------------------------------------------------


def _first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line or None


def _tool_version(argv: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(list(argv), capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _first_line(completed.stdout)


def toolchain_record(repo_root: Path) -> Dict[str, Optional[str]]:
    """The compiler, CMake and neuralspotx versions this build used.

    The compiler is resolved the way the build resolves it (the checkout's
    downloaded ARM GCC first, then PATH), so the recorded banner is the one that
    compiled the image and not whatever a login shell would find.
    """
    from .toolchain import arm_tool

    compiler = arm_tool("arm-none-eabi-gcc", repo_root)
    try:
        from importlib.metadata import version as package_version

        neuralspotx_version = package_version("neuralspotx")
    except Exception:
        neuralspotx_version = None
    return {
        "compiler": "arm-none-eabi-gcc",
        "compiler_version": _tool_version([compiler, "--version"]),
        "cmake_version": _tool_version(["cmake", "--version"]),
        "neuralspotx_version": neuralspotx_version,
    }


# --- the document -------------------------------------------------------------------


def workspace_inputs(render: AppRender) -> Dict[str, Any]:
    """The render inputs the fingerprint covers, spelled out for a reader.

    The digest alone answers "same build?"; these answer "different how?" without
    needing the app tree the digest was computed over, which a bundle read months
    later on another host does not have.
    """
    return {
        "board": render.board.id,
        "nsx_board": render.board.nsx_board,
        "cpu": render.board.cpu,
        "toolchain": "arm-none-eabi-gcc",
        "kernel_source": render.kernel_source.describe(),
        "kernel_options": render.kernel_options.cache_vars(),
        "modules": [spec.name for spec in render.modules],
    }


def build_provenance(
    render: AppRender,
    *,
    repo_root: Path,
    build_dir: Path,
    lock_mode: str,
    update_requested: bool = False,
    binary: Optional[Path] = None,
    build_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The provenance document for the firmware just built from `render`."""
    from .firmware_build import output_dir

    app_dir = render.app_dir
    lock_path = app_dir / "nsx.lock"
    if not lock_path.is_file():
        raise ProvenanceError(
            f"Cannot record dependency provenance without an exact NSX lock: {lock_path} is missing."
        )
    lock = read_lock(app_dir, render.board.nsx_board)
    modules = lock_modules(lock)

    baseline = render.baseline
    overrides: List[Dict[str, Any]] = []
    kernel_path = render.kernel_source.path
    if kernel_path is not None:
        locked = next((m for m in modules if m["name"] == CMSIS_NN_MODULE), None)
        override: Dict[str, Any] = {
            "scope": "module",
            "name": CMSIS_NN_MODULE,
            "mode": "path",
            "requested": str(kernel_path),
            "content_hash": locked["content_hash"] if locked else None,
            "local_checkout": local_checkout_state(kernel_path),
        }
        overrides.append(override)
    if baseline.path is not None and not _is_default_baseline(baseline.path, repo_root):
        overrides.append(
            {
                "scope": "baseline",
                "name": baseline.baseline_id,
                "mode": "file",
                "requested": str(baseline.path),
                "content_hash": _digest(baseline.fingerprint),
            }
        )

    # A path-overridden module has no baseline pin to be measured against -- the
    # override is the intent -- so it is excluded from the comparison and
    # disqualifies the build on its own below.
    skip = [CMSIS_NN_MODULE] if kernel_path is not None else []
    skip_projects = [m["project"] for m in modules if m["name"] in skip]
    mismatches = baseline_mismatches(modules, baseline, skip_projects=skip_projects)

    dirty_override = any(
        entry.get("local_checkout", {}).get("dirty") for entry in overrides if "local_checkout" in entry
    )
    disqualifiers: List[str] = list(mismatches)
    if kernel_path is not None:
        disqualifiers.append(f"{CMSIS_NN_MODULE} is built from the local path {kernel_path}")
    if dirty_override:
        disqualifiers.append("the local kernel checkout has uncommitted changes")
    qualification = QUALIFIED if not disqualifiers else DEVELOPMENT_OVERRIDES

    out_dir = output_dir(build_dir, render.board)
    images = []
    if binary is not None:
        image = build_image_record(
            role="benchmark-server",
            target_name=binary.stem,
            binary=binary,
            build_id=build_id,
            compile_commands=out_dir / "compile_commands.json",
            kernel_dir=synced_kernel_dir(app_dir),
            server_dir=repo_root / "cmake" / "hardware",
        )
        if image is not None:
            images.append(image)

    return {
        "schema": PROVENANCE_SCHEMA,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "workspace": {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "fingerprint": render.digest,
            "baseline_id": baseline.baseline_id,
            "baseline_fingerprint": baseline.fingerprint,
            "registry_hash": registry_digest(),
            "inputs": workspace_inputs(render),
        },
        "lock": {
            "mode": lock_mode,
            "update_requested": bool(update_requested),
            "offline": False,
            "frozen_sync": True,
            "schema_version": lock.schema_version,
            "sha256": _digest(_sha256_file(lock_path)),
            "manifest_hash": _digest(lock.manifest_hash),
        },
        "modules": modules,
        "overrides": overrides,
        "qualification": qualification,
        "unqualified_reasons": disqualifiers,
        "toolchain": toolchain_record(repo_root),
        "build_images": images,
    }


def _is_default_baseline(path: Path, repo_root: Path) -> bool:
    """Whether `path` is the repo's own baseline asset (i.e. `--baseline` was not used)."""
    from .dependency_baseline import default_baseline_path

    try:
        return Path(path).resolve() == default_baseline_path(repo_root).resolve()
    except OSError:
        return False


# --- on-disk record -----------------------------------------------------------------


def provenance_path(build_dir: Path, board) -> Path:
    """Where a build's provenance document lives: beside the image it describes.

    Next to the linked ELF, the build id and the `nsx.lock` snapshot rather than
    up in the app tree, because those four are one set: the app's own `nsx.lock`
    moves on the next time the app is re-locked, and a document that outlived the
    image it describes would be worse than none.
    """
    from .firmware_build import output_dir

    return output_dir(build_dir, board) / PROVENANCE_FILENAME


def write_provenance(build_dir: Path, board, document: Mapping[str, Any]) -> Path:
    path = provenance_path(build_dir, board)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(path, json.dumps(document, indent=2) + "\n")
    return path


def read_provenance(build_dir: Path, board) -> Optional[Dict[str, Any]]:
    """The document `hardware build` wrote for this build dir, or None when absent."""
    path = provenance_path(build_dir, board)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != PROVENANCE_SCHEMA:
        raise ProvenanceError(
            f"{path} is not a {PROVENANCE_SCHEMA} document; rebuild with `hardware build`."
        )
    return document


# --- reading the block back ---------------------------------------------------------


def recorded_build_id(document: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The build id of the image the document describes."""
    for image in (document or {}).get("build_images") or []:
        if isinstance(image, Mapping) and image.get("build_id"):
            return str(image["build_id"])
    return None


def module_for_project(document: Optional[Mapping[str, Any]], project: str) -> Optional[Mapping[str, Any]]:
    for module in (document or {}).get("modules") or []:
        if isinstance(module, Mapping) and module.get("project") == project:
            return module
    return None


def summarize_kernels(document: Optional[Mapping[str, Any]]) -> str:
    """One log line: which kernels the firmware was built from, and whether it is qualified."""
    from .dependency_baseline import CMSIS_NN_PROJECT

    if not document:
        return "unknown"
    module = module_for_project(document, CMSIS_NN_PROJECT)
    qualification = str(document.get("qualification") or "unknown")
    if module is None:
        return f"unknown ({qualification})"
    commit = module.get("peeled_commit")
    if commit:
        return f"{CMSIS_NN_PROJECT}@{commit} ({module.get('kind')}, {qualification})"
    content = (module.get("content_hash") or {}).get("value") or "unknown"
    return f"{CMSIS_NN_PROJECT}@content:{content[:16]} ({module.get('kind')}, {qualification})"


def describe_build(build_dir: Path, board) -> str:
    """One line for `doctor`: the qualification of the build in `build_dir`."""
    try:
        document = read_provenance(build_dir, board)
    except ProvenanceError as exc:
        return f"{board.id}: unreadable ({exc})"
    if document is None:
        return f"{board.id}: not built yet (no {PROVENANCE_FILENAME})"
    reasons = document.get("unqualified_reasons") or []
    detail = f" -- {'; '.join(str(reason) for reason in reasons)}" if reasons else ""
    return (
        f"{board.id}: {document.get('qualification')} against "
        f"{document.get('workspace', {}).get('baseline_id')}, "
        f"{summarize_kernels(document)}{detail}"
    )
