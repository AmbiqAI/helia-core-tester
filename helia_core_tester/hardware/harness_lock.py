"""Harness digest: what shapes measurement, minus kernels.

A build records the firmware inputs (tester files, NSX lock minus the
kernel module, build options and flags, toolchain); the bundle adds the
tester state at run time and hashes both into one hex digest, written
to the session manifest's top-level `harness_digest` (null for
unverified firmware). The inputs sit under `harness`. Two bundles
compare fairly only when their digests match. Scorers call
`harness_digest(manifest)` or `same_harness(a, b)`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml

# Session manifest keys; scorers read the digest.
HARNESS_FIELD = "harness_digest"
HARNESS_INPUTS = "harness"
HARNESS_SCHEMA = 1

# Tester files the firmware build reads.
FIRMWARE_INPUTS = (
    "cmake/hardware",
    "assets/templates/hardware/nsx",
    "scripts/generate_kernel_symbol_refs.py",
    "scripts/patch_build_id.py",
)
# NSX-owned app trees: board, flags.
APP_TREES = ("boards", "cmake/nsx")
# Kernel module location: ref vs root.
_KERNEL_MODULE_DIR = re.compile(r'(NSX_APP_MODULE_DIR_nsx_cmsis_nn\s+)"(?:modules/nsx-cmsis-nn|modules/ns-cmsis-nn/nsx)"')
_KERNEL_PROJECT_DIR = re.compile(r"(?m)^[ \t]*modules/nsx?-cmsis-nn[ \t]*\n")
# Cache entries that set compile flags.
_FLAG_CACHE = re.compile(
    r"^((?:CMAKE_(?:C|CXX|ASM)_FLAGS|CMAKE_EXE_LINKER_FLAGS)\w*|CMAKE_BUILD_TYPE|NSX_CMSIS_NN_\w+|ARM_NN_\w+)"
    r":[^=-]*=(.*)$",
    re.MULTILINE,
)
# Lock keys that vary per sync.
_LOCK_VOLATILE = ("generated_at", "acquired_at", "manifest")


def _git(root: Path, *args: str) -> Optional[bytes]:
    """Git stdout as bytes, or None."""
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def tester_state(repo_root: Path) -> dict[str, Any]:
    """Tester commit, dirty flag, diff hash."""
    from ..generation.reuse import _is_git_toplevel

    state: dict[str, Any] = {"commit": None, "dirty": None, "diff": None}
    if not _is_git_toplevel(repo_root):
        return state
    head = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain", "-z", "--untracked-files=all")
    state["commit"] = head.decode().strip() if head else None
    if status is None:
        return state
    hidden = _hidden_files(repo_root)
    if hidden is None:
        return state
    state["dirty"] = bool(status or hidden)
    if state["dirty"]:
        state["diff"] = _dirty_hash(repo_root, status, hidden)
    return state


def _hidden_files(repo_root: Path) -> Optional[list[bytes]]:
    """Tracked files git status skips."""
    listing = _git(repo_root, "ls-files", "-v", "-z")
    if listing is None:
        return None
    # S: skip-worktree; lowercase: assume-unchanged.
    return [entry[2:] for entry in listing.split(b"\0") if entry and (entry[:1] == b"S" or entry[:1].islower())]


def _dirty_hash(repo_root: Path, status: bytes, hidden: list[bytes]) -> Optional[str]:
    """Hash tracked edits, hidden and untracked files."""
    diff = _git(repo_root, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv")
    if diff is None:
        return None
    digest = hashlib.sha256(diff)
    for rel in hidden:
        path = repo_root / rel.decode()
        digest.update(b"hidden\0" + rel + b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        rel = entry[3:]
        path = repo_root / rel.decode()
        digest.update(rel + b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def path_hash(path: Path) -> Optional[str]:
    """Content hash of a file or tree."""
    from neuralspotx.nsx_lock import hash_file, hash_tree

    if path.is_dir():
        digest = hashlib.sha256(hash_tree(path, exclude_names=frozenset({"modules.cmake"})).encode())
        for modules in sorted(path.rglob("modules.cmake")):
            digest.update(modules.relative_to(path).as_posix().encode() + b"\0")
            digest.update(modules_print(modules.read_text(encoding="utf-8")).encode())
        return "sha256:" + digest.hexdigest()
    return hash_file(path) if path.is_file() else None


def modules_print(text: str) -> str:
    """modules.cmake minus the kernel module location."""
    text = _KERNEL_MODULE_DIR.sub(r'\1"<kernels>"', text)
    return _KERNEL_PROJECT_DIR.sub("", text)


def cache_flags(build_dir: Path) -> dict[str, str]:
    """Flag entries from CMakeCache.txt."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    text = cache.read_text(encoding="utf-8", errors="ignore")
    return {name: value for name, value in _FLAG_CACHE.findall(text)}


def lock_modules(app_dir: Path) -> Optional[dict[str, Any]]:
    """nsx.lock targets, minus kernels and timestamps."""
    from .nsx_app import CMSIS_NN_MODULE, CMSIS_NN_PROJECT

    try:
        lock = yaml.safe_load((app_dir / "nsx.lock").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    targets = lock.get("targets") if isinstance(lock, dict) else None
    if not isinstance(targets, dict):
        return None
    kernels = {CMSIS_NN_MODULE, CMSIS_NN_PROJECT}
    kept: dict[str, Any] = {}
    for name, target in targets.items():
        target = {k: v for k, v in (target or {}).items() if k not in _LOCK_VOLATILE}
        modules = {}
        for module, entry in (target.get("modules") or {}).items():
            entry = dict(entry or {})
            if module in kernels or entry.get("project") in kernels:
                continue
            resolved = {k: v for k, v in (entry.get("resolved") or {}).items() if k not in _LOCK_VOLATILE}
            modules[module] = {**entry, "resolved": resolved}
        kept[name] = {**target, "modules": modules}
    return kept


def firmware_record(repo_root: Path, build_dir: Path, options: Any, toolchain: Any, nsx_version: str) -> dict:
    """Firmware inputs a build used, minus kernels."""
    from .firmware_build import nsx_app_dir

    app_dir = nsx_app_dir(build_dir)
    # Kernel source fields stay out.
    switches = {
        name: getattr(options, name)
        for name in ("requantize_inline_asm", "enable_f32", "enable_f16", "build_size_probe", "placement")
    }
    return {
        "tester": tester_state(repo_root),
        "sources": {rel: path_hash(repo_root / rel) for rel in FIRMWARE_INPUTS},
        "options": switches,
        "cache_flags": cache_flags(build_dir),
        "nsx_version": nsx_version,
        "nsx_lock": lock_modules(app_dir),
        "app_trees": {rel: path_hash(app_dir / rel) for rel in APP_TREES},
        "module_trees": module_trees(app_dir),
        "toolchain": toolchain,
    }


def module_trees(app_dir: Path) -> dict[str, str]:
    """Content hash of each synced non-kernel module."""
    from neuralspotx.nsx_lock import hash_tree

    from .nsx_app import CMSIS_NN_MODULE, CMSIS_NN_PROJECT

    root = app_dir / "modules"
    if not root.is_dir():
        return {}
    # Lock pins revisions; sources can drift.
    kernels = {CMSIS_NN_MODULE, CMSIS_NN_PROJECT}
    return {path.name: hash_tree(path) for path in sorted(root.iterdir()) if path.is_dir() and path.name not in kernels}


def harness_record(firmware: Optional[dict], repo_root: Path) -> tuple[Optional[str], dict[str, Any]]:
    """Hex digest, plus inputs for the manifest."""
    host = tester_state(repo_root)
    # Unknown state counts as dirty.
    clean = host["dirty"] is False and (not firmware or firmware["tester"]["dirty"] is False)
    record = {"schema": HARNESS_SCHEMA, "tester_dirty": not clean, "inputs": None}
    if not firmware:
        return None, record
    record["inputs"] = {"firmware": firmware, "host": host}
    canonical = json.dumps(record["inputs"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), record


def harness_digest(manifest: dict) -> Optional[str]:
    """A session manifest's harness digest."""
    digest = manifest.get(HARNESS_FIELD)
    return digest if isinstance(digest, str) and digest else None


def same_harness(first: dict, second: dict) -> bool:
    """Both manifests carry one equal digest."""
    digest = harness_digest(first)
    return digest is not None and digest == harness_digest(second)


def kernel_digest(manifest: dict) -> Optional[str]:
    """The kernel tree hash a bundle built."""
    kernels = (manifest.get("build") or {}).get("kernels") or {}
    return kernels.get("tree_hash")
