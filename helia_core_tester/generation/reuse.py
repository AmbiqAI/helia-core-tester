"""Per-case reuse stamps for the generation step.

TFLite conversion plus interpreter inference dominates generation wall time, and
every input that can change a case's emitted bytes is knowable before that work
starts: the descriptor document, the per-case generation inputs (cpu, suite,
float precision, seed), the generator itself and the ns-cmsis-nn checkout the
generator reads. A stamp folds all of those into one digest; a case whose
on-disk stamp still matches, and whose artifacts still hash to what that stamp
recorded, is reused verbatim. See issue #107.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from helia_core_tester.core.discovery import find_repo_root, find_tester_templates_dir
from helia_core_tester.generation.artifact_identity import generated_case_artifact_sha256
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

STAMP_FILENAME = ".stamp"

# Version prefix of the stamp payload itself. Bump when the payload layout
# changes so old stamps cannot accidentally validate against new semantics.
_STAMP_SCHEMA = "helia-core-tester/generation-stamp/2"

# Packages whose codegen or numerics feed the emitted bytes; a version bump in
# either can change a converted model or a reference output.
_PINNED_PACKAGES = ("tensorflow", "ai-edge-litert")

# Modules outside generation/ that still decide emitted bytes: the CPU profile
# (has_mve gates which template variant a case emits) and the path layout that
# places a case in the tree.
_EXTERNAL_GENERATOR_SOURCES = (
    Path("helia_core_tester") / "core" / "cpu_targets.py",
    Path("helia_core_tester") / "core" / "path_layout.py",
)

# Subtrees of the ns-cmsis-nn checkout that are generation inputs: the public
# headers drive the temp-sizer probe's choice of template variant, and the
# UnitTest data is read directly as LSTM/GRU goldens.
_CMSIS_NN_INPUT_SUBTREES = ("Include", "Tests/UnitTest/TestCases/TestData")

_version_hash_cache: Optional[str] = None
_checkout_identity_cache: Optional[Dict[str, str]] = None


def _iter_generator_sources() -> Iterator[Path]:
    """Files whose content defines what the generator emits for any case.

    Descriptors are deliberately excluded: they are hashed into each case's own
    stamp, so folding them in here as well would make one descriptor edit
    invalidate every other case.

    Unresolvable templates raise rather than yielding a shorter list: a stamp
    computed without the templates would validate a case emitted by a different
    template, and the tester cannot generate anything without them anyway.
    """
    generation_root = Path(__file__).resolve().parent
    for path in sorted(generation_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path

    repo_root = find_repo_root()
    for relative in _EXTERNAL_GENERATOR_SOURCES:
        yield repo_root / relative

    templates_root = find_tester_templates_dir(repo_root)
    if not templates_root.is_dir():
        raise FileNotFoundError(
            f"Generator templates directory not found: {templates_root}. "
            f"A reuse stamp computed without the templates would validate cases "
            f"emitted by a different template."
        )
    for path in sorted(templates_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path


def _pinned_package_versions() -> Dict[str, Optional[str]]:
    """Installed versions of the pinned converter packages.

    An unresolvable version is recorded as JSON ``null``, which no real version
    string can equal: "installed but not queryable" and "version X" stay
    distinguishable states in the digest.
    """
    from importlib import metadata

    versions: Dict[str, Optional[str]] = {}
    for package in _PINNED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = None
    return versions


def generator_version_hash() -> str:
    """Digest of the generator sources, templates and pinned converter versions.

    Cached for the process: the inputs cannot change under a running generation.
    """
    global _version_hash_cache
    if _version_hash_cache is not None:
        return _version_hash_cache

    digest = hashlib.sha256()
    digest.update(_STAMP_SCHEMA.encode("utf-8"))
    try:
        repo_root = find_repo_root()
    except Exception:
        repo_root = None
    for path in _iter_generator_sources():
        try:
            label = str(path.relative_to(repo_root)) if repo_root else path.name
        except ValueError:
            label = path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(
        json.dumps(_pinned_package_versions(), sort_keys=True).encode("utf-8")
    )
    _version_hash_cache = digest.hexdigest()
    return _version_hash_cache


def _git_output(root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _is_git_toplevel(root: Path) -> bool:
    """True only when ``root`` is itself the top of a git working tree.

    ``git -C`` walks upwards, so a non-git copy of the checkout unpacked inside
    another repository (a gitignored scratch path, a vendored drop) would
    otherwise answer with the OUTER repository's commit and report itself clean
    no matter what was edited under it.
    """
    toplevel = _git_output(root, "rev-parse", "--show-toplevel")
    if toplevel is None or not toplevel.strip():
        return False
    try:
        return Path(toplevel.strip()).resolve() == root.resolve()
    except OSError:
        return False


def _subtree_digest(root: Path, subtrees: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for subtree in subtrees:
        digest.update(subtree.encode("utf-8"))
        digest.update(b"\0")
        base = root / subtree
        if not base.is_dir():
            digest.update(b"<absent>\0")
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def cmsis_nn_checkout_identity() -> Dict[str, str]:
    """Identity of the ns-cmsis-nn checkout the generator will read.

    The checkout is an input to the emitted bytes, not only to the build: the
    LSTM/GRU goldens are read straight out of its UnitTest TestData, and the
    temp-sizer probe (generation/utils/temp_sizer_probe.py) reads its public
    Include/ headers to pick which template variant a case emits. Two runs
    against different checkouts must therefore never share a stamp.

    A clean git checkout is identified by its commit, which is cheap and exact.
    A dirty one cannot be: the commit says nothing about the edits on top of it,
    so the input subtrees are hashed as well. A root that is not itself the top
    of a git working tree (an unpacked release, a vendored copy, a copy sitting
    inside some unrelated repository) is identified by content alone.

    Cached for the process: the checkout cannot change under a running
    generation.
    """
    global _checkout_identity_cache
    if _checkout_identity_cache is not None:
        return _checkout_identity_cache

    root = resolve_cmsis_nn_root()
    if root is None:
        identity: Dict[str, str] = {"state": "absent"}
    else:
        head = _git_output(root, "rev-parse", "HEAD") if _is_git_toplevel(root) else None
        if head is None:
            identity = {
                "state": "content",
                "content": _subtree_digest(root, _CMSIS_NN_INPUT_SUBTREES),
            }
        else:
            status = _git_output(root, "status", "--porcelain")
            if status is None or status.strip():
                identity = {
                    "state": "git-dirty",
                    "commit": head.strip(),
                    "content": _subtree_digest(root, _CMSIS_NN_INPUT_SUBTREES),
                }
            else:
                identity = {"state": "git-clean", "commit": head.strip()}
    _checkout_identity_cache = identity
    return identity


def case_stamp(
    descriptor: Dict[str, Any],
    *,
    case_name: str,
    cpu: str,
    suite: str,
    float_precision: str,
    seed: int,
    version_hash: str,
) -> str:
    """Digest of everything that determines a case's generated bytes."""
    payload = {
        "schema": _STAMP_SCHEMA,
        "descriptor": descriptor,
        "case_name": case_name,
        "cpu": cpu,
        "suite": suite,
        "float_precision": float_precision,
        "seed": seed,
        "generator_version": version_hash,
        "cmsis_nn_checkout": cmsis_nn_checkout_identity(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_stamp(test_dir: Path) -> Optional[Dict[str, str]]:
    """Return the stamp record written for this case, or None."""
    try:
        record = json.loads((test_dir / STAMP_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("schema") != _STAMP_SCHEMA:
        return None
    return record


def write_stamp(test_dir: Path, stamp: str) -> None:
    """Record a case as reusable. Must be the last write of a case.

    The record pins the emitted artifacts as well as the inputs, so a case that
    is edited or truncated after generation stops being reusable.
    """
    record = {
        "schema": _STAMP_SCHEMA,
        "stamp": stamp,
        "artifacts_sha256": generated_case_artifact_sha256(test_dir),
    }
    (test_dir / STAMP_FILENAME).write_text(json.dumps(record, sort_keys=True) + "\n")


def reset_case_dir(test_dir: Path) -> None:
    """Empty a case directory so regeneration starts from nothing.

    Overwriting in place is not enough: a descriptor that keeps its name but
    changes operator emits differently named sources, and the generated
    CMakeLists globs ``*_*.c``, so a file left over from the previous shape
    would be compiled alongside the new one and duplicate its symbols. Removing
    the directory also removes the stamp, so an interrupted regeneration leaves
    nothing that can be mistaken for a complete case.
    """
    if test_dir.exists():
        shutil.rmtree(test_dir)


def case_reusable(test_dir: Path, stamp: str) -> bool:
    """True iff this case was generated from ``stamp`` and is still intact.

    Presence is not integrity: a truncated .c, a deleted .tflite, a missing
    header or a lost CMakeLists.txt each leave a case that still looks generated
    but no longer builds, or no longer tests what the stamp claims. Comparing
    the recorded artifact digest (generation/artifact_identity.py, the same
    identity the reporting path binds results to) against the tree makes any
    such damage a regeneration.
    """
    record = read_stamp(test_dir)
    if record is None or record.get("stamp") != stamp:
        return False
    recorded = record.get("artifacts_sha256")
    if not recorded:
        return False
    try:
        return generated_case_artifact_sha256(test_dir) == recorded
    except (OSError, ValueError):
        return False


def prune_unlisted_cases(generated_tests_dir: Path, keep_relative_dirs: set[str]) -> int:
    """Delete case directories outside the current run's set, returning the
    number actually removed.

    Generation used to wipe the whole output tree on every run, which made the
    tree an exact record of the active filter. Reuse means the tree survives, so
    the pruning has to be explicit or a narrower run would leave a wider run's
    cases behind for build and run steps to pick up. A directory that cannot be
    removed raises: a survivor counted as pruned is exactly the stale binary the
    pruning exists to prevent.
    """
    removed = 0
    keep = {str(Path(item)) for item in keep_relative_dirs}
    for family_dir in sorted(generated_tests_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        for case_dir in sorted(family_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            relative = str(case_dir.relative_to(generated_tests_dir))
            if relative in keep:
                continue
            shutil.rmtree(case_dir)
            removed += 1
        if not any(family_dir.iterdir()):
            family_dir.rmdir()
    return removed
