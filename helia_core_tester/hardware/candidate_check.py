"""Check a candidate kernel diff stays in bounds.

A candidate may change kernel C, headers and assembly under Source/ and
Include/, nothing else. Rules, each a finding in the JSON report:

- outside_allowlist: any path outside Source/ or Include/ (Tests/,
  cmake/, nsx/, CMakeLists, scripts, build files).
- frozen_file: public API headers the adapters compile against, and
  arm_nntables.c, which the s16 golden generator reads.
- file_type: Source/Include files that are not .c/.h/.s/.S.
- symlink: a symlink in the candidate change set.
- attribute: added attributes beyond a safe list (inline, unused,
  aligned, packed...), so no section, optimize or target attributes.
- pragma: added pragmas other than once, GCC unroll, GCC diagnostic.
- special_section: .section/.pushsection, TCM and RAMFUNC macros.
- measurement_access: DWT, PMU, SysTick, SCB, NVIC, HAL calls, IRQ
  masking: anything that could touch the timer or the clock.
- harness_reference: added lines naming harness or golden internals.
- include_escape: #include with an absolute path, "..", or a macro;
  .incbin and .include in assembly.
- hidden_index_entry: any path flagged skip-worktree or
  assume-unchanged, which git diff and status would skip.
- guard_change: an added #if/#ifdef/#else/#define/#undef in a file
  that already holds a forbidden construct, which it could enable.

Rules also run on text with adjacent string literals joined, per line
and over all added lines of a file, as C joins them before asm sees them.

Literal tensor shapes from descriptors are not grepped: too many false
hits on common dims. Trees the build copies (Source, Include, cmake, nsx)
and Tests/ are compared file by file on disk against `git ls-tree`, so
index flags, ignore rules and diff config cannot hide a change. Other
paths come from a hardened `git diff`. Pass --base as a full commit SHA:
a branch or tag lives in the candidate repo and can be moved.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import typer

from .nsx_app import KERNEL_TREES

ALLOWED_DIRS = ("Source/", "Include/")
ALLOWED_SUFFIXES = (".c", ".h", ".s", ".S")
# Adapters and goldens read these.
FROZEN_FILES = (
    "Include/arm_nnfunctions.h",
    "Include/arm_nnfunctions_flt.h",
    "Include/arm_nn_types.h",
    "Include/arm_nn_types_flt.h",
    "Source/NNSupportFunctions/arm_nntables.c",
)
# Trees the build or generation reads.
WATCHED = (*KERNEL_TREES, "nsx", "Tests")
SAFE_ATTRIBUTES = frozenset((
    "always_inline", "noinline", "noipa", "unused", "maybe_unused", "aligned", "packed", "fallthrough",
    "const", "pure", "nonnull", "may_alias", "inline",
))
_ATTRIBUTE = re.compile(r"__attribute__\s*\(\((.*?)\)\)|\[\[(.*?)\]\]")
_ADJACENT_LITERALS = re.compile(r'"\s*"')
_ATTRIBUTE_HINT = re.compile(r"__attribute|__declspec|\[\[")
# "%:" is the "#" digraph.
_PRAGMA = re.compile(r"(?:#|%:)\s*pragma|_Pragma|__pragma")
# Checked per occurrence; _Pragma is never safe.
_SAFE_PRAGMA = re.compile(r"(?:#|%:)\s*pragma\s+(?:once|GCC\s+unroll\s+\d+|GCC\s+diagnostic\b)")
_COMMENT = re.compile(r"/\*.*?\*/|//.*$")
_GUARD = re.compile(r"^\s*(?:#|%:)\s*(?:if|ifdef|ifndef|elif|elifdef|elifndef|else|endif|define|undef)\b")
LINE_RULES = (
    ("special_section", re.compile(
        r"\.(?:push)?section\b|\b_*section_*\s*\(|\b(?:ITCM|DTCM|__RAMFUNC|RAMFUNC|AM_SHARED_RW|NS_PUT_IN_TCM)\b",
    )),
    ("measurement_access", re.compile(
        r"\b(?:DWT|CoreDebug|DCB|SysTick|ITM|TPI|NVIC|SCB|PMU|MEMSYSCTL|CYCCNT)\b|\bARM_PMU_|\bam_hal_"
        r"|__(?:disable|enable)_(?:irq|fault_irq)|__WF[IE]\b|__set_(?:BASEPRI|PRIMASK|FAULTMASK)"
        # Same, as assembly.
        r"|(?i:\bcpsi[de]\b|\bmsr\s+(?:primask|basepri(?:_max)?|faultmask|control)\b|\bwf[ie]\b)",
    )),
    ("harness_reference", re.compile(
        r"golden|\bhct_|\bhctp|benchmark_server|unittest|RefactoredTestGen", re.IGNORECASE,
    )),
    ("include_escape", re.compile(r'(?:#|%:)\s*include\s*(?:["<](?:/|[^">]*\.\.)|[^"<\s])|\.(?:incbin|include)\b')),
)
# Ignore user and system git config.
_GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1"}


class CheckError(RuntimeError):
    """The check could not run."""


def _git(tree: Path, *args: str) -> bytes:
    done = subprocess.run(
        ["git", "-C", str(tree), *args], capture_output=True, check=False, env={**os.environ, **_GIT_ENV},
    )
    if done.returncode != 0:
        raise CheckError(done.stderr.decode(errors="replace").strip() or f"git {args[0]} failed")
    return done.stdout


def _split(out: bytes) -> list[str]:
    return [part for part in out.decode().split("\0") if part]


def base_files(tree: Path, commit: str) -> dict[str, tuple[str, str]]:
    """Path to (mode, blob) at base."""
    files = {}
    for entry in _split(_git(tree, "ls-tree", "-r", "-z", "--full-tree", commit, "--", *WATCHED)):
        meta, path = entry.split("\t", 1)
        mode, _, blob = meta.split()
        files[path] = (mode, blob)
    return files


def _blob_id(path: Path, algo: str) -> str:
    data = path.read_bytes()
    return hashlib.new(algo, b"blob %d\0" % len(data) + data).hexdigest()


def _disk_files(tree: Path, skip: set[str]) -> Iterator[str]:
    """Files and symlinks under watched trees."""
    for top in WATCHED:
        # A symlinked root is a change.
        if os.path.islink(tree / top):
            yield top
            continue
        for root, dirs, names in os.walk(tree / top):
            # Submodules (a nested tester) have their own check.
            dirs[:] = [name for name in dirs if Path(root, name).relative_to(tree).as_posix() not in skip]
            # Report symlinked dirs; never follow.
            names += [name for name in dirs if os.path.islink(os.path.join(root, name))]
            for name in names:
                yield Path(root, name).relative_to(tree).as_posix()


def changed_paths(tree: Path, commit: str) -> dict[str, str]:
    """Path to status, vs base."""
    algo = _git(tree, "rev-parse", "--show-object-format").decode().strip()
    base = base_files(tree, commit)
    submodules = {rel for rel, (mode, _) in base.items() if mode == "160000"}
    changes: dict[str, str] = {}
    for rel in _disk_files(tree, submodules):
        path = tree / rel
        if rel not in base:
            changes[rel] = "A"
        elif path.is_symlink() or base[rel][0] == "120000" or _blob_id(path, algo) != base[rel][1]:
            changes[rel] = "M"
    changes.update((rel, "D") for rel in base if not os.path.lexists(tree / rel))
    for rel in submodules & changes.keys():
        del changes[rel]
    # Elsewhere: hardened git diff.
    outside = [f":(exclude){top}" for top in WATCHED]
    out = _split(_git(tree, "diff", "--name-status", "--no-renames", "--no-ext-diff", "-z", commit, "--", ".", *outside))
    changes.update(zip(out[1::2], out[0::2]))
    untracked = _split(_git(tree, "ls-files", "--others", "--exclude-standard", "-z", "--", ".", *outside))
    changes.update((rel, "A") for rel in untracked)
    return changes


def path_findings(path: str, status: str, tree: Path) -> Iterator[dict]:
    """Rule hits from the path alone."""
    if not path.startswith(ALLOWED_DIRS):
        yield {"rule": "outside_allowlist", "path": path, "message": "only Source/ and Include/ may change"}
    elif path in FROZEN_FILES:
        yield {"rule": "frozen_file", "path": path, "message": "harness reads this file"}
    elif not path.endswith(ALLOWED_SUFFIXES):
        yield {"rule": "file_type", "path": path, "message": "only .c .h .s .S files"}
    if status != "D" and (tree / path).is_symlink():
        yield {"rule": "symlink", "path": path, "message": "symlinks are not allowed"}


def _logical_lines(lines: list[str]) -> list[int]:
    """Each line's spliced-line start index."""
    starts, start = [], 0
    for index, line in enumerate(lines):
        starts.append(start)
        if not line.endswith("\\"):
            start = index + 1
    return starts


def added_lines(tree: Path, base: dict, path: str, status: str) -> Iterator[tuple[int, str]]:
    """Spliced lines holding an added line.

    C and assembly join backslash-newline before parsing, so rules see the
    joined text; the number is the first added line in it.
    """
    if status == "D":
        return
    new = (tree / path).read_text(encoding="utf-8", errors="replace").splitlines()
    old = _git(tree, "cat-file", "blob", base[path][1]).decode(errors="replace").splitlines() if path in base else []
    starts = _logical_lines(new)
    seen: set[int] = set()
    for tag, _, _, first, last in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag not in ("replace", "insert"):
            continue
        for index in range(first, last):
            start = starts[index]
            if start in seen:
                continue
            seen.add(start)
            end = start
            while end + 1 < len(new) and starts[end + 1] == start:
                end += 1
            text = "".join(line[:-1] if line.endswith("\\") else line for line in new[start:end + 1])
            yield index + 1, text


def _unsafe_attribute(text: str) -> bool:
    """Any attribute outside the safe list."""
    if not _ATTRIBUTE_HINT.search(text):
        return False
    found = _ATTRIBUTE.findall(text)
    if not found:
        return True
    for groups in found:
        body = re.sub(r"\([^()]*\)?", "", "".join(groups))
        names = (name.strip().removeprefix("gnu::").strip("_") for name in body.split(","))
        if any(name not in SAFE_ATTRIBUTES for name in names):
            return True
    return False


def line_rules(text: str) -> Iterator[str]:
    """Rule names one added line hits."""
    seen = set()
    # C joins adjacent string literals.
    for variant in (text, _ADJACENT_LITERALS.sub("", text)):
        for rule in _raw_rules(variant):
            if rule not in seen:
                seen.add(rule)
                yield rule


def _raw_rules(text: str) -> Iterator[str]:
    if _unsafe_attribute(text):
        yield "attribute"
    code = _COMMENT.sub(" ", text)
    if any(not _SAFE_PRAGMA.match(code, hit.start()) for hit in _PRAGMA.finditer(code)):
        yield "pragma"
    yield from (rule for rule, pattern in LINE_RULES if pattern.search(text))


def _grandfathered(tree: Path, path: str) -> bool:
    """File already holds a rule hit."""
    lines = (tree / path).read_text(encoding="utf-8", errors="replace").splitlines()
    starts = _logical_lines(lines)
    joined: dict[int, str] = {}
    for index, line in enumerate(lines):
        joined[starts[index]] = joined.get(starts[index], "") + (line[:-1] if line.endswith("\\") else line)
    return any(next(line_rules(text), None) for text in joined.values())


def hidden_entries(tree: Path) -> list[str]:
    """Paths with skip-worktree or assume-unchanged."""
    out = _split(_git(tree, "ls-files", "-v", "-z"))
    return [entry[2:] for entry in out if entry[:1] == "S" or entry[:1].islower()]


def check_candidate(tree: Path, base: str) -> dict:
    """The JSON report for one candidate."""
    tree = tree.resolve()
    commit = _git(tree, "rev-parse", "--verify", f"{base}^{{commit}}").decode().strip()
    changes = changed_paths(tree, commit)
    base_blobs = base_files(tree, commit)
    findings: list[dict] = []
    for path, status in sorted(changes.items()):
        hits = list(path_findings(path, status, tree))
        findings += hits
        if hits:
            continue
        added = list(added_lines(tree, base_blobs, path, status))
        hit_rules: set[str] = set()
        for line_no, text in added:
            for rule in line_rules(text):
                hit_rules.add(rule)
                findings.append({"rule": rule, "path": path, "line": line_no, "text": text.strip()[:200]})
        # Guard edits can enable old lines.
        if any(_GUARD.match(text) for _, text in added) and _grandfathered(tree, path):
            line_no = next(n for n, text in added if _GUARD.match(text))
            findings.append({"rule": "guard_change", "path": path, "line": line_no,
                             "text": "preprocessor change in a file with forbidden constructs"})
        # Literals split across lines.
        if added:
            joined = " ".join(text.strip() for _, text in added)
            findings += ({"rule": rule, "path": path, "line": added[0][0], "text": "joined added lines"}
                         for rule in line_rules(joined) if rule not in hit_rules)
    findings += ({"rule": "hidden_index_entry", "path": path, "message": "skip-worktree or assume-unchanged set"}
                 for path in hidden_entries(tree))
    return {
        "schema": "hct.candidate_check",
        "schema_version": 1,
        "tree": str(tree),
        "base": base,
        "base_commit": commit,
        "ok": not findings,
        "files": [{"path": path, "status": status} for path, status in sorted(changes.items())],
        "findings": findings,
    }


candidate_app = typer.Typer(help="Check candidate kernel trees.", no_args_is_help=True)


@candidate_app.command("check")
def check_command(
    tree: Path = typer.Argument(..., exists=True, file_okay=False, help="Candidate ns-cmsis-nn checkout."),
    base: str = typer.Option(..., "--base", help="Base commit SHA the candidate started from."),
) -> None:
    """Fail when the candidate diff leaves Source/Include."""
    try:
        report = check_candidate(tree, base)
    except CheckError as exc:
        typer.echo(json.dumps({"ok": False, "error": str(exc)}))
        raise typer.Exit(2)
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(0 if report["ok"] else 1)
