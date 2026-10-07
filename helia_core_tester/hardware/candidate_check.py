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
- build_probe: #line, line markers, __has_include and macros the
  build flags set (__OPTIMIZE__, __FAST_MATH__), which can hide code
  from the gcc -E scan. Also an added #if/#elif that calls a macro or
  pastes, and an added #define that calls or pastes when any #if in
  Source/ or Include/ reaches it: these can build a probe.
- hidden_index_entry: any path flagged skip-worktree or
  assume-unchanged, which git diff and status would skip.
- guard_change: an added or removed #if/#ifdef/#else/#define/#undef
  in a file that already holds a forbidden construct, which it could
  enable.

Rules also run on text with adjacent string literals joined, per line
and over all added lines of a file, as C joins them before asm sees them,
and on text with `##` pastes joined. candidate_scan then reruns the
rules on `gcc -E` output of base and candidate.

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
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Iterator, Optional

import typer

from .candidate_scan import preprocess_findings
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
_ATTRIBUTE = re.compile(r"__attribute(?:__)?\s*\(\s*\((.*?)\)\s*\)|\[\[(.*?)\]\]", re.DOTALL)
_ADJACENT_LITERALS = re.compile(r'"\s*"')
_PASTE = re.compile(r"\s*##\s*")
_ESCAPE = re.compile(r"\\([0-7]{1,3}|[xX][0-9a-fA-F]+)")
_ATTRIBUTE_HINT = re.compile(r"__attribute|__declspec|\[\[")
# "%:" is the "#" digraph.
_PRAGMA = re.compile(r"(?:#|%:)\s*pragma|_Pragma|__pragma")
# Checked per occurrence; _Pragma is never safe.
_SAFE_PRAGMA = re.compile(r"(?:#|%:)\s*pragma\s+(?:once|GCC\s+unroll\s+\d+|GCC\s+diagnostic\b)")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_GUARD = re.compile(r"^\s*(?:#|%:)\s*(?:if|ifdef|ifndef|elif|elifdef|elifndef|else|endif|define|undef)\b")
_CONDITION = re.compile(r"^\s*(?:#|%:)\s*(?:if|elif)\b(.*)", re.DOTALL)
_DEFINE = re.compile(r"^\s*(?:#|%:)\s*define\s+([A-Za-z_]\w*)(.*)", re.DOTALL)
_IDENT = re.compile(r"[A-Za-z_]\w*")
# Pastes, or calls besides defined/__has_*.
_MACRO_CALL = re.compile(r"##|%:%:|\b(?!defined\b|__has_\w+\b)[A-Za-z_]\w*\s*\(")
# Integer literals, any base and suffix.
_INTEGER = re.compile(r"(?<![\w.'])(0[xX][0-9a-fA-F']+|0[bB][01']+|\d[\d']*)[uUlLzZ]*(?![\w.])")
# SCS: debug, timer and NVIC registers.
SCS_RANGE = range(0xE0000000, 0xE0100000)
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
    ("build_probe", re.compile(
        r"(?:^|\n)\s*(?:#|%:)\s*(?:line\b|\d)|__has_include|__OPTIMIZE(?:_SIZE)?__|__FAST_MATH__|__NO_INLINE__",
    )),
)
# An empty tar: 1024 zero bytes.
_EMPTY_TAR = bytes(1024)
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


def _removed_guard(tree: Path, base: dict, path: str, status: str) -> bool:
    """A deleted line was a guard."""
    if status == "D" or path not in base:
        return False
    new = (tree / path).read_text(encoding="utf-8", errors="replace").splitlines()
    old = _git(tree, "cat-file", "blob", base[path][1]).decode(errors="replace").splitlines()
    for tag, first, last, _, _ in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag in ("replace", "delete") and any(_GUARD.match(line) for line in old[first:last]):
            return True
    return False


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
    # C joins literals and pasted tokens.
    for variant in (text, _literal_text(text), _PASTE.sub("", text)):
        for rule in _raw_rules(variant):
            if rule not in seen:
                seen.add(rule)
                yield rule


def _literal_text(text: str, comments: bool = True) -> str:
    """Text as the compiler sees literals."""
    joined = _ADJACENT_LITERALS.sub("", _COMMENT.sub(" ", text) if comments else text)
    return _ESCAPE.sub(_unescape, joined)


def _unescape(match: re.Match) -> str:
    code = match.group(1)
    value = int(code[1:], 16) if code[0] in "xX" else int(code, 8)
    return chr(value) if value < 0x110000 else match.group(0)


def _scs_count(text: str) -> int:
    """Integer literals inside SCS_RANGE."""
    count = 0
    for match in _INTEGER.finditer(text):
        digits = match.group(1).replace("'", "").lower()
        base = {"0x": 16, "0b": 2}.get(digits[:2], 8 if digits[:1] == "0" else 10)
        try:
            count += int(digits, base) in SCS_RANGE
        except ValueError:
            continue
    return count


def _raw_rules(text: str) -> Iterator[str]:
    if _unsafe_attribute(text):
        yield "attribute"
    code = _COMMENT.sub(" ", text)
    if any(not _SAFE_PRAGMA.match(code, hit.start()) for hit in _PRAGMA.finditer(code)):
        yield "pragma"
    if _scs_count(text):
        yield "measurement_access"
    yield from (rule for rule, pattern in LINE_RULES if pattern.search(text))


def _grandfathered(tree: Path, path: str) -> bool:
    """File already holds a rule hit."""
    lines = (tree / path).read_text(encoding="utf-8", errors="replace").splitlines()
    starts = _logical_lines(lines)
    joined: dict[int, str] = {}
    for index, line in enumerate(lines):
        joined[starts[index]] = joined.get(starts[index], "") + (line[:-1] if line.endswith("\\") else line)
    return any(next(line_rules(text), None) for text in joined.values())


def rule_counts(source: str, preprocessed: bool = False) -> Counter:
    """Rule matches over a whole file.

    Preprocessed text has no comments: a "//" literal stays.
    """
    lines = source.splitlines()
    text = _literal_text("\n".join(line[:-1] if line.endswith("\\") else line for line in lines), not preprocessed)
    counts: Counter = Counter()
    found = _ATTRIBUTE.findall(text)
    # Unparsed attribute spellings count as unsafe.
    counts["attribute"] = sum(_unsafe_attribute("".join(groups).join(("__attribute__((", "))"))) for groups in found)
    counts["attribute"] += max(0, len(_ATTRIBUTE_HINT.findall(text)) - len(found))
    code = text if preprocessed else _COMMENT.sub(" ", text)
    counts["pragma"] = sum(not _SAFE_PRAGMA.match(code, hit.start()) for hit in _PRAGMA.finditer(code))
    for rule, pattern in LINE_RULES:
        counts[rule] += len(pattern.findall(text))
    counts["measurement_access"] += _scs_count(text)
    return counts


def _tree_macros(tree: Path) -> set[str]:
    """Names any #if reaches via defines."""
    used: set[str] = set()
    bodies: dict[str, set[str]] = {}
    for top in ALLOWED_DIRS:
        for path in sorted((tree / top).rglob("*")):
            if path.is_symlink() or not path.is_file() or not path.name.endswith(ALLOWED_SUFFIXES):
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            starts = _logical_lines(lines)
            joined: dict[int, str] = {}
            for index, line in enumerate(lines):
                joined[starts[index]] = joined.get(starts[index], "") + (line[:-1] if line.endswith("\\") else line)
            for text in joined.values():
                code = _COMMENT.sub(" ", text)
                if match := _CONDITION.match(code):
                    used |= set(_IDENT.findall(match.group(1)))
                elif match := _DEFINE.match(code):
                    bodies.setdefault(match.group(1), set()).update(_IDENT.findall(match.group(2)))
    # Names reached through macro bodies.
    todo = list(used)
    while todo:
        for name in bodies.get(todo.pop(), ()):
            if name not in used:
                used.add(name)
                todo.append(name)
    return used


def probe_findings(tree: Path, added: dict[str, list[tuple[int, str]]]) -> Iterator[dict]:
    """Conditionals built from macro calls.

    gcc -E drops code under a false #if, so a probe the scan cannot
    see (a pasted __has_include) can hide a pragma.
    """
    used: Optional[set[str]] = None
    for path, lines in sorted(added.items()):
        for line_no, text in lines:
            code = _COMMENT.sub(" ", text)
            if (match := _CONDITION.match(code)) and _MACRO_CALL.search(match.group(1)):
                yield {"rule": "build_probe", "path": path, "line": line_no, "text": "macro call in #if"}
            elif (match := _DEFINE.match(code)) and _MACRO_CALL.search(match.group(2)):
                used = _tree_macros(tree) if used is None else used
                if match.group(1) in used:
                    yield {"rule": "build_probe", "path": path, "line": line_no, "text": "#if reaches computed macro"}


def hidden_entries(tree: Path) -> list[str]:
    """Paths with skip-worktree or assume-unchanged."""
    out = _split(_git(tree, "ls-files", "-v", "-z"))
    return [entry[2:] for entry in out if entry[:1] == "S" or entry[:1].islower()]


def check_candidate(tree: Path, base: str) -> dict:
    """The JSON report for one candidate."""
    tree = tree.resolve()
    commit = _git(tree, "rev-parse", "--verify", f"{base}^{{commit}}").decode().strip()
    # Branches and tags can be moved.
    if base.lower() != commit:
        raise CheckError(f"--base must be a full commit SHA, got {base!r}")
    changes = changed_paths(tree, commit)
    base_blobs = base_files(tree, commit)
    findings: list[dict] = []
    added_by_path: dict[str, list[tuple[int, str]]] = {}
    for path, status in sorted(changes.items()):
        hits = list(path_findings(path, status, tree))
        findings += hits
        if hits:
            continue
        added = added_by_path[path] = list(added_lines(tree, base_blobs, path, status))
        hit_rules: set[str] = set()
        for line_no, text in added:
            for rule in line_rules(text):
                hit_rules.add(rule)
                findings.append({"rule": rule, "path": path, "line": line_no, "text": text.strip()[:200]})
        # Guard edits can enable old lines.
        guard_line = next((n for n, text in added if _GUARD.match(text)), None)
        if guard_line is None and _removed_guard(tree, base_blobs, path, status):
            guard_line = added[0][0] if added else 0
        if guard_line is not None and _grandfathered(tree, path):
            findings.append({"rule": "guard_change", "path": path, "line": guard_line,
                             "text": "preprocessor change in a file with forbidden constructs"})
        # Literals split across lines.
        if added:
            joined = " ".join(text.strip() for _, text in added)
            for rule in line_rules(joined):
                if rule not in hit_rules:
                    hit_rules.add(rule)
                    findings.append({"rule": rule, "path": path, "line": added[0][0], "text": "joined added lines"})
            # Edits inside unchanged constructs.
            old = _git(tree, "cat-file", "blob", base_blobs[path][1]).decode(errors="replace") if path in base_blobs else ""
            before, after = rule_counts(old), rule_counts((tree / path).read_text(encoding="utf-8", errors="replace"))
            findings += ({"rule": rule, "path": path, "line": added[0][0], "text": "more matches in whole file"}
                         for rule in sorted(after) if after[rule] > before[rule] and rule not in hit_rules)
    findings += probe_findings(tree, added_by_path)
    findings += ({"rule": "hidden_index_entry", "path": path, "message": "skip-worktree or assume-unchanged set"}
                 for path in hidden_entries(tree))
    if any(path.startswith(ALLOWED_DIRS) for path in changes):
        tops = sorted({path.split("/", 1)[0] + "/" for path in base_blobs if path.startswith(ALLOWED_DIRS)})
        archive = _git(tree, "archive", commit, "--", *tops) if tops else _EMPTY_TAR
        findings += preprocess_findings(tree, archive, partial(rule_counts, preprocessed=True))
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
