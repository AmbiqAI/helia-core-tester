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
  from the gcc -E scan. Since the scan configs and the real build can
  disagree on which branches are live, also: an added #if/#elif that
  calls a macro or pastes; an added or removed #define/#undef of a name
  any conditional in Source/ or Include/ reaches, through macro bodies,
  whatever its body (include guards excepted); and an added line that
  uses a macro whose body (transitively) pastes.
- hidden_index_entry: any path flagged skip-worktree or
  assume-unchanged, which git diff and status would skip.
- guard_change: an added or removed #if/#ifdef/#else/#define/#undef
  in a file that already holds a forbidden construct, which it could
  enable.

Every rule reads one normalized view of a file, as translation
phases 1-3 make it: line splices joined, then each comment turned into
one space (keeping its newlines), string and char literals respected.
Line numbers in findings stay those of the original file.

Rules also run on text with adjacent string literals joined, per line
and over all added lines of a file, as C joins them before asm sees them,
and on text with `##` (or `%:%:`) pastes joined. candidate_scan then reruns the
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
# "%:%:" is the "##" digraph.
_PASTE = re.compile(r"\s*(?:##|%:%:)\s*")
_ESCAPE = re.compile(r"\\([0-7]{1,3}|[xX][0-9a-fA-F]+)")
_ATTRIBUTE_HINT = re.compile(r"__attribute|__declspec|\[\[")
# "%:" is the "#" digraph.
_PRAGMA = re.compile(r"(?:#|%:)\s*pragma|_Pragma|__pragma")
# Checked per occurrence; _Pragma is never safe.
_SAFE_PRAGMA = re.compile(r"(?:#|%:)\s*pragma\s+(?:once|GCC\s+unroll\s+\d+|GCC\s+diagnostic\b)")
_NEWLINE = re.compile(r"\r\n|\r|\n")
# gcc also splices after trailing blanks.
_SPLICE = re.compile(r"\\[ \t\f\v]*$")
# Code runs, literals, comments: phase 3.
_LEX = re.compile(
    r"(?P<code>(?:[^\W\d]\w*|\.?\d(?:[eEpP][+-]|'\w|[\w.])*|[^\"'/\w]|/(?![*/]))+)"
    r"|(?P<literal>\"(?:\\[^\n]|[^\"\\\n])*\"?|'(?:\\[^\n]|[^'\\\n])*'?)"
    r"|(?P<block>/\*.*?(?:\*/|\Z))"
    r"|(?P<line>//[^\n]*)",
    re.DOTALL,
)
_GUARD = re.compile(r"^\s*(?:#|%:)\s*(?:if|ifdef|ifndef|elif|elifdef|elifndef|else|endif|define|undef)\b")
_CONDITION = re.compile(r"^\s*(?:#|%:)\s*(?:if|elif)\b(.*)", re.DOTALL)
_NAME_TEST = re.compile(r"^\s*(?:#|%:)\s*(?:el)?ifn?def\s+([A-Za-z_]\w*)")
_DEFINE = re.compile(r"^\s*(?:#|%:)\s*define\s+([A-Za-z_]\w*)(\([^)]*\))?(.*)", re.DOTALL)
_IFNDEF = re.compile(r"^\s*(?:#|%:)\s*ifndef\s+([A-Za-z_]\w*)\s*$")
_DEFINE_UNDEF = re.compile(r"^\s*(?:#|%:)\s*(?:define|undef)\s+([A-Za-z_]\w*)")
_PASTES = re.compile(r"##|%:%:")
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
        r"(?:^|\n)\s*(?:#|%:)\s*(?:line\b|\d)|\?\?[=/'()!<>-]|__has_include|__OPTIMIZE(?:_SIZE)?__|__FAST_MATH__|__NO_INLINE__",
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


def _physical(source: str) -> list[str]:
    """Lines as gcc splits them."""
    return _NEWLINE.split(source)


def _blank(match: re.Match) -> str:
    if match.lastgroup == "block":
        return "\n" * match.group().count("\n") + " "
    return " " if match.lastgroup == "line" else match.group()


def normalize(source: str) -> tuple[list[str], list[int]]:
    """Logical lines after phases 1-3.

    Also each physical line's logical index.
    """
    logical: list[str] = []
    owner: list[int] = []
    parts: list[str] = []
    for line in _physical(source):
        owner.append(len(logical))
        spliced = _SPLICE.sub("", line)
        parts.append(spliced)
        if spliced == line:
            logical.append("".join(parts))
            parts = []
    if parts:
        logical.append("".join(parts))
    return _LEX.sub(_blank, "\n".join(logical)).split("\n"), owner


def _sources(tree: Path, base: dict, path: str) -> tuple[str, str]:
    """Base and candidate text of a path."""
    new = (tree / path).read_text(encoding="utf-8", errors="replace")
    old = _git(tree, "cat-file", "blob", base[path][1]).decode(errors="replace") if path in base else ""
    return old, new


def _changed(old: list[str], new: list[str], removed: bool = False) -> Iterator[int]:
    """Indexes of added (or removed) lines."""
    kinds = ("replace", "delete") if removed else ("replace", "insert")
    for tag, old_first, old_last, first, last in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag in kinds:
            yield from range(old_first, old_last) if removed else range(first, last)


def added_lines(tree: Path, base: dict, path: str, status: str) -> Iterator[tuple[int, str]]:
    """Normalized logical lines that changed.

    The number is the first added physical line in it, else its first.
    """
    if status == "D":
        return
    old_source, new_source = _sources(tree, base, path)
    new, owner = normalize(new_source)
    old = normalize(old_source)[0] if old_source else []
    numbers: dict[int, int] = {}
    for index in _changed(_physical(old_source) if old_source else [], _physical(new_source)):
        numbers.setdefault(owner[index], index)
    for index, logical in enumerate(owner):
        numbers.setdefault(logical, index)
    for logical in _changed(old, new):
        yield numbers[logical] + 1, new[logical]


def _removed_guard(tree: Path, base: dict, path: str, status: str) -> bool:
    """A deleted line was a guard."""
    if status == "D" or path not in base:
        return False
    old, new = (normalize(text)[0] for text in _sources(tree, base, path))
    return any(_GUARD.match(old[index]) for index in _changed(old, new, removed=True))


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


def _literal_text(text: str) -> str:
    """Text as the compiler sees literals."""
    return _ESCAPE.sub(_unescape, _ADJACENT_LITERALS.sub("", text))


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
    if any(not _SAFE_PRAGMA.match(text, hit.start()) for hit in _PRAGMA.finditer(text)):
        yield "pragma"
    if _scs_count(text):
        yield "measurement_access"
    yield from (rule for rule, pattern in LINE_RULES if pattern.search(text))


def _grandfathered(tree: Path, path: str) -> bool:
    """File already holds a rule hit."""
    lines = normalize((tree / path).read_text(encoding="utf-8", errors="replace"))[0]
    return any(next(line_rules(text), None) for text in lines)


def rule_counts(source: str) -> Counter:
    """Rule matches over a whole file."""
    text = _literal_text("\n".join(normalize(source)[0]))
    counts: Counter = Counter()
    found = _ATTRIBUTE.findall(text)
    # Unparsed attribute spellings count as unsafe.
    counts["attribute"] = sum(_unsafe_attribute("".join(groups).join(("__attribute__((", "))"))) for groups in found)
    counts["attribute"] += max(0, len(_ATTRIBUTE_HINT.findall(text)) - len(found))
    counts["pragma"] = sum(not _SAFE_PRAGMA.match(text, hit.start()) for hit in _PRAGMA.finditer(text))
    for rule, pattern in LINE_RULES:
        counts[rule] += len(pattern.findall(text))
    counts["measurement_access"] += _scs_count(text)
    return counts


def _closure(seed: set[str], edges: dict[str, set[str]]) -> set[str]:
    """Names reachable from seed."""
    found, todo = set(seed), list(seed)
    while todo:
        for name in edges.get(todo.pop(), ()):
            if name not in found:
                found.add(name)
                todo.append(name)
    return found


def _tree_macros(tree: Path) -> tuple[set[str], set[str]]:
    """Names conditionals reach; pasting macros."""
    tested: set[str] = set()
    bodies: dict[str, set[str]] = {}
    users: dict[str, set[str]] = {}
    pasting: set[str] = set()
    for top in ALLOWED_DIRS:
        for path in sorted((tree / top).rglob("*")):
            if path.is_symlink() or not path.is_file() or not path.name.endswith(ALLOWED_SUFFIXES):
                continue
            for code in normalize(path.read_text(encoding="utf-8", errors="replace"))[0]:
                if match := _CONDITION.match(code):
                    tested |= set(_IDENT.findall(match.group(1)))
                elif match := _NAME_TEST.match(code):
                    tested.add(match.group(1))
                elif match := _DEFINE.match(code):
                    name, body = match.group(1), match.group(3)
                    bodies.setdefault(name, set()).update(_IDENT.findall(body))
                    for ref in _IDENT.findall(body):
                        users.setdefault(ref, set()).add(name)
                    if _PASTES.search(body):
                        pasting.add(name)
    # Users of pasting macros paste too.
    return _closure(tested, bodies), _closure(pasting, users)


def _guard_name(lines: list[str]) -> Optional[str]:
    """The file's include guard, if any."""
    code = [line for line in lines if line.strip()]
    if len(code) < 2 or not (match := _IFNDEF.match(code[0])):
        return None
    define = _DEFINE.match(code[1])
    if define and define.group(1) == match.group(1) and not define.group(2) and not define.group(3).strip():
        return match.group(1)
    return None


def removed_lines(tree: Path, base: dict, path: str, status: str) -> list[tuple[int, str]]:
    """Base logical lines the candidate dropped."""
    if path not in base:
        return []
    old_source = _git(tree, "cat-file", "blob", base[path][1]).decode(errors="replace")
    old, owner = normalize(old_source)
    new = [] if status == "D" else normalize((tree / path).read_text(encoding="utf-8", errors="replace"))[0]
    first = {logical: index for index, logical in reversed(list(enumerate(owner)))}
    return [(first[logical] + 1, old[logical]) for logical in _changed(old, new, removed=True)]


def probe_findings(tree: Path, added: dict[str, list[tuple[int, str]]],
                   removed: dict[str, list[tuple[int, str]]]) -> Iterator[dict]:
    """Edits that can flip a build-only branch.

    gcc -E drops code under a false #if, and the scan configs need not
    match the real build, so code behind a probe stays unseen.
    """
    if not any(added.values()) and not any(removed.values()):
        return
    reached, pasting = _tree_macros(tree)
    for path in sorted(added.keys() | removed.keys()):
        guard = None
        if (tree / path).is_file():
            guard = _guard_name(normalize((tree / path).read_text(encoding="utf-8", errors="replace"))[0])
        for line_no, code in added.get(path, []):
            define = _DEFINE.match(code)
            if (match := _CONDITION.match(code)) and _MACRO_CALL.search(match.group(1)):
                yield {"rule": "build_probe", "path": path, "line": line_no, "text": "macro call in #if"}
            elif (match := _DEFINE_UNDEF.match(code)) and match.group(1) in reached and match.group(1) != guard:
                yield {"rule": "build_probe", "path": path, "line": line_no, "text": f"conditional reaches {match.group(1)}"}
            elif (names := set(_IDENT.findall(code)) & pasting - {define.group(1) if define else ""}):
                yield {"rule": "build_probe", "path": path, "line": line_no, "text": f"uses pasting macro {min(names)}"}
        for line_no, code in removed.get(path, []):
            if (match := _DEFINE_UNDEF.match(code)) and match.group(1) in reached:
                yield {"rule": "build_probe", "path": path, "line": line_no,
                       "text": f"removed {match.group(1)}, base line {line_no}"}


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
    removed_by_path: dict[str, list[tuple[int, str]]] = {}
    for path, status in sorted(changes.items()):
        hits = list(path_findings(path, status, tree))
        findings += hits
        if hits:
            continue
        added = added_by_path[path] = list(added_lines(tree, base_blobs, path, status))
        removed_by_path[path] = removed_lines(tree, base_blobs, path, status)
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
    findings += probe_findings(tree, added_by_path, removed_by_path)
    findings += ({"rule": "hidden_index_entry", "path": path, "message": "skip-worktree or assume-unchanged set"}
                 for path in hidden_entries(tree))
    if any(path.startswith(ALLOWED_DIRS) for path in changes):
        tops = sorted({path.split("/", 1)[0] + "/" for path in base_blobs if path.startswith(ALLOWED_DIRS)})
        archive = _git(tree, "archive", commit, "--", *tops) if tops else _EMPTY_TAR
        findings += preprocess_findings(tree, archive, rule_counts)
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
