"""Check a candidate kernel diff stays in bounds.

A candidate may change kernel C, headers and assembly under Source/ and
Include/, nothing else. Rules, each a finding in the JSON report:

- outside_allowlist: any path outside Source/ or Include/ (Tests/,
  cmake/, nsx/, CMakeLists, scripts, build files).
- frozen_file: public API headers the adapters compile against, and
  arm_nntables.c, which the s16 golden generator reads.
- file_type: Source/Include files that are not .c/.h/.s/.S, and any
  CMakeLists.txt or *.cmake.
- symlink: a symlink in the candidate change set.
- special_section: added lines that place code or data in a named
  section or TCM (section attributes, section pragmas, .section,
  ITCM/RAMFUNC-style macros).
- build_flags: added optimize/target pragmas or attributes.
- harness_reference: added lines naming harness or golden internals
  (golden, hct_, hctp, benchmark_server, UnitTest, RefactoredTestGen).
- include_escape: #include with an absolute path or "..".

Literal tensor shapes from descriptors are not grepped: too many false
hits on common dims. The diff is the working tree (tracked edits plus
untracked files, ignored ones too under copied trees) against --base.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import typer

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
# Trees the build or generation copies.
_COPIED = ("Source", "Include", "cmake", "nsx", "Tests")
LINE_RULES = (
    ("special_section", re.compile(
        r"__attribute__\s*\(\(.*\bsection\b|#\s*pragma\s+(?:arm\s+section|section|location|GCC\s+section)"
        r"|\.section\b|\b(?:ITCM|DTCM|__RAMFUNC|RAMFUNC|AM_SHARED_RW|NS_PUT_IN_TCM)\b",
    )),
    ("build_flags", re.compile(
        r"#\s*pragma\s+(?:GCC|clang)\s+(?:optimize|push_options|target)|#\s*pragma\s+O[0-3s]\b"
        r"|__attribute__\s*\(\(.*\b(?:optimize|target)\s*\(",
    )),
    ("harness_reference", re.compile(
        r"golden|\bhct_|\bhctp|benchmark_server|unittest|RefactoredTestGen", re.IGNORECASE,
    )),
    ("include_escape", re.compile(r'#\s*include\s*["<](?:/|[^">]*\.\.)')),
)


class CheckError(RuntimeError):
    """The check could not run."""


def _git(tree: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise CheckError(done.stderr.strip() or f"git {args[0]} failed")
    return done.stdout


def changed_paths(tree: Path, base: str) -> dict[str, str]:
    """Path to status, vs base, untracked included."""
    out = _git(tree, "diff", "--name-status", "--no-renames", "-z", base)
    parts = [part for part in out.split("\0") if part]
    changes = dict(zip(parts[1::2], parts[0::2]))
    untracked = _git(tree, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    # Copies include ignored files.
    untracked += _git(tree, "ls-files", "--others", "-z", "--", *_COPIED).split("\0")
    changes.update((path, "?") for path in untracked if path and path not in changes)
    return changes


def path_findings(path: str, status: str, tree: Path) -> Iterator[dict]:
    """Rule hits from the path alone."""
    name = path.rsplit("/", 1)[-1]
    if not path.startswith(ALLOWED_DIRS):
        yield {"rule": "outside_allowlist", "path": path, "message": "only Source/ and Include/ may change"}
    elif path in FROZEN_FILES:
        yield {"rule": "frozen_file", "path": path, "message": "harness reads this file"}
    elif name == "CMakeLists.txt" or name.endswith(".cmake") or not name.endswith(ALLOWED_SUFFIXES):
        yield {"rule": "file_type", "path": path, "message": "only .c .h .s .S files"}
    if status != "D" and (tree / path).is_symlink():
        yield {"rule": "symlink", "path": path, "message": "symlinks are not allowed"}


def added_lines(tree: Path, base: str, path: str, status: str) -> Iterator[tuple[Optional[int], str]]:
    """Added lines with new line numbers."""
    if status == "D" or (tree / path).is_symlink():
        return
    if status == "?":
        text = (tree / path).read_text(encoding="utf-8", errors="replace")
        yield from enumerate(text.splitlines(), start=1)
        return
    line_no: Optional[int] = None
    for line in _git(tree, "diff", "-U0", "--no-color", "--no-renames", base, "--", path).splitlines():
        hunk = re.match(r"@@ -\S+ \+(\d+)", line)
        if hunk:
            line_no = int(hunk.group(1))
        elif line.startswith("+") and not line.startswith("+++") and line_no is not None:
            yield line_no, line[1:]
            line_no += 1


def check_candidate(tree: Path, base: str) -> dict:
    """The JSON report for one candidate."""
    tree = tree.resolve()
    base_commit = _git(tree, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
    changes = changed_paths(tree, base_commit)
    findings: list[dict] = []
    for path, status in sorted(changes.items()):
        hits = list(path_findings(path, status, tree))
        findings += hits
        if hits:
            continue
        for line_no, text in added_lines(tree, base_commit, path, status):
            for rule, pattern in LINE_RULES:
                if pattern.search(text):
                    findings.append({"rule": rule, "path": path, "line": line_no, "text": text.strip()[:200]})
    return {
        "schema": "hct.candidate_check",
        "schema_version": 1,
        "tree": str(tree),
        "base": base,
        "base_commit": base_commit,
        "ok": not findings,
        "files": [{"path": path, "status": status} for path, status in sorted(changes.items())],
        "findings": findings,
    }


candidate_app = typer.Typer(help="Check candidate kernel trees.", no_args_is_help=True)


@candidate_app.command("check")
def check_command(
    tree: Path = typer.Argument(..., exists=True, file_okay=False, help="Candidate ns-cmsis-nn checkout."),
    base: str = typer.Option(..., "--base", help="Base ref the candidate started from."),
) -> None:
    """Fail when the candidate diff leaves Source/Include."""
    try:
        report = check_candidate(tree, base)
    except CheckError as exc:
        typer.echo(json.dumps({"ok": False, "error": str(exc)}))
        raise typer.Exit(2)
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(0 if report["ok"] else 1)
