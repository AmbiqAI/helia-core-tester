"""
Patch application for mutation scoring (issue #76).

Mutants are applied to a *copy* of the ns-cmsis-nn tree, never to the
user's checkout. Application is transactional per mutant: original file
contents are captured before editing and restored afterwards, so the copied
tree returns to pristine state between mutants. A patch that does not match
its target (source drifted since the catalog was written) raises
MutantApplyError -- the runner reports it loudly and fails the run; nothing
is silently skipped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from helia_core_tester.mutation.catalog import Mutant


class MutantApplyError(RuntimeError):
    """A catalogued mutant no longer matches the target source."""


class AppliedMutant:
    """Context manager that applies a mutant to a tree and restores it on exit."""

    def __init__(self, tree_root: Path, mutant: Mutant):
        self.tree_root = Path(tree_root)
        self.mutant = mutant
        self._originals: Dict[Path, str] = {}

    def __enter__(self) -> "AppliedMutant":
        try:
            self._apply()
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._restore()

    def _apply(self) -> None:
        for edit in self.mutant.edits:
            path = self.tree_root / edit.relpath
            if not path.is_file():
                raise MutantApplyError(
                    f"mutant '{self.mutant.mutant_id}': target file not found: {edit.relpath}"
                )
            content = path.read_text()
            if edit.regex:
                new_content, n = re.subn(edit.pattern, edit.replacement, content)
            else:
                n = content.count(edit.pattern)
                new_content = content.replace(edit.pattern, edit.replacement)
            if n != edit.count:
                raise MutantApplyError(
                    f"mutant '{self.mutant.mutant_id}': pattern matched {n} time(s) in "
                    f"{edit.relpath}, expected exactly {edit.count}. The kernel source has "
                    f"drifted; update the catalog entry."
                )
            if path not in self._originals:
                self._originals[path] = content
            path.write_text(new_content)

    def _restore(self) -> None:
        for path, content in self._originals.items():
            path.write_text(content)
        self._originals.clear()


def verify_pristine(tree_root: Path, mutants) -> None:
    """Assert no mutant marker comments remain anywhere in the copied tree."""
    markers = ["/* MUTANT "]
    for path in Path(tree_root).rglob("*.[ch]"):
        text = path.read_text(errors="replace")
        for marker in markers:
            if marker in text:
                raise RuntimeError(f"tree not pristine: {path} still contains '{marker}'")
