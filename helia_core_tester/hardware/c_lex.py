"""C translation phases 1-3, enough for line rules.

`strip_comments` splices backslash-newlines and replaces each comment
with one space, keeping a block comment's newlines. String and char
literals and #include header names stay as they are, so a `/*` or `//`
inside them is not a comment.
"""

from __future__ import annotations

import re

_NEWLINE = re.compile(r"\r\n|\r|\n")
_SPLICE = re.compile(r"\\[ \t\f\v]*$")
# Header names, code, literals, comments.
_LEX = re.compile(
    r"(?P<header>(?:#|%:)[ \t]*include[ \t]*<[^>\n]*>)"
    r"|(?P<code>(?:[^\W\d]\w*|\.?\d(?:[eEpP][+-]|'\w|[\w.])*|[^\"'/\w#%]|/(?![*/]))+)"
    r"|(?P<literal>\"(?:\\[^\n]|[^\"\\\n])*\"?|'(?:\\[^\n]|[^'\\\n])*'?)"
    r"|(?P<block>/\*.*?(?:\*/|\Z))"
    r"|(?P<line>//[^\n]*)",
    re.DOTALL,
)


def _blank(match: re.Match) -> str:
    if match.lastgroup == "block":
        return "\n" * match.group().count("\n") + " "
    return " " if match.lastgroup == "line" else match.group()


def strip_comments(source: str) -> str:
    """Source with lines spliced, comments blanked."""
    logical, parts = [], []
    for line in _NEWLINE.split(source):
        spliced = _SPLICE.sub("", line)
        parts.append(spliced)
        if spliced == line:
            logical.append("".join(parts))
            parts = []
    if parts:
        logical.append("".join(parts))
    return _LEX.sub(_blank, "\n".join(logical))
