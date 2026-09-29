"""Data an operator hands the generic harness: declarations, buffers, providers and the
call values, described as values rather than template text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


class HarnessError(ValueError):
    """An ArgumentPool that cannot render a well-formed harness."""


@dataclass(frozen=True)
class ArrayLiteral:
    """A preformatted C element list (the generator's formatted array)."""

    body: str


Initializer = Union[str, ArrayLiteral, Mapping[str, "Initializer"]]


@dataclass(frozen=True)
class Declaration:
    """A file-scope C declaration: `storage ctype name[] = init;`."""

    name: str
    ctype: str
    init: Optional[Initializer] = None
    storage: str = "static const"
    array: bool = False
    comment: str = ""


@dataclass(frozen=True)
class GuardedBuffer:
    """A writable buffer framed by HELIA guard canaries, sized by a `#define`."""

    name: str
    ctype: str
    count_macro: str
    count_value: str
    label: str = ""


@dataclass(frozen=True)
class Provider:
    """A value that needs set-up (weight sums, vector sums, LUTs). Emitted only when the bound
    prototype takes `param`: its declarations and buffers at file scope, its `setup` statements
    before the kernel call, its buffers guarded and checked."""

    param: str
    expr: str
    declarations: Sequence[Declaration] = ()
    buffers: Sequence[GuardedBuffer] = ()
    setup: str = ""


@dataclass(frozen=True)
class ArgumentPool:
    """Everything one generated case can pass to a kernel or its scratch query."""

    name: str
    values: Mapping[str, str]
    header: Sequence[Declaration] = ()
    source: Sequence[Declaration] = ()
    providers: Sequence[Provider] = ()
    input_param: str = "input_data"
    output_param: str = "output_data"
    output_count: str = "0"

    def validate(self) -> None:
        names: set[str] = set()
        for decl in [*self.header, *self.source, *(d for p in self.providers for d in p.declarations)]:
            if not _IDENT_RE.match(decl.name):
                raise HarnessError(f"{self.name}: declaration name {decl.name!r} is not a C identifier")
            if decl.name in names:
                raise HarnessError(f"{self.name}: {decl.name} is declared twice")
            names.add(decl.name)
        for buffer in (b for p in self.providers for b in p.buffers):
            if buffer.name in names:
                raise HarnessError(f"{self.name}: {buffer.name} is declared twice")
            names.add(buffer.name)
        params = set(self.values)
        for provider in self.providers:
            if provider.param in params:
                raise HarnessError(f"{self.name}: {provider.param} is both a pool value and a provider")
            params.add(provider.param)
        for param in (self.input_param, self.output_param):
            if param in self.values:
                raise HarnessError(f"{self.name}: {param} is supplied per call site, not as a pool value")
        for param, expr in self.values.items():
            if not isinstance(expr, str) or not expr.strip():
                raise HarnessError(f"{self.name}: pool value {param!r} is empty")


def _render_init(value: Initializer, depth: int = 0) -> str:
    if isinstance(value, ArrayLiteral):
        return "{\n" + value.body + "\n}"
    if isinstance(value, Mapping):
        if depth == 0:
            inner = ",\n".join(f"    .{key} = {_render_init(item, depth + 1)}" for key, item in value.items())
            return "{\n" + inner + "\n}"
        return "{" + ", ".join(f".{key} = {_render_init(item, depth + 1)}" for key, item in value.items()) + "}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_declaration(decl: Declaration) -> str:
    head = " ".join(part for part in (decl.storage, decl.ctype, decl.name + ("[]" if decl.array else "")) if part)
    text = head + ";" if decl.init is None else f"{head} = {_render_init(decl.init)};"
    return (f"// {decl.comment}\n" if decl.comment else "") + text
