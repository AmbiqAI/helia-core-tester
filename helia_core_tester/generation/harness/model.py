"""Data an operator hands the generic harness: declarations, buffers, providers and the
call values, described as values rather than template text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    """A writable buffer framed by HELIA guard canaries. `count` is its element count: a
    macro that `count_value` defines, or a literal expression when `count_value` is None."""

    name: str
    ctype: str
    count: str
    count_value: Optional[str] = None
    label: str = ""


@dataclass(frozen=True)
class RuleCheck:
    """A public predicate the case asserts before validating outputs (for example a planar
    rule): bound from the pool like the kernel, its answer must equal `expected`."""

    fn: str
    expected: int
    result_var: str


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
class FaultEdit:
    """A fault case as an edit of the pool: `values` replace what the kernel call (only) is
    passed, `declarations` and `setup` build the replacements after the providers run, and
    `no_scratch` hands the kernel a NULL scratch buffer. `requires` names parameters the edit
    assumes the kernel takes (the context an edit clears, a provider whose context the setup
    touches). The sizer and rule checks still see the unedited values, so the case differs from
    its passing sibling in the faulted argument."""

    kind: str
    values: Mapping[str, str] = field(default_factory=dict)
    declarations: Sequence[Declaration] = ()
    setup: str = ""
    no_scratch: bool = False
    requires: Sequence[str] = ()


@dataclass(frozen=True)
class ArgumentPool:
    """Everything one generated case can pass to a kernel or its scratch query."""

    name: str
    values: Mapping[str, str]
    header: Sequence[Declaration] = ()
    source: Sequence[Declaration] = ()
    providers: Sequence[Provider] = ()
    checks: Sequence[RuleCheck] = ()
    input_param: str = "input_data"
    output_param: str = "output_data"
    output_count: str = "0"
    benchmark: bool = True
    fault: Optional[FaultEdit] = None

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
        if self.fault is not None:
            fault = self.fault
            if not (fault.values or fault.setup.strip() or fault.no_scratch):
                raise HarnessError(f"{self.name}: fault {fault.kind!r} edits nothing")
            for decl in fault.declarations:
                if not _IDENT_RE.match(decl.name) or decl.name in names:
                    raise HarnessError(f"{self.name}: fault declaration {decl.name!r} is not a free C identifier")
                names.add(decl.name)
            supplied = params | {self.input_param, self.output_param}
            for param, expr in fault.values.items():
                if param not in supplied:
                    raise HarnessError(f"{self.name}: fault {fault.kind!r} edits {param!r}, which the pool does not supply")
                if not isinstance(expr, str) or not expr.strip():
                    raise HarnessError(f"{self.name}: fault {fault.kind!r} value for {param!r} is empty")
        for check in self.checks:
            if not _IDENT_RE.match(check.result_var) or check.result_var in names:
                raise HarnessError(f"{self.name}: rule check variable {check.result_var!r} is not a free C identifier")


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
