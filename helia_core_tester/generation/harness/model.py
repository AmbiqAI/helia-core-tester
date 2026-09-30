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
    extent: Optional[str] = None


@dataclass(frozen=True)
class HarnessInput:
    """One input tensor of the case: the kernel parameter it answers to, the `_run` argument
    that carries it, and the header array the test passes in."""

    param: str
    local: str
    array: str
    ctype: Optional[str] = None


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
class OutputSlot:
    """One of several kernel outputs: `<name>_output` holds `count` elements (one element of
    storage when zero), is compared with `<name>_expected_output`, and when empty must stay
    untouched."""

    name: str
    count: int


@dataclass(frozen=True)
class RuleCheck:
    """A public predicate the case asserts before validating outputs (for example a planar
    rule): bound from the pool like the kernel, its answer must equal `expected`."""

    fn: str
    expected: int
    result_var: str


@dataclass(frozen=True)
class SizeQuery:
    """A provider's own scratch query: bound from the pool like the main sizer, its answer is
    held in `result_var` and checked (negative sentinel, then against `capacity`) with the
    main sizer's answer, before any context is populated."""

    fn: str
    result_var: str
    capacity: str


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
    aliases: Sequence[str] = ()
    size_query: Optional[SizeQuery] = None

    @property
    def names(self) -> tuple[str, ...]:
        return (self.param, *self.aliases)


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
    inputs: Optional[Sequence[HarnessInput]] = None
    no_scratch: bool = False
    context_setup: str = ""
    scratch_buffer: bool = True
    prototype_from: Optional[str] = None
    test_prologue: str = ""
    extra_checks: str = ""
    validation: Optional[str] = None
    output_poison: bool = False
    output_untouched: bool = False
    owns_ctx: bool = False
    includes: Sequence[str] = ()
    guarded: Sequence[GuardedBuffer] = ()
    output_capacity: Optional[str] = None
    output_ctype: Optional[str] = None
    # An ordered list of kernel calls, each a mapping of parameter name to a C expression in terms
    # of the _run locals that replaces the call-site value for that call; the run returns the first
    # failing status (a void kernel is called as statements and the run reports success).
    # Benchmarks and fault edits are not defined over a call list.
    calls: Optional[Sequence[Mapping[str, str]]] = None
    # Several outputs the kernel reaches through a pointer array the pool declares: the harness
    # then has no `output` local and guards, checks and validates each slot in turn.
    outputs: Sequence[OutputSlot] = ()
    # C statements the run executes before the kernel call (over the _run locals and the pool's
    # declarations) and after every call succeeded; a post-pass forces the status-checked form.
    pre_call: str = ""
    post_call: str = ""

    @property
    def harness_inputs(self) -> Sequence[HarnessInput]:
        if self.inputs is None:
            return (HarnessInput(self.input_param, "input", f"{self.name}_input"),)
        return tuple(self.inputs)

    def validate(self) -> None:
        names: set[str] = set()
        for decl in [*self.header, *self.source, *(d for p in self.providers for d in p.declarations)]:
            if not _IDENT_RE.match(decl.name):
                raise HarnessError(f"{self.name}: declaration name {decl.name!r} is not a C identifier")
            if decl.name in names:
                raise HarnessError(f"{self.name}: {decl.name} is declared twice")
            names.add(decl.name)
        harness_owned = [f"{self.name}_output"] + ([f"{self.name}_buffer"] if self.scratch_buffer else [])
        for buffer in self.guarded:
            if buffer.name in harness_owned:
                raise HarnessError(f"{self.name}: guarded buffer {buffer.name} is the harness's own buffer")
        for buffer in (*self.guarded, *(b for p in self.providers for b in p.buffers)):
            if buffer.name in names:
                raise HarnessError(f"{self.name}: {buffer.name} is declared twice")
            names.add(buffer.name)
        params = set(self.values)
        for provider in self.providers:
            for param in provider.names:
                if param in params:
                    raise HarnessError(f"{self.name}: {param} is both a pool value and a provider")
                params.add(param)
            query = provider.size_query
            if query is not None and (not _IDENT_RE.match(query.result_var) or query.result_var in names
                                      or query.result_var == "required_buffer_size"):
                raise HarnessError(f"{self.name}: size query variable {query.result_var!r} is not a free C identifier")
            if query is not None:
                names.add(query.result_var)
        call_site = [i.param for i in self.harness_inputs] + ([] if self.outputs else [self.output_param])
        for param in call_site:
            if param in self.values:
                raise HarnessError(f"{self.name}: {param} is supplied per call site, not as a pool value")
        if len(set(call_site)) != len(call_site):
            raise HarnessError(f"{self.name}: call-site parameters {call_site} repeat")
        locals_ = [i.local for i in self.harness_inputs] + ([] if self.outputs else ["output"])
        if len(set(locals_)) != len(locals_) or not all(_IDENT_RE.match(n) for n in locals_):
            raise HarnessError(f"{self.name}: _run argument names {locals_} must be distinct C identifiers")
        if self.no_scratch and self.context_setup.strip():
            raise HarnessError(f"{self.name}: no_scratch and context_setup both set the context")
        for include in self.includes:
            if not re.fullmatch(r'<[\w./]+>|"[\w./]+"', include):
                raise HarnessError(f"{self.name}: include {include!r} is not <header> or \"header\"")
        if self.owns_ctx and "ctx" not in self.values:
            raise HarnessError(f"{self.name}: owns_ctx needs the pool to supply ctx")
        if self.output_untouched and not self.output_poison:
            raise HarnessError(f"{self.name}: an untouched-output check needs the output poisoned first")
        if not self.scratch_buffer and self.context_setup.strip():
            raise HarnessError(f"{self.name}: context_setup needs the scratch buffer it replaces")
        for param, expr in self.values.items():
            if not isinstance(expr, str) or not expr.strip():
                raise HarnessError(f"{self.name}: pool value {param!r} is empty")
        if self.outputs:
            if self.benchmark or self.fault is not None or self.calls is not None:
                raise HarnessError(f"{self.name}: output slots have no benchmark, fault or call-list form")
            for field in ("output_poison", "output_untouched", "output_capacity", "output_ctype", "validation"):
                if getattr(self, field):
                    raise HarnessError(f"{self.name}: {field} describes the single output, which output slots replace")
            slots = [slot.name for slot in self.outputs]
            if len(set(slots)) != len(slots) or not all(_IDENT_RE.match(n) for n in slots):
                raise HarnessError(f"{self.name}: output slots {slots} must be distinct C identifiers")
            for slot in slots:
                if f"{slot}_output" in names:
                    raise HarnessError(f"{self.name}: output slot {slot} collides with the declaration {slot}_output")
            if any(slot.count < 0 for slot in self.outputs):
                raise HarnessError(f"{self.name}: an output slot cannot hold a negative element count")
        if (self.pre_call.strip() or self.post_call.strip()) and (self.benchmark or self.fault is not None):
            raise HarnessError(f"{self.name}: pre- and post-passes have no benchmark or fault form")
        if self.calls is not None:
            if not self.calls:
                raise HarnessError(f"{self.name}: a call list needs at least one call")
            if self.benchmark or self.fault is not None:
                raise HarnessError(f"{self.name}: a call list has no benchmark or fault form")
            for index, call in enumerate(self.calls):
                for param, expr in call.items():
                    if not isinstance(expr, str) or not expr.strip():
                        raise HarnessError(f"{self.name}: call {index} gives {param!r} an empty expression")
        if self.fault is not None:
            fault = self.fault
            if not (fault.values or fault.setup.strip() or fault.no_scratch):
                raise HarnessError(f"{self.name}: fault {fault.kind!r} edits nothing")
            for decl in fault.declarations:
                if not _IDENT_RE.match(decl.name) or decl.name in names:
                    raise HarnessError(f"{self.name}: fault declaration {decl.name!r} is not a free C identifier")
                names.add(decl.name)
            supplied = params | set(call_site)
            for param, expr in fault.values.items():
                if param not in supplied:
                    raise HarnessError(f"{self.name}: fault {fault.kind!r} edits {param!r}, which the pool does not supply")
                if not isinstance(expr, str) or not expr.strip():
                    raise HarnessError(f"{self.name}: fault {fault.kind!r} value for {param!r} is empty")
        for check in self.checks:
            if not _IDENT_RE.match(check.result_var) or check.result_var in names:
                raise HarnessError(f"{self.name}: rule check variable {check.result_var!r} is not a free C identifier")
            names.add(check.result_var)


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
    suffix = f"[{decl.extent or ''}]" if decl.array else ""
    head = " ".join(part for part in (decl.storage, decl.ctype, decl.name + suffix) if part)
    text = head + ";" if decl.init is None else f"{head} = {_render_init(decl.init)};"
    return (f"// {decl.comment}\n" if decl.comment else "") + text
