"""Resolve an ArgumentPool against the kernel contract into exactly what the harness
template prints: which providers the prototype needs, and the bound sizer and kernel calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

from helia_core_tester.contract.bind import bind, takes
from helia_core_tester.contract.ir import ContractSet
from helia_core_tester.contract.render import render_call, require_bound_symbol
from helia_core_tester.generation.harness.model import ArgumentPool, HarnessError, Provider, render_declaration


@dataclass(frozen=True)
class ProviderBlock:
    """An active provider, its declarations already rendered."""

    provider: Provider
    declarations: Sequence[str]

    @property
    def buffers(self):
        return self.provider.buffers

    @property
    def setup(self) -> str:
        return self.provider.setup


@dataclass(frozen=True)
class HarnessPlan:
    header_declarations: Sequence[str]
    source_declarations: Sequence[str]
    providers: Sequence[ProviderBlock]
    sizer_fn: Optional[str]
    sizer_call: Optional[str]
    scratch_bytes: int
    run_call: str
    bench_call: str
    checks: Sequence[tuple[str, str, str, int]] = ()
    size_queries: Sequence[tuple[str, str, str, str]] = ()
    fault_kind: Optional[str] = None
    fault_declarations: Sequence[str] = ()
    fault_setup: str = ""
    no_scratch: bool = False
    inputs: Sequence[tuple[str, str, str]] = ()
    context_setup: str = ""
    uses_ctx: bool = True
    scratch_buffer: bool = True
    local_prototype: str = ""
    prototype_from: str = ""
    void_return: bool = False
    run_calls: Sequence[str] = ()


def plan_harness(pool: ArgumentPool, *, kernel_fn: str, sizer_fn: Optional[str], scratch_bytes: Optional[int],
                 contracts: ContractSet, indent: str = "        ") -> HarnessPlan:
    """Bind `kernel_fn` (and `sizer_fn`) from `pool`; exactly one of sizer_fn and scratch_bytes."""
    pool.validate()
    if (sizer_fn is None) == (scratch_bytes is None):
        raise HarnessError(f"{pool.name}: give either a scratch query or entry_scratch bytes, not both or neither")
    local_prototype = ""
    if pool.prototype_from:
        if contracts.find(kernel_fn) is not None:
            raise HarnessError(f"{pool.name}: {kernel_fn} is public; bind it from the contract, not from "
                               f"{pool.prototype_from}'s prototype")
        kernel = replace(require_bound_symbol(contracts, pool.prototype_from), name=kernel_fn)
        local_prototype = render_prototype(kernel)
    else:
        kernel = require_bound_symbol(contracts, kernel_fn)
    # A pool that owns its context (it passes NULL, or a context of its own) keeps the harness
    # from declaring and populating one.
    uses_ctx = takes(kernel, "ctx") and not pool.owns_ctx
    if not pool.scratch_buffer:
        if sizer_fn is not None or scratch_bytes:
            raise HarnessError(f"{pool.name}: a case without a scratch buffer cannot query or claim scratch")
        if uses_ctx and not pool.no_scratch:
            raise HarnessError(f"{pool.name}: {kernel_fn} takes a context but the case has no scratch buffer; "
                               "set no_scratch so the context is empty")
    elif not uses_ctx:
        raise HarnessError(f"{pool.name}: {kernel_fn} takes no context, so the case has no use for a scratch buffer")
    providers = [p for p in pool.providers if any(takes(kernel, name) for name in p.names)]
    values = dict(pool.values)
    values.update({name: p.expr for p in providers for name in p.names})

    fault = pool.fault
    if fault is not None:
        for param in (*fault.values, *fault.requires):
            if not takes(kernel, param):
                raise HarnessError(f"{pool.name}: fault {fault.kind!r} edits {param!r}, which {kernel.name} does not take")

    inputs = pool.harness_inputs
    if kernel.returns.strip() == "void" and (fault is not None or pool.checks):
        raise HarnessError(f"{pool.name}: {kernel.name} returns void, so a fault edit or rule check has no status "
                           "to assert")
    if pool.calls and kernel.returns.strip() == "void":
        raise HarnessError(f"{pool.name}: a call list needs a status to stop on, but {kernel.name} returns void")

    def call(bench: bool, overrides: Mapping[str, str] = {}) -> str:
        site = {**values, pool.output_param: f"{pool.name}_output" if bench else "output"}
        site.update({i.param: i.array if bench else i.local for i in inputs})
        if fault is not None:
            site.update(fault.values)
        for param in overrides:
            if not takes(kernel, param):
                raise HarnessError(f"{pool.name}: a call overrides {param!r}, which {kernel.name} does not take")
        site.update(overrides)
        return render_call(kernel, bind(kernel, site), indent=indent)

    checks = []
    for check in pool.checks:
        rule = require_bound_symbol(contracts, check.fn)
        checks.append((check.result_var, render_call(rule, bind(rule, values), indent=indent), rule.name,
                       int(check.expected)))

    size_queries = []
    for provider in providers:
        query = provider.size_query
        if query is None:
            continue
        decl = require_bound_symbol(contracts, query.fn)
        if decl.kind != "sizer":
            raise HarnessError(f"{pool.name}: {query.fn} is a {decl.kind}, not a scratch-size query")
        size_queries.append((query.result_var, render_call(decl, bind(decl, values), indent=indent), decl.name,
                             query.capacity))

    sizer_call = None
    if sizer_fn is not None:
        sizer = require_bound_symbol(contracts, sizer_fn)
        if sizer.kind != "sizer":
            raise HarnessError(f"{pool.name}: {sizer_fn} is a {sizer.kind}, not a scratch-size query")
        sizer_call = render_call(sizer, bind(sizer, values), indent=indent)
    return HarnessPlan(
        header_declarations=[render_declaration(d) for d in pool.header],
        source_declarations=[render_declaration(d) for d in pool.source],
        providers=[ProviderBlock(p, [render_declaration(d) for d in p.declarations]) for p in providers],
        sizer_fn=sizer_fn,
        sizer_call=sizer_call,
        scratch_bytes=int(scratch_bytes or 0),
        run_call=call(False),
        bench_call=call(True),
        checks=checks,
        size_queries=size_queries,
        fault_kind=fault.kind if fault else None,
        fault_declarations=[render_declaration(d) for d in fault.declarations] if fault else [],
        fault_setup=fault.setup if fault else "",
        no_scratch=bool(pool.no_scratch or (fault and fault.no_scratch)),
        inputs=[(i.local, i.ctype or "", i.array) for i in inputs],
        context_setup=pool.context_setup,
        uses_ctx=uses_ctx,
        scratch_buffer=pool.scratch_buffer,
        local_prototype=local_prototype,
        prototype_from=pool.prototype_from or "",
        void_return=kernel.returns.strip() == "void",
        run_calls=[call(False, overrides) for overrides in pool.calls or ()],
    )


def render_prototype(decl) -> str:
    """The C declaration of `decl` (a public function's prototype under another name)."""
    def param(p) -> str:
        c_type = p.c_type.strip()
        joiner = "" if c_type.endswith("*") else " "
        return f"{c_type}{joiner}{p.name}{p.extent or ''}"

    params = ", ".join(param(p) for p in decl.params) or "void"
    return f"{decl.returns} {decl.name}({params});"
