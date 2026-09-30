"""Resolve an ArgumentPool against the kernel contract into exactly what the harness
template prints: which providers the prototype needs, and the bound sizer and kernel calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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
    fault_kind: Optional[str] = None
    fault_declarations: Sequence[str] = ()
    fault_setup: str = ""
    no_scratch: bool = False


def plan_harness(pool: ArgumentPool, *, kernel_fn: str, sizer_fn: Optional[str], scratch_bytes: Optional[int],
                 contracts: ContractSet, indent: str = "        ") -> HarnessPlan:
    """Bind `kernel_fn` (and `sizer_fn`) from `pool`; exactly one of sizer_fn and scratch_bytes."""
    pool.validate()
    if (sizer_fn is None) == (scratch_bytes is None):
        raise HarnessError(f"{pool.name}: give either a scratch query or entry_scratch bytes, not both or neither")
    kernel = require_bound_symbol(contracts, kernel_fn)
    providers = [p for p in pool.providers if takes(kernel, p.param)]
    values = dict(pool.values)
    values.update({p.param: p.expr for p in providers})

    fault = pool.fault
    if fault is not None:
        for param in (*fault.values, *fault.requires):
            if not takes(kernel, param):
                raise HarnessError(f"{pool.name}: fault {fault.kind!r} edits {param!r}, which {kernel.name} does not take")

    def call(input_expr: str, output_expr: str) -> str:
        site = {**values, pool.input_param: input_expr, pool.output_param: output_expr}
        if fault is not None:
            site.update(fault.values)
        return render_call(kernel, bind(kernel, site), indent=indent)

    checks = []
    for check in pool.checks:
        rule = require_bound_symbol(contracts, check.fn)
        checks.append((check.result_var, render_call(rule, bind(rule, values), indent=indent), rule.name,
                       int(check.expected)))

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
        run_call=call("input", "output"),
        bench_call=call(f"{pool.name}_input", f"{pool.name}_output"),
        checks=checks,
        fault_kind=fault.kind if fault else None,
        fault_declarations=[render_declaration(d) for d in fault.declarations] if fault else [],
        fault_setup=fault.setup if fault else "",
        no_scratch=bool(fault and fault.no_scratch),
    )
