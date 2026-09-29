"""Resolve a descriptor's `entry:` (the public ns-cmsis-nn function a case calls).

Entries listed in kernel_dispatch.DIRECT_ENTRIES keep the table's resolution unchanged. Any
other public kernel is resolved from the kernel contract, for operators whose template binds
its call from the contract (CONTRACT_BOUND_OPERATORS): the entry must be a declared kernel,
its tensor pointers must match the descriptor's dtypes, and its scratch comes from
`<entry>_get_buffer_size[_mve|_dsp]`, from `entry_sizer`, or is declared absent with
`entry_scratch`. Every other case fails at generation, naming what is missing.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from helia_core_tester.contract import render
from helia_core_tester.contract.bind import ContractBindError, check_types
from helia_core_tester.contract.ir import ContractSet
from helia_core_tester.generation.kernel_dispatch import DIRECT_ENTRIES, resolve_direct_entry

# Operators whose template renders its kernel and sizer calls by binding from the contract.
CONTRACT_BOUND_OPERATORS: frozenset[str] = frozenset({"Convolve", "DepthwiseConv"})

ENTRY_SCRATCH_NONE = "none"


class EntryError(ValueError):
    """A descriptor's entry cannot be called by the operator's generated case."""


def entry_scratch_bytes(value: Any, where: str) -> int:
    if isinstance(value, str) and value.strip().lower() == ENTRY_SCRATCH_NONE:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EntryError(f"{where}: entry_scratch must be 'none' or a non-negative byte count, got {value!r}")
    return value


def resolve_entry(
    operator: str,
    entry: str,
    *,
    activation_dtype: str,
    weight_dtype: str,
    cpu: str,
    desc: Mapping[str, Any],
    extra_roles: Optional[Mapping[str, str]] = None,
    contracts: Optional[ContractSet] = None,
) -> Dict[str, Any]:
    """Kernel-info overrides for a case that calls `entry` instead of the operator's wrapper."""
    entry = str(entry).strip()
    where = f"{desc.get('name', '<unnamed>')}: entry {entry!r}"
    sizer, scratch = desc.get("entry_sizer"), desc.get("entry_scratch")
    if sizer is not None and scratch is not None:
        raise EntryError(f"{where}: set entry_sizer or entry_scratch, not both")

    if entry in DIRECT_ENTRIES:
        if sizer is not None or scratch is not None:
            raise EntryError(f"{where} is a kernel_dispatch.DIRECT_ENTRIES entry, whose scratch query "
                             "is fixed there; drop entry_sizer and entry_scratch")
        return resolve_direct_entry(operator, entry, activation_dtype, weight_dtype)

    if operator not in CONTRACT_BOUND_OPERATORS:
        known = sorted(name for name, spec in DIRECT_ENTRIES.items() if spec.operator == operator)
        raise EntryError(f"{where} is not in kernel_dispatch.DIRECT_ENTRIES, and {operator} does not yet "
                         f"bind its call from the kernel contract; known {operator} entries: {known}")

    contracts = contracts if contracts is not None else render.load_current_contracts()
    if not contracts.present:
        raise EntryError(f"{where} is resolved from the kernel contract, but the ns-cmsis-nn checkout "
                         f"({contracts.root or 'unresolved'}) has no Tests/KernelContracts/kernel_contracts.json")
    decl = contracts.find(entry)
    if decl is None:
        raise EntryError(f"{where} is not a public function of this ns-cmsis-nn checkout ({contracts.path})")
    if decl.kind != "kernel":
        raise EntryError(f"{where} is a {decl.kind}, not a kernel")
    roles = {"input": activation_dtype, "output": activation_dtype, "filter": weight_dtype, **dict(extra_roles or {})}
    try:
        check_types(decl, roles)
    except ContractBindError as error:
        raise EntryError(f"{where}: {error}") from None

    resolved: Dict[str, Any] = {"kernel_fn": entry, "entry_family": "contract"}
    if scratch is not None:
        resolved["kernel_get_buffer_size_fn"] = None
        resolved["entry_scratch_bytes"] = entry_scratch_bytes(scratch, where)
        return resolved
    if sizer is not None:
        sizer_decl = contracts.find(str(sizer))
        if sizer_decl is None:
            raise EntryError(f"{where}: entry_sizer {sizer!r} is not a public function of this checkout")
        if sizer_decl.kind != "sizer":
            raise EntryError(f"{where}: entry_sizer {sizer!r} is a {sizer_decl.kind}, not a scratch-size query")
        resolved["kernel_get_buffer_size_fn"] = sizer_decl.name
        return resolved
    found = contracts.sizer_for(entry, cpu)
    if found is None:
        raise EntryError(f"{where}: this checkout declares no {entry}_get_buffer_size; set entry_sizer to the "
                         "public query that sizes its scratch, or entry_scratch: none when it takes none")
    resolved["kernel_get_buffer_size_fn"] = found
    return resolved
