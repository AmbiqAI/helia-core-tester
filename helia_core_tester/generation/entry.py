"""Resolve a descriptor's `entry:` (the public ns-cmsis-nn function a case calls).

Every entry is resolved from the kernel contract, for operators whose pool binds its call from
the contract (CONTRACT_BOUND_OPERATORS): the entry must be a declared kernel, its tensor
pointers must match the descriptor's dtypes, and its scratch comes from
`<entry>_get_buffer_size[_mve|_dsp]`, from `entry_sizer`, or is declared absent with
`entry_scratch`; the operators whose cases pass no context (SCRATCHLESS_OPERATORS) take
neither. Every other case fails at generation, naming what is missing. Whether an entry takes
weight sums, a struct-typed bias or a layout follows from its prototype, so no per-entry table
exists any more.

An entry the implementation manifest has a record for takes those terms from the record
instead (`apply_entry_manifest`): the scratch query of the buffer bound to its `ctx`, the CPU
capabilities its `requires` name and whether it declines under ARM_MATH_AUTOVECTORIZE. Its
descriptor must not set them itself, so the manifest stays the one authority.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from helia_core_tester.contract import render
from helia_core_tester.contract.bind import ContractBindError, check_types, takes
from helia_core_tester.contract.ir import ContractSet, ImplementationDecl

# Operators whose template renders its kernel and sizer calls by binding from the contract.
CONTRACT_BOUND_OPERATORS: frozenset[str] = frozenset(
    {"Convolve", "DepthwiseConv", "FullyConnected", "TransposeConv", "Add", "Mul", "Dequantize"})
# Operators whose cases pass no context: an entry of theirs declares no scratch at all.
SCRATCHLESS_OPERATORS: frozenset[str] = frozenset({"Add", "Mul", "Dequantize"})

ENTRY_SCRATCH_NONE = "none"
# Descriptor keys the implementation manifest decides for an entry it has a record for.
MANIFEST_ENTRY_KEYS = ("entry_sizer", "entry_scratch", "required_capabilities", "autovectorize_declines")
# The record that decided them, stamped on the descriptor the generation step works from.
ENTRY_IMPLEMENTATION_KEY = "entry_implementation"
# The capability a CPU needs for each positive manifest feature token (see core/cpu_targets.py).
FEATURE_CAPABILITIES = {"mvei": ("mve",), "mvef": ("mve",), "mve_f16": ("mve", "fp16_execution"),
                        "dsp": ("dsp",), "f32_api": ("fp32_execution",), "f16_api": ("fp16_execution",)}
SUCCESS = "ARM_CMSIS_NN_SUCCESS"


class EntryError(ValueError):
    """A descriptor's entry cannot be called by the operator's generated case."""


def manifest_record(contracts: ContractSet, entry: str, where: str) -> Optional[ImplementationDecl]:
    records = contracts.implementations_of(entry) if contracts.present else []
    if len(records) > 1:
        raise EntryError(f"{where}: the implementation manifest has {len(records)} records for it "
                         f"({', '.join(r.id for r in records)}); a direct call needs exactly one")
    return records[0] if records else None


def apply_entry_manifest(desc: Mapping[str, Any], contracts: Optional[ContractSet] = None) -> Dict[str, Any]:
    """The descriptor with the terms the manifest decides for its entry filled in, stamped with the
    record id; unchanged when it names no entry or the manifest has no record for it.

    Scratch: the size query of the record's buffer bound to `ctx`, or `entry_scratch: none` when
    the entry is given no ctx or one without a query. A case that expects the full result also gets
    the capabilities of the record's positive `requires` tokens, and declines under
    ARM_MATH_AUTOVECTORIZE when the record needs `!autovectorize` and declines on a build cause. A
    case expecting ARM_CMSIS_NN_NO_IMPL_ERROR runs wherever the entry declines, so it gets neither.
    """
    entry = str(desc.get("entry") or "").strip()
    if not entry:
        return dict(desc)
    where = f"{desc.get('name', '<unnamed>')}: entry {entry!r}"
    contracts = contracts if contracts is not None else render.load_current_contracts()
    record = manifest_record(contracts, entry, where)
    stamped = desc.get(ENTRY_IMPLEMENTATION_KEY)
    if stamped is not None:
        if record is None or record.id != stamped:
            raise EntryError(f"{where}: stamped with manifest record {stamped!r}, but this checkout's manifest "
                             f"has {record.id if record else 'none'}")
        return dict(desc)
    if record is None:
        return dict(desc)
    authored = [key for key in MANIFEST_ENTRY_KEYS if key in desc]
    if authored:
        raise EntryError(f"{where}: the implementation manifest record {record.id} ({contracts.manifest_path}) "
                         f"decides {authored}; remove them from the descriptor")
    derived: Dict[str, Any] = {ENTRY_IMPLEMENTATION_KEY: record.id}
    ctx = [b for b in record.buffers if b.param == "ctx" and b.lifetime != "prepare"]
    if len(ctx) > 1:
        raise EntryError(f"{where}: record {record.id} binds {len(ctx)} buffers to ctx "
                         f"({', '.join(b.name for b in ctx)}); one sizes the scratch")
    if ctx and ctx[0].size_query is not None:
        derived["entry_sizer"] = ctx[0].size_query
    else:
        derived["entry_scratch"] = ENTRY_SCRATCH_NONE
    expected = str(desc.get("expected_status", SUCCESS)).strip()
    if expected != "ARM_CMSIS_NN_NO_IMPL_ERROR":
        capabilities: List[str] = []
        for token in sorted(t for t in record.requires if not t.startswith("!")):
            if token not in FEATURE_CAPABILITIES:
                raise EntryError(f"{where}: record {record.id} requires {token!r}, which no CPU capability expresses")
            capabilities.extend(c for c in FEATURE_CAPABILITIES[token] if c not in capabilities)
        if capabilities:
            derived["required_capabilities"] = capabilities
    if expected == SUCCESS and "!autovectorize" in record.requires and record.declines_on_build:
        derived["autovectorize_declines"] = True
    return {**desc, **derived}


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
    if operator not in CONTRACT_BOUND_OPERATORS:
        raise EntryError(f"{where}: {operator} does not resolve entries from the kernel contract; the operators "
                         f"that do are {sorted(CONTRACT_BOUND_OPERATORS)}")

    contracts = contracts if contracts is not None else render.load_current_contracts()
    desc = apply_entry_manifest({**desc, "entry": entry}, contracts) if contracts.present else desc
    sizer, scratch = desc.get("entry_sizer"), desc.get("entry_scratch")
    if sizer is not None and scratch is not None:
        raise EntryError(f"{where}: set entry_sizer or entry_scratch, not both")
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

    resolved: Dict[str, Any] = {"kernel_fn": entry, "entry_family": "contract",
                                "kernel_needs_layout": takes(decl, "layout"), "buffer_size_needs_layout": False}
    if operator in SCRATCHLESS_OPERATORS:
        if sizer is not None or scratch is not None:
            raise EntryError(f"{where}: {operator} cases pass no scratch, so an entry takes neither entry_sizer "
                             "nor entry_scratch")
        resolved["kernel_get_buffer_size_fn"] = None
        resolved["entry_scratch_bytes"] = 0
        return resolved
    if scratch is not None:
        resolved["kernel_get_buffer_size_fn"] = None
        resolved["entry_scratch_bytes"] = entry_scratch_bytes(scratch, where)
        return resolved
    if sizer is not None:
        if not isinstance(sizer, str) or not sizer.strip():
            raise EntryError(f"{where}: entry_sizer must name one scratch-size query, got {sizer!r}")
        decl_sizer = _sizer_decl(contracts, sizer.strip(), where)
        resolved["kernel_get_buffer_size_fn"] = decl_sizer.name
        resolved["buffer_size_needs_layout"] = takes(decl_sizer, "layout")
        return resolved
    found = contracts.sizer_for(entry, cpu)
    if found is None:
        raise EntryError(f"{where}: this checkout declares no {entry}_get_buffer_size; set entry_sizer to the "
                         "public query that sizes its scratch, or entry_scratch: none when it takes none")
    resolved["kernel_get_buffer_size_fn"] = found
    resolved["buffer_size_needs_layout"] = takes(contracts.find(found), "layout")
    return resolved


def _sizer_decl(contracts: ContractSet, sizer: str, where: str):
    sizer_decl = contracts.find(sizer)
    if sizer_decl is None:
        raise EntryError(f"{where}: entry_sizer {sizer!r} is not a public function of this checkout")
    if sizer_decl.kind != "sizer":
        raise EntryError(f"{where}: entry_sizer {sizer!r} is a {sizer_decl.kind}, not a scratch-size query")
    return sizer_decl


FLOAT_DTYPES = frozenset({"FP32", "FP16"})


def check_entry_fault(desc: Mapping[str, Any], resolved: Mapping[str, Any]) -> None:
    """Reject a fault on an entry case, except an invalid layout for a float entry that takes one."""
    fault = desc.get("fault")
    if not fault:
        return
    is_float = str(desc.get("activation_dtype", "S8")).upper() in FLOAT_DTYPES
    if fault == "invalid_layout" and is_float and resolved.get("kernel_needs_layout"):
        return
    raise EntryError(
        f"{desc.get('name')}: entry {resolved['kernel_fn']!r} supports only fault: invalid_layout, "
        f"and only as a float entry that takes a layout argument; got fault {fault!r}"
    )
