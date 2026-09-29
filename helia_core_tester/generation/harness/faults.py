"""Fault edits every operator spells the same way (iteration 1b, G7). An operator adds its
own for the guards only it has, usually with `struct_copy`."""

from __future__ import annotations

from typing import Mapping, Optional

from helia_core_tester.generation.harness.model import ArgumentPool, Declaration, FaultEdit, HarnessError


def struct_copy(pool: ArgumentPool, kind: str, param: str, ctype: str, source: str,
                fields: Mapping[str, object]) -> FaultEdit:
    """Pass the kernel a copy of `source` (a file-scope struct) with `fields` overwritten."""
    if not fields:
        raise HarnessError(f"{pool.name}: fault {kind!r} copies {source} without changing a field")
    copy = f"{pool.name}_fault_{param}"
    setup = [f"    // Fault {kind}: the kernel gets {source} with {', '.join(fields)} changed.",
             f"    {copy} = {source};"]
    setup += [f"    {copy}.{path} = {value};" for path, value in fields.items()]
    return FaultEdit(kind=kind, values={param: f"&{copy}"},
                     declarations=(Declaration(copy, ctype, storage="static"),), setup="\n".join(setup))


def common_fault(pool: ArgumentPool, kind: str, *, layout: Optional[str] = None) -> Optional[FaultEdit]:
    """The edit for a fault kind shared across operators, or None when the kind is the operator's own."""
    if kind == "null_input":
        return FaultEdit(kind=kind, values={pool.input_param: "NULL"})
    if kind == "null_output":
        return FaultEdit(kind=kind, values={pool.output_param: "NULL"})
    if kind == "null_ctx_buf":
        return FaultEdit(kind=kind, no_scratch=True)
    if kind == "invalid_layout":
        if not layout:
            raise HarnessError(f"{pool.name}: fault {kind!r} needs the case's layout")
        return FaultEdit(kind=kind, values={"layout": f"(arm_nn_tensor_layout)({layout} + 1)"})
    return None


def null_context_buffer(pool: ArgumentPool, kind: str, param: str, context_var: str) -> FaultEdit:
    """Hand the kernel, through `param`, a context whose buffer is NULL (the pointer stays valid)."""
    return FaultEdit(kind=kind, requires=(param,), setup=(f"    // Fault {kind}: {context_var} carries no buffer.\n"
                                       f"    {context_var}.buf = NULL;\n    {context_var}.size = 0;"))


def with_fault(pool: ArgumentPool, fault: FaultEdit) -> ArgumentPool:
    """The pool a fault case renders from: the passing case's pool plus the edit, no benchmark."""
    from dataclasses import replace

    edited = replace(pool, fault=fault, benchmark=False)
    edited.validate()
    return edited
