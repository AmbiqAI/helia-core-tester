"""Fault edits every operator spells the same way (iteration 1b, G7). An operator adds its
own for the guards only it has, usually with `struct_copy`."""

from __future__ import annotations

from typing import Mapping, Optional

from helia_core_tester.generation.harness.model import ArgumentPool, FaultEdit, HarnessError


def struct_copy(pool: ArgumentPool, kind: str, param: str, ctype: str, source: str,
                fields: Mapping[str, object]) -> FaultEdit:
    """Pass the kernel a copy of `source` (a file-scope struct) with `fields` overwritten.

    The copy is a local initialised from `source` in the fault set-up, which runs in `_run` just
    before the kernel call: several parameter structs have const members (cmsis_nn_bmm_params_f32
    adj_x/adj_y), so a copy can be initialised but never assigned as a whole. Fault cases carry no
    benchmark path, so the copy is never needed outside `_run`.
    """
    if not fields:
        raise HarnessError(f"{pool.name}: fault {kind!r} copies {source} without changing a field")
    copy = f"{pool.name}_fault_{param}"
    setup = [f"    // Fault {kind}: the kernel gets {source} with {', '.join(fields)} changed.",
             f"    {ctype} {copy} = {source};"]
    setup += [f"    {copy}.{path} = {value};" for path, value in fields.items()]
    return FaultEdit(kind=kind, values={param: f"&{copy}"}, setup="\n".join(setup))


def common_fault(pool: ArgumentPool, kind: str, *, layout: Optional[str] = None) -> Optional[FaultEdit]:
    """The edit for a fault kind shared across operators, or None when the kind is the operator's own."""
    if kind == "null_input":
        return FaultEdit(kind=kind, values={pool.harness_inputs[0].param: "NULL"})
    if kind == "null_output":
        return FaultEdit(kind=kind, values={pool.output_param: "NULL"})
    if kind == "null_ctx_buf":
        return FaultEdit(kind=kind, no_scratch=True, requires=("ctx",))
    if kind == "small_ctx_size":
        return FaultEdit(kind=kind, setup=(f"    // Fault {kind}: the context claims one byte of scratch.\n"
                                           f"    {pool.name}_ctx.size = 1;"), requires=("ctx",))
    if kind == "invalid_layout":
        if not layout:
            raise HarnessError(f"{pool.name}: fault {kind!r} needs the case's layout")
        return FaultEdit(kind=kind, values={"layout": f"(arm_nn_tensor_layout)({layout} + 1)"})
    return None


def null_context_buffer(pool: ArgumentPool, kind: str, param: str, context_var: str) -> FaultEdit:
    """Hand the kernel, through `param`, a context whose buffer is NULL (the pointer stays valid).
    The context must come from a provider: a plain pool value (NULL when the case computes no
    sums) declares no `context_var` for the setup to clear."""
    if not any(param in provider.names for provider in pool.providers):
        raise HarnessError(f"{pool.name}: fault {kind!r} clears {context_var}, but no provider supplies {param!r}")
    return FaultEdit(kind=kind, requires=(param,), setup=(f"    // Fault {kind}: {context_var} carries no buffer.\n"
                                       f"    {context_var}.buf = NULL;\n    {context_var}.size = 0;"))


def with_fault(pool: ArgumentPool, fault: FaultEdit) -> ArgumentPool:
    """The pool a fault case renders from: the passing case's pool plus the edit, no benchmark."""
    from dataclasses import replace

    edited = replace(pool, fault=fault, benchmark=False)
    edited.validate()
    return edited
