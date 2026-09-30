"""The pool shared by Split and Unpack: one input, an int32 shape array, and one output slot per
slice reached through a pointer array the source declares."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, HarnessInput, OutputSlot
from helia_core_tester.generation.harness.simple import dims_declaration


def slice_argument_pool(context: Mapping[str, Any], *, values: Mapping[str, Any],
                        extra_header: Sequence[Declaration] = ()) -> ArgumentPool:
    n, dtype = context["name"], context["output_dtype"]
    outputs = list(context["outputs"])
    if not outputs:
        raise ValueError(f"{n}: a slicing case needs at least one output slice")
    header = [*extra_header,
              Declaration(f"{n}_input_shape", "int32_t", ArrayLiteral(context["input_shape_array"]), array=True,
                          comment="Input shape"),
              Declaration(f"{n}_input", context["input_dtype"], ArrayLiteral(context["input_data_array"]), array=True,
                          comment="Input data (for testing)")]
    slots = []
    for index, out in enumerate(outputs):
        header.append(Declaration(f"{out['name']}_expected_output", dtype, ArrayLiteral(out["expected_output_array"]),
                                  array=True, comment=f"Expected output {index}"))
        slots.append(OutputSlot(out["name"], int(out["size"])))
    pointers = Declaration(f"{n}_output_ptrs", f"{dtype}*",
                           ArrayLiteral("\n".join(f"    {slot.name}_output," for slot in slots)), array=True,
                           storage="static")
    pool_values = {"input_dims": str(context["input_dims_count"]), "input_shape": f"{n}_input_shape",
                   "axis": str(context["axis"]), "output_data": f"{n}_output_ptrs"}
    pool_values.update({k: str(v) for k, v in values.items()})
    return ArgumentPool(name=n, values=pool_values, header=header, source=(pointers,), outputs=tuple(slots),
                        inputs=(HarnessInput("input_data", "input", f"{n}_input"),), benchmark=False,
                        scratch_buffer=False)


def split_argument_pool(context: Mapping[str, Any]) -> ArgumentPool:
    n = context["name"]
    dims = dims_declaration(f"{n}_input_dims", context["input_dims"], comment="Input dimensions")
    split_dims = Declaration(f"{n}_split_dims", "int32_t", ArrayLiteral(context["split_dims_array"]), array=True,
                             comment="Split dimensions")
    return slice_argument_pool(context, values={"num_splits": context["num_splits"], "split_dims": f"{n}_split_dims"},
                               extra_header=(dims, split_dims))


def unpack_argument_pool(context: Mapping[str, Any]) -> ArgumentPool:
    return slice_argument_pool(context, values={})
