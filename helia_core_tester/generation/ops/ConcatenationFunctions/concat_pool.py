"""The Concatenation pool: the any-rank kernels take every input in one call through a pointer
array; the legacy per-axis kernels (void, one input each) are called once per input with that
input's NHWC extents and its offset along the axis."""

from __future__ import annotations

from typing import Any, Mapping

from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, HarnessInput
from helia_core_tester.generation.harness.simple import dims_declaration

AXIS_STYLES = {"axis_x": "output_x", "axis_y": "output_y", "axis_z": "output_z", "axis_w": None}
SINGLE_CALL_STYLES = ("per_tensor", "any_rank")


def concatenation_argument_pool(context: Mapping[str, Any]) -> ArgumentPool:
    n, dtype = context["name"], context["input_dtype"]
    count = int(context["num_inputs"])
    if count < 1:
        raise ValueError(f"{n}: Concatenation needs at least one input")
    style = str(context.get("call_style") or "per_tensor").lower()
    if style not in AXIS_STYLES and style not in SINGLE_CALL_STYLES:
        raise ValueError(f"{n}: call_style {style!r} is not one of {sorted((*AXIS_STYLES, *SINGLE_CALL_STYLES))}")
    header = [dims_declaration(f"{n}_output_dims", context["output_dims"], comment="Output dimensions"),
              Declaration(f"{n}_input_concat_dims", "int32_t", ArrayLiteral(context["input_concat_dims_array"]),
                          array=True, comment="Input concat dims"),
              Declaration(f"{n}_output_shape", "int32_t", ArrayLiteral(context["output_shape_array"]), array=True,
                          comment="Output shape")]
    for axis, what in (("x", "width"), ("y", "height"), ("z", "channels"), ("w", "batch")):
        header.append(Declaration(f"{n}_input_{axis}", "int32_t", ArrayLiteral(context[f"input_{axis}_array"]),
                                  array=True, comment=f"Per-input NHWC {what} extents"))
    header.append(Declaration(f"{n}_offsets", "int32_t", ArrayLiteral(context["offsets_array"]), array=True,
                              comment="Per-input offsets along the concatenation axis"))
    arrays = context["input_data_arrays"]
    if len(arrays) != count:
        raise ValueError(f"{n}: {count} inputs declared but {len(arrays)} input arrays supplied")
    for i, body in enumerate(arrays):
        header.append(Declaration(f"{n}_input{i + 1}", dtype, ArrayLiteral(body), array=True))
    header.append(Declaration(f"{n}_expected_output", context["output_dtype"],
                              ArrayLiteral(context["expected_output_array"]), array=True, comment="Expected output"))
    pointers = Declaration(f"{n}_input_ptrs", f"{dtype}*",
                           ArrayLiteral("\n".join(f"    {n}_input{i + 1}," for i in range(count))), array=True,
                           comment="Array of input pointers")
    d = context["output_dims"]
    common = dict(name=n, header=header, source=(pointers,), benchmark=False, scratch_buffer=False,
                  output_count=f"({d['n']} * {d['h']} * {d['w']} * {d['c']})",
                  inputs=(HarnessInput("input_data", "input_ptrs", f"{n}_input_ptrs", f"{dtype}* const"),))
    if style in SINGLE_CALL_STYLES:
        values = {"inputs_count": str(count), "num_inputs": str(count), "input_concat_dims": f"{n}_input_concat_dims",
                  "axis_sizes": f"{n}_input_concat_dims", "axis": str(context["axis"]),
                  "output_dims": str(context["output_rank"]), "output_shape": f"{n}_output_shape"}
        return ArgumentPool(values=values, **common)
    axis = style[-1]
    calls = []
    for i in range(count):
        call = {"input": f"input_ptrs[{i}]", "output": "output", f"offset_{axis}": f"(uint32_t){n}_offsets[{i}]"}
        for extent in "xyzw":
            call[f"input_{extent}"] = f"(uint16_t){n}_input_{extent}[{i}]"
        if AXIS_STYLES[style]:
            call[f"output_{axis}"] = f"(uint16_t){context[AXIS_STYLES[style]]}"
        calls.append(call)
    return ArgumentPool(values={}, calls=calls, output_param="output", **common)
