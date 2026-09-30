"""The pool of a case whose kernel makes one call over tensors and scalars, with no context and
no scratch: the header carries the named dims, any extra arrays, the input and the golden, and
the operator supplies the kernel's scalar arguments by parameter name."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from helia_core_tester.generation.harness.model import ArgumentPool, ArrayLiteral, Declaration, HarnessInput


def dims_declaration(name: str, dims: Mapping[str, Any], comment: str = "") -> Declaration:
    return Declaration(name, "cmsis_nn_dims", {k: dims[k] for k in ("n", "h", "w", "c")}, comment=comment)


def tensor_case_pool(
    context: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    dims: Sequence[str] = ("input_dims", "output_dims"),
    extra_header: Sequence[Declaration] = (),
    output_count: Optional[str] = None,
    inputs: Optional[Sequence[HarnessInput]] = None,
    input_array: str = "input",
    output_param: str = "output_data",
    **pool_fields: Any,
) -> ArgumentPool:
    """`values` maps kernel parameter names to C expressions (scalars as literals); `dims` names
    context dims emitted as `<name>_<dims>` and offered to the kernel as `&<name>_<dims>`."""
    n = context["name"]
    header = [dims_declaration(f"{n}_{d}", context[d]) for d in dims]
    header += list(extra_header)
    header += [
        Declaration(f"{n}_{input_array}", context["input_dtype"], ArrayLiteral(context["input_data_array"]),
                    array=True, comment="Input data (for testing)"),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]),
                    array=True, comment="Expected output (golden)"),
    ]
    pool_values = {d: f"&{n}_{d}" for d in dims}
    pool_values.update({k: str(v) for k, v in values.items()})
    if output_count is None:
        output_count = str(context["output_size"])
    return ArgumentPool(
        name=n, values=pool_values, header=header, output_count=output_count, benchmark=False,
        scratch_buffer=False, output_param=output_param,
        inputs=tuple(inputs) if inputs is not None else (HarnessInput("input_data", "input", f"{n}_{input_array}"),),
        **pool_fields,
    )


def dims_count(dims: Mapping[str, Any]) -> str:
    return f"({dims['n']} * {dims['h']} * {dims['w']} * {dims['c']})"


def binary_case_pool(context: Mapping[str, Any], *, validation: Optional[str] = None,
                     extra_header: Sequence[Declaration] = (), **pool_fields: Any) -> ArgumentPool:
    """The pool of an elementwise or broadcast binary case (inputs `<name>_input1/2`). The binary
    prototypes spell their dims and scalars two ways (`input1_offset`, `input_1_offset`), so the
    pool offers both; bind picks the one the prototype uses."""
    n = context["name"]
    values: dict[str, Any] = {}
    for i in (1, 2):
        values[f"input{i}_dims"] = values[f"input_{i}_dims"] = f"&{n}_input{i}_dims"
        for key in ("offset", "mult", "shift"):
            if f"input{i}_{key}" in context:
                values[f"input{i}_{key}"] = values[f"input_{i}_{key}"] = context[f"input{i}_{key}"]
    for key in ("left_shift", "out_offset", "out_mult", "out_shift", "block_size"):
        if key in context:
            values[key] = context[key]
    if context.get("float_kernel"):
        for bound in ("min", "max"):
            if f"out_activation_{bound}_literal" in context:
                values[f"out_activation_{bound}"] = context[f"out_activation_{bound}_literal"]
    else:
        for bound in ("min", "max"):
            if f"out_activation_{bound}" in context:
                values[f"out_activation_{bound}"] = context[f"out_activation_{bound}"]
    header = [dims_declaration(f"{n}_{d}", context[d]) for d in ("input1_dims", "input2_dims", "output_dims")]
    header += [
        Declaration(f"{n}_input1", context["input_dtype"], ArrayLiteral(context["input1_data_array"]), array=True),
        Declaration(f"{n}_input2", context["input_dtype"], ArrayLiteral(context["input2_data_array"]), array=True),
        Declaration(f"{n}_expected_output", context["output_dtype"], ArrayLiteral(context["expected_output_array"]),
                    array=True, comment="Expected output (golden)"),
        *extra_header,
    ]
    values["output_dims"] = f"&{n}_output_dims"
    return ArgumentPool(
        name=n, values={k: str(v) for k, v in values.items()}, header=header, benchmark=False, scratch_buffer=False,
        output_count=dims_count(context["output_dims"]), validation=validation,
        inputs=(HarnessInput("input_1_data", "input1", f"{n}_input1"), HarnessInput("input_2_data", "input2", f"{n}_input2")),
        **pool_fields,
    )
