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
    inputs: Sequence[HarnessInput] = (),
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
        inputs=tuple(inputs) or (HarnessInput("input_data", "input", f"{n}_{input_array}"),), **pool_fields,
    )


def dims_count(dims: Mapping[str, Any]) -> str:
    return f"({dims['n']} * {dims['h']} * {dims['w']} * {dims['c']})"
