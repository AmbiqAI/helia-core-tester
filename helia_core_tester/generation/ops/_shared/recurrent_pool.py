"""The pool of a recurrent case (LSTM, GRU): the body stays a Jinja fragment (temp-buffer sizer
checks, per-gate kernel sums, state seeding, stream chunks, fault variants) and the kernel call
goes through `_run(input, output, params, buffers)`, whose struct types come from the contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from helia_core_tester.contract.bind import param_type
from helia_core_tester.contract.render import load_current_contracts, require_bound_symbol
from helia_core_tester.generation.harness import ArgumentPool, HarnessError, HarnessInput, fragment


def recurrent_argument_pool(context: Mapping[str, Any], *, body: str, header: str, ctype: str,
                            includes: Sequence[str] = ()) -> ArgumentPool:
    """`body` and `header` are fragment paths under assets/templates; `ctype` is the tensor
    element type of the kernel's input and output."""
    n, kernel_fn = context["name"], context["kernel_fn"]
    decl = require_bound_symbol(load_current_contracts(), kernel_fn)
    run_params = []
    for local in ("params", "buffers"):
        c_type = param_type(decl, local)
        if not c_type:
            raise HarnessError(f"{n}: {kernel_fn} takes no {local!r}, so it is not a recurrent kernel this pool can call")
        run_params.append((local, c_type))
    return ArgumentPool(
        name=n, values={local: local for local, _ in run_params},
        inputs=(HarnessInput("input", "input", f"{n}_input", ctype),), output_param="output", output_ctype=ctype,
        run_params=tuple(run_params), header_text=fragment(header, "header"), file_scope=fragment(body, "file_scope"),
        test_body=fragment(body, "test_body"), includes=tuple(includes), benchmark=False, scratch_buffer=False,
    )
