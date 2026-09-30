"""The pool of a layout-preserving copy (Reshape, Squeeze): one void kernel call over the whole
block (possibly empty), validated against the golden."""

from __future__ import annotations

from typing import Any, Mapping

from helia_core_tester.generation.harness import ArgumentPool
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool


def copy_argument_pool(context: Mapping[str, Any]) -> ArgumentPool:
    n, total = context["name"], int(context["total_size"])
    if total < 0:
        raise ValueError(f"{n}: a copy cannot have a negative element count")
    dims = context["output_dims"]
    elements = int(dims["n"]) * int(dims["h"]) * int(dims["w"]) * int(dims["c"])
    if total != elements:
        raise ValueError(f"{n}: the copy moves {total} elements but the output holds {elements}")
    # An empty copy compares nothing; one element of storage keeps the guard a real array.
    return tensor_case_pool(context, {"total_size": str(total)}, output_count=dims_count(dims),
                            output_capacity="1" if total == 0 else None)
