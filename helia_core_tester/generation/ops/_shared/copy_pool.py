"""The pool of a layout-preserving copy (Reshape, Squeeze): one void kernel call over the whole
block (possibly empty), validated against the golden."""

from __future__ import annotations

from typing import Any, Mapping

from helia_core_tester.generation.harness import ArgumentPool
from helia_core_tester.generation.harness.simple import dims_count, tensor_case_pool


def copy_argument_pool(context: Mapping[str, Any]) -> ArgumentPool:
    total = int(context["total_size"])
    if total < 0:
        raise ValueError(f"{context['name']}: a copy cannot have a negative element count")
    return tensor_case_pool(context, {"total_size": str(total)}, output_count=dims_count(context["output_dims"]))
