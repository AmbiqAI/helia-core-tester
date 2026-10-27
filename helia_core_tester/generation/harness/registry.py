"""Pool builders keyed by the template an operator used to render from (iteration 1b).

An operator that writes its case through OperationBase._write_op_outputs registers a pool
builder for its former .c template path; _write_op_outputs then renders the case through the
generic harness instead, with the same validation rules (keyed by that path) and sidecar."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from helia_core_tester.generation.harness.model import ArgumentPool, HarnessError

PoolBuilder = Callable[[Mapping[str, Any]], ArgumentPool]
_POOLS: Dict[str, Tuple[PoolBuilder, str]] = {}


def harness_pool(template: str, *, label: str) -> Callable[[PoolBuilder], PoolBuilder]:
    """Register `builder` as the pool of every case that used to render `template`."""
    def register(builder: PoolBuilder) -> PoolBuilder:
        existing = _POOLS.get(template)
        if existing is not None and existing[0] is not builder:
            raise HarnessError(f"{template} already has a harness pool ({existing[0].__qualname__})")
        _POOLS[template] = (builder, label)
        return builder
    return register


def lookup(template: str) -> Optional[Tuple[PoolBuilder, str]]:
    return _POOLS.get(template)
