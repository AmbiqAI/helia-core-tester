"""Operation registry access with lazy import."""

from __future__ import annotations

from typing import Dict, Type

_OP_MAP_CACHE: Dict[str, Type] | None = None


def get_op_map() -> Dict[str, Type]:
    """Return operator map, importing heavy op modules only when needed."""
    global _OP_MAP_CACHE
    if _OP_MAP_CACHE is None:
        from .op_registry import OP_MAP

        _OP_MAP_CACHE = OP_MAP
    return _OP_MAP_CACHE


__all__ = ["get_op_map"]
