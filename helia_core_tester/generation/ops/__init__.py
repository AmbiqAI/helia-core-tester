"""Operation registry access with lazy import."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Dict, Type

from .catalog import get_operator_spec, iter_operator_specs
from .op_registry import OP_CLASS_SPECS


class _LazyOpMap(Mapping[str, Type]):
    def __init__(self) -> None:
        self._cache: Dict[str, Type] = {}

    def __getitem__(self, key: str) -> Type:
        if key not in OP_CLASS_SPECS:
            raise KeyError(key)
        if key not in self._cache:
            module_name, class_name = OP_CLASS_SPECS[key]
            module = import_module(module_name)
            self._cache[key] = getattr(module, class_name)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(OP_CLASS_SPECS)

    def __len__(self) -> int:
        return len(OP_CLASS_SPECS)


_OP_MAP = _LazyOpMap()


def get_op_map() -> Mapping[str, Type]:
    """Return the lazy operator map keyed by canonical operator name."""
    return _OP_MAP


__all__ = ["get_op_map", "get_operator_spec", "iter_operator_specs"]
