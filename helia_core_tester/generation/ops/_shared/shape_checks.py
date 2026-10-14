"""Descriptor shape checks shared by operators that validate declared geometry."""

import numbers
from typing import Any


def require_positive_int_dims(name: str, operator: str, **shapes: Any) -> None:
    """Raise ValueError unless every dimension of every named shape is a positive integer.

    Shape arity is the caller's job; this checks the element type and value only.
    """
    for shape in shapes.values():
        for dim in shape:
            # numpy integers are Integral; bool is too, but never a dimension.
            if isinstance(dim, bool) or not isinstance(dim, numbers.Integral) or dim <= 0:
                listed = " ".join(f"{key}={value!r}" for key, value in shapes.items())
                raise ValueError(f"{name}: {operator} shapes must be positive integers, got {listed}")
