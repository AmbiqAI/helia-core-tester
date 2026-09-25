"""Loader for the shared kernel-id registry (assets/kernel_registry.yaml).

This is the single Python-side entry point for looking up the `kernel_id` a bridged
(family, operator, dtype) tuple should send over HCTP in CASE_META -- callers (currently
`generated_test_bridge.py`) must not hardcode kernel_id integers directly, so the mapping
stays centralized. Each row also carries the `c_define` name of the firmware's
`HCT_KERNEL_ID_*` macro, which `scripts/generate_kernel_catalog.py` renders into
`cmake/hardware/benchmark_server_adapters.h`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_REGISTRY_RELATIVE_PATH = Path("assets/kernel_registry.yaml")
_C_DEFINE_RE = re.compile(r"^HCT_KERNEL_ID_[A-Z0-9_]+$")
_C_FUNCTION_RE = re.compile(r"^arm_[a-z0-9_]+$")


class KernelRegistryError(ValueError):
    """Raised when assets/kernel_registry.yaml is malformed or contradicts itself."""


@dataclass(frozen=True)
class KernelEntry:
    kernel_id: int
    family: str | None
    operator: str
    dtype: str
    weight_dtype: str | None
    cmsis_function: str
    c_define: str


class UnknownKernelError(Exception):
    """Raised when a (family, operator, dtype) tuple has no registered kernel_id."""


class AmbiguousKernelError(UnknownKernelError):
    """Raised when the registry holds more than one entry for the same tuple.

    A subclass so existing ``except UnknownKernelError`` handlers keep catching it, but a
    distinct type so a caller that means "this dtype is not registered, skip the case" can
    avoid swallowing a registry that contradicts itself. Those are opposite situations: the
    first is an expected gap, the second is corrupt data that would otherwise silently drop
    every case for the duplicated tuple.
    """


def _registry_path(project_root: Path) -> Path:
    return project_root / _REGISTRY_RELATIVE_PATH


def _parse_entry(path: Path, index: int, entry: object) -> KernelEntry:
    where = f"{path}: kernels[{index}]"
    if not isinstance(entry, dict):
        raise KernelRegistryError(f"{where} is not a mapping")
    try:
        kernel_id = int(entry["kernel_id"])
    except (KeyError, TypeError, ValueError):
        raise KernelRegistryError(f"{where} has no integer kernel_id") from None
    if kernel_id <= 0:
        raise KernelRegistryError(f"{where} kernel_id={kernel_id} must be positive")
    where = f"{path}: kernel_id={kernel_id}"
    for key in ("operator", "dtype"):
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise KernelRegistryError(f"{where} has no {key}")
    cmsis_function = entry.get("cmsis_function")
    if not isinstance(cmsis_function, str) or not _C_FUNCTION_RE.match(cmsis_function):
        raise KernelRegistryError(f"{where} cmsis_function {cmsis_function!r} must match {_C_FUNCTION_RE.pattern}")
    c_define = entry.get("c_define")
    if not isinstance(c_define, str) or not _C_DEFINE_RE.match(c_define):
        raise KernelRegistryError(f"{where} c_define {c_define!r} must match {_C_DEFINE_RE.pattern}")
    weight_dtype = entry.get("weight_dtype")
    return KernelEntry(
        kernel_id=kernel_id,
        family=None if entry.get("family") is None else str(entry["family"]),
        operator=entry["operator"],
        dtype=entry["dtype"],
        weight_dtype=None if weight_dtype is None else str(weight_dtype),
        cmsis_function=cmsis_function,
        c_define=c_define,
    )


def load_kernel_registry(project_root: Path) -> list[KernelEntry]:
    """Load and validate the registry: every row carries an integer kernel_id, operator,
    dtype, cmsis_function and c_define, and no kernel_id or c_define appears twice."""
    path = _registry_path(project_root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("kernels"), list):
        raise KernelRegistryError(f"{path}: expected a mapping with a `kernels` list")
    entries = [_parse_entry(path, index, entry) for index, entry in enumerate(data["kernels"])]
    for attribute in ("kernel_id", "c_define"):
        seen: dict[object, int] = {}
        for entry in entries:
            value = getattr(entry, attribute)
            if value in seen:
                raise KernelRegistryError(
                    f"{path}: {attribute} {value!r} is used by kernel_id={seen[value]} and kernel_id={entry.kernel_id}"
                )
            seen[value] = entry.kernel_id
    return entries


def lookup_kernel_id(
    project_root: Path,
    *,
    family: str,
    operator: str,
    dtype: str = "S8",
    weight_dtype: str | None = None,
) -> int:
    """Look up the kernel_id for a bridged (family, operator, dtype[, weight_dtype]) tuple.

    Raises UnknownKernelError if the tuple isn't registered -- callers should treat that as
    an UnsupportedGeneratedTestError-worthy condition, not silently default to any kernel_id.
    Raises AmbiguousKernelError, a subclass, if the registry holds more than one entry for
    the tuple. That one is corrupt data rather than an expected gap and should stay fatal.
    """
    candidates: list[KernelEntry] = []
    for entry in load_kernel_registry(project_root):
        if entry.family == family and entry.operator == operator and entry.dtype == dtype:
            candidates.append(entry)

    if weight_dtype is not None:
        matches = [entry for entry in candidates if entry.weight_dtype == weight_dtype]
        if len(matches) > 1:
            raise AmbiguousKernelError(
                f"Ambiguous kernel registry entries for family={family!r} operator={operator!r} "
                f"dtype={dtype!r} weight_dtype={weight_dtype!r}: "
                f"{[e.cmsis_function for e in matches]} -- registry must have at most one "
                f"entry per (family, operator, dtype, weight_dtype) tuple."
            )
        if matches:
            return matches[0].kernel_id

    unweighted_matches = [entry for entry in candidates if entry.weight_dtype is None]
    if len(unweighted_matches) > 1:
        raise AmbiguousKernelError(
            f"Ambiguous kernel registry entries for family={family!r} operator={operator!r} "
            f"dtype={dtype!r} (no weight_dtype): {[e.cmsis_function for e in unweighted_matches]} -- "
            f"registry must have at most one entry per (family, operator, dtype, weight_dtype) tuple."
        )
    if unweighted_matches:
        return unweighted_matches[0].kernel_id

    raise UnknownKernelError(
        f"No registered kernel_id for family={family!r} operator={operator!r} "
        f"dtype={dtype!r} weight_dtype={weight_dtype!r}"
    )
