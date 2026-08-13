"""Loader for the shared kernel-id registry (assets/kernel_registry.yaml).

This is the single Python-side entry point for looking up the `kernel_id` a bridged
(family, operator, dtype) tuple should send over HCTP in CASE_META -- callers (currently
`generated_test_bridge.py`) must not hardcode kernel_id integers directly, so the mapping
stays centralized and testable against the firmware's `HCT_KERNEL_ID_*` defines in
`cmake/perf_stream/benchmark_server_session.h`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_REGISTRY_RELATIVE_PATH = Path("assets/kernel_registry.yaml")


@dataclass(frozen=True)
class KernelEntry:
    kernel_id: int
    family: str | None
    operator: str
    dtype: str
    weight_dtype: str | None
    cmsis_function: str


class UnknownKernelError(Exception):
    """Raised when a (family, operator, dtype) tuple has no registered kernel_id."""


def _registry_path(project_root: Path) -> Path:
    return project_root / _REGISTRY_RELATIVE_PATH


def load_kernel_registry(project_root: Path) -> list[KernelEntry]:
    path = _registry_path(project_root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        KernelEntry(
            kernel_id=int(entry["kernel_id"]),
            family=entry.get("family"),
            operator=str(entry["operator"]),
            dtype=str(entry["dtype"]),
            weight_dtype=(None if entry.get("weight_dtype") is None else str(entry["weight_dtype"])),
            cmsis_function=str(entry.get("cmsis_function", "")),
        )
        for entry in data.get("kernels", [])
    ]


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
    """
    candidates: list[KernelEntry] = []
    for entry in load_kernel_registry(project_root):
        if entry.family == family and entry.operator == operator and entry.dtype == dtype:
            candidates.append(entry)

    if weight_dtype is not None:
        for entry in candidates:
            if entry.weight_dtype == weight_dtype:
                return entry.kernel_id

    for entry in candidates:
        if entry.weight_dtype is None:
            return entry.kernel_id

    raise UnknownKernelError(
        f"No registered kernel_id for family={family!r} operator={operator!r} "
        f"dtype={dtype!r} weight_dtype={weight_dtype!r}"
    )
