"""
CPU target parsing and capability helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_CPU_ALIASES = {
    "m0": "cortex-m0",
    "m4": "cortex-m4",
    "m55": "cortex-m55",
    "cortex-m0": "cortex-m0",
    "cortex-m4": "cortex-m4",
    "cortex-m55": "cortex-m55",
}


@dataclass(frozen=True)
class CpuProfile:
    cpu: str
    has_dsp: bool
    has_mve: bool


def normalize_cpu(cpu: str) -> str:
    key = cpu.strip().lower()
    if key not in _CPU_ALIASES:
        raise ValueError(f"Unsupported CPU target: {cpu}")
    return _CPU_ALIASES[key]


def parse_cpu_list(cpu_str: str | Iterable[str]) -> list[str]:
    if isinstance(cpu_str, str):
        raw = [c.strip() for c in cpu_str.split(",") if c.strip()]
    else:
        raw = [str(c).strip() for c in cpu_str if str(c).strip()]
    if not raw:
        raise ValueError("At least one CPU target is required")
    normalized: list[str] = []
    seen = set()
    for cpu in raw:
        canon = normalize_cpu(cpu)
        if canon not in seen:
            normalized.append(canon)
            seen.add(canon)
    return normalized


def get_cpu_profile(cpu: str) -> CpuProfile:
    canon = normalize_cpu(cpu)
    if canon == "cortex-m55":
        return CpuProfile(cpu=canon, has_dsp=True, has_mve=True)
    if canon == "cortex-m4":
        return CpuProfile(cpu=canon, has_dsp=True, has_mve=False)
    if canon == "cortex-m0":
        return CpuProfile(cpu=canon, has_dsp=False, has_mve=False)
    raise ValueError(f"Unsupported CPU target: {cpu}")