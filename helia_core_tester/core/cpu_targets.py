"""
CPU target parsing and capability helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


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
    capabilities: frozenset[str]

    def supports_capability(self, capability: str) -> bool:
        return str(capability).lower() in self.capabilities

    def supports_execution_dtype(self, dtype: str) -> bool:
        normalized = str(dtype).upper()
        if normalized == "FP32":
            return self.supports_capability("fp32_execution")
        if normalized == "FP16":
            return self.supports_capability("fp16_execution")
        return True


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
        return CpuProfile(
            cpu=canon,
            has_dsp=True,
            has_mve=True,
            capabilities=frozenset({"dsp", "mve", "fp32_execution", "fp16_execution"}),
        )
    if canon == "cortex-m4":
        return CpuProfile(
            cpu=canon,
            has_dsp=True,
            has_mve=False,
            capabilities=frozenset({"dsp", "fp32_execution"}),
        )
    if canon == "cortex-m0":
        return CpuProfile(
            cpu=canon,
            has_dsp=False,
            has_mve=False,
            capabilities=frozenset(),
        )
    raise ValueError(f"Unsupported CPU target: {cpu}")


def missing_required_capabilities(cpu: str, required_capabilities: Sequence[str]) -> list[str]:
    profile = get_cpu_profile(cpu)
    missing: list[str] = []
    for capability in required_capabilities:
        capability_name = str(capability).strip().lower()
        if capability_name and not profile.supports_capability(capability_name):
            missing.append(capability_name)
    return missing
