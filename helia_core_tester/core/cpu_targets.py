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
    # Same physical Cortex-M55/Apollo510 silicon as "cortex-m55", but built
    # with MVE force-disabled (compiler flag `-mcpu=cortex-m55+nomve`, see
    # _TARGET_CPU_CMAKE_OVERRIDES below) so the DSP-only fallback code path
    # can be benchmarked/tested on the same hardware as the MVE path, rather
    # than only comparing across different chips (e.g. m4 vs. m55).
    "m55-dsp": "cortex-m55-dsp",
    "cortex-m55-dsp": "cortex-m55-dsp",
}

# Maps a canonical cpu name (as used for test generation, build-dir naming,
# and generated_tests_dir layout) to the actual `-mcpu=...`/`TARGET_CPU`
# CMake value passed to the compiler, when they differ. "cortex-m55-dsp" is
# a helia-core-tester-only pseudo target: identical generated test sources
# and identical physical target as "cortex-m55", just compiled with MVE
# disabled -- see CMakeLists.txt's ARM_CPU/ARM_FEATURES derivation, which
# strips this "+nomve" suffix before selecting the CMSIS device header
# (the physical chip doesn't change, only the instruction-set the compiler
# targets).
_TARGET_CPU_CMAKE_OVERRIDES = {
    "cortex-m55-dsp": "cortex-m55+nomve",
}


def target_cpu_cmake_value(cpu: str) -> str:
    """Returns the value to pass as `-DTARGET_CPU=...` for a canonical cpu
    name, translating helia-core-tester-only pseudo targets (like
    "cortex-m55-dsp") to the real `-mcpu` value GCC expects."""
    canon = normalize_cpu(cpu)
    return _TARGET_CPU_CMAKE_OVERRIDES.get(canon, canon)


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
    if canon == "cortex-m55-dsp":
        return CpuProfile(
            cpu=canon,
            has_dsp=True,
            has_mve=False,
            capabilities=frozenset({"dsp", "fp32_execution", "fp16_execution"}),
        )
    if canon == "cortex-m4":
        return CpuProfile(
            cpu=canon,
            has_dsp=True,
            has_mve=False,
            capabilities=frozenset({"dsp", "fp32_execution"}),
        )
    if canon == "cortex-m0":
        # fp32_execution without fp16_execution: v6-m has no FPU at all, so f32 runs
        # through the soft-float ABI (-mfloat-abi=soft) and is a buildable
        # configuration -- ns-cmsis-nn's own CI runs cortex-m0 int-only -- while f16
        # has no kernel path here. This is the only soft-float leg in the matrix,
        # which makes it the only one that exercises __aeabi_* float conversion rather
        # than VCVT -- the two differ on non-finite operands.
        #
        # soft_float is a positive capability rather than an inferred negation of
        # has_dsp/has_mve because ns-cmsis-nn guards some behaviour on
        # `#if !defined(__ARM_FP)`: a descriptor whose contract exists only in the
        # soft-float compilation of a kernel requires it, and every hard-float target
        # capability-skips instead of asserting an unspecified result.
        return CpuProfile(
            cpu=canon,
            has_dsp=False,
            has_mve=False,
            capabilities=frozenset({"fp32_execution", "soft_float"}),
        )
    raise ValueError(f"Unsupported CPU target: {cpu}")


def known_capabilities() -> frozenset[str]:
    """Union of the capabilities any CPU profile declares."""
    names: set[str] = set()
    for canon in set(_CPU_ALIASES.values()):
        names |= get_cpu_profile(canon).capabilities
    return frozenset(names)


def missing_required_capabilities(cpu: str, required_capabilities: Sequence[str]) -> list[str]:
    profile = get_cpu_profile(cpu)
    known = known_capabilities()
    missing: list[str] = []
    for capability in required_capabilities:
        capability_name = str(capability).strip().lower()
        if not capability_name:
            continue
        # A name no profile declares is unsatisfiable everywhere, so a typo like
        # "soft-float" would skip the descriptor on the whole matrix and report as
        # covered. Fail generation instead.
        if capability_name not in known:
            raise ValueError(
                f"Unknown required capability {capability_name!r}; "
                f"known capabilities are {sorted(known)}"
            )
        if not profile.supports_capability(capability_name):
            missing.append(capability_name)
    return missing
