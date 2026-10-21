"""Whether the target ns-cmsis-nn checkout can build a descriptor's case.

Every kernel symbol a case calls (test_ops._required_kernel_symbols) must be in the
checkout's signature export. Where the implementation manifest has a record for a symbol,
its ``since`` must not be after the checkout's library version (a contradiction, never a
skip), and its ``requires`` decide what the case meets on the target CPU: when they do not
hold and the record declines on a build cause, the entry is still callable and returns
ARM_CMSIS_NN_NO_IMPL_ERROR there, so a case expecting that status runs (``declined``) and any
other is skipped (``unmet``); without a build decline the implementation is absent from that
build and the case is skipped.

``requires`` tokens split by who decides them. The CPU profile decides the ISA and float API
tokens. ``autovectorize`` and ``single_rounding`` are compile options of one generated tree
(the coverage legs add ARM_MATH_AUTOVECTORIZE to the same sources), so generation does not
decide them; the rendered case declines under them (`autovectorize_declines`). A token
outside both sets fails closed: the manifest grew a feature this tester does not map.

A checkout without the export gives no answer here: every template renders its call from
the contract and refuses there, naming the ns-cmsis-nn it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from helia_core_tester.contract.ir import ContractSet
from helia_core_tester.core.cpu_targets import get_cpu_profile

BUILD_DECIDED_FEATURES = frozenset({"autovectorize", "single_rounding"})
CPU_DECIDED_FEATURES = frozenset({"mvei", "dsp", "mvef", "mve_f16", "f32_api", "f16_api"})


@dataclass(frozen=True)
class UnmetRequirement:
    symbol: str
    implementation: str
    unmet: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "implementation": self.implementation, "unmet": list(self.unmet)}


@dataclass(frozen=True)
class GateResult:
    missing_symbols: tuple[str, ...] = ()
    unmet: tuple[UnmetRequirement, ...] = ()
    contradictions: tuple[str, ...] = ()
    declined: tuple[UnmetRequirement, ...] = ()
    decided: bool = True

    @property
    def passes(self) -> bool:
        return not (self.missing_symbols or self.unmet or self.contradictions)


def cpu_features(cpu: str) -> frozenset[str]:
    """The manifest feature tokens the tester's build for ``cpu`` defines."""
    profile = get_cpu_profile(cpu)
    features = set()
    if profile.has_mve:
        features |= {"mvei", "mvef"}
        if profile.supports_capability("fp16_execution"):
            features.add("mve_f16")
    if profile.has_dsp:
        features.add("dsp")
    if profile.supports_capability("fp32_execution"):
        features.add("f32_api")
    if profile.supports_capability("fp16_execution"):
        features.add("f16_api")
    return frozenset(features)


def _version_key(version: str) -> tuple[int, ...]:
    parts = str(version).split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{version!r} is not a MAJOR.MINOR.PATCH version")
    return tuple(int(part) for part in parts)


def unmet_features(requires: Iterable[str], features: frozenset[str]) -> tuple[str, ...]:
    """The ``requires`` tokens the build does not satisfy; raises ValueError on a token this
    tester does not map to the CPU or the build."""
    unmet = []
    for token in sorted(requires):
        negated = token.startswith("!")
        feature = token[1:] if negated else token
        if feature in BUILD_DECIDED_FEATURES:
            continue
        if feature not in CPU_DECIDED_FEATURES:
            raise ValueError(f"manifest feature token {token!r} is not one helia-core-tester maps "
                             f"(CPU: {sorted(CPU_DECIDED_FEATURES)}, build: {sorted(BUILD_DECIDED_FEATURES)})")
        if (feature in features) == negated:
            unmet.append(token)
    return tuple(unmet)


NO_IMPL = "ARM_CMSIS_NN_NO_IMPL_ERROR"


def gate_kernel_symbols(contracts: ContractSet, symbols: Iterable[str], cpu: str,
                        *, features: Optional[frozenset[str]] = None,
                        expected_status: str = "ARM_CMSIS_NN_SUCCESS") -> GateResult:
    symbols = list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
    if not contracts.present:
        return GateResult(decided=False)
    target = cpu_features(cpu) if features is None else features
    missing = tuple(symbol for symbol in symbols if contracts.find(symbol) is None)
    unmet: list[UnmetRequirement] = []
    declined: list[UnmetRequirement] = []
    contradictions: list[str] = []
    for symbol in symbols:
        if symbol in missing:
            continue
        for impl in contracts.implementations_of(symbol):
            try:
                if contracts.library_version is None or \
                        _version_key(impl.since) > _version_key(contracts.library_version):
                    contradictions.append(
                        f"{symbol}: manifest record {impl.id} is since {impl.since}, after the checkout's "
                        f"library version {contracts.library_version} ({contracts.manifest_path})")
                    continue
                failed = unmet_features(impl.requires, target)
            except ValueError as error:
                contradictions.append(f"{symbol}: manifest record {impl.id}: {error}")
                continue
            if failed:
                requirement = UnmetRequirement(symbol=symbol, implementation=impl.id, unmet=failed)
                runs_to_decline = impl.declines_on_build and expected_status.strip() == NO_IMPL
                (declined if runs_to_decline else unmet).append(requirement)
    return GateResult(missing_symbols=missing, unmet=tuple(unmet), contradictions=tuple(contradictions),
                      declined=tuple(declined))
