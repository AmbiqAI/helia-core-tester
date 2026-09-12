"""Host-side sample normalization and PMU pass planning.

Adapted from helia-profiler concepts:
- counter catalog + pass planning mirrors helia_profiler.counters (catalog in pmu_catalog.py)
- min-duration auto-calibration mirrors helia_profiler.stages.plan_power._derive_inference_count
- robust median-based aggregation aligns with helia_profiler.capture.parser
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Iterable

from .pmu_catalog import (  # noqa: F401 -- CounterDescriptor is re-exported for callers
    CPU_CYCLES_EVENT_ID,
    CPU_CYCLES_NAME,
    DEFAULT_SELECTIONS,
    GROUPS,
    CounterDescriptor,
    counters_in_group,
)

# Cortex-M55 has 8 PMU event counters, each 16 bits wide. Chaining an even/odd pair
# (odd slot programmed with ARM_PMU_CHAIN) yields one 32-bit counter, so four chained
# counters per pass is the full width of the PMU.
MAX_COUNTERS_PER_PASS = 4


@dataclass(frozen=True)
class CounterPass:
    group: str
    pass_index: int
    counters: tuple[CounterDescriptor, ...]
    chained: bool = True

    @property
    def name(self) -> str:
        return f"{self.group}_{self.pass_index}"

    @property
    def slots_required(self) -> int:
        return (2 if self.chained else 1) * len(self.counters)


@dataclass(frozen=True)
class RawCounterValue:
    name: str
    event_id: int
    value: int
    overflow: bool = False
    supported: bool = True


@dataclass(frozen=True)
class RawSample:
    sample_index: int
    iterations: int
    cycles: int
    counters: tuple[RawCounterValue, ...]
    pass_name: str = "cpu_0"


@dataclass(frozen=True)
class NormalizedSample:
    sample_index: int
    iterations: int
    cycles: int
    cycles_per_invocation: float
    counters_per_invocation: dict[str, float]
    overflow: bool
    unsupported_counters: tuple[str, ...]
    pass_name: str


@dataclass(frozen=True)
class SampleStatistics:
    sample_count: int
    min_cycles: float
    median_cycles: float
    p90_cycles: float
    p99_cycles: float
    mad_cycles: float
    valid_for_regression: bool
    overflow_detected: bool
    unsupported_counters: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationResult:
    iterations: int
    estimated_cycles: int
    capped: bool


class StatefulKernelRestrictionError(ValueError):
    pass


class UnsupportedCounterError(ValueError):
    pass


def resolve_counter_selection(selection: dict[str, str | list[str]]) -> list[CounterDescriptor]:
    """Expand `{group: "all" | "default" | [names]}` into catalog descriptors.

    The dict is ordered: passes are planned in the order groups appear here, and the
    names within a group keep their catalog order for "all"/"default" and their given
    order for explicit lists. Duplicates collapse to the first mention.
    """
    resolved: list[CounterDescriptor] = []
    seen: set[str] = set()
    for group, requested in selection.items():
        available = {counter.name: counter for counter in counters_in_group(group)}
        if not available:
            raise UnsupportedCounterError(
                f"Unsupported counter group: {group} (expected one of: {', '.join(GROUPS)})"
            )
        if isinstance(requested, str):
            keyword = requested.strip().lower()
            if keyword == "default":
                names = list(DEFAULT_SELECTIONS[group])
            elif keyword == "all":
                names = list(available)
            else:
                names = [requested]
        else:
            names = list(requested)
        for name in names:
            if name not in available:
                raise UnsupportedCounterError(
                    f"Unsupported counter '{name}' for group '{group}'. "
                    f"Valid names: {', '.join(available)}"
                )
            if name not in seen:
                resolved.append(available[name])
                seen.add(name)
    return resolved


def plan_counter_passes(
    counters: list[CounterDescriptor],
    max_per_pass: int = MAX_COUNTERS_PER_PASS,
    *,
    chained: bool = True,
) -> list[CounterPass]:
    """Split counters into per-group passes of at most `max_per_pass` event counters.

    ARM_PMU_CPU_CYCLES never takes an event-counter slot -- firmware reports it from
    CCNTR as the first entry of every sample -- so it is stripped here. A selection
    that leaves nothing else still yields one empty `cpu_0` pass so the case is timed
    (DWT + CCNTR) at all.
    """
    by_group: dict[str, list[CounterDescriptor]] = {}
    for counter in counters:
        if counter.event_id == CPU_CYCLES_EVENT_ID:
            continue
        by_group.setdefault(counter.group, []).append(counter)
    passes: list[CounterPass] = []
    for group, group_counters in by_group.items():
        for index in range(0, len(group_counters), max_per_pass):
            batch = tuple(group_counters[index : index + max_per_pass])
            passes.append(CounterPass(group=group, pass_index=index // max_per_pass, counters=batch, chained=chained))
    if not passes:
        passes.append(CounterPass(group="cpu", pass_index=0, counters=(), chained=chained))
    return passes


def counter_passes_for_selection(selection: dict[str, str | list[str]], *, chained: bool = True) -> tuple[CounterPass, ...]:
    """`resolve_counter_selection` + `plan_counter_passes` in one call."""
    return tuple(plan_counter_passes(resolve_counter_selection(selection), chained=chained))


def counter_names_for_passes(passes: Iterable[CounterPass]) -> list[str]:
    """Every counter name the passes will report, CPU_CYCLES first."""
    names = [CPU_CYCLES_NAME]
    for counter_pass in passes:
        for counter in counter_pass.counters:
            if counter.name not in names:
                names.append(counter.name)
    return names


def auto_calibrate_iterations(
    *,
    base_cycles: int,
    min_cycles: int,
    max_iterations: int,
    stateful: bool,
) -> CalibrationResult:
    if stateful:
        if min_cycles > base_cycles:
            raise StatefulKernelRestrictionError(
                "Stateful kernels cannot auto-calibrate above one invocation per sample."
            )
        return CalibrationResult(iterations=1, estimated_cycles=base_cycles, capped=False)
    if base_cycles <= 0:
        return CalibrationResult(iterations=1, estimated_cycles=0, capped=False)
    iterations = max(1, math.ceil(min_cycles / base_cycles))
    capped = iterations > max_iterations
    iterations = min(iterations, max_iterations)
    return CalibrationResult(iterations=iterations, estimated_cycles=iterations * base_cycles, capped=capped)


def normalize_samples(samples: Iterable[RawSample]) -> list[NormalizedSample]:
    normalized: list[NormalizedSample] = []
    for sample in samples:
        iterations = max(1, sample.iterations)
        counters_per_invocation: dict[str, float] = {}
        unsupported: list[str] = []
        overflow = False
        for counter in sample.counters:
            if not counter.supported:
                unsupported.append(counter.name)
                continue
            counters_per_invocation[counter.name] = counter.value / iterations
            overflow = overflow or counter.overflow
        normalized.append(
            NormalizedSample(
                sample_index=sample.sample_index,
                iterations=iterations,
                cycles=sample.cycles,
                cycles_per_invocation=sample.cycles / iterations,
                counters_per_invocation=counters_per_invocation,
                overflow=overflow,
                unsupported_counters=tuple(sorted(unsupported)),
                pass_name=sample.pass_name,
            )
        )
    return normalized


def compute_sample_statistics(samples: Iterable[NormalizedSample]) -> SampleStatistics:
    """Aggregate per-sample cycle counts into summary statistics.

    Uses each sample's `cycles_per_invocation` (already normalized by `iterations` in
    `normalize_samples()`), not raw `cycles` -- every case bundle sends a fixed
    `iterations_per_sample` (never 0, so firmware's own on-device calibration never
    triggers), so aggregating raw `cycles` here previously reported every statistic at
    ~`iterations`x the true per-invocation cost.
    """
    materialized = list(samples)
    if not materialized:
        return SampleStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, ())
    cycle_values = [sample.cycles_per_invocation for sample in materialized]
    median_cycles = float(statistics.median(cycle_values))
    abs_deviation = [abs(value - median_cycles) for value in cycle_values]
    unsupported = sorted({name for sample in materialized for name in sample.unsupported_counters})
    overflow = any(sample.overflow for sample in materialized)
    return SampleStatistics(
        sample_count=len(materialized),
        min_cycles=min(cycle_values),
        median_cycles=median_cycles,
        p90_cycles=float(_percentile(cycle_values, 90.0)),
        p99_cycles=float(_percentile(cycle_values, 99.0)),
        mad_cycles=float(statistics.median(abs_deviation)),
        valid_for_regression=not overflow,
        overflow_detected=overflow,
        unsupported_counters=tuple(unsupported),
    )


def compute_counter_medians(samples: Iterable[NormalizedSample]) -> dict[str, float]:
    """Median per-invocation value of every supported counter across all samples.

    ARM_PMU_CPU_CYCLES appears in every pass, so its median spans every pass; other
    counters only appear in the pass that counted them.
    """
    values: dict[str, list[float]] = {}
    for sample in samples:
        for name, value in sample.counters_per_invocation.items():
            values.setdefault(name, []).append(value)
    return {name: float(statistics.median(series)) for name, series in values.items()}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)
