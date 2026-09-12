"""Armv8.1-M PMU event catalog (assets/pmu/armv8m_pmu_events.json).

The catalog is the host-side source of truth for counter names: firmware only ever
speaks in 16-bit event ids (SESSION_PLAN sends ids, SAMPLE_RESULT echoes them with an
empty name) and the host resolves names from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

_CATALOG_RELATIVE_PATH = Path("assets/pmu/armv8m_pmu_events.json")

CPU_CYCLES_NAME = "ARM_PMU_CPU_CYCLES"
CPU_CYCLES_EVENT_ID = 0x0011

GROUPS: Tuple[str, ...] = ("cpu", "memory", "mve")

# `GROUP:default` on the CLI. CPU_CYCLES is listed for the cpu group so the default
# selection reads naturally, but it is delivered from CCNTR in every pass and never
# occupies an event-counter slot (see plan_counter_passes in measurement.py).
DEFAULT_SELECTIONS: Dict[str, Tuple[str, ...]] = {
    "cpu": (CPU_CYCLES_NAME, "ARM_PMU_INST_RETIRED", "ARM_PMU_STALL_FRONTEND", "ARM_PMU_STALL_BACKEND"),
    "memory": ("ARM_PMU_MEM_ACCESS", "ARM_PMU_L1D_CACHE_REFILL", "ARM_PMU_BUS_ACCESS", "ARM_PMU_BUS_CYCLES"),
    "mve": ("ARM_PMU_MVE_INST_RETIRED", "ARM_PMU_MVE_INT_MAC_RETIRED", "ARM_PMU_MVE_LDST_RETIRED", "ARM_PMU_MVE_STALL"),
}


@dataclass(frozen=True)
class CounterDescriptor:
    name: str
    event_id: int
    group: str
    description: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_pmu_events(path: Optional[Path] = None) -> Tuple[CounterDescriptor, ...]:
    """Parse the catalog in file order. Names and event ids must both be unique."""
    path = path or (_repo_root() / _CATALOG_RELATIVE_PATH)
    rows = json.loads(path.read_text(encoding="utf-8"))
    descriptors = []
    seen_names: set = set()
    seen_ids: set = set()
    for row in rows:
        descriptor = CounterDescriptor(
            name=str(row["name"]),
            event_id=int(str(row["event_id"]), 16),
            group=str(row["group"]),
            description=str(row.get("description", "")),
        )
        if descriptor.group not in GROUPS:
            raise ValueError(f"{path}: {descriptor.name} has unknown group {descriptor.group!r}")
        if descriptor.name in seen_names:
            raise ValueError(f"{path}: duplicate counter name {descriptor.name}")
        if descriptor.event_id in seen_ids:
            raise ValueError(f"{path}: duplicate event id 0x{descriptor.event_id:04x}")
        seen_names.add(descriptor.name)
        seen_ids.add(descriptor.event_id)
        descriptors.append(descriptor)
    return tuple(descriptors)


def counters_in_group(group: str) -> Tuple[CounterDescriptor, ...]:
    return tuple(counter for counter in load_pmu_events() if counter.group == group)


def counter_by_name(name: str) -> Optional[CounterDescriptor]:
    for counter in load_pmu_events():
        if counter.name == name:
            return counter
    return None


def counter_by_event_id(event_id: int) -> Optional[CounterDescriptor]:
    for counter in load_pmu_events():
        if counter.event_id == event_id:
            return counter
    return None


def counter_name_for_event_id(event_id: int) -> str:
    """Catalog name for an event id, or a stable placeholder for ids the catalog does
    not know (firmware reports whatever it was asked to count)."""
    counter = counter_by_event_id(event_id)
    return counter.name if counter is not None else f"event_0x{event_id:04x}"


def default_selection() -> Dict[str, str]:
    """The `--pmu-counters` default: every group at its default selection."""
    return {group: "default" for group in GROUPS}
