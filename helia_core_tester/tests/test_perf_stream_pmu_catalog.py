from __future__ import annotations

import json
from pathlib import Path

from helia_core_tester.perf_stream.pmu_catalog import (
    CPU_CYCLES_EVENT_ID,
    CPU_CYCLES_NAME,
    DEFAULT_SELECTIONS,
    GROUPS,
    counter_by_event_id,
    counter_by_name,
    counter_name_for_event_id,
    counters_in_group,
    default_selection,
    load_pmu_events,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_matches_the_transcribed_hpx_event_table() -> None:
    events = load_pmu_events()
    assert len(events) == 70
    assert {group: len(counters_in_group(group)) for group in GROUPS} == {"cpu": 21, "memory": 15, "mve": 34}
    raw = json.loads((PROJECT_ROOT / "assets" / "pmu" / "armv8m_pmu_events.json").read_text())
    assert [e.name for e in events] == [row["name"] for row in raw]
    assert all(isinstance(e.event_id, int) and 0 <= e.event_id <= 0xFFFF for e in events)
    assert len({e.event_id for e in events}) == 70 and len({e.name for e in events}) == 70


def test_lookups_by_name_and_event_id() -> None:
    cycles = counter_by_name(CPU_CYCLES_NAME)
    assert cycles is not None and cycles.event_id == CPU_CYCLES_EVENT_ID == 0x11 and cycles.group == "cpu"
    assert counter_by_event_id(0x0200).name == "ARM_PMU_MVE_INST_RETIRED"
    assert counter_by_event_id(0x0008).name == "ARM_PMU_INST_RETIRED"
    assert counter_by_event_id(0x0013).name == "ARM_PMU_MEM_ACCESS"
    assert counter_by_name("nope") is None and counter_by_event_id(0xBEEF) is None
    assert counter_name_for_event_id(0x0200) == "ARM_PMU_MVE_INST_RETIRED"
    assert counter_name_for_event_id(0xBEEF) == "event_0xbeef"


def test_default_selections_are_valid_and_grouped() -> None:
    assert set(DEFAULT_SELECTIONS) == set(GROUPS)
    assert default_selection() == {"cpu": "default", "memory": "default", "mve": "default"}
    for group, names in DEFAULT_SELECTIONS.items():
        assert len(names) == 4
        for name in names:
            counter = counter_by_name(name)
            assert counter is not None and counter.group == group, name
    assert DEFAULT_SELECTIONS["cpu"][0] == CPU_CYCLES_NAME
