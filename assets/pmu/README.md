# Armv8.1-M PMU event catalog

`armv8m_pmu_events.json` lists the 70 Cortex-M55 PMU events the hardware host can
request (34 `mve`, 21 `cpu`, 15 `memory`). Each entry carries the CMSIS `ARM_PMU_*`
name, the architectural event id (hex string), the counter group used for
`--pmu-counters GROUP:SELECTION`, and a one-line description.

The file is a byte-for-byte copy of `data/armv8m_pmu_events.json` from the NSX module
[AmbiqAI/nsx-pmu-armv8m](https://github.com/AmbiqAI/nsx-pmu-armv8m) at tag `v0.2.0`
(commit `5725c065a0c3603132f1064ee2684d1fa8587c88`), the revision neuralspotx 0.8.1's
registry locks for the hardware app. heliaPROFILER syncs the same file the same way.
Loaded by `helia_core_tester/hardware/pmu_catalog.py`, which records the tag as
`PMU_MODULE_REF`.

To re-sync after a module bump, re-resolve the app's modules (a plain build reuses a
current `nsx.lock`), then copy the file:

```bash
uv run helia_core_tester hardware build --board apollo510_evb --update-dependencies
uv run python scripts/sync_pmu_catalog.py   # or pass the source path
```

and update the tag above and `PMU_MODULE_REF`. `test_hardware_pmu_catalog.py` checks
the tag against the neuralspotx registry and, when a synced app exists, the file
against the module copy.

`ARM_PMU_CPU_CYCLES` (0x0011) is special: firmware always reports it from the PMU
cycle counter (CCNTR) and it never occupies one of the eight 16-bit event-counter
slots, so selecting it is a no-op beyond the always-present first entry.
