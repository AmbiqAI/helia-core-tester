# Armv8.1-M PMU event catalog

`armv8m_pmu_events.json` lists the 70 Cortex-M55 PMU events the perf-stream host can
request (34 `mve`, 21 `cpu`, 15 `memory`). Each entry carries the CMSIS `ARM_PMU_*`
name, the architectural event id (hex string), the counter group used for
`--pmu-counters GROUP:SELECTION`, and a one-line description.

The file was transcribed verbatim from heliaPROFILER v0.1.6
(`src/helia_profiler/data/armv8m_pmu_events.json`) so counter names, ids and groups
match the hpx tooling. Loaded by `helia_core_tester/perf_stream/pmu_catalog.py`.

`ARM_PMU_CPU_CYCLES` (0x0011) is special: firmware always reports it from the PMU
cycle counter (CCNTR) and it never occupies one of the eight 16-bit event-counter
slots, so selecting it is a no-op beyond the always-present first entry.
