# Streaming hardware performance testing design

## Scope and repository ownership

This design follows the existing repository boundaries.

- **helia-core-tester** owns host orchestration, plan/build metadata, case generation, correctness comparison, reports, and the streaming protocol host implementation.
- **ns-cmsis-nn** continues to own public kernel APIs and kernel implementations. The streaming path calls those APIs through runtime adapters; descriptor-specific arrays do not move into firmware.
- **neuralspotx** remains the build/flash/transport integration layer for Apollo targets. The streaming firmware is an NSX-built application/profile, not a custom side channel that bypasses NSX.
- **helia-profiler** owns reusable PMU/DWT, RTT transport, overflow handling, result provenance, and memory/section reporting concepts that the streaming pipeline should reuse where practical.

## Current baseline

Today helia-core-tester generates standalone C tests per descriptor and builds/runs one ELF per case. That model is good for correctness but scales poorly for hardware performance work because each case carries tensors in flash and requires separate build/flash/run orchestration.

## Target architecture

The default architecture is **one universal benchmark-server firmware per target profile**:

- board
- CPU
- toolchain
- optimization/LTO
- integer/F16/F32 feature set
- cache/memory placement policy
- protocol version

That firmware contains:

- the selected ns-cmsis-nn library build
- a kernel adapter registry/catalog
- the HCTP streaming protocol server
- transport glue (RTT first)
- PMU/DWT measurement support
- a bounded runtime arena
- compact catalog metadata

It does **not** contain descriptor-specific tensors or expected outputs.

## Universal size-probe sizing rule

Before adding sharding, build a real linked firmware for Apollo510/Cortex-M55 and measure:

1. integer only
2. integer + F32
3. integer + F16
4. integer + F16 + F32

The proof artifact is the final ELF plus map/bin/size/nm/objdump reports and a machine-readable `memory_report.json`.

## Host/target split

### Host responsibilities

helia-core-tester will:

- generate streamable case bundles directly from the same NumPy/TFLite data used for standalone tests
- build ordered execution plans
- validate firmware TARGET_INFO/catalog compatibility
- stream case metadata and blobs in bounded chunks
- reconstruct streamed outputs
- reuse existing comparison rules
- compute statistics from raw samples
- write result bundles and JUnit/CSV/JSON artifacts

### Target responsibilities

Firmware will:

- expose a versioned TARGET_INFO block (capabilities, PMU width, session limits)
- expose a versioned kernel catalog with stable numeric IDs
- request cases/blobs from the host (target-driven pull)
- bind streamed blobs into validated adapter metadata
- run one correctness invocation
- stream outputs back once per case
- run warmups + measured iterations only after the host's CORRECTNESS_ACK
- collect DWT cycles and PMU samples outside transfer/protocol work
- rewind the arena and request the next case without rebooting

## Wire protocol

The device wire format is **HCTP**, a binary little-endian framed protocol.

### Header goals

- fixed-size header
- explicit payload length
- header CRC and payload CRC
- session ID and sequence ID
- bounded payloads
- explicit state validation

### Directionality

The protocol is target-driven after plan load:

1. target sends `TARGET_INFO`
2. host sends `TARGET_INFO_ACK`
3. target sends `KERNEL_CATALOG` (one or more pages)
4. host sends `SESSION_PLAN`
5. target sends `REQUEST_CASE`
6. host sends `CASE_META`
7. target requests blobs chunk-by-chunk (`REQUEST_BLOB`)
8. host sends `BLOB_CHUNK`
9. target sends `CASE_READY`
10. host sends `RUN_CORRECTNESS`
11. target streams output + correctness result
12. host sends `CORRECTNESS_ACK`
13. host sends `RUN_PERFORMANCE` when allowed
14. target streams raw sample results
15. target sends `CASE_COMPLETE`
16. loop until `SESSION_COMPLETE`

### Messages (HCTP v3)

Protocol version 3 (`hctp.SUPPORTED_VERSION` / `HCTP_SUPPORTED_VERSION`); a peer on
another version is refused at the header. Message ids are compact and in protocol
order; every payload is encoded and decoded on the host by exactly one pair of
functions in `helia_core_tester/perf_stream/wire.py`, which the host session and the
fake target both use, and which the host-compiled C harnesses check against the
firmware byte for byte.

| id | message | direction | payload |
| --- | --- | --- | --- |
| 1 | `TARGET_INFO` | target -> host | build id, catalog hash, capabilities, PMU width, session limits (below) |
| 2 | `TARGET_INFO_ACK` | host -> target | empty |
| 3 | `KERNEL_CATALOG` | target -> host | one page of catalog entries; `HCTP_FLAG_MORE` on every non-final page |
| 4 | `SESSION_PLAN` | host -> target | timing plan, PMU passes, case list (below) |
| 5 | `REQUEST_CASE` | target -> host | `u16 case_index` |
| 6 | `CASE_META` | host -> target | case id, kernel id, comparison config, scalars, blob descriptors, scratch bytes |
| 7 | `REQUEST_BLOB` | target -> host | `u32 blob_id, u32 offset, u16 max_length` |
| 8 | `BLOB_CHUNK` | host -> target | `u32 blob_id, u32 offset, raw data` |
| 9 | `CASE_READY` | target -> host | `u32 blob_id, u32 bytes_received` |
| 10 | `RUN_CORRECTNESS` | host -> target | empty |
| 11 | `CORRECTNESS_RESULT` | target -> host | `i32 status` |
| 12 | `OUTPUT_BEGIN` | target -> host | `u32 offset (0), u32 length` |
| 13 | `OUTPUT_CHUNK` | target -> host | `u32 offset, u32 length, bytes` |
| 14 | `OUTPUT_END` | target -> host | `u32 length, u32 checksum` |
| 15 | `CORRECTNESS_ACK` | host -> target | `u8 passed` (informational) |
| 16 | `RUN_PERFORMANCE` | host -> target | empty |
| 17 | `SAMPLE_RESULT` | target -> host | one sample of one pass (below) |
| 18 | `CASE_COMPLETE` | target -> host | `text case_id, u8, u8, u32 workspace_used_bytes` |
| 19 | `SESSION_COMPLETE` | target -> host | `u16 case_count` |
| 20 | `ERROR` | target -> host | `text message` |

All integers are little-endian; `text` is `u16 length + UTF-8 bytes`; `raw` is
`u32 length + bytes`.

`TARGET_INFO` (target -> host): `text build_id`, 32-byte catalog SHA-256,
`u32 max_frame_payload`, `u32 runtime_arena_capacity`, `u8 transfer_mode`,
`u8 output_mode`, `text board_id`, `text target_cpu`, `u8 transport_kind`,
`u32 capability_flags`, `u8 pmu_counter_slots`, `u32 max_rx_payload`,
`u16 max_cases_per_session`, `u8 max_passes`.
`capability_flags` bit 6 is `HCT_CAP_PMU_ARMV8M`, set only when the firmware was
built for a core whose device header declares `__PMU_PRESENT == 1`;
`pmu_counter_slots` is `__PMU_NUM_EVENTCNT` (8 on Cortex-M55, 0 without a PMU).
`max_rx_payload` is the largest frame payload the target's fixed receive buffer can
hold (`HCT_SERVER_RX_BUFFER_BYTES - HCTP_HEADER_SIZE`, 2016 today);
`max_cases_per_session` and `max_passes` are the firmware's `HCT_SERVER_MAX_CASES`
(32) and `HCT_SERVER_MAX_PASSES` (16). The host keeps no copy of these limits: it
derives its batching (`session.TargetLimits`) from every session's `TARGET_INFO`,
cuts each batch so the plan stays within all three, and checks its chained-pair
planning rule (four counters per pass) against `pmu_counter_slots / 2`.

`SESSION_PLAN` (host -> target): `u16 case_count`, `u8 transfer_mode`, `u16 warmups`,
`u16 samples`, `u32 iterations_per_sample`, `u32 min_cycles`, `u32 max_iterations`,
`u8 pass_count` and per pass `text pass_name`, `u8 chained`,
`u8 counter_count`, `u16 event_id[counter_count]`, then per case `text case_id`,
`u32 kernel_id`. The firmware rejects `pass_count > max_passes`, `counter_count > 4`,
`case_count > max_cases_per_session` and (when it has a PMU) a pass needing more
slots than it advertised (`chained ? 2 * counter_count : counter_count`); event ids
are not validated against a list -- whatever the host asks for is programmed and
reported back.

`SAMPLE_RESULT` (target -> host, one per sample per pass): `u16 sample_index`,
`u32 iterations`, `u64 cycles`, `text pass_name`, `u8 counter_count`, then per
counter `text name`, `u16 event_id`, `u64 value`, `u8 overflow`, `u8 supported`.
`cycles` is the DWT `CYCCNT` delta around the timed loop and is kept as an
independent cross-check. The first counter entry is always `ARM_PMU_CPU_CYCLES`
(event `0x0011`) read from the PMU cycle counter `CCNTR`, with `overflow` = bit 31
of the PMU overflow status register; the remaining entries are the pass's event
counters in plan order. The firmware sends every `name` empty and the host resolves
names from `assets/pmu/armv8m_pmu_events.json` by event id (unknown ids become
`event_0x....`). On a DWT-only build the cycle entry is the DWT value and every
event counter comes back `supported = 0`.

### PMU passes, chained counters and overflow

The host plans counters into passes (`measurement.plan_counter_passes`): per counter
group, at most four event counters per pass, named `<group>_<n>`. Each pass reruns
the case's warmups and samples with its counters programmed, so a selection like
`mve:all` (34 events) costs nine passes. `ARM_PMU_CPU_CYCLES` never occupies an
event-counter slot -- it is reported from `CCNTR` in every pass -- so it is stripped
when planning and a cycles-only selection still yields one empty `cpu_0` pass.

Armv8.1-M event counters are 16 bits wide. Passes are chained by default: counter
`i` is programmed into slot `2i` and slot `2i+1` is programmed with `ARM_PMU_CHAIN`
(event `0x001E`), which increments on the even slot's overflow, so the pair reads as
`(high << 16) | low`, a 32-bit counter. Four chained counters use all eight slots.
Per sample the firmware disables the PMU, resets the event counters and `CCNTR`,
clears the overflow status (`ARM_PMU_Set_CNTR_OVS(0xFFFFFFFF)`), enables the pass's
slots plus `CCNTR`, runs the timed loop, disables the counters, reads the values and
`ARM_PMU_Get_CNTR_OVS()`, and clears the bits that were set. A chained counter's
overflow is the odd slot's bit; an unchained counter's is its own slot's bit. Any
overflow in any sample of a case sets `overflow_detected` and clears
`valid_for_regression` for that case in the result bundle; the DWT cycle statistics
are unaffected.

## Kernel catalog

Each target build emits a catalog entry per supported runtime adapter with:

- stable kernel ID
- canonical kernel name
- operator family
- API version
- supported dtypes/capabilities
- adapter schema version
- stateless/repeated-invocation safety
- mutation/reset behavior
- scratch sizing behavior
- optional route-trace support

The target `TARGET_INFO` includes a hash of the full catalog. The host refuses plans that reference missing IDs.

## Adapter model

Adapters are reusable public-API/operator-family bindings, not one function per descriptor. Descriptor data stays on the host; firmware rebuilds only when:

- a new adapter is introduced
- a schema changes
- a kernel/catalog ID mapping changes

Adapters must declare mutation/reset/repeatability semantics so performance loops stay correct for stateful kernels.

## Memory model

The firmware uses bounded memory only:

- no unbounded dynamic allocation
- aligned static protocol buffers
- bounded runtime arena
- explicit max blob size/rank/output size
- per-case validation before copy/use
- arena rewind after each case

Runtime RAM is sized for the **largest active case**, not all descriptors.

## Transport

Default transport is bidirectional SEGGER RTT. In the current live Apollo510 implementation, HCTP uses RTT channel 0 in both directions and does not multiplex console logs onto that channel. Transport abstraction mirrors the fake/loopback split so hardware and simulated paths share the same host session logic.

### Current RTT implementation status

- **Live-real on Apollo510:** the benchmark-server target now boots on real Apollo510 hardware, initializes the real `SEGGER_RTT` target sources from `neuralspotx/examples/coremark/src/rtt/`, emits TARGET_INFO over RTT, accepts host frames, requests blobs, and streams correctness/performance results back to the host.
- **Host implementation:** the host now has a real J-Link RTT transport using `pylink-square`. It resolves `_SEGGER_RTT` from the linked ELF and starts RTT with an explicit control-block address.
- **Observed limitation:** SEGGER CLI auto-discovery (`JLinkRTTLogger`) did not find the control block on this board/firmware, so the working hardware path currently uses explicit RTT block-address startup rather than auto-discovery.

## PMU/DWT reuse

helia-profiler already provides useful patterns to reuse:

- counter registry/group planning
- RTT control-block discovery and direct read/write
- overflow-aware result models
- section-size probing and memory reporting concepts

The streaming implementation aligns with those semantics instead of inventing
incompatible PMU naming or overflow behavior: the event catalog
(`assets/pmu/armv8m_pmu_events.json`) is transcribed from heliaPROFILER, the
`--pmu-counters GROUP:SELECTION` syntax is hpx's, and the firmware's per-sample
reset/clear-OVS/read/clear-set-bits sequence mirrors the hpx PMU profiler. The
firmware uses raw CMSIS `pmu_armv8.h` rather than the NSX PMU module.

## Timing boundaries

Performance measurements must exclude:

- host-side plan construction
- transport framing and CRC work
- target-side frame parsing and blob-copy protocol handling
- output streaming back to the host

Performance measurements may include:

- adapter argument validation that is intrinsic to a kernel invocation
- required scratch-buffer zeroing/prepare steps when they are part of the kernel contract
- repeated kernel invocations within one calibrated sample window

The host/fake-target path still simulates timing for hardware-independent tests (the fake target honours the 16/32-bit counter widths so overflow handling is testable). The live Apollo510 firmware path performs real DWT cycle capture and real PMU event capture (CCNTR plus up to four chained event counters per pass) after correctness passes.

## Adding a new adapter

1. Pick a stable public CMSIS-NN API and assign a stable numeric kernel ID.
2. Add a catalog entry in the firmware-side catalog (`benchmark_server_catalog.*`).
3. Define a compact adapter metadata schema:
   - scalar fields
   - streamed blobs
   - scratch requirements
   - repeated-invocation/stateful constraints
4. Add host-side case-bundle generation in `helia_core_tester/perf_stream/case_bundle.py`.
5. Add host-side fake-target support for the same operator so protocol tests stay hardware-independent.
6. Add firmware-side dispatch code that calls the real CMSIS-NN API.
7. Prove retention in the linked firmware image with `arm-none-eabi-nm`.
8. Add:
   - framing/transfer tests if the blob mix is new
   - end-to-end fake-target session tests
   - firmware byte-compat or host-harness tests when practical

Current examples:

- `arm_abs_s8`: full host fake-target slice + real firmware C dispatch + host C session harness
- `arm_convolve_s8`: full host fake-target slice + real firmware C dispatch compiled/linked into the benchmark-server image

## Firmware sizing methodology

Two sizing checkpoints now exist:

1. **Universal size probe** (`memory_report.build_size_probe`)
   - goal: prove the whole retained ns-cmsis-nn library fits for a target profile
   - artifact: `artifacts/perf_stream/size_probe/*/memory_report.json`
2. **Real benchmark-server firmware image** (`hardware memory-report`, `memory_report.generate_memory_report`)
   - goal: measure the actual streaming skeleton with protocol, RTT binding, catalog, session state, and adapters
   - artifact: `artifacts/perf_stream/benchmark_server/memory_report.json`, copied into every result bundle

Both reports come from one analysis (`helia_core_tester/perf_stream/memory_report.py`) of:

- the final linked ELF
- `arm-none-eabi-size`
- `arm-none-eabi-nm`
- `arm-none-eabi-objdump -h`
- the board's NSX linker script memory regions -- the SoC directory (`soc`) and the
  flash/RAM region names (`flash_region`, `ram_region`) come from the board's row in
  `assets/hardware_boards.yaml`

Reported percentages are computed against:

- `MCU_MRAM` for flash image bytes
- `MCU_TCM` for static TCM usage before heap

## Result bundle

The streaming run writes a portable bundle under:

`artifacts/reports/performance_stream/<session_id>/`

Key files:

- `session_manifest.json`
- `session_summary.json`
- `memory_report.json`
- `kernel_catalog.json`
- `cases.json`
- `case_summary.csv`
- `raw_samples.csv`
- `protocol_trace.jsonl`
- `correctness/<case_id>.json`
- `outputs/<case_id>.bin`
- `logs/host.log`
- `logs/target.log`
- `junit.xml`

## Real vs simulated status by layer

- **Host HCTP framing/CRC/session validation:** real and unit-tested in Python.
- **Host case generation/comparison/statistics:** real and unit-tested in Python.
- **Loopback/fake-target transport and end-to-end sessions:** simulated, but executed for real in tests.
- **Firmware build/profile sizing:** real cross-compiled Cortex-M55 Apollo510 artifacts.
- **Firmware TARGET_INFO/catalog frame construction:** real C implementation, byte-for-byte decoded by the Python HCTP decoder on the host.
- **Firmware session state machine:** real C implementation for `TARGET_INFO_ACK -> KERNEL_CATALOG -> SESSION_PLAN -> REQUEST_CASE -> CASE_META -> REQUEST_BLOB* -> CASE_READY -> RUN_CORRECTNESS -> CORRECTNESS_RESULT/OUTPUT_* -> RUN_PERFORMANCE -> SAMPLE_RESULT -> CASE_COMPLETE -> SESSION_COMPLETE`; executed both in a host-compiled C harness (`arm_abs_s8`) and on real Apollo510 hardware (`arm_abs_s8` + `arm_convolve_s8`).
- **Firmware RTT transport binding:** real compile-time integration against neuralspotx's SEGGER RTT target sources; exercised on real Apollo510 hardware.
- **Firmware kernel adapter dispatch:** real C adapters compiled and linked against the real CMSIS-NN APIs. `arm_abs_s8` and `arm_convolve_s8` are now session-executed on real Apollo510 hardware.
- **Real flash/run/RTT/PMU data capture:** verified on Apollo510 for the current two-operator vertical slice.

## Incremental implementation plan

1. write this design and size-probe tooling
2. build a universal linked firmware size prototype
3. add host HCTP framing + loopback/fake transports
4. add one simple stateless adapter vertical slice
5. add correctness streaming + output reconstruction
6. add DWT/PMU sample transport and reporting
7. add one complex adapter with scratch/multi-blob metadata
8. integrate NSX flash/RTT session handling
9. expand catalog/adapters/tests/docs

## Verified and unverified boundaries

Current remaining boundary:

- Corstone-300 FVP execution may still be blocked by missing Linux-only binaries
- RTT auto-discovery via SEGGER CLI tools remains unreliable on this board/firmware; explicit `_SEGGER_RTT` address startup is the working path

Loopback/fake-target validation remains the hardware-independent proof path; Apollo510 live RTT now covers the first real-hardware proof path.

## Hardware commands

All hardware work goes through the board-keyed `hardware` CLI group (`--board`
selects a row of `assets/hardware_boards.yaml`, which supplies the CPU, NSX board
name, SEGGER device name, SWD speed and the `build/perf_stream/<board>` build dir;
`--serial-no` is optional and falls back to `$HPX_JLINK_SERIAL`, then to the single
connected J-Link probe enumerated through pylink).

Cross-build the benchmark-server firmware for the board (fetches nsx-ambiq-sdk,
neuralspotx and the toolchain file on first use):

```bash
uv run helia_core_tester hardware build --board apollo510_evb -j
```

Flash through the NSX-generated SEGGER target -- skipped automatically when the
ELF's sha256 matches the last flash to the same probe (`--force` overrides):

```bash
uv run helia_core_tester hardware flash --board apollo510_evb
```

Stream the generated suite to the flashed firmware and write a result bundle
(or run generate -> build -> flash -> stream in one go with `hardware run`):

```bash
uv run helia_core_tester hardware stream --board apollo510_evb --suite both
uv run helia_core_tester hardware run --board apollo510_evb --precision fp16 --json
```

The two-kernel synthetic demo session (`arm_abs_s8` + `arm_convolve_s8`) is still
available as library code, `session_runner.run_demo_session()`, and is covered by
the fake-target tests; it is no longer a CLI command.

## Host modules

- `hctp.py`: framing (header, CRCs, sequence/session validation), `ByteWriter`/`ByteReader`.
- `wire.py`: the payload codec -- one encode/decode pair per message, shared by the
  host session and the fake target.
- `session.py`: `HostSession` (handshake, plan, per-case streaming) and `TargetLimits`,
  the batching limits derived from `TARGET_INFO`.
- `session_runner.py`: one RTT session per batch on a `BoardSpec`, case discovery
  from the generated-test tree, result-bundle writing.
- `hardware_pipeline.py`: generate -> build -> flash -> stream orchestration behind
  `hardware run` / `hardware stream`.
- `memory_report.py`: the flash/RAM report and the universal size probe.
- `fake_target.py`: the host-side target double the deterministic tests run against.
